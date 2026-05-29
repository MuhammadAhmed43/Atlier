"""
ATELIER — FastAPI backend for controllable fashion image synthesis.

Endpoints:
  GET  /health                  → model status
  GET  /samples                 → list reference image ids
  GET  /samples/{sample_id}     → raw PNG of a reference sample
  POST /generate                → final image only (JSON in, JSON out with base64 image)
  GET  /generate/stream         → SSE: per-step previews + final image
"""

import os
# Must set BEFORE importing transformers / diffusers so they skip TF / Flax backends.
# This sidesteps the system-wide TensorFlow installation's protobuf version mismatch.
os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TORCH", "1")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "warning")

import asyncio
import base64
import io
import json
import logging
import random
import sys
import time
import traceback
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("atelier")

import cv2
import numpy as np
import torch
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from PIL import Image
from pydantic import BaseModel, Field
from sse_starlette.sse import EventSourceResponse

# ─────────────────────────── Configuration ───────────────────────────

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent

# Default LoRA: 30k+3k denoising 20 (balanced).
DEFAULT_LORA_DIR = (
    PROJECT_ROOT
    / "Img-synthesis" / "assets" / "weights"
    / "Model_weights_for_inference"
    / "fashion_image_generation_30k+3k_denoising20"
)
LORA_WEIGHT_FILE = "pytorch_lora_weights.safetensors"

SAMPLES_DIR = ROOT / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_canny"
RESOLUTION = 256

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

# Stable Diffusion v1.5 + ControlNet require dimensions divisible by 8.
# We keep the longer side at TRAINING_RESOLUTION (the LoRA was trained at 256²),
# and scale the shorter side to preserve the input's aspect ratio.
TRAINING_RESOLUTION = 256
SIZE_DIVISOR = 8
MIN_SIDE = 128

# ─────────────────────────── Global state ───────────────────────────

class State:
    pipe = None
    lora_loaded: Optional[str] = None
    loading: bool = False
    error: Optional[str] = None
    gen_lock = asyncio.Lock()  # one generation at a time (single GPU)


state = State()


# ─────────────────────────── Helpers ───────────────────────────

def canny_edge(image: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    arr = np.array(image.convert("RGB"))
    edges = cv2.Canny(arr, low, high)
    return Image.fromarray(np.stack([edges] * 3, axis=-1))


def pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def latents_to_pil(pipe, latents: torch.Tensor) -> Image.Image:
    """Decode the *current* latents through the VAE to get a noisy preview.
    Costs ~one VAE forward per call — only invoke for the steps you want to stream.
    """
    with torch.no_grad():
        latents = latents.to(pipe.vae.dtype)
        decoded = pipe.vae.decode(latents / pipe.vae.config.scaling_factor).sample
        decoded = (decoded.clamp(-1, 1) + 1) / 2
        decoded = (decoded.cpu().permute(0, 2, 3, 1).float().numpy() * 255).round().astype(np.uint8)
    return Image.fromarray(decoded[0])


def load_pipeline(lora_dir: Path) -> None:
    from diffusers import (
        ControlNetModel,
        StableDiffusionControlNetPipeline,
        UniPCMultistepScheduler,
    )

    state.loading = True
    state.error = None
    try:
        log.info(f"loading ControlNet on {DEVICE} ({DTYPE})")
        controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL, torch_dtype=DTYPE)

        log.info(f"loading SD pipeline {BASE_MODEL}")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            BASE_MODEL,
            controlnet=controlnet,
            torch_dtype=DTYPE,
            safety_checker=None,
            requires_safety_checker=False,
        ).to(DEVICE)
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

        # 8GB VRAM is tight — slice attention to be safe.
        try:
            pipe.enable_attention_slicing()
        except Exception:
            pass

        weight_path = lora_dir / LORA_WEIGHT_FILE
        if not weight_path.is_file():
            raise FileNotFoundError(f"LoRA weights not found at {weight_path}")
        log.info(f"loading LoRA from {lora_dir}")
        pipe.load_lora_weights(str(lora_dir), weight_name=LORA_WEIGHT_FILE)

        state.pipe = pipe
        state.lora_loaded = lora_dir.name
        log.info("pipeline ready")
    except Exception as e:
        state.error = f"{type(e).__name__}: {e}"
        log.exception(f"load failed: {state.error}")
        raise
    finally:
        state.loading = False


# ─────────────────────────── App lifecycle ───────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    if not DEFAULT_LORA_DIR.is_dir():
        log.warning(f"default LoRA dir missing: {DEFAULT_LORA_DIR}")
        state.error = f"LoRA dir missing: {DEFAULT_LORA_DIR}"
    else:
        try:
            load_pipeline(DEFAULT_LORA_DIR)
        except Exception as e:
            state.error = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
            log.exception("lifespan load_pipeline failed")
    yield


app = FastAPI(title="Atelier API", version="0.1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Mount the Vite-built React assets if the bundle exists.
_FRONTEND_DIST = PROJECT_ROOT / "atelier-frontend" / "dist"
if (_FRONTEND_DIST / "assets").is_dir():
    from fastapi.staticfiles import StaticFiles
    app.mount(
        "/assets",
        StaticFiles(directory=str(_FRONTEND_DIST / "assets")),
        name="frontend-assets",
    )

    @app.get("/favicon.ico")
    async def _favicon_ico():
        for name in ("favicon.ico", "favicon.svg", "vite.svg"):
            p = _FRONTEND_DIST / name
            if p.is_file():
                return FileResponse(p)
        raise HTTPException(404)

    @app.get("/vite.svg")
    async def _vite_svg():
        p = _FRONTEND_DIST / "vite.svg"
        if p.is_file():
            return FileResponse(p)
        raise HTTPException(404)


# ─────────────────────────── Models ───────────────────────────

class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=500)
    negative_prompt: Optional[str] = (
        "blurry, low quality, distorted, deformed, watermark, text"
    )
    sample_id: Optional[str] = Field(None, description="reference sample id (filename without ext)")
    image_b64: Optional[str] = Field(None, description="user-uploaded reference (base64 PNG/JPEG)")
    steps: int = Field(20, ge=10, le=50)
    guidance: float = Field(7.5, ge=1.0, le=15.0)
    canny_low: int = Field(100, ge=50, le=200)
    canny_high: int = Field(200, ge=100, le=300)
    seed: int = Field(-1, description="-1 for random")


# ─────────────────────────── Endpoints ───────────────────────────

@app.get("/")
async def root():
    """Serve the production React app if built, else the dev test page.
    Looks for the Vite-built bundle at ../atelier-frontend/dist/index.html.
    """
    react_index = PROJECT_ROOT / "atelier-frontend" / "dist" / "index.html"
    if react_index.is_file():
        return FileResponse(react_index, media_type="text/html")
    test_page = ROOT / "test.html"
    if test_page.is_file():
        return FileResponse(test_page, media_type="text/html")
    return JSONResponse({
        "service": "atelier-backend",
        "endpoints": ["/health", "/samples", "/samples/{filename}", "/generate", "/generate/stream"],
    })


@app.get("/health")
async def health():
    return {
        "device": DEVICE,
        "dtype": str(DTYPE),
        "pipeline_loaded": state.pipe is not None,
        "loading": state.loading,
        "lora": state.lora_loaded,
        "error": state.error,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "vram_total_gb": (
            round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2)
            if torch.cuda.is_available() else None
        ),
    }


@app.get("/samples")
async def list_samples():
    items = []
    for p in sorted(SAMPLES_DIR.iterdir()):
        if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}:
            items.append({"id": p.stem, "url": f"/samples/{p.name}"})
    return {"count": len(items), "samples": items}


@app.get("/samples/{filename}")
async def get_sample(filename: str):
    p = SAMPLES_DIR / filename
    if not p.is_file():
        raise HTTPException(404, "sample not found")
    return FileResponse(p)


def _aspect_preserving_size(w: int, h: int) -> tuple[int, int]:
    """Compute (width, height) preserving aspect ratio, longer side at TRAINING_RESOLUTION,
    both rounded to a multiple of SIZE_DIVISOR (SD 1.5 requirement).
    """
    if w >= h:
        new_w = TRAINING_RESOLUTION
        new_h = max(MIN_SIDE, round(h * TRAINING_RESOLUTION / w / SIZE_DIVISOR) * SIZE_DIVISOR)
    else:
        new_h = TRAINING_RESOLUTION
        new_w = max(MIN_SIDE, round(w * TRAINING_RESOLUTION / h / SIZE_DIVISOR) * SIZE_DIVISOR)
    return new_w, new_h


def _load_reference(req: GenerateRequest) -> Image.Image:
    """Load and resize the reference image while preserving its aspect ratio."""
    if req.image_b64:
        try:
            data = base64.b64decode(req.image_b64)
            img = Image.open(io.BytesIO(data)).convert("RGB")
        except Exception as e:
            raise HTTPException(400, f"bad image_b64: {e}")
    elif req.sample_id:
        match = next(
            (p for p in SAMPLES_DIR.iterdir() if p.stem == req.sample_id), None
        )
        if not match:
            raise HTTPException(404, f"sample '{req.sample_id}' not found")
        img = Image.open(match).convert("RGB")
    else:
        raise HTTPException(400, "provide either sample_id or image_b64")

    target_w, target_h = _aspect_preserving_size(*img.size)
    return img.resize((target_w, target_h), Image.LANCZOS)


@app.post("/generate")
async def generate(req: GenerateRequest):
    if state.pipe is None:
        raise HTTPException(503, f"pipeline not ready ({state.error or 'loading'})")
    ref = _load_reference(req)
    edges = canny_edge(ref, req.canny_low, req.canny_high)
    seed = req.seed if req.seed != -1 else random.randint(0, 2_147_483_647)
    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))
    W, H = ref.size

    async with state.gen_lock:
        with torch.inference_mode():
            out = state.pipe(
                prompt=req.prompt,
                negative_prompt=req.negative_prompt or None,
                image=edges,
                width=W,
                height=H,
                num_inference_steps=req.steps,
                guidance_scale=req.guidance,
                generator=generator,
            )

    return {
        "seed": seed,
        "width": W,
        "height": H,
        "aspect": round(W / H, 4),
        "reference_b64": pil_to_b64(ref),
        "edges_b64": pil_to_b64(edges),
        "image_b64": pil_to_b64(out.images[0]),
    }


@app.get("/generate/stream")
async def generate_stream(
    prompt: str = Query(..., min_length=1, max_length=500),
    sample_id: Optional[str] = None,
    image_b64: Optional[str] = None,
    negative_prompt: Optional[str] = "blurry, low quality, distorted, deformed, watermark, text",
    steps: int = Query(20, ge=10, le=50),
    guidance: float = Query(7.5, ge=1.0, le=15.0),
    canny_low: int = Query(100, ge=50, le=200),
    canny_high: int = Query(200, ge=100, le=300),
    seed: int = -1,
    preview_every: int = Query(2, ge=1, le=10, description="decode latents every N steps"),
):
    """Server-Sent Events stream of generation progress.

    Event types emitted (data is JSON):
      - ready      : reference + edges base64
      - step       : {index, total, preview_b64?}  (preview only on stride)
      - done       : {image_b64, seed, elapsed_s}
      - error      : {message}
    """
    if state.pipe is None:
        raise HTTPException(503, f"pipeline not ready ({state.error or 'loading'})")

    req = GenerateRequest(
        prompt=prompt,
        negative_prompt=negative_prompt,
        sample_id=sample_id,
        image_b64=image_b64,
        steps=steps,
        guidance=guidance,
        canny_low=canny_low,
        canny_high=canny_high,
        seed=seed,
    )
    ref = _load_reference(req)
    edges = canny_edge(ref, req.canny_low, req.canny_high)
    used_seed = req.seed if req.seed != -1 else random.randint(0, 2_147_483_647)
    generator = torch.Generator(device=DEVICE).manual_seed(int(used_seed))
    W, H = ref.size

    # Bridge the synchronous diffusers callback to the async SSE stream
    # via a thread-safe queue + a background thread running the pipeline.
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()

    def put(event: str, data: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, (event, data))

    def diffusers_callback(pipe_, step_index: int, timestep, callback_kwargs):
        latents = callback_kwargs.get("latents")
        emit_preview = (
            (step_index + 1) % preview_every == 0
            or step_index == req.steps - 1
            or step_index == 0
        )
        payload = {"index": step_index + 1, "total": req.steps}
        if emit_preview and latents is not None:
            try:
                payload["preview_b64"] = pil_to_b64(latents_to_pil(pipe_, latents), "JPEG")
            except Exception as e:
                payload["preview_error"] = str(e)
        put("step", payload)
        return callback_kwargs

    def run_pipeline():
        t0 = time.time()
        try:
            with torch.inference_mode():
                out = state.pipe(
                    prompt=req.prompt,
                    negative_prompt=req.negative_prompt or None,
                    image=edges,
                    width=W,
                    height=H,
                    num_inference_steps=req.steps,
                    guidance_scale=req.guidance,
                    generator=generator,
                    callback_on_step_end=diffusers_callback,
                    callback_on_step_end_tensor_inputs=["latents"],
                )
            put("done", {
                "image_b64": pil_to_b64(out.images[0]),
                "seed": used_seed,
                "elapsed_s": round(time.time() - t0, 2),
            })
        except Exception as e:
            put("error", {"message": f"{type(e).__name__}: {e}"})
        finally:
            put("__close__", {})

    async def event_generator():
        # ready frame first
        yield {"event": "ready", "data": json.dumps({
            "reference_b64": pil_to_b64(ref),
            "edges_b64": pil_to_b64(edges),
            "total_steps": req.steps,
            "seed_used": used_seed,
            "width": W,
            "height": H,
            "aspect": round(W / H, 4),
        })}

        async with state.gen_lock:
            # Run pipeline in a background thread so the event loop stays responsive.
            asyncio.create_task(asyncio.to_thread(run_pipeline))
            while True:
                event, data = await queue.get()
                if event == "__close__":
                    return
                yield {"event": event, "data": json.dumps(data)}

    return EventSourceResponse(event_generator())


# ─────────────────────────── Dev entrypoint ───────────────────────────

if __name__ == "__main__":
    import uvicorn

    # Force UTF-8 stdout on Windows so emojis in logs don't crash.
    if sys.platform == "win32":
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")

    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=int(os.environ.get("ATELIER_PORT", "8001")),
        reload=False,
        log_level="info",
    )
