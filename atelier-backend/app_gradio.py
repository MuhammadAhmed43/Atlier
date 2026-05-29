"""
ATELIER — Gradio demo wrapper.

Lightweight UI for the live demo that professors will actually click.
Reuses the same pipeline loading logic as server.py but skips the SSE plumbing
(Gradio's share-tunnel is what we need — a public URL pointing at this process).

Run locally:
    python app_gradio.py

The console will print a `https://xxxxx.gradio.live` URL that is publicly
reachable for ~72 hours, routing through Gradio's tunnel to your RTX 4070.
"""

import os
# Must set before importing transformers / diffusers (TF backend conflict).
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
os.environ.setdefault("USE_TORCH", "1")

import random
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
import torch
from diffusers import (
    ControlNetModel,
    StableDiffusionControlNetPipeline,
    UniPCMultistepScheduler,
)
from PIL import Image

# ───────────────────────── Config ─────────────────────────

ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
LORA_DIR = (
    PROJECT_ROOT / "Img-synthesis" / "assets" / "weights"
    / "Model_weights_for_inference"
    / "fashion_image_generation_30k+3k_denoising20"
)
LORA_WEIGHT = "pytorch_lora_weights.safetensors"
SAMPLES_DIR = ROOT / "samples"

BASE_MODEL = "runwayml/stable-diffusion-v1-5"
CONTROLNET_MODEL = "lllyasviel/control_v11p_sd15_canny"
TRAINING_RESOLUTION = 256
DIVISOR = 8
MIN_SIDE = 128

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

PRESETS = {
    "Navy wool blazer": "A navy wool blazer with gold buttons, structural shoulders, heavy fabric",
    "Black leather jacket": "A classic black leather jacket with silver zippers and fitted silhouette",
    "Red silk dress": "A bright red silk evening dress with elegant draping and flowing fabric",
    "Beige trench coat": "A structured beige trench coat with belted waist and storm flap",
    "Cashmere sweater": "A cozy beige cashmere sweater with ribbed texture",
    "Yellow summer dress": "A bright yellow summer sundress with flowing skirt and bright daylight",
}

# ───────────────────────── Pipeline load ─────────────────────────

print(f"[atelier-demo] device={DEVICE} dtype={DTYPE}")
print(f"[atelier-demo] loading ControlNet …")
controlnet = ControlNetModel.from_pretrained(CONTROLNET_MODEL, torch_dtype=DTYPE)

print(f"[atelier-demo] loading SD pipeline …")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    BASE_MODEL,
    controlnet=controlnet,
    torch_dtype=DTYPE,
    safety_checker=None,
    requires_safety_checker=False,
).to(DEVICE)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
try:
    pipe.enable_attention_slicing()
except Exception:
    pass

print(f"[atelier-demo] loading LoRA from {LORA_DIR}")
pipe.load_lora_weights(str(LORA_DIR), weight_name=LORA_WEIGHT)
print(f"[atelier-demo] pipeline ready ✓")


# ───────────────────────── Helpers ─────────────────────────

def aspect_size(w: int, h: int) -> tuple[int, int]:
    if w >= h:
        new_w = TRAINING_RESOLUTION
        new_h = max(MIN_SIDE, round(h * TRAINING_RESOLUTION / w / DIVISOR) * DIVISOR)
    else:
        new_h = TRAINING_RESOLUTION
        new_w = max(MIN_SIDE, round(w * TRAINING_RESOLUTION / h / DIVISOR) * DIVISOR)
    return new_w, new_h


def canny_edge(img: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    edges = cv2.Canny(arr, low, high)
    return Image.fromarray(np.stack([edges] * 3, axis=-1))


def list_sample_paths() -> list[str]:
    if not SAMPLES_DIR.is_dir():
        return []
    return sorted([str(p) for p in SAMPLES_DIR.iterdir()
                   if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}])


def generate(reference: Image.Image, prompt: str, steps: int, guidance: float, seed: int):
    if reference is None:
        raise gr.Error("Pick a reference image (upload, paste, or click a sample).")
    if not prompt.strip():
        raise gr.Error("Enter a prompt.")

    w, h = aspect_size(*reference.size)
    ref = reference.convert("RGB").resize((w, h), Image.LANCZOS)
    edges = canny_edge(ref)

    if seed == -1:
        seed = random.randint(0, 2_147_483_647)
    generator = torch.Generator(device=DEVICE).manual_seed(int(seed))

    with torch.inference_mode():
        out = pipe(
            prompt=prompt,
            negative_prompt="blurry, low quality, distorted, watermark, text",
            image=edges,
            width=w,
            height=h,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            generator=generator,
        )

    return ref, edges, out.images[0], f"seed={seed}  ·  {w}×{h}"


def use_preset(name: str) -> str:
    return PRESETS.get(name, "")


# ───────────────────────── UI ─────────────────────────

DESCRIPTION = """
# ATELIER — Controllable Fashion Image Synthesis
Stable Diffusion v1.5 + ControlNet (Canny) + LoRA fine-tuned on FashionGen.
Upload (or pick) a reference photo, describe a garment, and the model preserves
the silhouette while changing the design. Aspect ratio is preserved end-to-end.

Code: [github.com/MuhammadAhmed43/Atelier](https://github.com/MuhammadAhmed43/Atelier)
"""

with gr.Blocks(title="ATELIER — Controllable Fashion Synthesis", theme=gr.themes.Soft()) as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            reference = gr.Image(label="Reference image", type="pil", height=320, sources=["upload", "clipboard"])
            samples = list_sample_paths()
            if samples:
                gr.Examples(examples=samples[:8], inputs=reference, label="Reference samples")
            preset_dd = gr.Dropdown(choices=list(PRESETS.keys()), value=None, label="Preset prompts")
            prompt = gr.Textbox(label="Prompt", lines=3, placeholder="A navy wool blazer with gold buttons…")
            with gr.Row():
                steps = gr.Slider(10, 40, value=20, step=1, label="Steps")
                guidance = gr.Slider(1.0, 12.0, value=7.5, step=0.5, label="Guidance")
            seed = gr.Number(value=-1, label="Seed (-1 random)", precision=0)
            go = gr.Button("Generate", variant="primary", size="lg")

        with gr.Column(scale=1):
            out_ref = gr.Image(label="Reference (cropped)", height=260)
            out_edges = gr.Image(label="Extracted edges (Canny)", height=260)
            out_image = gr.Image(label="Generated", height=320)
            meta = gr.Textbox(label="Run info", interactive=False)

    preset_dd.change(use_preset, inputs=preset_dd, outputs=prompt)
    go.click(generate, inputs=[reference, prompt, steps, guidance, seed],
             outputs=[out_ref, out_edges, out_image, meta])

    gr.Markdown(
        "Built by Syed Muhammed Ahmed — NUST CS, Year 3. "
        "Read more in the [project repository](https://github.com/MuhammadAhmed43/Atelier)."
    )

if __name__ == "__main__":
    demo.queue(max_size=4).launch(
        server_name="0.0.0.0",
        share=True,             # public gradio.live tunnel
        show_error=True,
    )
