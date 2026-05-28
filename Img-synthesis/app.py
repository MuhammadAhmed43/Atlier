"""
Controllable Fashion Image Synthesis — Gradio Web Application
=============================================================
A professional, standalone UI for generating fashion images using
Stable Diffusion + ControlNet + LoRA.

Usage:
    python app.py --weights /path/to/lora/weights/folder

The app will launch a local web server with a shareable link.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

import argparse
import os
import random
import numpy as np
import cv2
from PIL import Image

import torch
from diffusers import (
    StableDiffusionControlNetPipeline,
    ControlNetModel,
    UniPCMultistepScheduler,
)

import gradio as gr

# ──────────────────────────── Argument Parser ────────────────────────────

parser = argparse.ArgumentParser(description="Fashion Image Synthesis App")
parser.add_argument(
    "--weights",
    type=str,
    default=None,
    help="Path to folder containing pytorch_lora_weights.safetensors",
)
parser.add_argument("--share", action="store_true", help="Create a public Gradio link")
parser.add_argument(
    "--port", type=int, default=7860, help="Port for the local server"
)
args = parser.parse_args()

# ──────────────────────────── Auto-Detect Weights ────────────────────────

LORA_WEIGHTS_PATH = args.weights

if LORA_WEIGHTS_PATH is None:
    # Try common locations
    candidates = [
        os.path.join(os.path.dirname(__file__), "assets", "weights"),
        os.path.join(
            os.path.dirname(__file__),
            "controllable-fashion-image-synthesis-project-AMD",
            "working",
            "fashion_lora_output",
        ),
        "/kaggle/working/fashion_lora_output",
    ]
    for c in candidates:
        safetensor = os.path.join(c, "pytorch_lora_weights.safetensors")
        if os.path.isfile(safetensor):
            LORA_WEIGHTS_PATH = c
            break

    # Search recursively from script directory
    if LORA_WEIGHTS_PATH is None:
        for root, dirs, files in os.walk(os.path.dirname(__file__)):
            if "pytorch_lora_weights.safetensors" in files:
                LORA_WEIGHTS_PATH = root
                break

if LORA_WEIGHTS_PATH is None:
    print(
        "⚠️  Could not auto-detect LoRA weights. "
        "Use --weights /path/to/folder or place them in assets/weights/"
    )
    print("    The app will still load but generation will fail until weights are set.")

# ──────────────────────────── Load Pipeline ──────────────────────────────

print("⚙️  Loading ControlNet …")
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if device == "cuda" else torch.float32

controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/control_v11p_sd15_canny", torch_dtype=dtype
)

print("⚙️  Loading Stable Diffusion pipeline …")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype=dtype,
    safety_checker=None,
).to(device)
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)

if LORA_WEIGHTS_PATH:
    print(f"⚙️  Loading LoRA weights from: {LORA_WEIGHTS_PATH}")
    pipe.load_lora_weights(
        LORA_WEIGHTS_PATH, weight_name="pytorch_lora_weights.safetensors"
    )
    print("✅  LoRA weights loaded!")
else:
    print("⚠️  Running without LoRA weights (baseline mode)")

print(f"✅  Pipeline ready on {device.upper()}")

# ──────────────────────────── Helper Functions ───────────────────────────


def extract_canny(image: Image.Image, low: int = 100, high: int = 200) -> Image.Image:
    """Apply Canny edge detection to extract structural map."""
    img_array = np.array(image.convert("RGB"))
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
    edges = cv2.Canny(img_array, low, high)
    edges_rgb = np.stack([edges] * 3, axis=-1)
    return Image.fromarray(edges_rgb)


def generate(
    input_image: Image.Image,
    prompt: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    canny_low: int,
    canny_high: int,
    seed: int,
):
    """Core generation function."""
    if input_image is None:
        raise gr.Error("Please upload a reference image first!")
    if not prompt.strip():
        raise gr.Error("Please enter a text prompt!")

    # Resize to 256x256
    input_image = input_image.convert("RGB").resize((256, 256), Image.LANCZOS)

    # Extract edges
    edge_image = extract_canny(input_image, canny_low, canny_high)

    # Seed
    if seed == -1:
        seed = random.randint(0, 2147483647)
    generator = torch.Generator(device=device).manual_seed(int(seed))

    # Generate
    with torch.inference_mode():
        output = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt if negative_prompt.strip() else None,
            image=edge_image,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            generator=generator,
        )

    generated_image = output.images[0]
    return edge_image, generated_image, f"Seed: {seed}"


def batch_generate(
    input_image: Image.Image,
    prompts_text: str,
    negative_prompt: str,
    steps: int,
    guidance: float,
    canny_low: int,
    canny_high: int,
    seed: int,
):
    """Generate multiple images from newline-separated prompts."""
    if input_image is None:
        raise gr.Error("Please upload a reference image first!")

    prompts = [p.strip() for p in prompts_text.strip().split("\n") if p.strip()]
    if not prompts:
        raise gr.Error("Please enter at least one prompt (one per line)!")
    prompts = prompts[:8]  # Cap at 8

    input_image = input_image.convert("RGB").resize((256, 256), Image.LANCZOS)
    edge_image = extract_canny(input_image, canny_low, canny_high)

    results = []
    for i, prompt in enumerate(prompts):
        gen_seed = seed + i if seed != -1 else random.randint(0, 2147483647)
        generator = torch.Generator(device=device).manual_seed(int(gen_seed))

        with torch.inference_mode():
            output = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt if negative_prompt.strip() else None,
                image=edge_image,
                num_inference_steps=int(steps),
                guidance_scale=float(guidance),
                generator=generator,
            )
        results.append((output.images[0], prompt))

    return edge_image, results


# ──────────────────────────── Preset Prompts ─────────────────────────────

PRESETS = {
    "🧥 Black Leather Jacket": "A classic black leather jacket with silver zippers and fitted silhouette",
    "👔 White Linen Shirt": "A crisp white linen shirt with relaxed fit and clean lines",
    "👗 Red Silk Dress": "A bright red silk evening dress with elegant draping and flowing fabric",
    "🧥 Navy Blazer": "A tailored navy blue blazer with gold buttons and sharp lapels",
    "👕 Striped T-Shirt": "A classic navy and white striped cotton t-shirt with crew neck",
    "🧥 Brown Bomber Jacket": "A brown leather bomber jacket with shearling collar and ribbed cuffs",
    "👗 Little Black Dress": "A sophisticated black fitted dress with clean lines and minimal design",
    "🧶 Cashmere Sweater": "A cozy beige cashmere sweater with ribbed texture and soft finish",
    "👖 Dark Denim Jeans": "Classic dark indigo straight-leg denim jeans with clean stitching",
    "👗 Velvet Cocktail Dress": "A deep burgundy velvet cocktail dress with subtle sheen",
}


def load_preset(preset_name):
    """Return the prompt text for a selected preset."""
    return PRESETS.get(preset_name, "")


# ──────────────────────────── Custom CSS ─────────────────────────────────

CUSTOM_CSS = """
/* ─── Global Theme ─── */
.gradio-container {
    max-width: 1200px !important;
    margin: auto;
}

/* ─── Header ─── */
#app-title {
    text-align: center;
    padding: 24px 16px;
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
    border-radius: 16px;
    margin-bottom: 16px;
    border: 1px solid rgba(255, 255, 255, 0.08);
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
}

#app-title h1 {
    background: linear-gradient(135deg, #e2e2e2, #c084fc, #818cf8);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0 0 6px 0;
    letter-spacing: -0.5px;
}

#app-title p {
    color: #94a3b8;
    font-size: 0.95rem;
    margin: 0;
    font-weight: 400;
}

/* ─── Section Cards ─── */
.panel-card {
    border: 1px solid rgba(255, 255, 255, 0.06) !important;
    border-radius: 12px !important;
    background: rgba(30, 30, 46, 0.5) !important;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.15) !important;
}

/* ─── Generate Button ─── */
#generate-btn {
    background: linear-gradient(135deg, #7c3aed, #6366f1) !important;
    border: none !important;
    color: white !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    padding: 14px 32px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(99, 102, 241, 0.4) !important;
    transition: all 0.3s ease !important;
    letter-spacing: 0.3px;
}

#generate-btn:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(99, 102, 241, 0.5) !important;
}

#batch-btn {
    background: linear-gradient(135deg, #059669, #10b981) !important;
    border: none !important;
    color: white !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    padding: 14px 32px !important;
    border-radius: 12px !important;
    box-shadow: 0 4px 20px rgba(16, 185, 129, 0.4) !important;
}

/* ─── Info Badge ─── */
.info-badge {
    background: rgba(99, 102, 241, 0.1);
    border: 1px solid rgba(99, 102, 241, 0.25);
    border-radius: 10px;
    padding: 14px 18px;
    margin: 8px 0;
    font-size: 0.85rem;
    color: #c4b5fd;
    line-height: 1.6;
}

/* ─── Footer ─── */
.footer-text {
    text-align: center;
    color: #64748b;
    font-size: 0.8rem;
    padding: 16px;
    border-top: 1px solid rgba(255, 255, 255, 0.05);
    margin-top: 24px;
}
"""

# ──────────────────────────── Gradio UI ──────────────────────────────────

with gr.Blocks(
    title="Fashion Image Synthesis",
) as demo:

    # ── Header ──
    gr.HTML(
        """
        <div id="app-title">
            <h1>👗 Controllable Fashion Image Synthesis</h1>
            <p>Generate stunning fashion images from text prompts with structural guidance &nbsp;·&nbsp;
            Stable Diffusion + ControlNet + LoRA</p>
        </div>
        """
    )

    with gr.Tabs():
        # ━━━━━━━━━━━━━━━━━━━ TAB 1: Single Generation ━━━━━━━━━━━━━━━━━━━
        with gr.TabItem("🎨 Single Generation", id="single"):
            with gr.Row(equal_height=False):
                # ── Left Column: Inputs ──
                with gr.Column(scale=1):
                    gr.HTML('<div class="info-badge">📷 <b>Step 1:</b> Upload any fashion reference image. The structural outline will be extracted automatically.</div>')

                    input_image = gr.Image(
                        label="Reference Image",
                        type="pil",
                        height=280,
                        sources=["upload", "clipboard"],
                    )

                    gr.HTML('<div class="info-badge">✏️ <b>Step 2:</b> Describe the garment you want to generate, or pick a preset below.</div>')

                    preset_dropdown = gr.Dropdown(
                        choices=["— Custom Prompt —"] + list(PRESETS.keys()),
                        value="— Custom Prompt —",
                        label="Quick Presets",
                        interactive=True,
                    )

                    prompt_input = gr.Textbox(
                        label="Text Prompt",
                        placeholder="A sleek black leather jacket with silver zippers …",
                        lines=3,
                    )

                    negative_prompt = gr.Textbox(
                        label="Negative Prompt (optional)",
                        placeholder="blurry, low quality, distorted, watermark …",
                        value="blurry, low quality, distorted, deformed, watermark, text",
                        lines=2,
                    )

                    with gr.Accordion("⚙️ Advanced Settings", open=False):
                        steps_slider = gr.Slider(
                            minimum=10,
                            maximum=50,
                            step=5,
                            value=20,
                            label="Inference Steps (Quality)",
                            info="More steps = higher quality, slower generation",
                        )
                        guidance_slider = gr.Slider(
                            minimum=1.0,
                            maximum=15.0,
                            step=0.5,
                            value=7.5,
                            label="Guidance Scale (Prompt Strength)",
                            info="Higher = follows text more strictly",
                        )
                        with gr.Row():
                            canny_low = gr.Slider(
                                minimum=50,
                                maximum=200,
                                step=10,
                                value=100,
                                label="Canny Low Threshold",
                            )
                            canny_high = gr.Slider(
                                minimum=100,
                                maximum=300,
                                step=10,
                                value=200,
                                label="Canny High Threshold",
                            )
                        seed_input = gr.Number(
                            value=-1,
                            label="Seed (-1 = random)",
                            precision=0,
                        )

                    generate_btn = gr.Button(
                        "🚀 Generate Image",
                        variant="primary",
                        size="lg",
                        elem_id="generate-btn",
                    )

                # ── Right Column: Outputs ──
                with gr.Column(scale=1):
                    gr.HTML('<div class="info-badge">🖼️ <b>Results:</b> The extracted structure (edges) and the AI-generated fashion image.</div>')

                    with gr.Row():
                        edge_output = gr.Image(label="Extracted Structure", height=256)
                        generated_output = gr.Image(
                            label="✨ Generated Fashion Image", height=256
                        )

                    seed_info = gr.Textbox(label="Generation Info", interactive=False)

            # ── Events (Single) ──
            preset_dropdown.change(
                fn=load_preset,
                inputs=[preset_dropdown],
                outputs=[prompt_input],
            )

            generate_btn.click(
                fn=generate,
                inputs=[
                    input_image,
                    prompt_input,
                    negative_prompt,
                    steps_slider,
                    guidance_slider,
                    canny_low,
                    canny_high,
                    seed_input,
                ],
                outputs=[edge_output, generated_output, seed_info],
            )

        # ━━━━━━━━━━━━━━━━━━━ TAB 2: Batch Generation ━━━━━━━━━━━━━━━━━━━
        with gr.TabItem("📦 Batch Generation", id="batch"):
            with gr.Row(equal_height=False):
                with gr.Column(scale=1):
                    gr.HTML('<div class="info-badge">📦 Generate up to 8 different garments from the same reference image. Enter one prompt per line.</div>')

                    batch_image = gr.Image(
                        label="Reference Image",
                        type="pil",
                        height=250,
                        sources=["upload", "clipboard"],
                    )

                    batch_prompts = gr.Textbox(
                        label="Prompts (one per line, max 8)",
                        lines=8,
                        value="A classic black leather jacket\nA white silk blouse with subtle sheen\nA bright red evening dress\nA tailored navy blue blazer\nA cozy beige cashmere sweater\nA brown leather bomber jacket",
                    )

                    batch_negative = gr.Textbox(
                        label="Negative Prompt (optional)",
                        value="blurry, low quality, distorted, deformed, watermark, text",
                        lines=2,
                    )

                    with gr.Accordion("⚙️ Advanced Settings", open=False):
                        batch_steps = gr.Slider(10, 50, 20, step=5, label="Steps")
                        batch_guidance = gr.Slider(
                            1.0, 15.0, 7.5, step=0.5, label="Guidance"
                        )
                        with gr.Row():
                            batch_canny_low = gr.Slider(
                                50, 200, 100, step=10, label="Canny Low"
                            )
                            batch_canny_high = gr.Slider(
                                100, 300, 200, step=10, label="Canny High"
                            )
                        batch_seed = gr.Number(value=-1, label="Seed", precision=0)

                    batch_btn = gr.Button(
                        "📦 Generate Batch",
                        variant="primary",
                        size="lg",
                        elem_id="batch-btn",
                    )

                with gr.Column(scale=2):
                    batch_edge_output = gr.Image(label="Extracted Structure", height=200)
                    batch_gallery = gr.Gallery(
                        label="✨ Generated Results",
                        columns=3,
                        height=500,
                        object_fit="contain",
                        show_label=True,
                    )

            batch_btn.click(
                fn=batch_generate,
                inputs=[
                    batch_image,
                    batch_prompts,
                    batch_negative,
                    batch_steps,
                    batch_guidance,
                    batch_canny_low,
                    batch_canny_high,
                    batch_seed,
                ],
                outputs=[batch_edge_output, batch_gallery],
            )

        # ━━━━━━━━━━━━━━━━━━━ TAB 3: About ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
        with gr.TabItem("ℹ️ About", id="about"):
            gr.Markdown(
                """
                ## 🏗️ Architecture

                This application combines three deep learning components:

                | Component | Role | Trainable? |
                |-----------|------|-----------|
                | **Stable Diffusion v1.5** | Base image generation (860M params) | ❄️ Frozen |
                | **ControlNet (Canny)** | Structural guidance via edge maps | ❄️ Frozen |
                | **LoRA Adapter** | Fashion domain adaptation (~4M params) | ✅ Trained |

                ## 🔄 Pipeline Flow

                ```
                Reference Image → Canny Edge Detection → ControlNet
                                                            ↓
                Text Prompt → CLIP Encoder → U-Net (with LoRA) → VAE Decoder → Generated Image
                ```

                ## 📊 Training Details

                - **Dataset:** FashionGen (256×256 fashion images with text descriptions)
                - **Method:** LoRA fine-tuning (Rank 4) on U-Net cross-attention layers
                - **Steps:** 5,000 training steps with cosine LR schedule
                - **Loss:** MSE between predicted and actual noise

                ## 🎛️ Parameter Guide

                - **Inference Steps:** More steps = higher quality but slower (20 is a good default)
                - **Guidance Scale:** How strongly the model follows your text (7.5 is balanced)
                - **Canny Thresholds:** Control edge sensitivity (100/200 works for most images)
                - **Seed:** Set a specific number for reproducible results, or -1 for random
                """
            )

    # ── Footer ──
    gr.HTML(
        """
        <div class="footer-text">
            Controllable Fashion Image Synthesis &nbsp;·&nbsp;
            Stable Diffusion v1.5 + ControlNet + LoRA &nbsp;·&nbsp;
            Deep Learning Project
        </div>
        """
    )

# ──────────────────────────── Launch ─────────────────────────────────────

if __name__ == "__main__":
    demo.launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="purple",
            neutral_hue="slate",
            font=gr.themes.GoogleFont("Inter"),
        )
    )
