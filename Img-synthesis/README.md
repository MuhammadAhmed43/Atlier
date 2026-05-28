# Controllable Fashion Image Synthesis

A comprehensive deep learning project for generating fashion images from text descriptions using Stable Diffusion, ControlNet, and LoRA fine-tuning. This project provides complete implementations optimized for both **AMD GPUs (ROCm)** and **Kaggle GPU environments (NVIDIA)**.

---

## 🎯 Project Overview

This project implements a state-of-the-art controllable image synthesis system that combines:

- **Stable Diffusion v1.5** - Base generative model
- **ControlNet** - Structure guidance via Canny edge maps
- **LoRA (Low-Rank Adaptation)** - Efficient fine-tuning for fashion domain
- **Comprehensive Evaluation** - Quantitative metrics (FID, KID, LPIPS, CLIP)

### Key Capabilities

✅ **Text-to-Image Generation** - Generate fashion images from natural language descriptions  
✅ **Structure Control** - Control image structure using edge maps from reference images  
✅ **Domain Adaptation** - Fine-tuned specifically for fashion images  
✅ **Multiple Interfaces** - GUI, CLI, and programmatic APIs  
✅ **Comprehensive Evaluation** - Full metric suite for quality assessment  
✅ **Cross-Platform** - Works on AMD (ROCm) and NVIDIA (CUDA) GPUs  
✅ **Pre-trained Weights** - Ready-to-use weights available for immediate inference  

---

## 🎨 Example Outputs

The system generates high-quality fashion images that:
- Match the structure of reference images
- Follow text descriptions accurately
- Maintain fashion domain characteristics
- Preserve fine details and textures

### Sample Batch Generation

The following example demonstrates the system's ability to generate diverse fashion items from a single reference image using different text prompts:

![Batch Generation Example](assets/images/batch_generation_example.png)

**What this shows:**
- **Top Left:** Reference image with a model in a navy shirt
- **Top Right:** Extracted Canny edge structure (pose and silhouette)
- **Bottom 6 panels:** Generated fashion items using different text prompts:
  - Classic black leather jacket
  - Blue wool hood
  - Sleeveless green cotton T-shirt
  - Brown bomber jacket
  - Red Christmas sweater
  - Purple silky sleeping gown

All generated images maintain the same pose and structure from the reference while accurately following the text descriptions.

---



## 🏗️ Architecture

### System Components

```
┌─────────────────────────────────────────────────────────────┐
│                    Text Prompt                              │
│          "A sleek black leather jacket"                     │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Reference Image (Fashion Item)                 │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Canny Edge Detection                           │
│         (Structure Extraction)                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│         ControlNet + Stable Diffusion v1.5                  │
│         (Base Model)                                        │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              LoRA Fine-Tuned Weights                        │
│         (Fashion Domain Adaptation)                         │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│           Generated Fashion Image                           │
│     (Matches structure + text description)                  │
└─────────────────────────────────────────────────────────────┘
```

### Technical Stack

- **Base Model:** `runwayml/stable-diffusion-v1-5`
- **ControlNet:** `lllyasviel/control_v11p_sd15_canny`
- **Fine-tuning Method:** LoRA (Low-Rank Adaptation)
- **Image Resolution:** 256×256 pixels
- **Precision:** FP32 (for stability across platforms)

---

## 📦 Project Structure

```
controllable-fashion-image-synthesis-project/
│
├── README.md                          # This file - Project overview
│
├── assets/                             # Project assets
│   ├── images/                        # Sample output images
│   │   └── batch_generation_example.png
│   └── weights/                       # Pre-trained model weights
│       └── Model_weights_for_inference-Kaggle.rar
│
├── controllable-fashion-image-synthesis-project-AMD/
│   ├── README.md                      # AMD-specific documentation
│   ├── setup.sh                       # Automated setup script
│   ├── verify_setup.py                # Environment verification
│   ├── requirements.txt               # Python dependencies
│   ├── .gitignore                     # Git ignore rules
│   │
│   ├── Notebooks:
│   │   ├── controllable-fashion-image-synthesis-v1.ipynb
│   │   │   └── Training notebook (100k samples)
│   │   ├── controllable-fashion-image-synthesis-v1-evaluation.ipynb
│   │   │   └── Evaluation notebook (10k samples)
│   │   ├── controllable-fashion-image-synthesis-app.ipynb
│   │   │   └── GUI application
│   │   └── controllable-fashion-image-synthesis-app-cli.ipynb
│   │       └── CLI application
│   │
│   └── working/                       # Working directory
│       ├── fashion_lora_output/      # Trained LoRA weights
│       ├── eval_data/                # Evaluation data
│       └── model_cache/               # Cached models
│
└── controllable-fashion-image-synthesis-project-kaggle/
    ├── README.md                      # Kaggle-specific documentation
    │
    └── Notebooks:
        ├── fashion-image-generation-30k-3k-denoising20.ipynb
        │   └── 30k training, 3k eval, 20 inference steps
        ├── fashion-image-generation-30k-3k-denoising40.ipynb
        │   └── 30k training, 3k eval, 40 inference steps
        └── fashion-image-generation-5k-0-5k-denoising20.ipynb
            └── 5k training, 500 eval, 20 inference steps
```

---

## 🚀 Quick Start Guide

### Choose Your Platform

This project provides two implementations:

1. **AMD GPU Version** - For local AMD GPU systems with ROCm
2. **Kaggle Version** - For cloud-based training on Kaggle GPUs

### AMD GPU Version

**Best for:**
- Local development and experimentation
- Systems with AMD GPUs (ROCm 5.7+)
- Long-term training runs
- Full control over environment

**Quick Start:**
```bash
cd controllable-fashion-image-synthesis-project-AMD
./setup.sh
source venv/bin/activate
jupyter notebook
# Open: controllable-fashion-image-synthesis-app.ipynb
```

**See:** [`controllable-fashion-image-synthesis-project-AMD/README.md`](controllable-fashion-image-synthesis-project-AMD/README.md) for detailed instructions.

### Kaggle Version

**Best for:**
- Quick experiments and prototyping
- No local GPU required
- Sharing and collaboration
- Limited compute resources

**Quick Start Options:**

**Option A: Using Pre-trained Weights (Recommended for Quick Start)**
1. Download pre-trained weights from `assets/weights/Model_weights_for_inference-Kaggle.rar`
2. Upload to Kaggle as a dataset
3. Create an inference notebook
4. Load weights and start generating immediately!

**Option B: Training from Scratch**
1. Upload FashionGen dataset to Kaggle
2. Upload one of the Kaggle notebooks
3. Enable GPU in notebook settings
4. Run all cells

**See:** [`controllable-fashion-image-synthesis-project-kaggle/README.md`](controllable-fashion-image-synthesis-project-kaggle/README.md) for detailed instructions.

---

## 🔬 Methodology

### Training Process

1. **Data Preparation**
   - Extract images and text descriptions from FashionGen dataset
   - Create train/eval splits
   - Generate metadata files

2. **LoRA Fine-tuning**
   - Freeze base Stable Diffusion model
   - Train low-rank adaptation matrices
   - Optimize for fashion domain

3. **Training Configuration**
   - **Learning Rate:** 1e-04 (0.0001)
   - **Training Steps:** 5,000
   - **Batch Size:** 2 (effective: 4 with gradient accumulation)
   - **Scheduler:** Cosine annealing with 500-step warmup
   - **LoRA Rank:** 4

4. **Evaluation**
   - Generate images for evaluation set
   - Calculate quantitative metrics
   - Visual comparisons

### Generation Process

1. **Input:** Text prompt + Reference image
2. **Edge Extraction:** Canny edge detection on reference
3. **Generation:** ControlNet + LoRA-guided diffusion
4. **Output:** Fashion image matching structure and description

---

## 📊 Training Specifications

### Common Parameters (Both Versions)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Base Model | `runwayml/stable-diffusion-v1-5` | Pre-trained Stable Diffusion |
| ControlNet | `lllyasviel/control_v11p_sd15_canny` | Canny edge control |
| LoRA Rank | 4 | Low-rank adaptation dimension |
| Image Resolution | 256×256 | Input/output image size |
| Learning Rate | 1e-04 | Optimizer learning rate |
| Max Steps | 5,000 | Total training steps |
| Batch Size | 2 | Per-GPU batch size |
| Gradient Accumulation | 2 | Effective batch size: 4 |
| LR Scheduler | Cosine | Learning rate schedule |
| Warmup Steps | 500 | Learning rate warmup |
| Precision | FP32 | Full precision (platform compatibility) |
| Random Seed | 42 | Reproducibility |

### Dataset Configurations

#### AMD Version
- **Training:** 100,000 samples
- **Evaluation:** 10,000 samples
- **Inference Steps:** 20 (default, configurable)

#### Kaggle Version
- **Option 1:** 30,000 training / 3,000 eval (20 or 40 inference steps)
- **Option 2:** 5,000 training / 500 eval (20 inference steps)

---

## 📈 Evaluation Metrics

The project uses comprehensive evaluation metrics:

### Quantitative Metrics

1. **FID (Fréchet Inception Distance)**
   - Measures image quality and realism
   - Lower is better
   - Uses Inception v3 features

2. **KID (Kernel Inception Distance)**
   - Measures image diversity
   - Lower is better
   - More robust than FID for small datasets

3. **LPIPS (Learned Perceptual Image Patch Similarity)**
   - Measures perceptual similarity
   - Lower is better
   - Uses AlexNet features

4. **CLIP Score**
   - Measures text-image alignment
   - Higher is better
   - Uses CLIP model embeddings

### Evaluation Process

1. Generate images for evaluation set
2. Compare with ground truth images
3. Calculate all metrics
4. Generate visual comparisons
5. Export comprehensive reports

---

## 🎨 Usage Examples

### Example 1: Generate from Text Prompt

```python
# Using the CLI interface
prompt = "A sleek black leather jacket with silver zippers"
reference_image = "sample_jacket.jpg"

# Generate image
result = generate_image(
    image=reference_image,
    prompt=prompt,
    steps=20,
    guidance=7.5,
    seed=42
)
```

### Example 2: Batch Generation

```python
prompts = [
    "A black leather jacket",
    "A white silk blouse",
    "A red evening dress"
]

results = batch_generate(
    reference_image_path="reference.jpg",
    prompts_list=prompts,
    steps=20,
    guidance=7.5
)
```

### Example 3: Programmatic Interface

```python
from diffusers import StableDiffusionControlNetPipeline
from PIL import Image

# Load pipeline
pipe = StableDiffusionControlNetPipeline.from_pretrained(...)
pipe.load_lora_weights("path/to/lora/weights")

# Generate
image = pipe(
    prompt="A vintage denim jacket",
    image=edge_map,
    num_inference_steps=20
).images[0]
```

---

## 🔄 Platform Comparison

| Feature | AMD Version | Kaggle Version |
|---------|-------------|----------------|
| **Platform** | Local AMD GPU (ROCm) | Kaggle Cloud (NVIDIA T4) |
| **Setup** | Manual setup required | Pre-configured environment |
| **Dataset Size** | 100k training / 10k eval | 30k or 5k training |
| **Training Time** | ~28 min (8x MI250X) | ~30-60 min (T4) |
| **Interfaces** | GUI + CLI + API | Notebook-based |
| **Persistence** | Local storage | Session-based |
| **Cost** | Hardware cost | Free (with limits) |
| **Best For** | Production, long runs | Experimentation, sharing |

---

## 🛠️ Technical Details

### Model Architecture

- **Base:** Stable Diffusion v1.5 (860M parameters)
- **ControlNet:** Canny edge control (additional ~1.4B parameters)
- **LoRA:** Low-rank adaptation (only ~4M trainable parameters)

### Training Efficiency

- **LoRA Advantage:** Only 4M parameters trained vs 860M+ full fine-tuning
- **Memory Efficient:** Can train on 16GB GPUs
- **Fast Training:** 5,000 steps in ~30-60 minutes

### Inference

- **Speed:** ~15-30 seconds per image (20 steps)
- **Quality:** Higher with more steps (40 steps = ~30-60 seconds)
- **Memory:** ~8-12GB GPU memory required

---

## 📋 Prerequisites

### AMD Version
- AMD GPU with ROCm 5.7+ support
- 16GB+ GPU memory (24GB+ recommended)
- Linux (Ubuntu 20.04+)
- Python 3.10+
- 50GB+ disk space

### Kaggle Version
- Kaggle account
- GPU-enabled notebook
- Internet connection
- FashionGen dataset uploaded to Kaggle

---

## 🚦 Getting Started Checklist

### For AMD Version:
- [ ] Install ROCm and PyTorch with ROCm support
- [ ] Run `setup.sh` script
- [ ] Verify installation with `verify_setup.py`
- [ ] Prepare FashionGen dataset
- [ ] Run training notebook
- [ ] Run evaluation notebook
- [ ] Use GUI/CLI apps for generation

### For Kaggle Version (Training):
- [ ] Create Kaggle account
- [ ] Upload FashionGen dataset
- [ ] Upload notebook to Kaggle
- [ ] Enable GPU in settings
- [ ] Run all cells
- [ ] Download results

### For Kaggle Version (Inference with Pre-trained Weights):
- [ ] Create Kaggle account
- [ ] Download `Model_weights_for_inference-Kaggle.rar` from `assets/weights/`
- [ ] Upload weights to Kaggle as a dataset
- [ ] Create inference notebook
- [ ] Load pre-trained weights
- [ ] Start generating images immediately!

---

## 📦 Pre-trained Weights

### Kaggle Inference Weights

**Pre-trained LoRA weights are available for immediate inference:**

- **Location:** `assets/weights/Model_weights_for_inference-Kaggle.rar`
- **Format:** RAR archive containing LoRA weights
- **Size:** ~3-5 MB (compressed)
- **Usage:** Extract and use for inference without training
- **Platform:** Optimized for Kaggle GPU environments

**How to Use:**
1. Download the RAR file from `assets/weights/`
2. Extract the LoRA weights
3. Upload to Kaggle as a dataset or extract in your notebook
4. Load in your inference code:
   ```python
   pipe.load_lora_weights("path/to/weights", 
                          weight_name="pytorch_lora_weights.safetensors")
   ```
5. Start generating fashion images immediately!

**Benefits:**
- ✅ Skip training time (save hours)
- ✅ Immediate image generation
- ✅ Perfect for testing and experimentation
- ✅ Same quality as trained models

---

## 📚 Documentation

### Detailed Documentation

- **AMD Version:** See [`controllable-fashion-image-synthesis-project-AMD/README.md`](controllable-fashion-image-synthesis-project-AMD/README.md)
  - Complete setup instructions
  - Training parameters
  - Evaluation specifications
  - Troubleshooting guide

- **Kaggle Version:** See [`controllable-fashion-image-synthesis-project-kaggle/README.md`](controllable-fashion-image-synthesis-project-kaggle/README.md)
  - Kaggle-specific setup
  - Notebook workflow
  - Platform limitations
  - Usage tips

### Key Concepts

- **Stable Diffusion:** Latent diffusion model for image generation
- **ControlNet:** Conditional control mechanism for structure guidance
- **LoRA:** Parameter-efficient fine-tuning method
- **Canny Edge Detection:** Edge extraction for structure control

---

## 🔧 Troubleshooting

### Common Issues

**GPU Not Detected (AMD)**
```bash
rocm-smi  # Check ROCm
python -c "import torch; print(torch.cuda.is_available())"
```

**Out of Memory**
- Reduce batch size
- Use fewer inference steps
- Enable attention slicing

**Import Errors**
```bash
pip install --upgrade -r requirements.txt
```

**NumPy 2.0 Conflicts (Kaggle)**
- Notebooks handle this automatically
- Restart kernel if issues persist

See platform-specific READMEs for detailed troubleshooting.

---

## 📊 Results & Performance

### Training Metrics

- **Training Loss:** Decreases from ~0.18 to ~0.008 over 5000 steps
- **Convergence:** Stable after ~3000 steps
- **Checkpoints:** Saved every 1000 steps

### Evaluation Results

Results vary by dataset and configuration. Typical improvements:
- **FID:** Lower with LoRA vs baseline
- **CLIP Score:** Higher text-image alignment
- **LPIPS:** Better perceptual similarity

---

## 🔬 Research & Applications

### Research Applications

- Fashion design and prototyping
- Virtual try-on systems
- E-commerce product visualization
- Style transfer and customization
- Dataset augmentation

### Extensions

- Multi-view generation
- Style mixing
- Category-specific fine-tuning
- Higher resolution generation
- Real-time inference optimization

---

## 📄 License

This project uses:
- **Stable Diffusion** (CreativeML Open RAIL-M License)
- **ControlNet** (Apache 2.0)
- **LoRA** (Apache 2.0)

Please review and comply with all licenses before use.

---

## 🙏 Acknowledgments

- **Hugging Face** - diffusers and transformers libraries
- **Stability AI** - Stable Diffusion model
- **lllyasviel** - ControlNet implementation
- **Microsoft** - LoRA methodology
- **Kaggle** - GPU infrastructure (Kaggle version)

---

## 📧 Support & Contributing

### Getting Help

1. Check platform-specific README files
2. Review troubleshooting sections
3. Check notebook comments
4. Verify environment setup

### Contributing

This is a research project. Contributions welcome:
- Bug reports
- Feature suggestions
- Performance improvements
- Documentation updates

---

## 🎯 Project Goals

### Primary Objectives

✅ **Accessibility** - Works on multiple GPU platforms  
✅ **Reproducibility** - Complete documentation and specifications  
✅ **Usability** - Multiple interfaces (GUI, CLI, API)  
✅ **Evaluation** - Comprehensive metric suite  
✅ **Efficiency** - LoRA for parameter-efficient training  

### Future Directions

- [ ] Higher resolution generation (512×512, 1024×1024)
- [ ] Multi-modal control (text + structure + style)
- [ ] Real-time inference optimization
- [ ] Additional evaluation metrics
- [ ] Web-based interface

---

**Happy Generating! 🎨✨**

For platform-specific instructions, see:
- [AMD GPU Version README](controllable-fashion-image-synthesis-project-AMD/README.md)
- [Kaggle Version README](controllable-fashion-image-synthesis-project-kaggle/README.md)
