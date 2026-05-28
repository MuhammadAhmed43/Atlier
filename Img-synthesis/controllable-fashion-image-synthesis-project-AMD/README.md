# Controllable Fashion Image Synthesis - AMD GPU Compatible

A complete implementation of controllable fashion image synthesis using Stable Diffusion with ControlNet and LoRA fine-tuning, optimized for AMD GPUs with ROCm.

## 🎯 Project Overview

This project enables you to:
- **Train** a LoRA model on fashion images with text prompts
- **Generate** fashion images from text descriptions using edge structure guidance
- **Evaluate** model performance with comprehensive metrics (FID, KID, LPIPS, CLIP)
- **Interact** with the model via GUI or CLI interfaces

---

## 📦 Package Contents

This package is **ready to deploy** on any AMD GPU system. Everything is self-contained.

### ✅ What's Included

**4 Complete Notebooks:**
- `controllable-fashion-image-synthesis-v1.ipynb` - Training notebook
- `controllable-fashion-image-synthesis-v1-evaluation.ipynb` - Evaluation notebook
- `controllable-fashion-image-synthesis-app.ipynb` - GUI application
- `controllable-fashion-image-synthesis-app-cli.ipynb` - CLI application

**Pre-trained Model:**
- LoRA weights in `working/fashion_lora_output/`
- Evaluation data in `working/eval_data/`
- Model cache in `working/model_cache/`

**Setup Scripts:**
- `setup.sh` - Automated installation
- `verify_setup.py` - Verification script

**Configuration:**
- `requirements.txt` - All Python dependencies
- `.gitignore` - Git ignore rules

### 📋 Setup Checklist

- [x] All notebooks copied
- [x] Requirements file included
- [x] Setup script created
- [x] Paths updated to relative (./working)
- [x] Documentation complete
- [x] Git ignore configured
- [x] Pre-trained model included (if available)

---

## 📋 Prerequisites

### Hardware Requirements
- **AMD GPU** with ROCm support (ROCm 5.7+ recommended)
- **Minimum 16GB GPU memory** (24GB+ recommended for training)
- **50GB+ free disk space** for models and data

### Software Requirements
- **Linux** (Ubuntu 20.04+ recommended)
- **Python 3.10+**
- **ROCm 5.7+** or **ROCm 6.0+**
- **CUDA toolkit** (for ROCm compatibility)

---

## 🚀 Quick Start

### For Users with Pre-trained Model

If you have the pre-trained model weights, you can skip training and go directly to generation!

#### 1. Setup (One-time)

```bash
# Run setup script
./setup.sh

# Or manually:
python3 -m venv venv
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
pip install -r requirements.txt
```

#### 2. Verify Installation

```bash
source venv/bin/activate
python verify_setup.py
```

#### 3. Generate Images (GUI)

```bash
# Activate environment
source venv/bin/activate

# Start Jupyter
jupyter notebook

# Open: controllable-fashion-image-synthesis-app.ipynb
# Run all cells
# Use the GUI to generate images!
```

#### 4. Generate Images (CLI)

```bash
# Activate environment
source venv/bin/activate

# Start Jupyter
jupyter notebook

# Open: controllable-fashion-image-synthesis-app-cli.ipynb
# Run all cells
# Follow terminal prompts
```

### For Users Training from Scratch

#### 1. Setup

Same as above - run `./setup.sh`

#### 2. Prepare Data

Place your FashionGen dataset at:
```
data/fashiongen_256_256_train.h5
```

Or update paths in the training notebook.

#### 3. Train Model

```bash
source venv/bin/activate
jupyter notebook

# Open: controllable-fashion-image-synthesis-v1.ipynb
# Run all cells (takes 2-4 hours)
```

#### 4. Evaluate Model

```bash
# Open: controllable-fashion-image-synthesis-v1-evaluation.ipynb
# Run all cells (takes 30-60 minutes)
```

#### 5. Generate Images

Follow steps 3-4 from "For Users with Pre-trained Model" above.

---

## 📋 Detailed Setup Instructions

### 1. Clone the Repository

```bash
git clone <repository-url>
cd controllable-fashion-image-synthesis-project
```

### 2. Install PyTorch with ROCm Support

**IMPORTANT:** Install PyTorch FIRST before other packages!

```bash
# Check your ROCm version
rocm-smi --showdriverversion

# For ROCm 6.0+ (recommended)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2

# For ROCm 5.7
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7
```

### 3. Install Dependencies

```bash
# Create virtual environment (recommended)
python3 -m venv venv
source venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

### 4. Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
```

Expected output should show:
- PyTorch version with `rocm` in the name
- `CUDA Available: True`
- Your AMD GPU name

---

## 📓 Notebooks Guide

Run notebooks **in order**:

### 1. **Training Notebook** (`controllable-fashion-image-synthesis-v1.ipynb`)

**Purpose:** Train LoRA weights on fashion images

**What it does:**
- Loads FashionGen dataset
- Fine-tunes Stable Diffusion with LoRA
- Saves trained weights to `working/fashion_lora_output/`
- Generates evaluation data

**Estimated Time:** 2-4 hours (depending on GPU and dataset size)

**Key Settings:**
- Training steps: Configurable (default: ~5000)
- Batch size: Adjust based on GPU memory
- Learning rate: 1e-4 (default)

### 2. **Evaluation Notebook** (`controllable-fashion-image-synthesis-v1-evaluation.ipynb`)

**Purpose:** Comprehensive model evaluation

**What it does:**
- Calculates quantitative metrics (FID, KID, LPIPS, CLIP)
- Performs statistical significance testing
- Generates visual comparisons
- Identifies best/worst cases
- Creates comprehensive evaluation report

**Estimated Time:** 30-60 minutes

**Outputs:**
- `working/final_metrics.json` - All metrics
- `working/comprehensive_evaluation_report.json` - Full report
- `working/per_sample_metrics.csv` - Per-sample analysis
- Visual comparison images

### 3. **GUI Application** (`controllable-fashion-image-synthesis-app.ipynb`)

**Purpose:** Interactive GUI for image generation

**What it does:**
- Loads trained model
- Provides widget-based interface
- Real-time image generation
- Visual preview and settings

**Usage:**
1. Run all cells
2. Use the GUI widgets to:
   - Select reference image
   - Enter or choose prompt
   - Adjust settings (steps, guidance, seed)
   - Click "Generate"

### 4. **CLI Application** (`controllable-fashion-image-synthesis-app-cli.ipynb`)

**Purpose:** Terminal-based interface for scripting

**What it does:**
- Command-line prompt input
- Batch generation support
- Programmatic control

**Usage:**
1. Run all cells
2. Follow terminal prompts
3. Type prompts and generate images

---

## 📁 Project Structure

```
controllable-fashion-image-synthesis-project/
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── setup.sh                           # Setup script
├── verify_setup.py                    # Verification script
├── .gitignore                         # Git ignore rules
│
├── Notebooks/
│   ├── controllable-fashion-image-synthesis-v1.ipynb              # Training
│   ├── controllable-fashion-image-synthesis-v1-evaluation.ipynb   # Evaluation
│   ├── controllable-fashion-image-synthesis-app.ipynb             # GUI App
│   └── controllable-fashion-image-synthesis-app-cli.ipynb          # CLI App
│
└── working/                           # Working directory
    ├── fashion_lora_output/           # Trained LoRA weights
    │   └── pytorch_lora_weights.safetensors
    ├── eval_data/                     # Evaluation data
    │   ├── gt/                        # Ground truth images
    │   ├── baseline/                  # Baseline outputs
    │   └── lora/                      # LoRA outputs
    ├── model_cache/                   # Cached model downloads
    └── [generated outputs]            # Generated images
```

---

## ⚙️ Configuration

### Path Configuration

All notebooks use relative paths. The default working directory is:
```python
WORKING_DIR = "./working"
```

To change paths, update the `WORKING_DIR` variable in each notebook's Cell 2.

### Model Configuration

- **Base Model:** `runwayml/stable-diffusion-v1-5`
- **ControlNet:** `lllyasviel/control_v11p_sd15_canny`
- **LoRA Rank:** 4 (default from training script)
- **Precision:** FP32 (full precision) for AMD GPUs (ROCm compatibility)

### Training Parameters

The model was trained with the following specifications (from `controllable-fashion-image-synthesis-v1.ipynb`):

#### Hyperparameters
- **Learning Rate:** `1e-04` (0.0001)
- **Max Training Steps:** `5000`
- **Batch Size:** `2` (per GPU)
- **Gradient Accumulation Steps:** `2` (effective batch size: 4)
- **Learning Rate Scheduler:** `cosine`
- **Learning Rate Warmup Steps:** `500`
- **Max Gradient Norm:** `1.0` (gradient clipping)
- **Mixed Precision:** `no` (FP32 - required for ROCm stability)

#### Model Architecture
- **LoRA Rank:** `4` (default, not explicitly set in training command)
- **LoRA Alpha:** `4` (typically equals rank)
- **Image Resolution:** `256x256`
- **Random Flip:** Enabled (data augmentation)

#### Training Configuration
- **Checkpointing Steps:** `1000` (saves checkpoint every 1000 steps)
- **Random Seed:** `42` (for reproducibility)
- **Report To:** `tensorboard` (training logs)
- **Training Data:** `./working/fashion_train` (100,000 samples)
- **Output Directory:** `./working/fashion_lora_output`

#### Evaluation Specifications

From `controllable-fashion-image-synthesis-v1-evaluation.ipynb`:

- **Evaluation Dataset Size:** `10,000` samples
- **Evaluation Metrics:**
  - **FID (Fréchet Inception Distance):** Image quality assessment
  - **KID (Kernel Inception Distance):** Image diversity measurement
  - **LPIPS (Learned Perceptual Image Patch Similarity):** Perceptual similarity
  - **CLIP Score:** Text-image alignment
- **Evaluation Data Structure:**
  - Ground truth images: `./working/eval_data/gt/`
  - Baseline outputs: `./working/eval_data/baseline/`
  - LoRA outputs: `./working/eval_data/lora/`
  - Evaluation configs: `./working/eval_data/eval_configs.json`

#### Hardware Specifications
- **GPU:** AMD Instinct MI250X/MI250 (8 GPUs used for training)
- **ROCm Version:** 6.2
- **PyTorch Version:** 2.5.1+rocm6.2
- **Training Time:** ~28 minutes for 5000 steps (on 8x MI250X GPUs)

---

## 🎨 Usage Examples

### Generate Single Image (GUI)

1. Open `controllable-fashion-image-synthesis-app.ipynb`
2. Run all cells
3. In the GUI:
   - Select image from dropdown
   - Enter prompt: "A sleek black leather jacket with silver zippers"
   - Click "Generate"
4. Wait ~15-30 seconds
5. View result and saved file

### Generate Single Image (CLI)

1. Open `controllable-fashion-image-synthesis-app-cli.ipynb`
2. Run all cells
3. Follow prompts:
   ```
   ✏️  Enter your fashion prompt: A sleek black leather jacket
   ```
4. Wait for generation
5. Check saved output

### Batch Generation

Use the programmatic interface in Cell 4 of the app notebooks:

```python
prompts = [
    "A black leather jacket",
    "A white silk blouse",
    "A red evening dress"
]

results = batch_generate(
    reference_image_path=sample_image_paths[sample_key],
    prompts_list=prompts,
    steps=20,
    guidance=7.5
)
```

---

## 📊 Evaluation Metrics

The evaluation notebook calculates:

- **FID (Fréchet Inception Distance)**: Image quality (lower is better)
- **KID (Kernel Inception Distance)**: Image diversity (lower is better)
- **LPIPS**: Perceptual similarity (lower is better)
- **CLIP Score**: Text-image alignment (higher is better)
- **Edge SSIM/IoU**: Structure preservation (higher is better)

---

## 🚀 Deployment Guide

### On Target System:

1. **Copy/Clone** this entire folder
   ```bash
   # If using git:
   git clone <repository-url>
   cd controllable-fashion-image-synthesis-project
   
   # Or copy the folder directly
   ```

2. **Run Setup**
   ```bash
   ./setup.sh
   ```
   This will:
   - Create virtual environment
   - Install PyTorch with ROCm
   - Install all dependencies
   - Create directory structure

3. **Verify Installation**
   ```bash
   source venv/bin/activate
   python verify_setup.py
   ```

4. **Start Using**
   ```bash
   source venv/bin/activate
   jupyter notebook
   ```

### ✅ Pre-flight Checklist

Before deploying, verify:

- [x] All 4 notebooks present
- [x] `requirements.txt` included
- [x] `setup.sh` executable
- [x] `working/` directory with model weights
- [x] All paths are relative (./working)
- [x] Documentation complete

### 🎯 Quick Test

After deployment, test with:

```bash
source venv/bin/activate
jupyter notebook
# Open: controllable-fashion-image-synthesis-app.ipynb
# Run all cells
# Generate an image via GUI
```

If this works, everything is set up correctly!

### 📊 Package Size

- **Code & Docs:** ~5MB
- **Model Weights:** ~50-100MB
- **Model Cache:** ~20GB (HuggingFace models)
- **Evaluation Data:** ~4GB
- **Total:** ~24GB

**Note:** Model cache can be re-downloaded if needed, reducing package size.

---

## 🔧 Troubleshooting

### GPU Not Detected

```bash
# Check ROCm installation
rocm-smi

# Verify PyTorch ROCm installation
python -c "import torch; print(torch.version.hip)"
```

### Out of Memory Errors

- Reduce batch size in training notebook
- Use fewer inference steps (10-20 instead of 20-40)
- Enable attention slicing (already enabled by default)

### Model Download Issues

Models are cached in `working/model_cache/`. If downloads fail:
- Check internet connection
- Clear cache: `rm -rf working/model_cache/`
- Re-run notebook to re-download

### Import Errors

```bash
# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Common Issues

**GPU not detected?**
```bash
rocm-smi  # Check ROCm
python -c "import torch; print(torch.cuda.is_available())"
```

**Out of memory?**
- Reduce batch size
- Use fewer inference steps
- Enable attention slicing (already enabled)

**Import errors?**
```bash
pip install --upgrade -r requirements.txt
```

---

## 📝 Notes

- **bitsandbytes** is NOT included (poor ROCm support)
- Training uses **FP16 mixed precision** automatically
- **XFormers** is optional but improves memory efficiency
- All paths are **relative** for portability
- Generated images are saved with seed numbers for reproducibility
- **All paths are relative** - works on any system
- **Model cache included** - faster first run
- **Pre-trained weights included** - skip training if desired
- **Virtual environment** - isolated dependencies

---

## 🔄 Updates

To update the package:
1. Pull latest changes
2. Run `./setup.sh` again
3. Verify with `python verify_setup.py`

---

## 🤝 Contributing

This is a research project. Feel free to:
- Report issues
- Suggest improvements
- Share your results

---

## 📄 License

This project uses:
- **Stable Diffusion** (CreativeML Open RAIL-M License)
- **ControlNet** (Apache 2.0)
- **LoRA** (Apache 2.0)

Please review and comply with all licenses.

---

## 🙏 Acknowledgments

- Hugging Face for diffusers and transformers
- Stability AI for Stable Diffusion
- lllyasviel for ControlNet
- Microsoft for LoRA implementation

---

## 📧 Support

For issues or questions:
1. Check the troubleshooting section
2. Review notebook comments
3. Check ROCm/PyTorch compatibility

---

**Happy Generating! 🎨✨**
