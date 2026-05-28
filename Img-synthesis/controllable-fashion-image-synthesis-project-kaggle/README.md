# Controllable Fashion Image Synthesis - Kaggle GPU Compatible

A complete implementation of controllable fashion image synthesis using Stable Diffusion with ControlNet and LoRA fine-tuning, optimized for Kaggle GPU environments (NVIDIA T4).

## 🎯 Project Overview

This project enables you to:
- **Train** a LoRA model on fashion images with text prompts
- **Generate** fashion images from text descriptions using edge structure guidance
- **Evaluate** model performance with comprehensive metrics (FID, KID, LPIPS, CLIP)
- **Run** everything on Kaggle's GPU infrastructure

---

## 📦 Package Contents

This package contains **3 complete training notebooks** with different configurations:

### Notebooks

1. **`fashion-image-generation-30k-3k-denoising20.ipynb`**
   - Training: 30,000 samples
   - Evaluation: 3,000 samples
   - Inference: 20 denoising steps

2. **`fashion-image-generation-30k-3k-denoising40.ipynb`**
   - Training: 30,000 samples
   - Evaluation: 3,000 samples
   - Inference: 40 denoising steps (higher quality, slower)

3. **`fashion-image-generation-5k-0-5k-denoising20.ipynb`**
   - Training: 5,000 samples
   - Evaluation: 500 samples
   - Inference: 20 denoising steps
   - Quick training option for testing

### Pre-trained Weights

**Pre-trained LoRA weights are available for inference:**
- **Location:** `../assets/weights/Model_weights_for_inference-Kaggle.rar`
- **Format:** RAR archive containing LoRA weights
- **Usage:** Extract and use for inference without training
- **Note:** These weights can be used directly in inference notebooks

---

## 📋 Prerequisites

### Hardware Requirements
- **Kaggle GPU** (NVIDIA T4 recommended)
- **Kaggle Notebook** with GPU enabled
- **Internet connection** for model downloads

### Software Requirements
- **Python 3.11+** (Kaggle default)
- **Kaggle account** with GPU access

---

## 🚀 Quick Start on Kaggle

### Option A: Using Pre-trained Weights (Inference Only)

**Skip training and go directly to inference!**

1. **Download Pre-trained Weights**
   - Download `Model_weights_for_inference-Kaggle.rar` from `../assets/weights/`
   - Upload to Kaggle as a dataset or extract in your notebook

2. **Create Inference Notebook**
   - Create a new Kaggle notebook
   - Load the pre-trained weights
   - Start generating images immediately

3. **Benefits:**
   - No training time required
   - Immediate image generation
   - Perfect for testing and experimentation

### Option B: Training from Scratch

### 1. Upload Dataset

Upload the FashionGen dataset to Kaggle:
- Dataset: `fashiongen_256_256_train.h5`
- Add as input dataset to your notebook

### 2. Open Notebook

1. Upload one of the notebooks to Kaggle
2. Enable GPU in notebook settings
3. Add the FashionGen dataset as input
4. Run all cells

### 3. Training Process

Each notebook follows this workflow:
1. **Data Preparation** - Extract training/evaluation samples
2. **Environment Setup** - Install dependencies (handles NumPy 2.0 conflicts)
3. **Training** - Fine-tune LoRA weights
4. **Loss Visualization** - Plot training curves
5. **Image Generation** - Generate baseline and LoRA images
6. **Evaluation** - Calculate metrics (FID, KID, LPIPS, CLIP)
7. **Visualization** - Create comparison images
8. **Export** - Zip results for download

---

## ⚙️ Configuration

### Model Configuration

- **Base Model:** `runwayml/stable-diffusion-v1-5`
- **ControlNet:** `lllyasviel/control_v11p_sd15_canny`
- **LoRA Rank:** 4 (default from training script)
- **Precision:** FP32 (full precision) - bitsandbytes removed for compatibility

### Training Parameters

All notebooks use the following training specifications:

#### Hyperparameters
- **Learning Rate:** `1e-04` (0.0001)
- **Max Training Steps:** `5000`
- **Batch Size:** `2` (per GPU)
- **Gradient Accumulation Steps:** `2` (effective batch size: 4)
- **Learning Rate Scheduler:** `cosine`
- **Learning Rate Warmup Steps:** `500`
- **Max Gradient Norm:** `1.0` (gradient clipping)
- **Mixed Precision:** `no` (FP32 - bitsandbytes removed for compatibility)

#### Model Architecture
- **LoRA Rank:** `4` (default, not explicitly set in training command)
- **LoRA Alpha:** `4` (typically equals rank)
- **Image Resolution:** `256x256`
- **Random Flip:** Enabled (data augmentation)

#### Training Configuration
- **Checkpointing Steps:** `1000` (saves checkpoint every 1000 steps)
- **Random Seed:** `42` (for reproducibility)
- **Report To:** `tensorboard` (training logs)
- **Output Directory:** `/kaggle/working/fashion_lora_output`

#### Dataset Configurations

**Notebook 1 & 2 (30k-3k):**
- **Training Data:** `30,000` samples
- **Evaluation Data:** `3,000` samples
- **Inference Steps:** `20` (notebook 1) or `40` (notebook 2)

**Notebook 3 (5k-0.5k):**
- **Training Data:** `5,000` samples
- **Evaluation Data:** `500` samples
- **Inference Steps:** `20`

#### Evaluation Specifications

- **Evaluation Metrics:**
  - **FID (Fréchet Inception Distance):** Image quality assessment
  - **KID (Kernel Inception Distance):** Image diversity measurement
  - **LPIPS (Learned Perceptual Image Patch Similarity):** Perceptual similarity
  - **CLIP Score:** Text-image alignment (with truncation fix)
- **Evaluation Data Structure:**
  - Ground truth images: `/kaggle/working/eval_data/gt/`
  - Baseline outputs: `/kaggle/working/eval_data/baseline/`
  - LoRA outputs: `/kaggle/working/eval_data/lora/`
  - Evaluation configs: `/kaggle/working/eval_data/eval_configs.json`

#### Hardware Specifications
- **GPU:** NVIDIA Tesla T4 (Kaggle default)
- **Python Version:** 3.11.13
- **Key Dependencies:**
  - `diffusers==0.26.3`
  - `transformers==4.38.2`
  - `accelerate==0.27.2`
  - `peft==0.9.0`
  - `numpy==1.26.4` (constrained to avoid NumPy 2.0 conflicts)

---

## 📓 Notebook Workflow

Each notebook contains the following cells:

### Cell 1: Data Preparation
- Loads FashionGen H5 dataset
- Extracts training and evaluation samples
- Creates metadata CSV for training
- Saves evaluation configs JSON

### Cell 2: Environment Setup
- Handles NumPy 2.0 compatibility issues
- Installs training dependencies with constraints
- Downloads training script from HuggingFace
- Removes broken bitsandbytes (falls back to FP32)

### Cell 3: Training
- Launches training with accelerate
- Trains LoRA weights for 5000 steps
- Saves checkpoints every 1000 steps
- Logs to TensorBoard

### Cell 3.5: Training Loss Visualization
- Extracts loss from TensorBoard logs
- Plots training curve with moving average
- Saves `training_loss.png`

### Cell 4: Evaluation Tools Installation
- Installs clean-fid, lpips, and other evaluation libraries
- Handles scikit-image version constraints

### Cell 5: Image Generation
- Loads ControlNet pipeline
- Generates baseline images (ControlNet only)
- Loads LoRA weights
- Generates LoRA-enhanced images
- Uses specified number of inference steps (20 or 40)

### Cell 6: Data Location
- Auto-detects evaluation data location
- Verifies all required directories exist

### Cell 7: Visual Comparison
- Creates side-by-side comparisons
- Shows ground truth, edge map, baseline, and LoRA results
- Includes text prompts in visualization
- Saves `qualitative_comparison_with_text.png`

### Cell 8: Metrics Calculation
- Calculates FID and KID scores
- Computes LPIPS perceptual similarity
- Calculates CLIP scores (with truncation fix)
- Saves `final_metrics_recalculated.json`

### Cell 9: Export Results
- Zips all results for download
- Includes metrics, visualizations, and weights

---

## 🔧 Key Features

### NumPy 2.0 Compatibility
- Explicitly constrains NumPy to `<2.0`
- Handles dependency conflicts automatically
- Ensures compatibility with diffusers 0.26.3

### bitsandbytes Handling
- Attempts to install bitsandbytes initially
- Automatically removes it if broken
- Falls back to FP32 precision (more stable)

### CLIP Score Fix
- Includes truncation and max_length parameters
- Prevents tokenizer errors
- Ensures reliable CLIP score calculation

### Auto-Detection
- Automatically finds evaluation data
- Works with Kaggle's input/output structure
- Handles different directory layouts

---

## 📊 Expected Outputs

After running a notebook, you'll get:

1. **Training Loss Plot** (`training_loss.png`)
   - Training curve over 5000 steps
   - Moving average overlay

2. **Qualitative Comparison** (`qualitative_comparison_with_text.png`)
   - 5 random samples with full pipeline visualization
   - Includes text prompts

3. **Metrics JSON** (`final_metrics_recalculated.json`)
   - FID, KID, LPIPS, CLIP scores
   - Separate metrics for baseline and LoRA

4. **Trained Weights** (`fashion_lora_output/pytorch_lora_weights.safetensors`)
   - LoRA weights for inference
   - Can be downloaded and used elsewhere

5. **Generated Images**
   - Baseline images: `/kaggle/working/eval_data/baseline/`
   - LoRA images: `/kaggle/working/eval_data/lora/`

6. **Zipped Results** (`final_evaluation_results.zip`)
   - All outputs packaged for download

### Using Pre-trained Weights

If you're using the pre-trained weights from `../assets/weights/Model_weights_for_inference-Kaggle.rar`:

1. **Extract the RAR file** to get the LoRA weights
2. **Upload to Kaggle** as a dataset or extract in your notebook
3. **Load in your inference notebook:**
   ```python
   pipe.load_lora_weights("path/to/extracted/weights", 
                          weight_name="pytorch_lora_weights.safetensors")
   ```
4. **Start generating** images immediately without training!

---

## 🎨 Usage Tips

### Choosing a Notebook

- **30k-3k-denoising20**: Best balance of quality and speed
- **30k-3k-denoising40**: Highest quality, slower inference
- **5k-0.5k-denoising20**: Quick testing and prototyping

### Kaggle GPU Limits

- **Session Time:** 9 hours maximum
- **GPU Memory:** ~16GB on T4
- **Disk Space:** ~30GB working directory

### Saving Progress

- Checkpoints saved every 1000 steps
- Can resume from checkpoint if session expires
- Download weights before session ends

---

## 🔍 Troubleshooting

### NumPy 2.0 Errors
- The notebook automatically handles this
- If issues persist, restart kernel and run Cell 2 again

### Out of Memory
- Reduce batch size in training command
- Use 20 inference steps instead of 40
- Clear variables between cells: `del variable_name`

### Import Errors
- Run Cell 2 again to reinstall dependencies
- Check Kaggle internet is enabled
- Verify dataset is properly attached

### Training Interrupted
- Checkpoints saved every 1000 steps
- Can modify training to resume from checkpoint
- Download weights from `/kaggle/working/fashion_lora_output/`

---

## 📝 Notes

- **bitsandbytes** is removed due to compatibility issues on Kaggle
- Training uses **FP32 precision** (slower but more stable)
- All paths use **Kaggle's standard directories** (`/kaggle/working`, `/kaggle/input`)
- **NumPy 2.0** is explicitly avoided with constraints
- **CLIP score** includes truncation fix for reliability
- Generated images are saved with **6-digit indices** matching evaluation configs
- **TensorBoard logs** are saved for loss visualization

---

## 🔄 Differences from AMD Version

- **Platform:** Kaggle (NVIDIA T4) vs AMD (ROCm)
- **Precision:** FP32 (bitsandbytes removed) vs FP32 (ROCm compatibility)
- **Paths:** Kaggle directories vs local filesystem
- **Dependencies:** Kaggle-optimized versions vs AMD-optimized
- **Dataset:** Uploaded to Kaggle vs local H5 file

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
- Kaggle for GPU infrastructure

---

## 📧 Support

For issues or questions:
1. Check the troubleshooting section
2. Review notebook comments
3. Check Kaggle GPU availability
4. Verify dataset is properly attached

---

**Happy Generating on Kaggle! 🎨✨**
