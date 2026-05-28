#!/bin/bash

# ============================================================
# Controllable Fashion Image Synthesis - Setup Script
# AMD GPU Compatible
# ============================================================

set -e  # Exit on error

echo "============================================================"
echo "🎨 Controllable Fashion Image Synthesis - Setup"
echo "============================================================"
echo ""

# Check Python version
echo "📋 Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "   Python: $python_version"

if ! python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "❌ Python 3.10+ required. Found: $python_version"
    exit 1
fi
echo "   ✅ Python version OK"
echo ""

# Check ROCm
echo "📋 Checking ROCm installation..."
if command -v rocm-smi &> /dev/null; then
    rocm_version=$(rocm-smi --showdriverversion 2>&1 | head -1)
    echo "   ROCm: $rocm_version"
    echo "   ✅ ROCm detected"
else
    echo "   ⚠️  ROCm not found. Please install ROCm first."
    echo "   Visit: https://rocm.docs.amd.com/"
fi
echo ""

# Create virtual environment
echo "📦 Creating virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "   ✅ Virtual environment created"
else
    echo "   ℹ️  Virtual environment already exists"
fi
echo ""

# Activate virtual environment
echo "🔌 Activating virtual environment..."
source venv/bin/activate
echo "   ✅ Activated"
echo ""

# Upgrade pip
echo "📦 Upgrading pip..."
pip install --upgrade pip --quiet
echo "   ✅ pip upgraded"
echo ""

# Install PyTorch with ROCm
echo "📦 Installing PyTorch with ROCm support..."
echo "   This may take a few minutes..."
echo ""

# Detect ROCm version and install appropriate PyTorch
if rocm-smi --showdriverversion 2>&1 | grep -q "6\."; then
    echo "   Detected ROCm 6.x - Installing PyTorch for ROCm 6.2..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2 --quiet
elif rocm-smi --showdriverversion 2>&1 | grep -q "5\."; then
    echo "   Detected ROCm 5.x - Installing PyTorch for ROCm 5.7..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7 --quiet
else
    echo "   ⚠️  Could not detect ROCm version. Installing for ROCm 6.2..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2 --quiet
fi

echo "   ✅ PyTorch installed"
echo ""

# Verify PyTorch installation
echo "🔍 Verifying PyTorch installation..."
python3 -c "
import torch
print(f'   PyTorch Version: {torch.__version__}')
print(f'   CUDA Available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'   GPU Count: {torch.cuda.device_count()}')
    print(f'   GPU 0: {torch.cuda.get_device_name(0)}')
else:
    print('   ⚠️  GPU not detected')
"
echo ""

# Install other requirements
echo "📦 Installing other dependencies..."
echo "   This may take several minutes..."
pip install -r requirements.txt --quiet
echo "   ✅ Dependencies installed"
echo ""

# Create working directory structure
echo "📁 Creating directory structure..."
mkdir -p working/fashion_lora_output
mkdir -p working/eval_data/gt
mkdir -p working/eval_data/baseline
mkdir -p working/eval_data/lora
mkdir -p working/model_cache
echo "   ✅ Directories created"
echo ""

# Check if model weights exist
if [ -f "working/fashion_lora_output/pytorch_lora_weights.safetensors" ]; then
    echo "✅ Pre-trained model weights found!"
    echo "   You can skip training and go directly to generation."
else
    echo "ℹ️  No pre-trained weights found."
    echo "   You'll need to run the training notebook first."
fi
echo ""

echo "============================================================"
echo "✅ Setup Complete!"
echo "============================================================"
echo ""
echo "📝 Next Steps:"
echo "   1. Activate virtual environment: source venv/bin/activate"
echo "   2. Start Jupyter: jupyter notebook"
echo "   3. Open notebooks in order:"
echo "      - controllable-fashion-image-synthesis-v1.ipynb (Training)"
echo "      - controllable-fashion-image-synthesis-v1-evaluation.ipynb (Evaluation)"
echo "      - controllable-fashion-image-synthesis-app.ipynb (GUI)"
echo "      - controllable-fashion-image-synthesis-app-cli.ipynb (CLI)"
echo ""
echo "💡 For detailed instructions, see README.md"
echo "============================================================"
