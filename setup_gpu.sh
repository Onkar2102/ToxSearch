#!/bin/bash
# ==============================================================================
# GPU Setup Script for eost-cam-llm
# ==============================================================================
# This script automates the installation of GPU-enabled dependencies
# for NVIDIA CUDA GPUs.
#
# Usage: bash setup_gpu.sh
# ==============================================================================

set -e  # Exit on error

echo "=========================================="
echo "GPU Setup for eost-cam-llm"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check if running in virtual environment
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo -e "${YELLOW}Warning: Not running in virtual environment${NC}"
    echo "Recommended: source venv/bin/activate"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check for CUDA installation
echo "Checking CUDA installation..."
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
    echo -e "${GREEN}✓ CUDA found: version $CUDA_VERSION${NC}"
else
    echo -e "${RED}✗ CUDA not found${NC}"
    echo "Please install CUDA Toolkit from:"
    echo "https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

# Check for nvidia-smi
echo "Checking NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -n 1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -n 1)
    echo -e "${GREEN}✓ GPU found: $GPU_NAME ($GPU_MEMORY)${NC}"
else
    echo -e "${RED}✗ nvidia-smi not found${NC}"
    echo "GPU may not be available"
    exit 1
fi

# Determine CUDA version for PyTorch
CUDA_MAJOR=$(echo $CUDA_VERSION | cut -d'.' -f1)
CUDA_MINOR=$(echo $CUDA_VERSION | cut -d'.' -f2)

if [ "$CUDA_MAJOR" -ge 12 ]; then
    TORCH_CUDA="cu121"
    echo "Using PyTorch with CUDA 12.1"
elif [ "$CUDA_MAJOR" -eq 11 ] && [ "$CUDA_MINOR" -ge 8 ]; then
    TORCH_CUDA="cu118"
    echo "Using PyTorch with CUDA 11.8"
else
    echo -e "${YELLOW}Warning: CUDA version may not be fully supported${NC}"
    echo "Attempting to use CUDA 11.8 packages"
    TORCH_CUDA="cu118"
fi

echo ""
echo "=========================================="
echo "Step 1: Uninstalling CPU versions"
echo "=========================================="
pip uninstall torch torchvision torchaudio llama-cpp-python -y || true

echo ""
echo "=========================================="
echo "Step 2: Installing PyTorch with CUDA"
echo "=========================================="
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 \
    --index-url https://download.pytorch.org/whl/${TORCH_CUDA}

echo ""
echo "=========================================="
echo "Step 3: Installing llama-cpp-python with CUDA"
echo "=========================================="
echo "This may take 5-10 minutes to compile..."
CMAKE_ARGS="-DLLAMA_CUDA=on" pip install llama-cpp-python==0.3.16 \
    --force-reinstall --no-cache-dir

echo ""
echo "=========================================="
echo "Step 4: Verifying GPU installation"
echo "=========================================="

# Verify PyTorch CUDA
echo "Testing PyTorch CUDA support..."
python3 << EOF
import torch
if torch.cuda.is_available():
    print(f"✓ PyTorch CUDA: Available")
    print(f"  Device: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("✗ PyTorch CUDA: Not available")
    exit(1)
EOF

# Verify llama-cpp-python
echo ""
echo "Testing llama-cpp-python..."
python3 << EOF
try:
    from llama_cpp import Llama
    print("✓ llama-cpp-python: Installed with GPU support")
except Exception as e:
    print(f"✗ llama-cpp-python: Error - {e}")
    exit(1)
EOF

# Verify device detection
echo ""
echo "Testing device detection..."
python3 << EOF
try:
    from utils.device_utils import get_device_info
    import json
    info = get_device_info()
    print("✓ Device detection working")
    print(f"  Detected device: {info['device']}")
    print(f"  CUDA available: {info.get('cuda_available', False)}")
    if info.get('cuda_available'):
        print(f"  CUDA device: {info.get('cuda_device_name', 'Unknown')}")
except Exception as e:
    print(f"⚠ Device detection: {e}")
EOF

echo ""
echo "=========================================="
echo "GPU Setup Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Configure your .env file with API keys"
echo "2. Run: python3 src/main.py --generations 1"
echo ""
echo "GPU will be automatically detected and used."
echo "Monitor GPU usage with: watch -n 1 nvidia-smi"
echo ""

