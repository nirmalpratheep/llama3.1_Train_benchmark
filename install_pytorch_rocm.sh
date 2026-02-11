#!/bin/bash

# Install PyTorch with ROCm support for AMD MI300X
# Using manual venv and ROCm 7.0 nightly build

echo "Installing PyTorch with ROCm 7.0 nightly for AMD MI300X..."
echo "============================================================"
echo ""

# Check ROCm version
if command -v rocminfo &> /dev/null; then
    echo "ROCm detected on system"
    rocm-smi --showproductname | grep "GPU\[0\]" | head -1
    echo ""
else
    echo "Warning: ROCm tools not found. Please install ROCm first."
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d "myenv" ]; then
    echo "Creating virtual environment 'myenv'..."
    python3 -m venv myenv
fi

# Activate virtual environment
source myenv/bin/activate

# Install system dependencies
echo "Installing system dependencies..."
sudo apt update
sudo apt install -y libjpeg-dev python3-dev python3-pip

# Install base Python packages
echo "Installing wheel and setuptools..."
pip3 install wheel setuptools

# Install PyTorch with ROCm 7.0 (nightly)
echo "Installing PyTorch nightly with ROCm 7.0..."
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.0

# Install other dependencies
echo "Installing other dependencies..."
pip3 install transformers datasets accelerate psutil pyyaml

echo ""
echo "Verifying installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}'); print(f'GPU name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "Installation complete!"
echo ""
echo "To use the environment, run: source myenv/bin/activate"
