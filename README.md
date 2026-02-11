# Llama 3.1 Training from Scratch

PyTorch-based training script for Llama 3.1 models (8B, 70B) from scratch with comprehensive metrics logging and FSDP support for distributed training on AMD MI300X GPUs.

## Features

- ✅ **Modular Configuration**: YAML-based configs for different model sizes
- ✅ **FSDP Support**: Fully Sharded Data Parallel for multi-GPU training
- ✅ **Detailed Metrics**: Real-time tracking of tokens/sec, GPU/CPU utilization, memory stats
- ✅ **AMD MI300X Optimized**: Uses BF16 and ROCm-compatible monitoring
- ✅ **Random Initialization**: Train from scratch without pretrained weights

## Quick Start

### Installation for AMD MI300X (ROCm)

**Important:** For AMD MI300X GPUs, you need PyTorch with ROCm support:

```bash
# Run the installation script (creates myenv venv and installs dependencies)
./install_pytorch_rocm.sh

# Activate the virtual environment
source myenv/bin/activate
```

**Or manually:**
```bash
python3 -m venv myenv
source myenv/bin/activate
sudo apt update
sudo apt install -y libjpeg-dev python3-dev python3-pip
pip3 install wheel setuptools
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.0
pip3 install transformers datasets accelerate psutil pyyaml
```

**Verify GPU detection:**
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### Standard Installation (NVIDIA or CPU)

```bash
pip install -e .
```

### Single GPU Training (8B Model)

```bash
# Make sure venv is activated first
source myenv/bin/activate

./run_training.sh
# or explicitly:
./run_training.sh configs/llama_8b.yaml
```

### Multi-GPU Training (70B Model with FSDP)

```bash
# Make sure venv is activated first
source myenv/bin/activate

# For 8 GPUs:
torchrun --nproc_per_node=8 train_llama_from_scratch.py --config configs/llama_70b.yaml
```

## Configuration Files

### `configs/llama_8b.yaml`
- Model: Llama 3.1 8B
- Batch size: 4 per device
- Sequence length: 8192
- FSDP: Disabled (single GPU)
- Optimized for: Single MI300X

### `configs/llama_70b.yaml`
- Model: Llama 3.1 70B  
- Batch size: 1 per device
- Sequence length: 8192
- FSDP: Enabled (full sharding)
- Optimized for: Multi-GPU (8x MI300X)

## Creating Custom Configs

Copy and modify an existing config:

```bash
cp configs/llama_8b.yaml configs/my_config.yaml
# Edit my_config.yaml as needed
./run_training.sh configs/my_config.yaml
```

### Key Configuration Options

```yaml
model:
  name: "meta-llama/Meta-Llama-3.1-8B"  # Model architecture
  sequence_length: 8192                   # Max sequence length

training:
  per_device_train_batch_size: 4          # Batch size per GPU
  gradient_accumulation_steps: 1          # Gradient accumulation
  learning_rate: 5.0e-5                   # Learning rate
  bf16: true                              # Use BF16 (recommended for MI300X)
  gradient_checkpointing: true            # Save memory

fsdp:
  enabled: true                           # Enable FSDP for multi-GPU
  fsdp_strategy: "full_shard"            # Sharding strategy
```

## Metrics Output

During training, you'll see detailed metrics for each step:

```
================================================================================
Step 1 Metrics (GPU: AMD Instinct MI300X VF):
================================================================================
  Loss:           11.9986
  Tokens/sec:     3883.50
  Step time:      8.438s
  GPU Util:       N/A
  GPU Mem:        45.02GB / 191.69GB (23.5%)
  CPU Util:       0.0%
  Process Mem:    6.05GB
  System Mem:     6.7%
================================================================================
```

**Performance on AMD MI300X:**
- **~3,880 tokens/sec** with batch size 4, sequence length 8192
- **45GB GPU memory** usage for 8B model
- **8.4s per training step** (32,768 tokens per step)

## Architecture

```
llama3.1_benchmark/
├── configs/
│   ├── llama_8b.yaml          # 8B model config
│   └── llama_70b.yaml         # 70B model config (FSDP)
├── train_llama_from_scratch.py  # Main training script
├── run_training.sh            # Convenience script
└── pyproject.toml             # Dependencies
```

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0 (with ROCm for MI300X)
- Transformers >= 4.40.0
- See `pyproject.toml` for full dependencies

## HuggingFace Authentication

To download Llama model configs, you need HuggingFace access:

```bash
huggingface-cli login
```

Then accept the Llama license on HuggingFace: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B

## Notes

- **BF16 vs FP16**: BF16 is recommended for MI300X and large models for better numerical stability
- **FSDP**: Required for 70B model to fit in memory across multiple GPUs
- **Gradient Checkpointing**: Trades compute for memory - essential for large models
- **Dataset**: Default is WikiText-2 for quick testing. For production, use larger datasets (C4, etc.)
