# Llama 3.1 Training from Scratch — AMD MI300X

PyTorch-based training script for Llama 3.1 models (8B, 70B) from scratch with comprehensive metrics logging, FSDP support, **GPU kernel profiling**, and **custom Triton kernel optimizations** for AMD MI300X.

## Features

- ✅ **Modular Configuration**: YAML-based configs for different model sizes
- ✅ **FSDP Support**: Fully Sharded Data Parallel for multi-GPU training
- ✅ **Detailed Metrics**: Real-time tracking of tokens/sec, GPU/CPU utilization, memory stats
- ✅ **AMD MI300X Optimized**: BF16, ROCm-compatible monitoring, TunableOp GEMM tuning
- ✅ **GPU Kernel Profiling**: PyTorch Profiler + rocprofv3 integration
- ✅ **Custom Triton Kernels**: Fused RMSNorm, RoPE, and Cross-Entropy for MI300X
- ✅ **torch.compile Support**: Additional kernel fusion via TorchInductor

---

## Quick Start

### Installation for AMD MI300X (ROCm)

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
sudo apt update && sudo apt install -y libjpeg-dev python3-dev python3-pip
pip3 install wheel setuptools
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/rocm7.0
pip3 install transformers datasets accelerate psutil pyyaml triton
```

**Verify GPU detection:**
```bash
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"
```

### Standard Installation (NVIDIA or CPU)

```bash
pip install -e .
```

---

## Training

### Single GPU Training (8B Model)

```bash
source myenv/bin/activate
./run_training.sh
# or:
./run_training.sh configs/llama_8b.yaml
```

### Multi-GPU Training (70B Model with FSDP)

```bash
source myenv/bin/activate
torchrun --nproc_per_node=8 train_llama_from_scratch.py --config configs/llama_70b.yaml
```

### Optimized Training (with Custom Kernels)

```bash
# All optimizations: custom Triton kernels + torch.compile + TunableOp
./run_training_optimized.sh

# Or run manually with specific flags:
python train_optimized.py --config configs/llama_8b.yaml --use-custom-kernels
python train_optimized.py --config configs/llama_8b.yaml --use-custom-kernels --use-torch-compile
python train_optimized.py --config configs/llama_8b.yaml --use-custom-kernels --tune-gemm
```

| Flag | Description |
|------|-------------|
| `--use-custom-kernels` | Enable fused Triton kernels (RMSNorm, RoPE, CrossEntropy) |
| `--use-torch-compile` | Enable `torch.compile` with Inductor backend |
| `--tune-gemm` | Auto-tune GEMM operations via TunableOp (slower first run) |
| `--no-tunable-op` | Disable TunableOp entirely |

---

## GPU Kernel Profiling

### Step 1: Profile with PyTorch Profiler

Captures GPU kernel-level traces and identifies hotspots:

```bash
python profile_training.py --config configs/llama_8b.yaml --num-steps 5
```

**Options:**
| Argument | Default | Description |
|----------|---------|-------------|
| `--num-steps` | 5 | Number of profiling steps |
| `--warmup-steps` | 3 | Warmup steps before profiling |
| `--output-dir` | `profiling_output` | Output directory |

**Output:**
- `profiling_output/chrome_trace.json` — Visual trace (load in [Perfetto UI](https://ui.perfetto.dev/))
- `profiling_output/kernel_summary.json` — Machine-readable kernel data
- Console: Top 30 GPU kernels sorted by total CUDA time

### Step 2: Analyze Profiling Results

Parse and categorize kernels with optimization recommendations:

```bash
python analyze_profile.py --input profiling_output
```

**Output includes:**
- Per-kernel ranking by GPU time
- Category breakdown (GEMM, Normalization, Attention, RoPE, Loss, etc.)
- Optimization candidates (non-GEMM kernels > 2% of total GPU time)
- Specific recommended actions for each category

### Step 3: Hardware-Level Profiling with rocprofv3

For deeper hardware counter analysis:

```bash
# HIP API + kernel dispatch tracing
./run_rocprof.sh trace

# Hardware performance counters (FLOPS, cache hits, occupancy)
./run_rocprof.sh counters

# With custom config
./run_rocprof.sh trace configs/llama_8b.yaml
```

---

## Custom Triton Kernels

Three fused Triton kernels optimized for MI300X, targeting the most common non-GEMM hotspots in Llama training:

### Fused RMSNorm (`custom_kernels/fused_rmsnorm.py`)
- **What it does:** Fuses variance computation, normalization, and weight scaling into a single GPU kernel
- **Why it matters:** Standard PyTorch RMSNorm launches 3+ separate kernels with intermediate global memory reads/writes
- **Where it's used:** Called 2× per transformer block (64 times total for 8B model's 32 layers)

### Fused RoPE (`custom_kernels/fused_rope.py`)
- **What it does:** Fuses sin/cos computation with the rotation application into one kernel
- **Why it matters:** Avoids materializing full sin/cos tensors in GPU global memory
- **Config:** Supports Llama 3.1 RoPE (theta=500000, non-interleaved)

### Fused Cross-Entropy (`custom_kernels/fused_cross_entropy.py`)
- **What it does:** Computes log-softmax + NLL loss in a single pass using online softmax
- **Why it matters:** Llama 3.1 has a 128K vocabulary — standard cross-entropy materializes a full `[batch×seq, 128256]` tensor (~2GB per batch at bf16). The fused version uses O(chunk_size) intermediate memory.

### Testing Kernel Correctness

```bash
python -m custom_kernels.test_kernels
```

Tests compare each Triton kernel against PyTorch reference implementations:
- Forward pass output matching
- Backward pass gradient matching
- Representative shapes from Llama 3.1 8B (hidden_size=4096, vocab_size=128256)
- BF16 precision tolerance

---

## MI300X Environment Optimizations

The optimized training script (`run_training_optimized.sh`) sets these environment variables:

| Variable | Value | Effect |
|----------|-------|--------|
| `HIP_FORCE_DEV_KERNARG` | `1` | Reduces kernel argument passing latency by 2-3μs |
| `HSA_ENABLE_SDMA` | `0` | Can improve memory transfer overlap |
| `PYTORCH_TUNABLEOP_ENABLED` | `1` | Enables GEMM auto-tuning via rocBLAS/hipBLASLt |
| `PYTORCH_HIP_ALLOC_CONF` | `expandable_segments:True` | Better memory pool management for large models |

### TunableOp GEMM Auto-Tuning

TunableOp automatically explores different GEMM algorithms on MI300X to find the fastest:

```bash
# First run: tune GEMMs (slower, saves results to CSV)
python train_optimized.py --config configs/llama_8b.yaml --use-custom-kernels --tune-gemm

# Subsequent runs: reuse tuning results (fast)
python train_optimized.py --config configs/llama_8b.yaml --use-custom-kernels
```

Tuning results are saved to `output/tunableop_results.csv` and reused automatically.

---

## Configuration Files

### `configs/llama_8b.yaml`
- Model: Llama 3.1 8B (4096 hidden, 32 layers, 32 heads)
- Batch size: 4 per device, sequence length: 8192
- FSDP: Disabled (single GPU)

### `configs/llama_70b.yaml`
- Model: Llama 3.1 70B
- Batch size: 1 per device, sequence length: 8192
- FSDP: Enabled (full sharding across 8 GPUs)

### Creating Custom Configs

```bash
cp configs/llama_8b.yaml configs/my_config.yaml
# Edit as needed, then:
python train_optimized.py --config configs/my_config.yaml --use-custom-kernels
```

---

## Metrics Output

During training, you'll see detailed metrics for each step:

```
================================================================================
Step 1 Metrics (GPU: AMD Instinct MI300X VF):
================================================================================
  Loss:           11.9986
  Tokens/sec:     5635.00
  Step time:      5.815s
  GPU Util:       N/A
  GPU Mem:        45.02GB / 191.69GB (23.5%)
  CPU Util:       2.1%
  Process Mem:    6.05GB
  System Mem:     6.7%
================================================================================
```

---

## Project Structure

```
llama3.1_Train_benchmark/
├── configs/
│   ├── llama_8b.yaml                  # 8B model config
│   └── llama_70b.yaml                 # 70B model config (FSDP)
├── custom_kernels/                    # Triton kernel optimizations
│   ├── __init__.py                    # patch_model() entry point
│   ├── fused_rmsnorm.py               # Fused RMSNorm kernel
│   ├── fused_rope.py                  # Fused Rotary Embedding kernel
│   ├── fused_cross_entropy.py         # Fused Cross-Entropy kernel
│   └── test_kernels.py               # Correctness tests
├── train_llama_from_scratch.py        # Baseline training script
├── train_optimized.py                 # Optimized training with custom kernels
├── profile_training.py                # PyTorch Profiler wrapper
├── analyze_profile.py                 # Profiling analysis & recommendations
├── run_training.sh                    # Baseline launch script
├── run_training_optimized.sh          # Optimized launch script (MI300X env vars)
├── run_rocprof.sh                     # rocprofv3 profiling launcher
└── install_pytorch_rocm.sh            # ROCm installation script
```

---

## Requirements

- Python >= 3.8
- PyTorch >= 2.0.0 (with ROCm for MI300X)
- Triton (for custom kernels — optional, falls back gracefully)
- Transformers >= 4.40.0
- See `install_pytorch_rocm.sh` for full dependencies

## HuggingFace Authentication

```bash
huggingface-cli login
```

Then accept the Llama license: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B

## Notes

- **BF16 vs FP16**: BF16 is recommended for MI300X for better numerical stability
- **FSDP**: Required for 70B model to fit in memory across multiple GPUs
- **Gradient Checkpointing**: Trades compute for memory — essential for large models
- **Custom Kernels**: Require Triton. If unavailable, training falls back to standard PyTorch ops
- **Dataset**: Default is WikiText-2 for quick testing. For production, use larger datasets (C4, The Pile, etc.)
