#!/bin/bash

# =============================================================================
# Optimized Training Launch Script for AMD MI300X
# =============================================================================
#
# Usage:
#   ./run_training_optimized.sh                                    # All optimizations
#   ./run_training_optimized.sh configs/llama_8b.yaml              # With config
#   ./run_training_optimized.sh configs/llama_8b.yaml --tune-gemm  # Auto-tune GEMMs
#
# =============================================================================

set -e

CONFIG="${1:-configs/llama_8b.yaml}"
shift 2>/dev/null || true
EXTRA_ARGS="$@"

# Create dirs
mkdir -p logs output

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training_optimized_${TIMESTAMP}.log"

echo "============================================================"
echo "Optimized Llama Training for AMD MI300X"
echo "============================================================"
echo "  Config:        ${CONFIG}"
echo "  Log file:      ${LOG_FILE}"
echo "  Extra args:    ${EXTRA_ARGS}"
echo ""

# ─── MI300X Environment Variables ─────────────────────────────────
# Kernel argument passing optimization
export HIP_FORCE_DEV_KERNARG=1

# Disable SDMA for potentially better overlap
export HSA_ENABLE_SDMA=0

# TunableOp: auto-tune GEMM kernels (rocBLAS/hipBLASLt)
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_FILENAME=./output/tunableop_results.csv

# Memory pool settings for large models
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True

echo "Environment variables set:"
echo "  HIP_FORCE_DEV_KERNARG=$HIP_FORCE_DEV_KERNARG"
echo "  HSA_ENABLE_SDMA=$HSA_ENABLE_SDMA"
echo "  PYTORCH_TUNABLEOP_ENABLED=$PYTORCH_TUNABLEOP_ENABLED"
echo "  PYTORCH_HIP_ALLOC_CONF=$PYTORCH_HIP_ALLOC_CONF"
echo ""

# ─── GPU Check ────────────────────────────────────────────────────
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB')
try:
    import triton
    print(f'Triton: {triton.__version__}')
except ImportError:
    print('Triton: NOT AVAILABLE (custom kernels will be disabled)')
" | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"

# ─── Run Training ────────────────────────────────────────────────
echo "Starting optimized training..." | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# For multi-GPU (FSDP):
# torchrun --nproc_per_node=8 train_optimized.py --config $CONFIG --use-custom-kernels --use-torch-compile $EXTRA_ARGS 2>&1 | tee -a "$LOG_FILE"

# Single GPU:
python train_optimized.py \
    --config "$CONFIG" \
    --use-custom-kernels \
    --use-torch-compile \
    $EXTRA_ARGS \
    2>&1 | tee -a "$LOG_FILE"

echo "" | tee -a "$LOG_FILE"
echo "Training completed!" | tee -a "$LOG_FILE"
echo "Full log: $LOG_FILE" | tee -a "$LOG_FILE"
