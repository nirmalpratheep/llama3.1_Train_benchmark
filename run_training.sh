#!/bin/bash

# Run Llama training from scratch with detailed metrics
# Usage: ./run_training.sh [config_file]
# Example: ./run_training.sh configs/llama_8b.yaml
# Example: ./run_training.sh configs/llama_70b.yaml

# Make sure to activate your virtual environment first:
# source myenv/bin/activate

# Default config
CONFIG="${1:-configs/llama_8b.yaml}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp for log file
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/training_${TIMESTAMP}.log"

echo "Starting Llama training from scratch..."
echo "================================================"
echo "Config file: $CONFIG"
echo "Log file: $LOG_FILE"
echo ""

# Check if CUDA is available
python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('GPU Count:', torch.cuda.device_count() if torch.cuda.is_available() else 0)" | tee -a "$LOG_FILE"
echo ""

# For multi-GPU (FSDP), use torchrun:
# torchrun --nproc_per_node=8 train_llama_from_scratch.py --config $CONFIG 2>&1 | tee -a "$LOG_FILE"

# For single GPU or CPU:
# Redirect both stdout and stderr to log file while also displaying on console
python train_llama_from_scratch.py --config "$CONFIG" 2>&1 | tee -a "$LOG_FILE"

echo ""
echo "Training completed!"
echo "Full log saved to: $LOG_FILE"
