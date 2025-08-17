#!/bin/bash
# A100 Training Launch Script
# ==========================

# Set environment variables for optimal A100 performance
export CUDA_LAUNCH_BLOCKING=0
export TORCH_CUDA_ARCH_LIST="8.0"  # A100 architecture
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

# Enable TF32 for A100
export TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1
export NVIDIA_TF32_OVERRIDE=1

# Memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
export CUDA_MODULE_LOADING=LAZY

# Detect A100 variant and adjust config
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -n1)
echo "Detected GPU Memory: ${GPU_MEM} MB"

if [ "$GPU_MEM" -gt "70000" ]; then
    echo "Detected A100 80GB - Using large batch configuration"
    export BATCH_SIZE=128
    export GRADIENT_ACCUMULATION_STEPS=2
    export EVAL_BATCH_SIZE=256
else
    echo "Detected A100 40GB - Using standard configuration"
    export BATCH_SIZE=64
    export GRADIENT_ACCUMULATION_STEPS=4
    export EVAL_BATCH_SIZE=128
fi

# Clear cache before starting
python -c "import torch; torch.cuda.empty_cache()"

# Launch training with configuration
echo "Starting A100-optimized training..."
echo "=================================="

# Check if custom config is provided
CONFIG_FILE=${1:-"configs/a100_config.env"}
echo "Using configuration: $CONFIG_FILE"

# Run the training
python train_comparison.py \
    --config "$CONFIG_FILE" \
    --device cuda \
    --mixed-precision \
    --compile \
    2>&1 | tee logs/a100_training_$(date +%Y%m%d_%H%M%S).log

# Post-training cleanup
python -c "import torch; torch.cuda.empty_cache()"

echo "Training completed!"
echo "Check outputs/a100_runs for results"