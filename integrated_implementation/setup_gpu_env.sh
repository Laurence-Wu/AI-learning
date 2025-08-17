#!/bin/bash
# GPU Environment Setup Script for BERT Training
# ==============================================
# This script sets up the environment for GPU-compatible BERT training

echo "🔧 Setting up GPU environment for BERT training..."

# Activate Triton environment
echo "📦 Activating Triton environment..."
source ~/triton-env-stable/bin/activate

# Set CUDA library paths
echo "🔗 Setting CUDA library paths..."
export LD_LIBRARY_PATH="/usr/lib/wsl/lib:$LD_LIBRARY_PATH"
export CUDA_HOME=/usr/local/cuda

# Set CUDA debugging and compatibility flags
echo "🐛 Setting CUDA debugging flags..."
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1

# Verify CUDA is available
echo "✅ Verifying CUDA setup..."
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name()}')
    print(f'CUDA version: {torch.version.cuda}')
    print(f'cuDNN version: {torch.backends.cudnn.version()}')
else:
    print('⚠️  CUDA not available')
"

echo ""
echo "🎯 Environment setup complete!"
echo "You can now run: python train.py --config config.env"
echo ""

# Keep the environment active
exec "$SHELL"
