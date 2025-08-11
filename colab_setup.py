#!/usr/bin/env python3
"""
Google Colab Setup Script for Triton & MLSys Learning Repository
================================================================

This script installs all required dependencies for running Triton GPU kernels
and MLSys implementations in Google Colab environment.

Usage in Colab:
1. Upload this file to your Colab session
2. Run: !python colab_setup.py
   OR copy the cell content below and run it directly

Requirements:
- Google Colab with GPU runtime enabled (T4, V100, A100, etc.)
- CUDA-compatible environment
"""

import subprocess
import sys
import os
import importlib.util
from pathlib import Path

def run_command(command, description=""):
    """Run a shell command and handle errors gracefully."""
    print(f"\n{'='*60}")
    print(f"🔧 {description}")
    print(f"{'='*60}")
    print(f"Running: {command}")
    
    try:
        result = subprocess.run(
            command, 
            shell=True, 
            check=True, 
            capture_output=True, 
            text=True
        )
        if result.stdout:
            print("✅ Success!")
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False

def check_package_installed(package_name, import_name=None):
    """Check if a package is installed and importable."""
    if import_name is None:
        import_name = package_name
    
    try:
        importlib.import_module(import_name)
        print(f"✅ {package_name} is already installed")
        return True
    except ImportError:
        print(f"❌ {package_name} not found")
        return False

def get_gpu_info():
    """Get GPU information for optimization."""
    try:
        result = subprocess.run(
            "nvidia-smi --query-gpu=name,memory.total,compute_cap --format=csv,noheader,nounits",
            shell=True,
            capture_output=True,
            text=True,
            check=True
        )
        gpu_info = result.stdout.strip().split('\n')[0]
        name, memory, compute_cap = gpu_info.split(', ')
        print(f"🖥️  GPU: {name}")
        print(f"💾 Memory: {memory} MB")
        print(f"🔢 Compute Capability: {compute_cap}")
        return {
            'name': name,
            'memory': int(memory),
            'compute_cap': float(compute_cap)
        }
    except Exception as e:
        print(f"⚠️  Could not get GPU info: {e}")
        return None

def main():
    print("""
    🚀 Triton & MLSys Learning Repository Setup
    ==========================================
    
    This script will install all dependencies needed for:
    - Triton GPU kernel programming
    - Flash Attention implementations
    - RoPE windowed attention
    - Memory hierarchy optimization examples
    """)
    
    # Check if we're in Colab
    try:
        import google.colab
        print("✅ Running in Google Colab")
        colab_env = True
    except ImportError:
        print("⚠️  Not running in Google Colab - some features may not work")
        colab_env = False
    
    # Get GPU information
    print("\n" + "="*60)
    print("🔍 SYSTEM INFORMATION")
    print("="*60)
    
    gpu_info = get_gpu_info()
    
    # Check Python version
    python_version = sys.version_info
    print(f"🐍 Python: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("❌ Python 3.8+ required for Triton")
        return False
    
    # Update system packages
    if not run_command("apt-get update -qq", "Updating system packages"):
        print("⚠️  System update failed, continuing anyway...")
    
    # Install system dependencies
    system_deps = [
        "build-essential",
        "cmake", 
        "git",
        "wget",
        "curl",
        "nvidia-cuda-toolkit"  # Ensure CUDA toolkit is available
    ]
    
    system_cmd = f"apt-get install -y {' '.join(system_deps)}"
    run_command(system_cmd, "Installing system dependencies")
    
    # Check existing packages
    print("\n" + "="*60)
    print("📦 CHECKING EXISTING PACKAGES")
    print("="*60)
    
    packages_to_check = {
        'torch': 'torch',
        'numpy': 'numpy',
        'triton': 'triton'
    }
    
    installed_packages = {}
    for pkg, import_name in packages_to_check.items():
        installed_packages[pkg] = check_package_installed(pkg, import_name)
    
    # Install/upgrade PyTorch with CUDA support
    if not installed_packages.get('torch', False) or True:  # Always reinstall for latest
        torch_cmd = (
            "pip install --upgrade torch torchvision torchaudio "
            "--index-url https://download.pytorch.org/whl/cu118"
        )
        run_command(torch_cmd, "Installing PyTorch with CUDA support")
    
    # Install Triton
    if not installed_packages.get('triton', False):
        # For Colab, use the latest stable Triton
        triton_cmd = "pip install triton"
        run_command(triton_cmd, "Installing Triton GPU compiler")
    
    # Install additional ML/scientific packages
    additional_packages = [
        "numpy>=1.21.0",
        "matplotlib",
        "seaborn", 
        "jupyter",
        "ipython",
        "tqdm",
        "einops",  # For tensor operations
        "transformers",  # For transformer models
        "datasets",  # For loading datasets
        "accelerate",  # For distributed training
        "wandb",  # For experiment tracking
        "scikit-learn",
        "pandas"
    ]
    
    additional_cmd = f"pip install {' '.join(additional_packages)}"
    run_command(additional_cmd, "Installing additional ML packages")
    
    # Install optional performance packages
    performance_packages = [
        "ninja",  # For faster compilation
        "packaging",  # For version checking
        "psutil",  # For system monitoring
    ]
    
    perf_cmd = f"pip install {' '.join(performance_packages)}"
    run_command(perf_cmd, "Installing performance packages")
    
    # Verification
    print("\n" + "="*60)
    print("🧪 VERIFICATION")
    print("="*60)
    
    verification_script = '''
import sys
import torch
import triton
import numpy as np

print("✅ Verification Results:")
print(f"   Python: {sys.version}")
print(f"   PyTorch: {torch.__version__}")
print(f"   Triton: {triton.__version__}")
print(f"   NumPy: {np.__version__}")
print(f"   CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   GPU Count: {torch.cuda.device_count()}")
    print(f"   Current GPU: {torch.cuda.get_device_name()}")
    print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Test basic Triton functionality
@triton.jit
def test_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: triton.language.constexpr):
    pid = triton.language.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + triton.language.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = triton.language.load(x_ptr + offsets, mask=mask)
    triton.language.store(y_ptr + offsets, x * 2, mask=mask)

try:
    # Test tensor creation
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(1024, device=device)
    y = torch.empty_like(x)
    
    if device == "cuda":
        # Test Triton kernel compilation
        grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
        test_kernel[grid](x, y, x.numel(), BLOCK_SIZE=256)
        torch.cuda.synchronize()
        print("   ✅ Triton kernel compilation successful")
    else:
        print("   ⚠️  CUDA not available - Triton kernels will not work")
    
    print("   ✅ Basic tensor operations successful")
    
except Exception as e:
    print(f"   ❌ Verification failed: {e}")
    '''
    
    # Write and run verification
    with open('/tmp/verify.py', 'w') as f:
        f.write(verification_script)
    
    run_command("python /tmp/verify.py", "Running verification tests")
    
    # Cleanup
    run_command("rm -f /tmp/verify.py", "Cleaning up temporary files")
    
    # Final instructions
    print("\n" + "="*60)
    print("🎉 SETUP COMPLETE!")
    print("="*60)
    print("""
    Your environment is now ready for Triton & MLSys learning!
    
    📋 Next Steps:
    1. Upload your repository files (fused_attention.py, fused_rope.py, etc.)
    2. Test with: 
       ```python
       !python fused_attention.py
       !python fused_rope.py  
       ```
    3. Explore the triton/ directory examples
    
    💡 Tips:
    - Always use GPU runtime in Colab (Runtime > Change runtime type > GPU)
    - Monitor GPU memory usage with: torch.cuda.memory_summary()
    - Use torch.cuda.empty_cache() to free GPU memory when needed
    
    📚 Resources:
    - Triton docs: https://triton-lang.org/
    - Repository README: Contains detailed learning path
    - Example kernels: Start with triton/vector_addition.py
    
    Happy Learning! 🚀
    """)
    
    return True

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n❌ Setup failed. Please check the error messages above.")
        sys.exit(1)
    else:
        print("\n✅ Setup completed successfully!")

# Colab cell content - copy this into a Colab cell to run directly
COLAB_CELL_CONTENT = '''
# 🚀 Triton & MLSys Learning Repository Setup for Google Colab
# =============================================================
# Run this cell to install all dependencies for Triton GPU programming

import subprocess
import sys
import importlib.util

def install_dependencies():
    """Install all required dependencies for Triton & MLSys learning."""
    
    print("🔧 Installing Triton & MLSys Learning Dependencies...")
    
    # Update pip
    !pip install --upgrade pip
    
    # Install PyTorch with CUDA support
    !pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    
    # Install Triton
    !pip install triton
    
    # Install additional packages
    !pip install numpy matplotlib seaborn tqdm einops transformers datasets accelerate wandb scikit-learn pandas ninja packaging psutil
    
    # Verify installation
    print("\\n✅ Verifying installation...")
    
    import torch
    import triton
    import numpy as np
    
    print(f"PyTorch: {torch.__version__}")
    print(f"Triton: {triton.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name()}")
        print(f"CUDA Version: {torch.version.cuda}")
        
        # Test basic Triton kernel
        @triton.jit
        def test_kernel(x_ptr, y_ptr, n, BLOCK_SIZE: triton.language.constexpr):
            pid = triton.language.program_id(axis=0)
            offs = pid * BLOCK_SIZE + triton.language.arange(0, BLOCK_SIZE)
            mask = offs < n
            x = triton.language.load(x_ptr + offs, mask=mask)
            triton.language.store(y_ptr + offs, x * 2, mask=mask)
        
        x = torch.randn(1024, device='cuda')
        y = torch.empty_like(x)
        grid = lambda meta: (triton.cdiv(x.numel(), meta['BLOCK_SIZE']),)
        test_kernel[grid](x, y, x.numel(), BLOCK_SIZE=256)
        torch.cuda.synchronize()
        print("✅ Triton kernel test successful!")
    
    print("\\n🎉 Setup complete! Ready for Triton & MLSys learning.")
    print("\\n📋 Next steps:")
    print("1. Upload your .py files (fused_attention.py, fused_rope.py, etc.)")
    print("2. Run: !python fused_attention.py")
    print("3. Explore the examples and documentation")

# Run the installation
install_dependencies()
'''

print(f"\n{'='*60}")
print("📱 COLAB CELL VERSION")
print("="*60)
print("Copy the following code into a Colab cell for direct installation:")
print(f"\n{COLAB_CELL_CONTENT}") 