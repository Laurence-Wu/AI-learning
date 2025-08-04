"""
Test script to verify fused_attention.py works in both local and Colab environments.

This script can be run in:
1. Local environment (macOS/CPU/MPS) - uses PyTorch fallback
2. Colab environment (CUDA/GPU) - can use Triton if available

Copy this code to a Colab notebook cell to test the Triton functionality.
"""

# Install requirements in Colab (uncomment if running in Colab)
# !pip install triton torch

import torch
import sys
import os

# Add current directory to path (for Colab)
if '/content' in os.getcwd():
    sys.path.append('/content')

# Import our fused attention implementation
try:
    from fused_attention import attention, HAS_TRITON, DEVICE
    print("✓ Successfully imported fused_attention module")
except ImportError as e:
    print(f"✗ Failed to import fused_attention: {e}")
    exit(1)

def run_comprehensive_test():
    """
    Run comprehensive test of the attention implementation.
    
    MEMORY HIERARCHY TESTING:
    =========================
    This test validates the implementation across different memory hierarchies:
    
    1. **Local System (macOS/MPS)**: Tests unified memory architecture
       - Uses Apple Silicon GPU with shared memory
       - PyTorch fallback with optimized cache usage
       
    2. **Google Colab (CUDA)**: Tests discrete GPU memory hierarchy  
       - HBM main memory + SRAM shared memory
       - Triton-accelerated Flash Attention kernels
       
    3. **CPU Systems**: Tests traditional cache hierarchy
       - L1/L2/L3 caches + system RAM
       - PyTorch with BLAS optimization
       
    The same code automatically adapts to each platform's memory characteristics.
    """
    
    print("Fused Attention Compatibility Test")
    print("=" * 50)
    
    print(f"Device: {DEVICE}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA capability: {torch.cuda.get_device_capability()}")
    print(f"Triton available: {HAS_TRITON}")
    print(f"PyTorch version: {torch.__version__}")
    print()
    
    # Test configurations
    test_configs = [
        {"name": "Small", "batch": 1, "heads": 2, "seq_len": 64, "head_dim": 32},
        {"name": "Medium", "batch": 2, "heads": 8, "seq_len": 256, "head_dim": 64},
        {"name": "Large", "batch": 1, "heads": 16, "seq_len": 512, "head_dim": 128},
    ]
    
    for config in test_configs:
        print(f"Testing {config['name']} configuration...")
        print(f"  Batch: {config['batch']}, Heads: {config['heads']}, Seq: {config['seq_len']}, Dim: {config['head_dim']}")
        
        # Create test tensors
        torch.manual_seed(42)
        q = torch.randn(config['batch'], config['heads'], config['seq_len'], config['head_dim'], 
                       dtype=torch.float16, device=DEVICE)
        k = torch.randn(config['batch'], config['heads'], config['seq_len'], config['head_dim'], 
                       dtype=torch.float16, device=DEVICE)
        v = torch.randn(config['batch'], config['heads'], config['seq_len'], config['head_dim'], 
                       dtype=torch.float16, device=DEVICE)
        
        # Test both causal and non-causal attention
        for causal in [True, False]:
            try:
                output = attention(q, k, v, causal=causal)
                print(f"  ✓ {'Causal' if causal else 'Non-causal'} attention successful: {output.shape}")
                
                # Verify output properties
                assert output.shape == q.shape, f"Shape mismatch: {output.shape} vs {q.shape}"
                assert torch.all(torch.isfinite(output)), "Output contains non-finite values"
                
            except Exception as e:
                print(f"  ✗ {'Causal' if causal else 'Non-causal'} attention failed: {e}")
                return False
        
        print(f"  ✓ {config['name']} configuration passed\n")
    
    # Test gradient computation
    print("Testing gradient computation...")
    q_grad = torch.randn(2, 4, 128, 64, dtype=torch.float16, device=DEVICE, requires_grad=True)
    k_grad = torch.randn(2, 4, 128, 64, dtype=torch.float16, device=DEVICE, requires_grad=True)
    v_grad = torch.randn(2, 4, 128, 64, dtype=torch.float16, device=DEVICE, requires_grad=True)
    
    try:
        output = attention(q_grad, k_grad, v_grad, causal=True)
        loss = output.sum()
        loss.backward()
        
        assert q_grad.grad is not None, "No gradients computed for q"
        assert k_grad.grad is not None, "No gradients computed for k"
        assert v_grad.grad is not None, "No gradients computed for v"
        
        print("✓ Gradient computation successful")
        
    except Exception as e:
        print(f"✗ Gradient computation failed: {e}")
        return False
    
    print("\n" + "=" * 50)
    print("✓ All tests passed successfully!")
    
    if HAS_TRITON:
        print("✨ Using Triton-accelerated kernels")
    else:
        print("🔄 Using PyTorch fallback (no Triton available)")
    
    return True

if __name__ == "__main__":
    success = run_comprehensive_test()
    
    if success:
        print("\n🎉 The fused attention implementation is working correctly!")
        print("📝 You can now use this code in your projects.")
        
        if not HAS_TRITON:
            print("\n📋 To enable Triton acceleration in Colab:")
            print("   1. Upload this file to Google Colab")
            print("   2. Run: !pip install triton")
            print("   3. Use a GPU runtime (Runtime -> Change runtime type -> GPU)")
            print("   4. Run the script again")
    else:
        print("\n❌ Some tests failed. Please check the error messages above.")
        sys.exit(1) 