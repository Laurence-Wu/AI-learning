#!/usr/bin/env python3
"""
Quick test script for the corrected RSE implementation
"""

import torch
import traceback

try:
    from triton_rse_attention_corrected import CorrectedRSEBERTAttention, CorrectedRSEReferenceImplementation
    print("✓ Successfully imported corrected RSE implementation")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    traceback.print_exc()
    exit(1)

def test_basic_functionality():
    """Test basic functionality of corrected RSE"""
    print("\nTesting basic functionality...")
    
    try:
        # Test parameters
        batch_size = 2
        seq_len = 32
        d_model = 128
        n_heads = 8
        
        # Create model
        model = CorrectedRSEBERTAttention(
            d_model=d_model, 
            n_heads=n_heads, 
            max_seq_len=64,
            use_efficient_approx=True  # Use O(n²) approximation
        )
        
        # Test input
        x = torch.randn(batch_size, seq_len, d_model)
        
        # Forward pass
        output = model(x)
        
        print(f"✓ Forward pass successful")
        print(f"  Input shape: {x.shape}")
        print(f"  Output shape: {output[0].shape}")
        print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
        
        # Test gradient computation
        loss = output[0].sum()
        loss.backward()
        
        print(f"✓ Gradient computation successful")
        print(f"  Lambda gradient: {model.lambda_param.grad.item():.6f}")
        
        # Display performance stats
        stats = model.get_performance_stats()
        print(f"\n✓ Performance statistics:")
        for key, value in stats.items():
            print(f"  {key}: {value}")
        
        return True
        
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
        traceback.print_exc()
        return False

def test_reference_implementation():
    """Test the corrected reference implementation directly"""
    print("\nTesting corrected reference implementation...")
    
    try:
        # Test parameters
        batch_size = 1
        n_heads = 4
        seq_len = 8
        head_dim = 16
        lambda_param = 0.1
        
        # Create test tensors
        q = torch.randn(batch_size, n_heads, seq_len, head_dim) * 0.1
        k = torch.randn(batch_size, n_heads, seq_len, head_dim) * 0.1
        v = torch.randn(batch_size, n_heads, seq_len, head_dim)
        
        # Create RoPE cache
        dim_half = head_dim // 2
        positions = torch.arange(seq_len, dtype=torch.float32)
        freq_idx = torch.arange(0, dim_half, dtype=torch.float32)
        inv_freq = 1.0 / (10000.0 ** (freq_idx * 2.0 / head_dim))
        angles = positions[:, None] * inv_freq[None, :]
        cos_cache = torch.cos(angles)
        sin_cache = torch.sin(angles)
        
        # Test exact implementation
        output_exact, attn_exact = CorrectedRSEReferenceImplementation.forward(
            q, k, v, cos_cache, sin_cache, lambda_param, causal=False, use_efficient=False
        )
        
        # Test efficient implementation
        output_efficient, attn_efficient = CorrectedRSEReferenceImplementation.forward(
            q, k, v, cos_cache, sin_cache, lambda_param, causal=False, use_efficient=True
        )
        
        # Check attention properties
        exact_row_sums = torch.sum(attn_exact, dim=-1)
        efficient_row_sums = torch.sum(attn_efficient, dim=-1)
        
        print(f"✓ Reference implementation test successful")
        print(f"  Exact attention row sums: {exact_row_sums.min():.3f} - {exact_row_sums.max():.3f}")
        print(f"  Efficient attention row sums: {efficient_row_sums.min():.3f} - {efficient_row_sums.max():.3f}")
        print(f"  Output difference (L2): {torch.norm(output_exact - output_efficient).item():.6f}")
        
        # Verify attention weights are valid
        assert torch.all(attn_exact >= 0), "Exact attention weights must be non-negative"
        assert torch.all(attn_efficient >= 0), "Efficient attention weights must be non-negative"
        
        print(f"✓ Attention weight validity checks passed")
        
        return True
        
    except Exception as e:
        print(f"✗ Reference implementation test failed: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("Corrected RSE Implementation Test")
    print("=" * 50)
    
    tests = [
        ("Basic Functionality", test_basic_functionality),
        ("Reference Implementation", test_reference_implementation),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * len(test_name))
        if test_func():
            passed += 1
    
    print(f"\n" + "=" * 50)
    print(f"Results: {passed}/{len(tests)} tests passed")
    
    if passed == len(tests):
        print("🎉 All tests passed! Corrected RSE implementation is working.")
    else:
        print("❌ Some tests failed. Please review the implementation.")
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)