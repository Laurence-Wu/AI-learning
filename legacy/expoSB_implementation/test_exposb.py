#!/usr/bin/env python3
"""
Test script for ExpoSB attention implementation
Validates the implementation without full training
"""

import sys
import torch
import traceback
from bert_config import BERTComparisonConfig
from triton_exposb_attention import ExpoSBBERTAttention
from triton_standard_attention import StandardBERTAttention

def test_basic_functionality():
    """Test basic forward pass"""
    print("Testing ExpoSB basic functionality...")
    
    try:
        # Test parameters - using config values
        batch_size = 2
        seq_len = 64  # Smaller for testing
        hidden_size = 512  # Match config
        num_heads = 8      # Match config
        
        # Create ExpoSB model
        exposb_model = ExpoSBBERTAttention(hidden_size, num_heads)
        
        # Test input (CPU for compatibility)
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        print(f"Input shape: {x.shape}")
        
        # Forward pass
        output = exposb_model(x)
        print(f"Output shape: {output[0].shape}")
        
        # Test backward pass
        loss = output[0].sum()
        loss.backward()
        print("Gradient computation successful!")
        
        # Parameter count
        exposb_params = sum(p.numel() for p in exposb_model.parameters())
        standard_params = sum(p.numel() for p in StandardBERTAttention(hidden_size, num_heads).parameters())
        
        print(f"ExpoSB parameters: {exposb_params:,}")
        print(f"Standard parameters: {standard_params:,}")
        print(f"Additional parameters: {exposb_params - standard_params:,}")
        
        return True
        
    except Exception as e:
        print(f"Error in basic functionality test: {e}")
        traceback.print_exc()
        return False

def test_configuration():
    """Test configuration loading"""
    print("\nTesting configuration loading...")
    
    try:
        config = BERTComparisonConfig.from_env("config.env")
        print("Configuration loaded successfully")
        
        # Print some key values
        print(f"Hidden size: {config.hidden_size}")
        print(f"Number of layers: {config.num_hidden_layers}")
        print(f"Attention heads: {config.num_attention_heads}")
        print(f"ExpoSB model save path: {config.exposb_model_save_path}")
        
        return True
        
    except Exception as e:
        print(f"Error in configuration test: {e}")
        traceback.print_exc()
        return False

def test_imports():
    """Test all required imports"""
    print("\nTesting imports...")
    
    try:
        # Test critical imports
        from transformers import BertConfig, BertForMaskedLM, BertTokenizer
        print("Transformers imports successful")
        
        from data_preprocessing import load_training_data
        print("Data preprocessing import successful")
        
        import triton
        import triton.language as tl
        print("Triton imports successful")
        
        return True
        
    except Exception as e:
        print(f"Error in import test: {e}")
        traceback.print_exc()
        return False

def test_dimension_consistency():
    """Test that dimensions are consistent across operations"""
    print("\nTesting dimension consistency...")
    
    try:
        batch_size = 1
        seq_len = 32
        hidden_size = 512  # Match config
        num_heads = 8      # Match config
        head_dim = hidden_size // num_heads
        
        model = ExpoSBBERTAttention(hidden_size, num_heads)
        x = torch.randn(batch_size, seq_len, hidden_size)
        
        # Test Q, K, V projections
        q = model.q_proj(x).view(batch_size, seq_len, num_heads, head_dim)
        k = model.k_proj(x).view(batch_size, seq_len, num_heads, head_dim)
        v = model.v_proj(x).view(batch_size, seq_len, num_heads, head_dim)
        
        print(f"Q shape: {q.shape}")
        print(f"K shape: {k.shape}")
        print(f"V shape: {v.shape}")
        
        assert q.shape == k.shape == v.shape, "Q, K, V shapes must match"
        
        # Test full forward pass
        output = model(x)
        assert output[0].shape == x.shape, "Output shape must match input shape"
        
        print("All dimension checks passed!")
        return True
        
    except Exception as e:
        print(f"Error in dimension consistency test: {e}")
        traceback.print_exc()
        return False

def main():
    """Run all tests"""
    print("=" * 60)
    print("ExpoSB Implementation Test Suite")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Configuration Test", test_configuration),
        ("Dimension Consistency Test", test_dimension_consistency),
        ("Basic Functionality Test", test_basic_functionality),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\n{test_name}")
        print("-" * len(test_name))
        success = test_func()
        results.append((test_name, success))
    
    print("\n" + "=" * 60)
    print("Test Results Summary")
    print("=" * 60)
    
    all_passed = True
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name}: {status}")
        if not success:
            all_passed = False
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 All tests passed! ExpoSB implementation is ready.")
        print("You can now run the full training with:")
        print("python bert_comparison_train.py")
    else:
        print("❌ Some tests failed. Please check the implementation.")
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())