#!/usr/bin/env python3
"""
Simple RSE attention test to isolate the triton.language.ones error
"""

import torch
import sys
sys.path.insert(0, 'src')

try:
    from src.attention.rse_attention import RSEBERTAttention
    print("✓ RSE import successful")
    
    # Create simple test
    model = RSEBERTAttention(64, 2, max_position_embeddings=128)
    x = torch.randn(1, 32, 64)
    
    print("Testing forward pass...")
    output = model(x)
    print(f"✓ Forward pass successful: {output[0].shape}")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()