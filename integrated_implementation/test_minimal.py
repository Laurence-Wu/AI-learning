#!/usr/bin/env python3
"""
Minimal test to isolate NaN loss issue
"""

import torch
import torch.nn as nn
from transformers import BertConfig, BertForMaskedLM, BertTokenizer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_basic_bert():
    """Test basic BERT without any modifications"""
    print("Testing basic BERT...")
    
    # Ultra-small configuration
    config = BertConfig(
        vocab_size=1000,  # Very small vocab
        hidden_size=32,   # Very small hidden size
        num_hidden_layers=1,  # Single layer
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
        attention_probs_dropout_prob=0.0,
        hidden_dropout_prob=0.0
    )
    
    # Create model
    model = BertForMaskedLM(config)
    
    # Ultra-conservative initialization
    def init_weights(module):
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.001)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.001)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
    
    model.apply(init_weights)
    
    # Test data
    input_ids = torch.randint(0, 999, (1, 10))
    attention_mask = torch.ones(1, 10)
    labels = input_ids.clone()
    labels[0, 3] = 100  # Mask one token
    
    print(f"Input shape: {input_ids.shape}")
    print(f"Vocab size: {config.vocab_size}")
    print(f"Input max token: {input_ids.max()}")
    
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        print(f"Initial loss: {outputs.loss.item():.6f}")
        print(f"Logits shape: {outputs.logits.shape}")
        print(f"Logits range: {outputs.logits.min():.6f} to {outputs.logits.max():.6f}")
    
    # Test training for a few steps
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6, weight_decay=0.0)
    
    for step in range(10):
        optimizer.zero_grad()
        
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"NaN/Inf loss at step {step}: {loss}")
            print(f"Logits stats: min={outputs.logits.min():.6f}, max={outputs.logits.max():.6f}")
            break
        
        print(f"Step {step}: Loss = {loss.item():.6f}")
        
        loss.backward()
        
        # Check gradients
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if torch.isnan(grad_norm) or torch.isinf(grad_norm):
            print(f"NaN/Inf gradient norm at step {step}: {grad_norm}")
            break
        
        print(f"  Grad norm: {grad_norm:.6f}")
        
        optimizer.step()
    
    print("Basic BERT test completed")

if __name__ == "__main__":
    test_basic_bert()