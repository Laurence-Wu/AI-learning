#!/usr/bin/env python3
"""
Simple test to verify MLM masking is working
"""

import sys
import torch
from pathlib import Path

# Add project root and src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

# Test direct import
print("Testing MLM masking...")
print("Python path:", sys.path[:3])

try:
    # Force reload imports
    import importlib
    
    # Import and test the dataset
    from src.data.mlm_patterns import BERTMLMDataset, MLMConfig
    print("✓ Successfully imported MLM classes")
    
    # Create a simple test tokenizer equivalent
    class MockTokenizer:
        def __init__(self):
            self.cls_token_id = 101
            self.sep_token_id = 102
            self.pad_token_id = 0
            self.mask_token_id = 103
            self.vocab = list(range(1000))
            
        def __len__(self):
            return len(self.vocab)
            
        def __call__(self, text, **kwargs):
            # Simple mock tokenization
            words = text.split()
            input_ids = [self.cls_token_id] + [200 + i for i in range(len(words))] + [self.sep_token_id]
            # Pad to max_length if specified
            max_length = kwargs.get('max_length', 128)
            while len(input_ids) < max_length:
                input_ids.append(self.pad_token_id)
            input_ids = input_ids[:max_length]
            
            attention_mask = [1 if id != self.pad_token_id else 0 for id in input_ids]
            
            return {
                'input_ids': torch.tensor(input_ids).unsqueeze(0),
                'attention_mask': torch.tensor(attention_mask).unsqueeze(0)
            }
    
    # Test the MLM dataset
    tokenizer = MockTokenizer()
    config = MLMConfig(mlm_probability=0.15)
    
    print(f"✓ Created MLM config with probability: {config.mlm_probability}")
    
    dataset = BERTMLMDataset(
        ["This is a test sentence for MLM masking verification."],
        tokenizer,
        max_length=64,
        mlm_config=config
    )
    
    print("✓ Created MLM dataset")
    
    # Test multiple samples to ensure masking is working
    for i in range(5):
        sample = dataset[0]
        labels = sample['labels']
        
        # Count non -100 labels (these are the masked tokens)
        masked_count = (labels != -100).sum().item()
        total_labels = len(labels)
        
        print(f"Sample {i+1}: {masked_count}/{total_labels} tokens masked")
        print(f"  Labels range: {labels.min().item()} to {labels.max().item()}")
        print(f"  Unique label values: {len(torch.unique(labels))}")
        
        if masked_count > 0:
            print(f"  ✓ SUCCESS: Masking is working!")
            break
        else:
            print(f"  ✗ ERROR: No tokens are masked!")
    
    print("\nTest completed.")
    
except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()