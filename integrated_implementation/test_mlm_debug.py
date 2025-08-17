#!/usr/bin/env python3
"""
Debug MLM masking issue
"""

import sys
import torch
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root / "src"))

try:
    from transformers import BertTokenizer
    from src.data.mlm_patterns import BERTMLMDataset, MLMConfig
    
    print("Testing MLM masking...")
    
    # Create tokenizer (fallback for testing)
    try:
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    except Exception as e:
        print(f"Could not load tokenizer: {e}")
        sys.exit(1)
    
    # Create MLM config
    config = MLMConfig(mlm_probability=0.15)
    print(f"MLM Config: probability={config.mlm_probability}, strategy={config.strategy}")
    
    # Create test dataset
    test_texts = ["This is a test sentence for MLM masking."]
    dataset = BERTMLMDataset(test_texts, tokenizer, max_length=128, mlm_config=config)
    
    # Test a few samples
    for i in range(3):
        print(f"\n--- Sample {i+1} ---")
        sample = dataset[0]  # Always test the same text for consistency
        
        print(f"Input IDs shape: {sample['input_ids'].shape}")
        print(f"Labels unique values: {torch.unique(sample['labels'])}")
        print(f"Labels range: {sample['labels'].min().item()} to {sample['labels'].max().item()}")
        
        # Count non -100 labels (these are the masked tokens)
        masked_count = (sample['labels'] != -100).sum().item()
        print(f"Masked tokens count: {masked_count}")
        
        if masked_count == 0:
            print("ERROR: No tokens are masked!")
        else:
            print(f"SUCCESS: {masked_count} tokens are masked")
            
except ImportError as e:
    print(f"Import error: {e}")
    print("Transformers library not available - this is expected in some environments")
except Exception as e:
    print(f"Error: {e}")