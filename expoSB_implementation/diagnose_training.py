#!/usr/bin/env python3
"""
Training diagnostics script to identify why loss might be linear
"""

import torch
import numpy as np
from data_preprocessing import load_training_data
from transformers import BertTokenizer
from bert_comparison_train import BERTDataset
from torch.utils.data import DataLoader

def diagnose_dataset():
    """Analyze the dataset for potential issues"""
    print("🔍 Dataset Diagnostics")
    print("=" * 50)
    
    # Load data
    texts = load_training_data("dummy.txt")
    print(f"Total texts: {len(texts)}")
    
    if len(texts) == 0:
        print("❌ No data loaded!")
        return
    
    # Analyze text lengths
    word_counts = [len(text.split()) for text in texts]
    char_counts = [len(text) for text in texts]
    
    print(f"\n📊 Text Statistics:")
    print(f"   Words per text: avg={np.mean(word_counts):.1f}, min={min(word_counts)}, max={max(word_counts)}")
    print(f"   Chars per text: avg={np.mean(char_counts):.1f}, min={min(char_counts)}, max={max(char_counts)}")
    
    # Check for repetitive content
    unique_texts = set(texts)
    print(f"   Unique texts: {len(unique_texts)} / {len(texts)} ({len(unique_texts)/len(texts)*100:.1f}% unique)")
    
    # Sample some texts
    print(f"\n📝 Sample texts:")
    for i, text in enumerate(texts[:3]):
        print(f"Text {i+1}: {text[:100]}...")
    
    return texts

def diagnose_tokenization():
    """Check tokenization issues"""
    print("\n🔤 Tokenization Diagnostics") 
    print("=" * 50)
    
    try:
        tokenizer = BertTokenizer.from_pretrained("./local_tokenizer")
        print("✓ Loaded local tokenizer")
    except:
        print("⚠️  Loading tokenizer from HuggingFace...")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    texts = load_training_data("dummy.txt")
    
    # Create sample dataset
    dataset = BERTDataset(texts[:100], tokenizer, max_length=256, mlm_probability=0.15)
    
    # Analyze tokenization
    sample_batch = []
    for i in range(min(10, len(dataset))):
        sample_batch.append(dataset[i])
    
    # Check input_ids statistics
    input_lengths = [torch.sum(item['attention_mask']).item() for item in sample_batch]
    label_counts = [torch.sum(item['labels'] != -100).item() for item in sample_batch]
    
    print(f"   Input lengths: avg={np.mean(input_lengths):.1f}, min={min(input_lengths)}, max={max(input_lengths)}")
    print(f"   Masked tokens per sample: avg={np.mean(label_counts):.1f}, min={min(label_counts)}, max={max(label_counts)}")
    
    # Check if we have proper masking
    if max(label_counts) == 0:
        print("❌ No tokens are being masked! MLM training will fail.")
    elif np.mean(label_counts) < 5:
        print("⚠️  Very few tokens being masked - training might be slow.")
    else:
        print("✓ Masking looks good")
    
    return dataset

def diagnose_model_capacity():
    """Estimate if model capacity matches dataset size"""
    print("\n🧠 Model Capacity Diagnostics")
    print("=" * 50)
    
    # Model parameters
    hidden_size = 384
    num_layers = 6 
    num_heads = 6
    vocab_size = 30522
    
    # Rough parameter count estimate
    attention_params = num_layers * (4 * hidden_size * hidden_size)  # Q, K, V, O projections
    ffn_params = num_layers * (2 * hidden_size * 1536)  # FFN layers
    embedding_params = vocab_size * hidden_size + 512 * hidden_size  # Token + position embeddings
    
    total_params = attention_params + ffn_params + embedding_params
    print(f"   Estimated model parameters: ~{total_params/1e6:.1f}M")
    
    # Dataset size
    texts = load_training_data("dummy.txt")
    total_words = sum(len(text.split()) for text in texts)
    print(f"   Dataset size: ~{total_words/1e3:.1f}k words")
    
    # Rule of thumb: need ~10-100 words per parameter for good generalization
    words_per_param = total_words / total_params
    print(f"   Words per parameter: {words_per_param:.2f}")
    
    if words_per_param < 0.1:
        print("❌ Severe overfitting likely - too few words per parameter")
    elif words_per_param < 1.0:
        print("⚠️  May overfit - consider more data or smaller model")
    else:
        print("✓ Good balance of model size and data")

def diagnose_learning_rate():
    """Suggest learning rate based on dataset size"""
    print("\n📈 Learning Rate Diagnostics")
    print("=" * 50)
    
    texts = load_training_data("dummy.txt")
    batch_size = 16
    grad_accum = 4
    effective_batch_size = batch_size * grad_accum
    
    steps_per_epoch = len(texts) // effective_batch_size
    total_steps = steps_per_epoch * 10  # 10 epochs
    
    print(f"   Dataset size: {len(texts)} samples")
    print(f"   Effective batch size: {effective_batch_size}")
    print(f"   Steps per epoch: {steps_per_epoch}")
    print(f"   Total training steps: {total_steps}")
    
    # Recommend learning rate based on effective batch size
    if effective_batch_size <= 32:
        suggested_lr = 5e-5
    elif effective_batch_size <= 128:
        suggested_lr = 1e-4
    else:
        suggested_lr = 2e-4
    
    print(f"   Suggested learning rate: {suggested_lr}")
    print(f"   Suggested warmup steps: {total_steps // 10}")

def main():
    """Run all diagnostics"""
    print("🏥 BERT Training Diagnostics")
    print("=" * 60)
    
    try:
        diagnose_dataset()
        diagnose_tokenization() 
        diagnose_model_capacity()
        diagnose_learning_rate()
        
        print("\n💡 Common causes of linear loss:")
        print("   1. ❌ Bad learning rate schedule (FIXED)")
        print("   2. ❌ No weight decay or wrong optimizer (FIXED)")
        print("   3. ⚠️  Model too large for dataset")
        print("   4. ⚠️  Learning rate too high")
        print("   5. ⚠️  No proper regularization")
        print("   6. ⚠️  Repetitive/poor quality data")
        
        print(f"\n✅ Run training with the updated configuration to see improvement!")
        
    except Exception as e:
        print(f"❌ Diagnostic failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()


