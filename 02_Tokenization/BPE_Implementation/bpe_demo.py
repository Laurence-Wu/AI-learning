#!/usr/bin/env python3
"""
BPE Tokenization Demo
Demonstrates various aspects of Byte Pair Encoding tokenization
"""

from context import BPE
import os

def load_bpe_model(codes_file):
    """Load BPE model from codes file"""
    with open(codes_file, 'r', encoding='utf-8') as f:
        return BPE(f)

def demonstrate_tokenization():
    """Demonstrate BPE tokenization with various examples"""
    
    # Load the trained BPE model
    bpe = load_bpe_model("current_vocab.txt")
    
    print("=== BPE Tokenization Demonstration ===\n")
    
    # Test cases demonstrating different aspects of BPE
    test_cases = [
        # Simple words
        ("hello", "Simple word"),
        ("world", "Another simple word"),
        
        # Compound words
        ("understanding", "Long word - should be broken down"),
        ("unbelievable", "Another long word"),
        
        # Common vs. rare words
        ("the", "Very common word"),
        ("pneumonoultramicroscopicsilicovolcanocon", "Very rare/long word"),
        
        # Numbers and punctuation
        ("123", "Numbers"),
        ("hello, world!", "With punctuation"),
        
        # Multiple words
        ("Hello world, how are you today?", "Full sentence"),
        ("The quick brown fox jumps over the lazy dog.", "Pangram sentence"),
        
        # Domain-specific text (if relevant to training data)
        ("mortality", "Word from training corpus"),
        ("philosophy", "Another domain word"),
    ]
    
    for text, description in test_cases:
        print(f"Text: '{text}' ({description})")
        tokens = bpe.segment(text).split()
        print(f"Tokens: {tokens}")
        print(f"Token count: {len(tokens)}")
        
        # Calculate compression
        original_chars = len(text.replace(' ', ''))
        token_chars = sum(len(token.replace('@@', '')) for token in tokens)
        print(f"Character preservation: {token_chars}/{original_chars}")
        print("-" * 50)

def interactive_tokenizer():
    """Interactive tokenizer for user input"""
    bpe = load_bpe_model("current_vocab.txt")
    
    print("\n=== Interactive BPE Tokenizer ===")
    print("Enter text to tokenize (type 'quit' to exit):")
    
    while True:
        text = input("\n> ").strip()
        if text.lower() in ['quit', 'exit', 'q']:
            break
        
        if text:
            tokens = bpe.segment(text).split()
            print(f"Tokens: {tokens}")
            print(f"Count: {len(tokens)}")
            
            # Show token breakdown
            reconstructed = ''.join(token.replace('@@', '') for token in tokens)
            print(f"Reconstructed: '{reconstructed}'")

def analyze_vocabulary():
    """Analyze the learned BPE vocabulary"""
    print("\n=== BPE Vocabulary Analysis ===")
    
    with open("current_vocab.txt", 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    # Skip the version line
    bpe_codes = [line.strip() for line in lines[1:] if line.strip()]
    
    print(f"Total BPE merge operations: {len(bpe_codes)}")
    print(f"First 10 operations:")
    for i, code in enumerate(bpe_codes[:10], 1):
        print(f"  {i}: {code}")
    
    print(f"\nLast 10 operations:")
    for i, code in enumerate(bpe_codes[-10:], len(bpe_codes)-9):
        print(f"  {i}: {code}")
    
    # Analyze patterns
    word_final_ops = [op for op in bpe_codes if '</w>' in op]
    word_internal_ops = [op for op in bpe_codes if '</w>' not in op]
    
    print(f"\nWord-final operations: {len(word_final_ops)}")
    print(f"Word-internal operations: {len(word_internal_ops)}")

def compare_with_without_bpe():
    """Compare text with and without BPE tokenization"""
    print("\n=== Comparison: With vs Without BPE ===")
    
    bpe = load_bpe_model("current_vocab.txt")
    
    sample_texts = [
        "This is a simple sentence.",
        "Understanding the complexity of natural language processing.",
        "The quick brown fox jumps over the lazy dog.",
    ]
    
    for text in sample_texts:
        print(f"\nText: '{text}'")
        
        # Without BPE (simple word splitting)
        simple_tokens = text.split()
        print(f"Simple tokens: {simple_tokens}")
        print(f"Simple token count: {len(simple_tokens)}")
        
        # With BPE
        bpe_tokens = bpe.segment(text).split()
        print(f"BPE tokens: {bpe_tokens}")
        print(f"BPE token count: {len(bpe_tokens)}")
        
        # Analysis
        ratio = len(bpe_tokens) / len(simple_tokens) if simple_tokens else 0
        print(f"BPE expansion ratio: {ratio:.2f}")

if __name__ == "__main__":
    if not os.path.exists("current_vocab.txt"):
        print("Error: current_vocab.txt not found. Please run Byte_Pair_Encoding.py first.")
        exit(1)
    
    # Run demonstrations
    demonstrate_tokenization()
    analyze_vocabulary()
    compare_with_without_bpe()
    
    # Interactive session
    interactive_tokenizer()
    
    print("\nDemo completed!")
