"""
Setup script to download and save BERT tokenizer locally
Run this once with internet connection to enable offline training
"""

import os
from transformers import BertTokenizer

def setup_tokenizer():
    """Download and save BERT tokenizer locally"""
    print("Downloading BERT tokenizer...")
    
    # Create local tokenizer directory
    tokenizer_dir = "./local_tokenizer"
    os.makedirs(tokenizer_dir, exist_ok=True)
    
    try:
        # Download tokenizer from Hugging Face
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        
        # Save locally
        tokenizer.save_pretrained(tokenizer_dir)
        
        print(f"✅ Tokenizer successfully saved to {tokenizer_dir}")
        print("You can now run the training script offline!")
        
        # Test loading from local directory
        print("\nTesting local tokenizer...")
        local_tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
        test_text = "Hello world, this is a test."
        tokens = local_tokenizer.tokenize(test_text)
        print(f"Test tokenization: {tokens}")
        print("✅ Local tokenizer works correctly!")
        
    except Exception as e:
        print(f"❌ Error downloading tokenizer: {e}")
        print("Make sure you have internet connection and try again.")

if __name__ == "__main__":
    setup_tokenizer()
