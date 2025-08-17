"""
Data preprocessing utilities for BERT training
Handles text cleaning, chunking, and dataset preparation
"""

import re
from typing import List, Optional
from pathlib import Path


def clean_gutenberg_text(text: str) -> str:
    """
    Clean Project Gutenberg text by removing headers/footers and formatting
    
    Args:
        text: Raw text from Project Gutenberg file
        
    Returns:
        Cleaned text with headers/footers removed and normalized formatting
    """
    lines = text.split('\n')
    
    # Find start of actual content
    start_idx = 0
    for i, line in enumerate(lines):
        if "*** START OF THIS PROJECT GUTENBERG" in line.upper():
            start_idx = i + 1
            break
    
    # Find end of actual content
    end_idx = len(lines)
    for i in range(len(lines) - 1, -1, -1):
        if "*** END OF THIS PROJECT GUTENBERG" in lines[i].upper():
            end_idx = i
            break
    
    # Extract main content
    content_lines = lines[start_idx:end_idx]
    content = '\n'.join(content_lines)
    
    # Clean up formatting
    content = re.sub(r'\r', '', content)  # Remove carriage returns
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)  # Normalize paragraph breaks
    content = re.sub(r'[ \t]+', ' ', content)  # Normalize spaces
    content = content.strip()
    
    return content


def split_into_training_chunks(text: str, target_length: int = 800, overlap: int = 100) -> List[str]:
    """
    Split text into overlapping chunks suitable for BERT training
    
    Args:
        text: Input text to split
        target_length: Target number of words per chunk
        overlap: Number of words to overlap between chunks
    
    Returns:
        List of text chunks suitable for BERT training
    """
    # Split into sentences for better context preservation
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
        
        # Count words in this sentence
        sentence_words = sentence.split()
        sentence_word_count = len(sentence_words)
        
        # If adding this sentence would make chunk too long, save current chunk
        if current_word_count + sentence_word_count > target_length and current_chunk:
            chunk_text = ' '.join(current_chunk)
            
            # Only save chunks that are substantial enough
            if current_word_count >= 30:  # Minimum 30 words
                chunks.append(chunk_text)
            
            # Create overlap for context continuity
            overlap_sentences = []
            overlap_words = 0
            
            # Take sentences from the end for overlap
            for sent in reversed(current_chunk):
                sent_words = len(sent.split())
                if overlap_words + sent_words <= overlap:
                    overlap_sentences.insert(0, sent)
                    overlap_words += sent_words
                else:
                    break
            
            # Start new chunk with overlap
            current_chunk = overlap_sentences
            current_word_count = overlap_words
        
        # Add current sentence to chunk
        current_chunk.append(sentence)
        current_word_count += sentence_word_count
    
    # Add final chunk if it's substantial
    if current_chunk and current_word_count >= 30:
        chunk_text = ' '.join(current_chunk)
        chunks.append(chunk_text)
    
    return chunks


def process_huckleberry_finn_data(book_file: str = "training_data/Adventures-of-Huckleberry-Finn_76-master/76.txt") -> Optional[List[str]]:
    """
    Process Huckleberry Finn book into training chunks
    
    Args:
        book_file: Path to the Huckleberry Finn text file
        
    Returns:
        List of training chunks or None if file not found
    """
    try:
        with open(book_file, 'r', encoding='utf-8', errors='ignore') as f:
            raw_text = f.read()
        
        print(f"✓ Loaded Huckleberry Finn: {len(raw_text):,} characters")
        
        # Clean the text
        cleaned_text = clean_gutenberg_text(raw_text)
        print(f"✓ Cleaned text: {len(cleaned_text):,} characters, {len(cleaned_text.split()):,} words")
        
        # Split into training chunks for longer sequences
        chunks = split_into_training_chunks(cleaned_text, target_length=800, overlap=100)
        print(f"✓ Generated {len(chunks)} training chunks")
        print(f"✓ Average chunk length: {sum(len(chunk.split()) for chunk in chunks) / len(chunks):.1f} words")
        
        return chunks
        
    except FileNotFoundError:
        print(f"Warning: {book_file} not found. Unable to process Huckleberry Finn data.")
        return None


def load_training_data(file_path: str) -> List[str]:
    """
    Load training texts from file or process Huckleberry Finn
    
    Args:
        file_path: Path to training data file (used as fallback)
        
    Returns:
        List of training text chunks
    """
    
    # First try to process the Huckleberry Finn book
    huck_chunks = process_huckleberry_finn_data()
    if huck_chunks:
        print(f"✅ Using processed Huckleberry Finn data: {len(huck_chunks)} chunks")
        return huck_chunks
    
    # Fallback to original file loading
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(texts)} training texts from {file_path}")
        return texts
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Using minimal fallback dataset.")
        return [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Transformers have revolutionized natural language processing.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning models have achieved remarkable performance on various tasks.",
            "Attention mechanisms have become fundamental in modern neural networks."
        ] * 50


def save_processed_data(chunks: List[str], output_file: str = "training_data/processed_training_data.txt") -> None:
    """
    Save processed chunks to a file for later use
    
    Args:
        chunks: List of text chunks to save
        output_file: Output file path
    """
    output_path = Path(output_file)
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(chunk + '\n')
    
    print(f"✓ Saved {len(chunks)} chunks to {output_path}")
    print(f"✓ File size: {output_path.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    # Demo the data processing
    print("📚 Data Preprocessing Demo")
    print("=" * 50)
    
    chunks = load_training_data("dummy_file.txt")
    
    if chunks:
        print(f"\n📊 Dataset Statistics:")
        word_counts = [len(chunk.split()) for chunk in chunks]
        print(f"   Total chunks: {len(chunks)}")
        print(f"   Average words per chunk: {sum(word_counts) / len(word_counts):.1f}")
        print(f"   Min words: {min(word_counts)}")
        print(f"   Max words: {max(word_counts)}")
        
        # Show sample chunks
        print(f"\n📝 Sample chunks:")
        for i, chunk in enumerate(chunks[:2]):
            print(f"\nChunk {i+1} ({len(chunk.split())} words):")
            print(f"   {chunk[:150]}...")
        
        # Save the processed data
        save_processed_data(chunks)
        
        print("\n✅ Data preprocessing completed successfully!")


