"""
Data preprocessing utilities for BERT training
Handles text cleaning, chunking, and dataset preparation
"""

import re
import random
from typing import List, Optional, Tuple, Dict, Any
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PreprocessingConfig:
    """Configuration for data preprocessing"""
    target_length: int = 200
    overlap: int = 30
    min_chunk_length: int = 30
    max_examples: Optional[int] = None
    shuffle: bool = True
    random_seed: int = 42


class DataProcessor:
    """Data processor for text preprocessing"""
    
    def __init__(self, config: PreprocessingConfig):
        self.config = config
    
    def process(self, texts: List[str]) -> List[str]:
        """Process a list of texts"""
        processed = []
        for text in texts:
            chunks = self._split_into_chunks(text)
            processed.extend(chunks)
        
        if self.config.shuffle:
            random.shuffle(processed)
        
        if self.config.max_examples:
            processed = processed[:self.config.max_examples]
        
        return processed
    
    def _split_into_chunks(self, text: str) -> List[str]:
        """Split text into chunks"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            sentence_words = sentence.split()
            sentence_word_count = len(sentence_words)
            
            if current_word_count + sentence_word_count > self.config.target_length and current_chunk:
                chunk_text = ' '.join(current_chunk)
                
                if current_word_count >= self.config.min_chunk_length:
                    chunks.append(chunk_text)
                
                # Create overlap for context continuity
                overlap_sentences = []
                overlap_words = 0
                
                for sent in reversed(current_chunk):
                    sent_words = len(sent.split())
                    if overlap_words + sent_words <= self.config.overlap:
                        overlap_sentences.insert(0, sent)
                        overlap_words += sent_words
                    else:
                        break
                
                current_chunk = overlap_sentences
                current_word_count = overlap_words
            
            current_chunk.append(sentence)
            current_word_count += sentence_word_count
        
        if current_chunk and current_word_count >= self.config.min_chunk_length:
            chunk_text = ' '.join(current_chunk)
            chunks.append(chunk_text)
        
        return chunks


def clean_gutenberg_text(text: str) -> str:
    """Clean Project Gutenberg text by removing headers/footers and formatting"""
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
    content = re.sub(r'\r', '', content)
    content = re.sub(r'\n\s*\n\s*\n+', '\n\n', content)
    content = re.sub(r'[ \t]+', ' ', content)
    content = content.strip()
    
    return content


def load_training_data(file_path: str, config: PreprocessingConfig, max_examples: Optional[int] = None) -> List[str]:
    """Load training texts from file"""
    
    # First try to process the Huckleberry Finn book if available
    book_path = Path(file_path).parent / "training_data/Adventures-of-Huckleberry-Finn_76-master/76.txt"
    if book_path.exists():
        try:
            with open(book_path, 'r', encoding='utf-8', errors='ignore') as f:
                raw_text = f.read()
            
            cleaned_text = clean_gutenberg_text(raw_text)
            processor = DataProcessor(config)
            chunks = processor._split_into_chunks(cleaned_text)
            
            if max_examples:
                chunks = chunks[:max_examples]
            
            print(f"✅ Using processed Huckleberry Finn data: {len(chunks)} chunks")
            return chunks
        except Exception as e:
            print(f"Warning: Could not process {book_path}: {e}")
    
    # Fallback to original file loading
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        
        processor = DataProcessor(config)
        processed_texts = processor.process(texts)
        
        if max_examples:
            processed_texts = processed_texts[:max_examples]
        
        print(f"Loaded {len(processed_texts)} training texts from {file_path}")
        return processed_texts
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Using minimal fallback dataset.")
        fallback_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Transformers have revolutionized natural language processing.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning models have achieved remarkable performance on various tasks.",
            "Attention mechanisms have become fundamental in modern neural networks.",
            "BERT uses bidirectional attention to understand context from both directions.",
            "Causal language modeling predicts the next token in a sequence autoregressively.",
            "Masked language modeling learns bidirectional representations for better understanding.",
            "The attention mechanism computes weighted representations of input sequences."
        ] * 50
        
        if max_examples:
            fallback_texts = fallback_texts[:max_examples]
        
        return fallback_texts


def split_dataset(texts: List[str], train_ratio: float = 0.8, val_ratio: float = 0.1, 
                 test_ratio: float = 0.1, shuffle: bool = True, random_seed: int = 42) -> Tuple[List[str], List[str], List[str]]:
    """Split dataset into train/val/test sets"""
    
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("Ratios must sum to 1.0")
    
    if shuffle:
        random.seed(random_seed)
        texts = texts.copy()
        random.shuffle(texts)
    
    total = len(texts)
    train_end = int(total * train_ratio)
    val_end = int(total * (train_ratio + val_ratio))
    
    train_texts = texts[:train_end]
    val_texts = texts[train_end:val_end]
    test_texts = texts[val_end:]
    
    return train_texts, val_texts, test_texts