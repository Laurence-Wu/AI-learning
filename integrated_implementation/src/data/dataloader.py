"""
Data Loading and Preparation Module
===================================

Handles loading training data from local files and creating appropriate
datasets and dataloaders for BERT training.
"""

import logging
import random
from pathlib import Path
from typing import List, Tuple
from torch.utils.data import DataLoader

from .mlm_patterns import BERTMLMDataset, get_dataloader, MLMConfig
from .clm_patterns import CLMDataset, get_clm_dataloader

logger = logging.getLogger(__name__)


def load_training_data(data_file: str = 'training_data.txt') -> List[str]:
    """
    Load training data from local file
    
    Args:
        data_file: Path to training data file
        
    Returns:
        List of text strings
    """
    logger.info(f"Loading training data from {data_file}")
    
    if not Path(data_file).exists():
        raise FileNotFoundError(f"Training data file not found: {data_file}")
    
    # Load all texts from file
    with open(data_file, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    logger.info(f"Loaded {len(texts)} texts from {data_file}")
    
    # Calculate basic statistics
    total_tokens = sum(len(text.split()) for text in texts)
    avg_tokens = total_tokens / len(texts) if texts else 0
    
    logger.info(f"Dataset statistics:")
    logger.info(f"  Total texts: {len(texts)}")
    logger.info(f"  Total tokens: {total_tokens:,}")
    logger.info(f"  Average tokens per text: {avg_tokens:.1f}")
    
    return texts


def split_train_validation(texts: List[str], train_ratio: float = 0.8, 
                          seed: int = 42) -> Tuple[List[str], List[str]]:
    """
    Split texts into training and validation sets
    
    Args:
        texts: List of text strings
        train_ratio: Fraction for training (default 0.8)
        seed: Random seed for reproducible splits
        
    Returns:
        Tuple of (train_texts, val_texts)
    """
    random.seed(seed)
    texts_shuffled = texts.copy()
    random.shuffle(texts_shuffled)
    
    split_idx = int(train_ratio * len(texts_shuffled))
    train_texts = texts_shuffled[:split_idx]
    val_texts = texts_shuffled[split_idx:]
    
    logger.info(f"Train/validation split: {len(train_texts)}/{len(val_texts)}")
    
    return train_texts, val_texts


def create_mlm_dataloaders(train_texts: List[str], val_texts: List[str], 
                          tokenizer, config) -> Tuple[DataLoader, DataLoader]:
    """
    Create MLM datasets and dataloaders
    
    Args:
        train_texts: Training text list
        val_texts: Validation text list  
        tokenizer: BERT tokenizer
        config: Configuration object
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create MLM configuration
    mlm_config = MLMConfig(mlm_probability=config.mlm_probability)
    
    # Create datasets
    train_dataset = BERTMLMDataset(
        train_texts, 
        tokenizer, 
        max_length=config.max_seq_length,
        mlm_config=mlm_config
    )
    
    val_dataset = BERTMLMDataset(
        val_texts, 
        tokenizer,
        max_length=config.max_seq_length,
        mlm_config=mlm_config
    )
    
    # Create dataloaders
    train_loader = get_dataloader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True
    )
    
    val_loader = get_dataloader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False
    )
    
    logger.info(f"Created MLM dataloaders: train={len(train_dataset)}, val={len(val_dataset)}")
    
    return train_loader, val_loader


def create_clm_dataloaders(train_texts: List[str], val_texts: List[str], 
                          tokenizer, config) -> Tuple[DataLoader, DataLoader]:
    """
    Create CLM datasets and dataloaders
    
    Args:
        train_texts: Training text list
        val_texts: Validation text list
        tokenizer: BERT tokenizer  
        config: Configuration object
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    # Create datasets
    train_dataset = CLMDataset(
        train_texts, 
        tokenizer,
        max_length=config.max_seq_length
    )
    
    val_dataset = CLMDataset(
        val_texts, 
        tokenizer,
        max_length=config.max_seq_length
    )
    
    # Create dataloaders
    train_loader = get_clm_dataloader(
        train_dataset, 
        batch_size=config.batch_size, 
        shuffle=True
    )
    
    val_loader = get_clm_dataloader(
        val_dataset, 
        batch_size=config.batch_size, 
        shuffle=False
    )
    
    logger.info(f"Created CLM dataloaders: train={len(train_dataset)}, val={len(val_dataset)}")
    
    return train_loader, val_loader


def create_dataloaders(objective: str, train_texts: List[str], val_texts: List[str], 
                      tokenizer, config) -> Tuple[DataLoader, DataLoader]:
    """
    Create appropriate dataloaders based on objective
    
    Args:
        objective: Training objective ('mlm' or 'clm')
        train_texts: Training text list
        val_texts: Validation text list
        tokenizer: BERT tokenizer
        config: Configuration object
        
    Returns:
        Tuple of (train_loader, val_loader)
    """
    if objective == "mlm":
        return create_mlm_dataloaders(train_texts, val_texts, tokenizer, config)
    elif objective == "clm":
        return create_clm_dataloaders(train_texts, val_texts, tokenizer, config)
    else:
        raise ValueError(f"Unknown objective: {objective}. Must be 'mlm' or 'clm'")


def load_and_prepare_data(config) -> Tuple[List[str], List[str]]:
    """
    Main function to load and prepare training data
    
    Args:
        config: Configuration object
        
    Returns:
        Tuple of (train_texts, val_texts)
    """
    # Load data from local file
    texts = load_training_data('training_data.txt')
    
    # Split into train/validation
    train_texts, val_texts = split_train_validation(texts)
    
    return train_texts, val_texts