"""
Causal Language Modeling (CLM) dataset and utilities
Implements various CLM strategies for BERT-style models adapted for CLM
"""

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass


class CLMStrategy(Enum):
    """CLM processing strategies"""
    STANDARD = "standard"
    PACKED = "packed"
    SLIDING = "sliding"


@dataclass
class CLMConfig:
    """Configuration for CLM training"""
    strategy: CLMStrategy = CLMStrategy.STANDARD
    max_length: int = 128
    stride: int = 64  # For sliding window
    pack_sequences: bool = False


class CLMDataset(Dataset):
    """Dataset for Causal Language Modeling"""
    
    def __init__(self, texts: List[str], tokenizer: BertTokenizer, 
                 max_length: int = 128, clm_config: CLMConfig = None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.clm_config = clm_config or CLMConfig()
        
        # Preprocess texts based on strategy
        self.examples = self._preprocess_texts()
        
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.examples[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        
        # For CLM, labels are shifted input_ids
        labels = input_ids.clone()
        
        # Mask padding tokens in labels
        labels[attention_mask == 0] = -100
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def _preprocess_texts(self) -> List[str]:
        """Preprocess texts based on CLM strategy"""
        if self.clm_config.strategy == CLMStrategy.SLIDING:
            return self._create_sliding_windows()
        elif self.clm_config.strategy == CLMStrategy.PACKED:
            return self._pack_sequences()
        else:
            # Standard: just return original texts
            return self.texts
    
    def _create_sliding_windows(self) -> List[str]:
        """Create sliding window sequences for CLM"""
        windows = []
        
        for text in self.texts:
            # Tokenize to get word-level sliding
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            if len(tokens) <= self.max_length - 2:  # Account for special tokens
                windows.append(text)
                continue
            
            # Create sliding windows
            stride = self.clm_config.stride
            for i in range(0, len(tokens) - self.max_length + 2, stride):
                window_tokens = tokens[i:i + self.max_length - 2]
                window_text = self.tokenizer.decode(window_tokens, skip_special_tokens=True)
                windows.append(window_text)
        
        return windows
    
    def _pack_sequences(self) -> List[str]:
        """Pack multiple short sequences together"""
        packed = []
        current_pack = []
        current_length = 0
        
        for text in self.texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            
            if current_length + len(tokens) + 1 <= self.max_length - 2:  # +1 for separator
                current_pack.append(text)
                current_length += len(tokens) + 1
            else:
                if current_pack:
                    packed_text = " [SEP] ".join(current_pack)
                    packed.append(packed_text)
                
                current_pack = [text]
                current_length = len(tokens)
        
        # Add remaining pack
        if current_pack:
            packed_text = " [SEP] ".join(current_pack)
            packed.append(packed_text)
        
        return packed


def get_clm_dataloader(dataset: CLMDataset, batch_size: int = 32, shuffle: bool = True,
                      num_workers: int = 0, clm_config: CLMConfig = None) -> DataLoader:
    """Create DataLoader for CLM dataset"""
    
    def collate_fn(batch):
        """Custom collate function for CLM batches"""
        input_ids = torch.stack([item['input_ids'] for item in batch])
        attention_mask = torch.stack([item['attention_mask'] for item in batch])
        labels = torch.stack([item['labels'] for item in batch])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=torch.cuda.is_available()
    )