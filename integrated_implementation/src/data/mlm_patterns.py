"""
Masked Language Modeling (MLM) dataset and utilities
Implements various MLM strategies for BERT training
"""

import random
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass


class MLMStrategy(Enum):
    """MLM masking strategies"""
    STANDARD = "standard"
    DYNAMIC = "dynamic"
    SPAN = "span"
    WHOLE_WORD = "whole_word"


@dataclass
class MLMConfig:
    """Configuration for MLM training"""
    strategy: MLMStrategy = MLMStrategy.STANDARD
    mlm_probability: float = 0.15
    mask_token_prob: float = 0.8
    replace_token_prob: float = 0.1
    keep_token_prob: float = 0.1
    max_span_length: int = 3
    

class BERTMLMDataset(Dataset):
    """Dataset for BERT Masked Language Modeling"""
    
    def __init__(self, texts: List[str], tokenizer: BertTokenizer, 
                 max_length: int = 128, mlm_config: MLMConfig = None):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_config = mlm_config or MLMConfig()
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.texts[idx]
        
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
        
        # Create MLM targets
        input_ids_mlm, labels = self._apply_mlm_masking(input_ids.clone())
        
        return {
            'input_ids': input_ids_mlm,
            'attention_mask': attention_mask,
            'labels': labels
        }
    
    def _apply_mlm_masking(self, input_ids: torch.Tensor) -> tuple:
        """Apply MLM masking based on strategy"""
        labels = input_ids.clone()
        
        # Don't mask special tokens
        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        
        # Handle potential missing token IDs gracefully
        if hasattr(self.tokenizer, 'cls_token_id') and self.tokenizer.cls_token_id is not None:
            special_tokens_mask[input_ids == self.tokenizer.cls_token_id] = True
        if hasattr(self.tokenizer, 'sep_token_id') and self.tokenizer.sep_token_id is not None:
            special_tokens_mask[input_ids == self.tokenizer.sep_token_id] = True
        if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
            special_tokens_mask[input_ids == self.tokenizer.pad_token_id] = True
        
        # Also handle common special token IDs directly
        special_tokens_mask[input_ids == 0] = True    # PAD
        special_tokens_mask[input_ids == 101] = True  # CLS  
        special_tokens_mask[input_ids == 102] = True  # SEP
        
        if self.mlm_config.strategy == MLMStrategy.STANDARD:
            masked_indices = self._get_random_mask_indices(input_ids, special_tokens_mask)
        elif self.mlm_config.strategy == MLMStrategy.SPAN:
            masked_indices = self._get_span_mask_indices(input_ids, special_tokens_mask)
        else:
            # Fallback to standard
            masked_indices = self._get_random_mask_indices(input_ids, special_tokens_mask)
        
        # Debug: Check if any tokens are masked
        num_masked = masked_indices.sum().item()
        total_valid = (~special_tokens_mask).sum().item()
        
        # CRITICAL FIX: Always ensure at least one token is masked for MLM
        if total_valid > 0:
            if num_masked == 0:
                # Force mask at least one token
                valid_positions = (~special_tokens_mask).nonzero().flatten()
                if len(valid_positions) > 0:
                    random_pos = valid_positions[torch.randint(0, len(valid_positions), (1,))]
                    masked_indices[random_pos] = True
                    num_masked = 1
            
            # Additional safety: if we still have no masked tokens, mask the first valid token
            if num_masked == 0:
                for i in range(len(input_ids)):
                    if not special_tokens_mask[i]:
                        masked_indices[i] = True
                        num_masked = 1
                        break
        
        # Set labels to -100 for non-masked tokens
        labels[~masked_indices] = -100
        
        # Final safety check: if all labels are -100, mask one token manually
        if (labels == -100).all() and len(input_ids) > 3:  # Ensure we have more than just special tokens
            # Find the first non-special token and mask it
            for i in range(1, len(input_ids) - 1):  # Skip first and last (usually CLS and SEP)
                if input_ids[i] != 0:  # Not PAD
                    labels[i] = input_ids[i]
                    masked_indices[i] = True
                    break
        
        # Apply masking transformations
        self._apply_mask_transformations(input_ids, masked_indices)
        
        return input_ids, labels
    
    def _get_random_mask_indices(self, input_ids: torch.Tensor, 
                                special_tokens_mask: torch.Tensor) -> torch.Tensor:
        """Get random mask indices"""
        # Create probability matrix
        probability_matrix = torch.full(input_ids.shape, self.mlm_config.mlm_probability)
        # Don't mask special tokens
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        # Apply masking with explicit debugging
        masked_indices = torch.bernoulli(probability_matrix).bool()
        
        # Ensure we have at least one token to mask (if possible)
        valid_positions = (~special_tokens_mask).nonzero().flatten()
        if len(valid_positions) > 0 and not masked_indices.any():
            # Force at least one mask if no tokens were randomly selected
            random_pos = valid_positions[torch.randint(0, len(valid_positions), (1,))]
            masked_indices[random_pos] = True
        
        return masked_indices
    
    def _get_span_mask_indices(self, input_ids: torch.Tensor, 
                              special_tokens_mask: torch.Tensor) -> torch.Tensor:
        """Get span-based mask indices"""
        masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)
        
        valid_positions = (~special_tokens_mask).nonzero().flatten()
        if len(valid_positions) == 0:
            return masked_indices
        
        # Calculate number of tokens to mask
        num_to_mask = int(len(valid_positions) * self.mlm_config.mlm_probability)
        
        while num_to_mask > 0:
            # Choose random start position
            start_idx = random.choice(valid_positions).item()
            
            # Determine span length
            span_length = min(
                random.randint(1, self.mlm_config.max_span_length),
                num_to_mask,
                len(input_ids) - start_idx
            )
            
            # Mask the span
            for i in range(span_length):
                pos = start_idx + i
                if pos < len(input_ids) and not special_tokens_mask[pos]:
                    masked_indices[pos] = True
                    num_to_mask -= 1
                    if num_to_mask <= 0:
                        break
        
        return masked_indices
    
    def _apply_mask_transformations(self, input_ids: torch.Tensor, 
                                   masked_indices: torch.Tensor) -> None:
        """Apply mask transformations (80% mask, 10% replace, 10% keep)"""
        # 80% of the time, replace with [MASK] token
        mask_token_indices = torch.bernoulli(
            torch.full(input_ids.shape, self.mlm_config.mask_token_prob)
        ).bool() & masked_indices
        input_ids[mask_token_indices] = self.tokenizer.mask_token_id
        
        # 10% of the time, replace with random token
        replace_token_indices = torch.bernoulli(
            torch.full(input_ids.shape, self.mlm_config.replace_token_prob)
        ).bool() & masked_indices & ~mask_token_indices
        
        random_tokens = torch.randint(
            low=0, high=len(self.tokenizer), size=input_ids.shape, dtype=torch.long
        )
        input_ids[replace_token_indices] = random_tokens[replace_token_indices]
        
        # 10% of the time, keep original token (already in place)


def get_dataloader(dataset: BERTMLMDataset, batch_size: int = 32, shuffle: bool = True,
                  num_workers: int = 0, mlm_config: MLMConfig = None) -> DataLoader:
    """Create DataLoader for MLM dataset"""
    
    def collate_fn(batch):
        """Custom collate function for MLM batches"""
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