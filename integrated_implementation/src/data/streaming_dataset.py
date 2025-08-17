"""
Streaming Dataset Implementation for Large-Scale BERT Training
============================================================

Provides memory-efficient streaming data loading for training BERT models
on large datasets without loading everything into memory.
"""

import torch
import random
import logging
from torch.utils.data import Dataset, DataLoader, IterableDataset
from transformers import BertTokenizer
from datasets import load_dataset, concatenate_datasets
from typing import Iterator, List, Dict, Optional, Union
from pathlib import Path
import json
from ..data.mlm_patterns import MLMConfig
from ..data.clm_patterns import CLMConfig

logger = logging.getLogger(__name__)


class StreamingTextDataset(IterableDataset):
    """
    Streaming dataset that loads text data on-demand without storing
    everything in memory. Supports multiple data sources and formats.
    """
    
    def __init__(
        self,
        data_sources: List[str],
        tokenizer: BertTokenizer,
        max_length: int = 512,
        buffer_size: int = 10000,
        preprocessing_fn: Optional[callable] = None,
        streaming: bool = True
    ):
        self.data_sources = data_sources
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.buffer_size = buffer_size
        self.preprocessing_fn = preprocessing_fn or self._default_preprocessing
        self.streaming = streaming
        
        # Initialize datasets
        self.datasets = self._load_datasets()
        
    def _load_datasets(self):
        """Load all specified datasets"""
        datasets = []
        
        for source in self.data_sources:
            try:
                if source.startswith("hf://"):
                    # Hugging Face dataset
                    dataset_name = source[5:]
                    dataset = load_dataset(dataset_name, split="train", streaming=self.streaming)
                    datasets.append(dataset)
                    logger.info(f"Loaded HF dataset: {dataset_name}")
                    
                elif source.startswith("file://"):
                    # Local file
                    file_path = source[7:]
                    dataset = self._load_text_file(file_path)
                    datasets.append(dataset)
                    logger.info(f"Loaded file: {file_path}")
                    
                elif source in ["wikipedia", "bookcorpus", "openwebtext", "c4"]:
                    # Predefined datasets
                    dataset = self._load_predefined_dataset(source)
                    if dataset:
                        datasets.append(dataset)
                        logger.info(f"Loaded predefined dataset: {source}")
                        
            except Exception as e:
                logger.warning(f"Failed to load dataset {source}: {e}")
                continue
                
        if not datasets:
            raise ValueError("No datasets could be loaded")
            
        return datasets
    
    def _load_predefined_dataset(self, name: str):
        """Load commonly used datasets"""
        try:
            if name == "wikipedia":
                return load_dataset("wikipedia", "20220301.en", split="train", streaming=self.streaming)
            elif name == "bookcorpus":
                return load_dataset("bookcorpus", split="train", streaming=self.streaming)
            elif name == "openwebtext":
                return load_dataset("openwebtext", split="train", streaming=self.streaming)
            elif name == "c4":
                return load_dataset("c4", "en", split="train", streaming=self.streaming)
            else:
                return None
        except:
            return None
    
    def _load_text_file(self, file_path: str):
        """Load a simple text file"""
        def text_generator():
            with open(file_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield {"text": line}
        
        return text_generator()
    
    def _default_preprocessing(self, text: str) -> str:
        """Default text preprocessing"""
        # Basic cleaning
        text = text.strip()
        
        # Remove very short or very long texts
        if len(text) < 10 or len(text) > 10000:
            return None
            
        # Basic deduplication (simple hash check could be added)
        return text
    
    def __iter__(self) -> Iterator[Dict[str, torch.Tensor]]:
        """Iterate through the dataset"""
        buffer = []
        
        # Create unified iterator from all datasets
        for dataset in self.datasets:
            for item in dataset:
                # Extract text from different dataset formats
                text = self._extract_text(item)
                if not text:
                    continue
                    
                # Apply preprocessing
                processed_text = self.preprocessing_fn(text)
                if not processed_text:
                    continue
                
                buffer.append(processed_text)
                
                # Yield from buffer when full
                if len(buffer) >= self.buffer_size:
                    # Shuffle buffer for randomness
                    random.shuffle(buffer)
                    for text in buffer:
                        yield self._tokenize_text(text)
                    buffer = []
        
        # Yield remaining items in buffer
        if buffer:
            random.shuffle(buffer)
            for text in buffer:
                yield self._tokenize_text(text)
    
    def _extract_text(self, item) -> Optional[str]:
        """Extract text from various dataset formats"""
        if isinstance(item, str):
            return item
        elif isinstance(item, dict):
            # Try common text field names
            for field in ["text", "content", "article", "document", "body"]:
                if field in item and item[field]:
                    return item[field]
        return None
    
    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize text for model input"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "text": text  # Keep original for debugging
        }


class StreamingMLMDataset(StreamingTextDataset):
    """Streaming dataset with MLM (Masked Language Modeling) preprocessing"""
    
    def __init__(
        self,
        data_sources: List[str],
        tokenizer: BertTokenizer,
        mlm_config: MLMConfig,
        max_length: int = 512,
        buffer_size: int = 10000
    ):
        self.mlm_config = mlm_config
        super().__init__(data_sources, tokenizer, max_length, buffer_size)
    
    def _tokenize_text(self, text: str) -> Dict[str, torch.Tensor]:
        """Tokenize and apply MLM masking"""
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # Apply MLM masking
        input_ids_mlm, labels = self._apply_mlm_masking(input_ids.clone())
        
        return {
            "input_ids": input_ids_mlm,
            "attention_mask": attention_mask,
            "labels": labels
        }
    
    def _apply_mlm_masking(self, input_ids: torch.Tensor) -> tuple:
        """Apply MLM masking (simplified version)"""
        labels = input_ids.clone()
        
        # Create mask for special tokens
        special_tokens_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        special_tokens_mask[input_ids == self.tokenizer.cls_token_id] = True
        special_tokens_mask[input_ids == self.tokenizer.sep_token_id] = True
        special_tokens_mask[input_ids == self.tokenizer.pad_token_id] = True
        
        # Random masking
        probability_matrix = torch.full(input_ids.shape, self.mlm_config.mlm_probability)
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100
        
        # 80% mask, 10% replace, 10% keep
        mask_token_indices = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[mask_token_indices] = self.tokenizer.mask_token_id
        
        replace_token_indices = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~mask_token_indices
        random_tokens = torch.randint(low=0, high=len(self.tokenizer), size=input_ids.shape, dtype=torch.long)
        input_ids[replace_token_indices] = random_tokens[replace_token_indices]
        
        return input_ids, labels


def get_large_scale_datasets() -> List[str]:
    """Get list of recommended large-scale datasets"""
    return [
        "hf://wikipedia",           # ~6GB, 6M articles
        "hf://openwebtext",         # ~40GB, 8M documents
        "hf://bookcorpus",          # ~5GB, 11K books
        "hf://c4",                  # 750GB+ (use with caution)
    ]


def get_medium_scale_datasets() -> List[str]:
    """Get list of medium-scale datasets for development"""
    return [
        "hf://wikitext",
        "hf://imdb",
        "hf://amazon_reviews_multi",
        "file://./training_data.txt"
    ]


def get_streaming_dataloader(
    data_sources: List[str],
    tokenizer: BertTokenizer,
    objective: str = "mlm",
    batch_size: int = 32,
    max_length: int = 512,
    mlm_config: Optional[MLMConfig] = None,
    num_workers: int = 0
) -> DataLoader:
    """Create streaming dataloader for large-scale training"""
    
    if objective == "mlm":
        if not mlm_config:
            mlm_config = MLMConfig()
        dataset = StreamingMLMDataset(
            data_sources=data_sources,
            tokenizer=tokenizer,
            mlm_config=mlm_config,
            max_length=max_length
        )
    else:
        # For CLM, use base streaming dataset
        dataset = StreamingTextDataset(
            data_sources=data_sources,
            tokenizer=tokenizer,
            max_length=max_length
        )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )


# Usage example function
def create_production_dataloader(config):
    """Create production-ready dataloader with large datasets"""
    
    # Choose dataset scale based on environment
    if config.debug:
        data_sources = ["file://./training_data.txt"]
    elif config.scale == "medium":
        data_sources = get_medium_scale_datasets()
    else:  # large scale
        data_sources = get_large_scale_datasets()
    
    # Load tokenizer
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Create streaming dataloader
    train_loader = get_streaming_dataloader(
        data_sources=data_sources,
        tokenizer=tokenizer,
        objective="mlm",  # or "clm"
        batch_size=config.batch_size,
        max_length=config.max_seq_length,
        num_workers=config.num_workers
    )
    
    logger.info(f"Created streaming dataloader with {len(data_sources)} data sources")
    return train_loader