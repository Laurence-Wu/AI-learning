"""
BERT Model Factory for Different Attention Mechanisms
====================================================

Creates BERT models with configurable attention mechanisms for comparison.
Supports both MLM and CLM training objectives.
"""

import torch
import torch.nn as nn
from transformers import BertConfig, BertForMaskedLM
from typing import Optional, Dict, Any
import logging

from ..attention import (
    StandardBERTAttention,
    RoPEBERTAttention, 
    ExpoSBBERTAttention,
    AbsoluteBERTAttention,
    get_attention_class
)

logger = logging.getLogger(__name__)


class ModifiedBERTModel(nn.Module):
    """
    Modified BERT model that can use different attention mechanisms
    """
    
    def __init__(self, config: BertConfig, attention_type: str = "standard"):
        super().__init__()
        self.config = config
        self.attention_type = attention_type
        
        # Create base BERT model
        self.bert = BertForMaskedLM(config)
        
        # Replace attention layers with our Triton implementations
        self._replace_attention_layers()
        
    def _replace_attention_layers(self):
        """Replace all attention layers with Triton implementations"""
        logger.info(f"Replacing {self.config.num_hidden_layers} attention layers with {self.attention_type} attention...")
        
        # Get attention class
        attention_class = get_attention_class(self.attention_type)
        
        for layer_idx in range(self.config.num_hidden_layers):
            # Get the attention layer
            layer = self.bert.bert.encoder.layer[layer_idx]
            original_attention = layer.attention.self
            
            # Create new attention layer
            if self.attention_type == "standard":
                new_attention = attention_class(
                    hidden_size=self.config.hidden_size,
                    num_heads=self.config.num_attention_heads,
                    max_position_embeddings=self.config.max_position_embeddings,
                    dropout=self.config.attention_probs_dropout_prob
                )
            elif self.attention_type == "rope":
                new_attention = attention_class(
                    hidden_size=self.config.hidden_size,
                    num_heads=self.config.num_attention_heads,
                    dropout=self.config.attention_probs_dropout_prob
                )
            elif self.attention_type == "exposb":
                new_attention = attention_class(
                    hidden_size=self.config.hidden_size,
                    num_heads=self.config.num_attention_heads,
                    max_position_embeddings=self.config.max_position_embeddings,
                    dropout=self.config.attention_probs_dropout_prob
                )
            elif self.attention_type == "absolute":
                new_attention = attention_class(
                    hidden_size=self.config.hidden_size,
                    num_heads=self.config.num_attention_heads,
                    max_position_embeddings=self.config.max_position_embeddings,
                    dropout=self.config.attention_probs_dropout_prob
                )
            else:
                raise ValueError(f"Unknown attention type: {self.attention_type}")
            
            # Replace the self-attention
            layer.attention.self = new_attention
            logger.debug(f"  Layer {layer_idx}: {type(original_attention).__name__} -> {type(new_attention).__name__}")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        """Forward pass through the modified BERT model"""
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


# CLM functionality has been moved to gpt_models.py for better separation of concerns


def create_bert_model(config: BertConfig, attention_type: str = "standard") -> ModifiedBERTModel:
    """
    Create a BERT model with specified attention mechanism for MLM training
    
    Args:
        config: BERT configuration
        attention_type: Type of attention mechanism ("standard", "rope", "exposb", "absolute")
        
    Returns:
        Modified BERT model with custom attention
    """
    logger.info(f"Creating BERT model with {attention_type} attention")
    
    # Validate attention type
    valid_types = ["standard", "rope", "exposb", "absolute"]
    if attention_type not in valid_types:
        raise ValueError(f"Invalid attention type: {attention_type}. Valid types: {valid_types}")
    
    model = ModifiedBERTModel(config, attention_type)
    
    # Log model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Created BERT model:")
    logger.info(f"  Attention type: {attention_type}")
    logger.info(f"  Hidden size: {config.hidden_size}")
    logger.info(f"  Layers: {config.num_hidden_layers}")
    logger.info(f"  Attention heads: {config.num_attention_heads}")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    return model


def create_clm_model(config: BertConfig, attention_type: str = "standard"):
    """
    Create a CLM (Causal Language Model) with specified attention mechanism
    
    Args:
        config: BERT configuration (will be adapted for CLM)
        attention_type: Type of attention mechanism ("standard", "rope", "exposb", "absolute")
        
    Returns:
        Modified CLM model with custom attention
    """
    # Import here to avoid circular imports
    from .gpt_models import create_gpt_model
    
    logger.info(f"Creating CLM model with {attention_type} attention (delegating to gpt_models)")
    
    # Use the dedicated GPT model implementation
    return create_gpt_model(config, attention_type)


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get detailed information about a model
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    info = {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "model_type": type(model).__name__,
        "device": next(model.parameters()).device if next(model.parameters(), None) is not None else "unknown"
    }
    
    if hasattr(model, 'attention_type'):
        info["attention_type"] = model.attention_type
    
    if hasattr(model, 'config'):
        config = model.config
        info.update({
            "hidden_size": getattr(config, 'hidden_size', 'unknown'),
            "num_layers": getattr(config, 'num_hidden_layers', 'unknown'),
            "num_attention_heads": getattr(config, 'num_attention_heads', 'unknown'),
            "vocab_size": getattr(config, 'vocab_size', 'unknown')
        })
    
    return info
