"""
Modular BERT Model Implementation
=================================

BERT model with swappable attention mechanisms.
"""

import torch
import torch.nn as nn
from transformers import BertForMaskedLM, BertConfig
from typing import Optional

from ..attention import get_attention_class


class ModularBERTModel(nn.Module):
    """
    BERT model with modular attention mechanism
    """
    
    def __init__(self, config: BertConfig, attention_type: str = "standard"):
        super().__init__()
        self.config = config
        self.attention_type = attention_type
        
        # Create base BERT model
        self.bert = BertForMaskedLM(config)
        
        # Replace attention layers with specified type
        self._replace_attention_layers(attention_type)
    
    def _replace_attention_layers(self, attention_type: str):
        """Replace all attention layers with specified type"""
        attention_class = get_attention_class(attention_type)
        
        for layer_idx in range(self.config.num_hidden_layers):
            layer = self.bert.bert.encoder.layer[layer_idx]
            
            # Create new attention layer
            new_attention = attention_class(
                hidden_size=self.config.hidden_size,
                num_heads=self.config.num_attention_heads,
                max_position_embeddings=self.config.max_position_embeddings,
                dropout=self.config.attention_probs_dropout_prob
            )
            
            # Replace the self-attention
            layer.attention.self = new_attention
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass through the model"""
        return self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            **kwargs
        )


class BERTMLMModel(ModularBERTModel):
    """
    BERT model specifically for Masked Language Modeling
    """
    
    def __init__(self, config: BertConfig, attention_type: str = "standard"):
        super().__init__(config, attention_type)
        
    def get_mlm_loss(self, input_ids, attention_mask=None, labels=None):
        """Get MLM loss for training"""
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss if hasattr(outputs, 'loss') else None