"""
GPT Model Factory for Different Attention Mechanisms
===================================================

Creates GPT-style models with configurable attention mechanisms for CLM training.
Implements proper causal language modeling with custom attention mechanisms.
"""

import torch
import torch.nn as nn
import math
from transformers import GPT2Config, GPT2LMHeadModel
from typing import Optional, Dict, Any
import logging

from ..attention import (
    StandardBERTAttention,
    RoPEBERTAttention, 
    ExpoSBBERTAttention,
    RSEBERTAttention,
    AbsoluteBERTAttention,
    get_attention_class
)

logger = logging.getLogger(__name__)


class GPTConfig:
    """
    Configuration class for GPT models with custom attention mechanisms
    """
    
    def __init__(
        self,
        vocab_size: int = 50257,
        max_position_embeddings: int = 1024,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        layer_norm_eps: float = 1e-5,
        initializer_range: float = 0.02,
        pad_token_id: int = 50256,
        bos_token_id: int = 50256,
        eos_token_id: int = 50256
    ):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.layer_norm_eps = layer_norm_eps
        self.initializer_range = initializer_range
        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        
        # Derived properties
        self.head_dim = hidden_size // num_attention_heads
    
    @classmethod
    def from_bert_config(cls, bert_config):
        """Create GPT config from BERT config"""
        return cls(
            vocab_size=bert_config.vocab_size,
            max_position_embeddings=bert_config.max_position_embeddings,
            hidden_size=bert_config.hidden_size,
            num_hidden_layers=bert_config.num_hidden_layers,
            num_attention_heads=bert_config.num_attention_heads,
            intermediate_size=bert_config.intermediate_size,
            hidden_dropout_prob=bert_config.hidden_dropout_prob,
            attention_probs_dropout_prob=bert_config.attention_probs_dropout_prob,
            layer_norm_eps=getattr(bert_config, 'layer_norm_eps', 1e-5),
            initializer_range=getattr(bert_config, 'initializer_range', 0.02),
            pad_token_id=getattr(bert_config, 'pad_token_id', 0),
            bos_token_id=getattr(bert_config, 'pad_token_id', 0),
            eos_token_id=getattr(bert_config, 'pad_token_id', 0)
        )
    
    def to_gpt2_config(self):
        """Convert to HuggingFace GPT2Config for compatibility"""
        return GPT2Config(
            vocab_size=self.vocab_size,
            n_positions=self.max_position_embeddings,
            n_ctx=self.max_position_embeddings,
            n_embd=self.hidden_size,
            n_layer=self.num_hidden_layers,
            n_head=self.num_attention_heads,
            resid_pdrop=self.hidden_dropout_prob,
            attn_pdrop=self.attention_probs_dropout_prob,
            layer_norm_epsilon=self.layer_norm_eps,
            initializer_range=self.initializer_range,
            bos_token_id=self.bos_token_id,
            eos_token_id=self.eos_token_id,
            pad_token_id=self.pad_token_id
        )


class CausalAttentionAdapter(nn.Module):
    """
    Adapter to make BERT attention mechanisms work with causal language modeling
    """
    
    def __init__(self, attention_layer, attention_type: str = "standard"):
        super().__init__()
        self.attention = attention_layer
        self.attention_type = attention_type
    
    def forward(self, hidden_states, past_key_value=None, cache_position=None,
                attention_mask=None, head_mask=None, output_attentions=False, **kwargs):
        """
        Forward pass with causal masking
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Handle different attribute naming conventions
        if hasattr(self.attention, 'q_proj'):
            # Standard/RoPE/ExpoSB attention modules
            q_proj = self.attention.q_proj
            k_proj = self.attention.k_proj
            v_proj = self.attention.v_proj
            num_heads = self.attention.num_heads
            head_dim = self.attention.head_dim
        else:
            # Absolute attention module (uses different naming)
            q_proj = self.attention.query
            k_proj = self.attention.key
            v_proj = self.attention.value
            num_heads = self.attention.num_attention_heads
            head_dim = self.attention.attention_head_size
        
        # Get attention projections
        q = q_proj(hidden_states).view(
            batch_size, seq_len, num_heads, head_dim
        ).transpose(1, 2)
        
        k = k_proj(hidden_states).view(
            batch_size, seq_len, num_heads, head_dim
        ).transpose(1, 2)
        
        v = v_proj(hidden_states).view(
            batch_size, seq_len, num_heads, head_dim
        ).transpose(1, 2)
        
        # Apply causal attention based on attention type
        scale = getattr(self.attention, 'scale', 1.0 / math.sqrt(head_dim))
        
        if self.attention_type == "standard":
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, scale=scale
            )
        elif self.attention_type == "rope":
            # Use RoPE with causal masking
            from ..attention.rope_attention import rope_attention
            attn_output = rope_attention(q, k, v, causal=True, sm_scale=scale)
        elif self.attention_type == "exposb":
            # Use ExpoSB with causal masking
            from ..attention.exposb_attention import exposb_attention
            attn_output = exposb_attention(q, k, v, causal=True, sm_scale=scale)
        elif self.attention_type == "absolute":
            # Use Absolute with causal masking
            from ..attention.absolute_attention import absolute_attention
            attn_output = absolute_attention(q, k, v, causal=True, sm_scale=scale)
        else:
            # Fallback to PyTorch's causal attention
            attn_output = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, attn_mask=None, dropout_p=0.0, is_causal=True, scale=scale
            )
        
        # Apply head mask if provided
        if head_mask is not None:
            attn_output = attn_output * head_mask
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, hidden_size)
        
        # Handle different output projection naming
        if hasattr(self.attention, 'out_proj'):
            output = self.attention.out_proj(attn_output)
        else:
            # For absolute attention, there's no separate output projection
            output = attn_output
        
        # Apply dropout
        dropout_layer = getattr(self.attention, 'dropout', torch.nn.Identity())
        output = dropout_layer(output)
        
        # Return tuple to match GPT2 attention interface
        outputs = (output, None)  # (hidden_states, past_key_value)
        
        if output_attentions:
            outputs = outputs + (None,)  # Add attention weights (not implemented)
        
        return outputs


class ModifiedGPTModel(nn.Module):
    """
    GPT model with configurable attention mechanisms for causal language modeling
    """
    
    def __init__(self, config: GPTConfig, attention_type: str = "standard"):
        super().__init__()
        self.config = config
        self.attention_type = attention_type
        
        # Convert our config to GPT2 config for base model
        gpt2_config = config.to_gpt2_config()
        
        # Create base GPT2 model
        self.gpt = GPT2LMHeadModel(gpt2_config)
        
        # Replace attention layers with custom attention mechanisms
        self._replace_attention_layers()
        
        # Log model creation
        self._log_model_info()
    
    def _replace_attention_layers(self):
        """Replace GPT2 attention layers with custom attention mechanisms"""
        
        if self.attention_type == "standard":
            logger.info("Using standard GPT2 attention (no replacement needed)")
            return
        
        logger.info(f"Replacing {self.config.num_hidden_layers} GPT attention layers with {self.attention_type} attention...")
        
        # Get attention class
        attention_class = get_attention_class(self.attention_type)
        
        for layer_idx in range(self.config.num_hidden_layers):
            # Get the GPT2 attention layer
            gpt_layer = self.gpt.transformer.h[layer_idx]
            original_attention = gpt_layer.attn
            
            # Create new BERT-style attention layer
            if self.attention_type == "rope":
                bert_attention = attention_class(
                    hidden_size=self.config.hidden_size,
                    num_heads=self.config.num_attention_heads,
                    dropout=self.config.attention_probs_dropout_prob
                )
            elif self.attention_type in ["exposb", "absolute"]:
                bert_attention = attention_class(
                    hidden_size=self.config.hidden_size,
                    num_heads=self.config.num_attention_heads,
                    max_position_embeddings=self.config.max_position_embeddings,
                    dropout=self.config.attention_probs_dropout_prob
                )
            else:
                bert_attention = attention_class(
                    hidden_size=self.config.hidden_size,
                    num_heads=self.config.num_attention_heads,
                    max_position_embeddings=self.config.max_position_embeddings,
                    dropout=self.config.attention_probs_dropout_prob
                )
            
            # Wrap with causal attention adapter
            causal_attention = CausalAttentionAdapter(bert_attention, self.attention_type)
            
            # Replace the GPT2 attention with our causal adapter
            gpt_layer.attn = causal_attention
            
            logger.debug(f"  Layer {layer_idx}: {type(original_attention).__name__} -> CausalAttentionAdapter({type(bert_attention).__name__})")
        
        logger.info(f"Successfully replaced GPT attention layers with {self.attention_type} + causal masking")
    
    def _log_model_info(self):
        """Log model information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        logger.info(f"Created GPT model:")
        logger.info(f"  Attention type: {self.attention_type}")
        logger.info(f"  Hidden size: {self.config.hidden_size}")
        logger.info(f"  Layers: {self.config.num_hidden_layers}")
        logger.info(f"  Attention heads: {self.config.num_attention_heads}")
        logger.info(f"  Vocabulary size: {self.config.vocab_size}")
        logger.info(f"  Max position embeddings: {self.config.max_position_embeddings}")
        logger.info(f"  Total parameters: {total_params:,}")
        logger.info(f"  Trainable parameters: {trainable_params:,}")
    
    def forward(self, input_ids, attention_mask=None, labels=None, **kwargs):
        """Forward pass through the modified GPT model"""
        return self.gpt(input_ids=input_ids, attention_mask=attention_mask, labels=labels, **kwargs)


def create_gpt_model(config, attention_type: str = "standard") -> ModifiedGPTModel:
    """
    Create a GPT model with specified attention mechanism for CLM training
    
    Args:
        config: GPTConfig or BertConfig (will be converted)
        attention_type: Type of attention mechanism ("standard", "rope", "exposb", "rse", "absolute")
        
    Returns:
        Modified GPT model with custom attention
    """
    logger.info(f"Creating GPT model with {attention_type} attention")
    
    # Convert BertConfig to GPTConfig if needed
    if hasattr(config, '__class__') and 'Bert' in config.__class__.__name__:
        gpt_config = GPTConfig.from_bert_config(config)
    else:
        gpt_config = config
    
    # Validate attention type
    valid_types = ["standard", "rope", "exposb", "rse", "absolute"]
    if attention_type not in valid_types:
        raise ValueError(f"Invalid attention type: {attention_type}. Valid types: {valid_types}")
    
    model = ModifiedGPTModel(gpt_config, attention_type)
    
    return model


def get_gpt_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Get detailed information about a GPT model
    
    Args:
        model: PyTorch GPT model
        
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
            "vocab_size": getattr(config, 'vocab_size', 'unknown'),
            "max_position_embeddings": getattr(config, 'max_position_embeddings', 'unknown')
        })
    
    return info
