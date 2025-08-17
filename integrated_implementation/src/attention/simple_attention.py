"""
Simple PyTorch-based attention implementations for debugging
No Triton kernels - pure PyTorch for maximum stability
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class SimpleBERTAttention(nn.Module):
    """Simple BERT attention using PyTorch built-ins only"""
    
    def __init__(self, hidden_size, num_heads, max_position_embeddings=512, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.attention_head_size)
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        cache_position=None,
    ):
        # Standard BERT attention computation
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Use PyTorch's stable scaled_dot_product_attention
        context_layer = F.scaled_dot_product_attention(
            query_layer, 
            key_layer, 
            value_layer,
            attn_mask=None,
            dropout_p=0.0 if not self.training else self.dropout.p,
            is_causal=False,
            scale=self.scale
        )
        
        # Reshape output
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        outputs = (context_layer, None) if output_attentions else (context_layer,)
        return outputs


class SimpleRoPEAttention(nn.Module):
    """Simple RoPE attention using PyTorch only"""
    
    def __init__(self, hidden_size, num_heads, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_heads
        self.attention_head_size = int(hidden_size / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        self.query = nn.Linear(hidden_size, self.all_head_size)
        self.key = nn.Linear(hidden_size, self.all_head_size)
        self.value = nn.Linear(hidden_size, self.all_head_size)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.attention_head_size)
        
        # RoPE parameters
        self.max_seq_len = 2048
        self.base = 10000
        
    def _create_rope_embeddings(self, seq_len, device):
        """Create RoPE embeddings"""
        pos = torch.arange(seq_len, device=device).unsqueeze(1)
        dim_half = self.attention_head_size // 2
        
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim_half, 2, device=device) / dim_half))
        sinusoid_inp = torch.einsum('i,j->ij', pos.squeeze(), inv_freq)
        
        sin = torch.sin(sinusoid_inp)
        cos = torch.cos(sinusoid_inp)
        
        return sin, cos
    
    def _apply_rope(self, x, sin, cos):
        """Apply RoPE to input tensor"""
        # x shape: [batch, heads, seq_len, head_dim]
        seq_len = x.size(2)
        head_dim = x.size(3)
        
        # Only apply to even dimensions for simplicity
        dim_half = head_dim // 2
        x1 = x[..., :dim_half]
        x2 = x[..., dim_half:dim_half*2] if head_dim > dim_half else torch.zeros_like(x1)
        
        # Apply rotation
        sin = sin[:seq_len].unsqueeze(0).unsqueeze(0)
        cos = cos[:seq_len].unsqueeze(0).unsqueeze(0)
        
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        rotated = torch.cat([rotated_x1, rotated_x2], dim=-1)
        
        # If head_dim is odd, append the last dimension unchanged
        if head_dim > dim_half * 2:
            rotated = torch.cat([rotated, x[..., dim_half*2:]], dim=-1)
        
        return rotated
    
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_value=None,
        output_attentions=False,
        cache_position=None,
    ):
        seq_len = hidden_states.size(1)
        
        # Compute Q, K, V
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # Apply RoPE to Q and K
        sin, cos = self._create_rope_embeddings(seq_len, hidden_states.device)
        query_layer = self._apply_rope(query_layer, sin, cos)
        key_layer = self._apply_rope(key_layer, sin, cos)
        
        # Use PyTorch's stable attention
        context_layer = F.scaled_dot_product_attention(
            query_layer, 
            key_layer, 
            value_layer,
            attn_mask=None,
            dropout_p=0.0 if not self.training else self.dropout.p,
            is_causal=False,
            scale=self.scale
        )
        
        # Reshape output
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        outputs = (context_layer, None) if output_attentions else (context_layer,)
        return outputs


# Create aliases for other attention types to fall back to simple implementations
SimpleExpoSBAttention = SimpleBERTAttention  # Fallback to standard for now
SimpleAbsoluteAttention = SimpleBERTAttention  # Fallback to standard for now