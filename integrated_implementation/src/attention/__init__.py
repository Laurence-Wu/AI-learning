"""
Attention Mechanism Implementations
==================================

Collection of Triton-optimized attention mechanisms for BERT comparison:
- StandardBERTAttention: Traditional scaled dot-product attention
- RoPEBERTAttention: Rotary Position Embedding attention
- ExpoSBBERTAttention: Exponential Stick Breaking attention  
- AbsoluteBERTAttention: Absolute positional encoding attention
"""

from .standard_attention import StandardBERTAttention
from .rope_attention import RoPEBERTAttention
from .exposb_attention import ExpoSBBERTAttention
from .absolute_attention import AbsoluteBERTAttention

# Import simple fallback implementations for debugging
from .simple_attention import (
    SimpleBERTAttention,
    SimpleRoPEAttention,
    SimpleExpoSBAttention,
    SimpleAbsoluteAttention
)

__all__ = [
    'StandardBERTAttention',
    'RoPEBERTAttention', 
    'ExpoSBBERTAttention',
    'AbsoluteBERTAttention',
    'SimpleBERTAttention',
    'SimpleRoPEAttention',
    'SimpleExpoSBAttention',
    'SimpleAbsoluteAttention'
]

# Attention mechanism registry for dynamic loading
# Using simple implementations for stability testing
ATTENTION_REGISTRY = {
    'standard': SimpleBERTAttention,  # Use simple version for now
    'rope': SimpleRoPEAttention,     # Use simple version for now
    'exposb': SimpleExpoSBAttention, # Use simple version for now
    'absolute': SimpleAbsoluteAttention  # Use simple version for now
}

# Original Triton implementations (can be restored later)
TRITON_ATTENTION_REGISTRY = {
    'standard': StandardBERTAttention,
    'rope': RoPEBERTAttention,
    'exposb': ExpoSBBERTAttention,
    'absolute': AbsoluteBERTAttention
}

def get_attention_class(attention_type: str):
    """Get attention class by name"""
    if attention_type not in ATTENTION_REGISTRY:
        raise ValueError(f"Unknown attention type: {attention_type}. "
                        f"Available: {list(ATTENTION_REGISTRY.keys())}")
    return ATTENTION_REGISTRY[attention_type]