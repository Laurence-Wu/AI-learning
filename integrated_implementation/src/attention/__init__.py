"""
Attention Mechanism Implementations
==================================

Collection of Triton-optimized attention mechanisms for BERT comparison:
- StandardBERTAttention: Traditional scaled dot-product attention
- RoPEBERTAttention: Rotary Position Embedding attention
- ExpoSBBERTAttention: Exponential Stick Breaking attention  
- RSEBERTAttention: Rotary Stick-breaking Encoding attention
- AbsoluteBERTAttention: Absolute positional encoding attention
"""

from .standard_attention import StandardBERTAttention
from .rope_attention import RoPEBERTAttention
from .exposb_attention import ExpoSBBERTAttention
from .rse_attention import RSEBERTAttention
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
    'RSEBERTAttention',
    'AbsoluteBERTAttention',
    'SimpleBERTAttention',
    'SimpleRoPEAttention',
    'SimpleExpoSBAttention',
    'SimpleAbsoluteAttention'
]

# Attention mechanism registry for dynamic loading
# Back to original Triton implementations
ATTENTION_REGISTRY = {
    'standard': StandardBERTAttention,
    'rope': RoPEBERTAttention,
    'exposb': ExpoSBBERTAttention,
    'rse': RSEBERTAttention,
    'absolute': AbsoluteBERTAttention
}

# Simple PyTorch implementations (fallback for debugging)
SIMPLE_ATTENTION_REGISTRY = {
    'standard': SimpleBERTAttention,
    'rope': SimpleRoPEAttention,
    'exposb': SimpleExpoSBAttention,
    'absolute': SimpleAbsoluteAttention
}

def get_attention_class(attention_type: str):
    """Get attention class by name"""
    if attention_type not in ATTENTION_REGISTRY:
        raise ValueError(f"Unknown attention type: {attention_type}. "
                        f"Available: {list(ATTENTION_REGISTRY.keys())}")
    return ATTENTION_REGISTRY[attention_type]