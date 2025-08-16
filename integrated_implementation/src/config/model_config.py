"""
Model Configuration
==================

Configuration for BERT model architecture.
"""

from dataclasses import dataclass
from typing import Optional,Dict
from .base_config import BaseConfig


@dataclass
class AttentionConfig(BaseConfig):
    """Attention mechanism configuration"""
    attention_type: str = "standard"
    num_heads: int = 12
    head_dim: Optional[int] = None
    attention_dropout: float = 0.1
    
    def __post_init__(self):
        super().__post_init__()
        if self.attention_type not in ["standard", "rope", "exposb", "absolute"]:
            raise ValueError(f"Unknown attention type: {self.attention_type}")


@dataclass
class ModelConfig(BaseConfig):
    """BERT model configuration"""
    # Architecture
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    
    # Regularization
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    layer_norm_eps: float = 1e-12
    
    # Initialization
    initializer_range: float = 0.02
    
    # Vocabulary
    vocab_size: int = 30522
    type_vocab_size: int = 2
    pad_token_id: int = 0
    
    # Attention configuration
    attention_type: str = "standard"
    
    def get_bert_config_dict(self) -> dict:
        """Get configuration dictionary for HuggingFace BERT"""
        return {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_hidden_layers": self.num_hidden_layers,
            "num_attention_heads": self.num_attention_heads,
            "intermediate_size": self.intermediate_size,
            "hidden_act": "gelu",
            "hidden_dropout_prob": self.hidden_dropout,
            "attention_probs_dropout_prob": self.attention_dropout,
            "max_position_embeddings": self.max_position_embeddings,
            "type_vocab_size": self.type_vocab_size,
            "initializer_range": self.initializer_range,
            "layer_norm_eps": self.layer_norm_eps,
            "pad_token_id": self.pad_token_id
        }
    
    @classmethod
    def from_env_dict(cls, env_dict: Dict[str, str]) -> 'ModelConfig':
        """Create from environment variables"""
        config = {}
        
        # Parse model fields
        field_mapping = {
            'HIDDEN_SIZE': 'hidden_size',
            'NUM_HIDDEN_LAYERS': 'num_hidden_layers',
            'NUM_ATTENTION_HEADS': 'num_attention_heads',
            'MAX_POSITION_EMBEDDINGS': 'max_position_embeddings',
            'VOCAB_SIZE': 'vocab_size'
        }
        
        for env_key, field_name in field_mapping.items():
            if env_key in env_dict:
                config[field_name] = int(env_dict[env_key])
        
        return cls(**config)