"""
Model Factory Functions
======================

Utilities for creating BERT models with different attention mechanisms.
"""

from transformers import BertConfig
from .bert_model import BERTMLMModel
from .attention_models import StandardBERT, RoPEBERT, ExpoSBBERT, AbsoluteBERT


MODEL_CLASSES = {
    "standard": StandardBERT,
    "rope": RoPEBERT,
    "exposb": ExpoSBBERT,
    "absolute": AbsoluteBERT
}


def create_bert_model(config: BertConfig, attention_type: str = "standard"):
    """
    Create a BERT model with specified attention type
    
    Args:
        config: BERT configuration
        attention_type: Type of attention ("standard", "rope", "exposb", "absolute")
    
    Returns:
        BERT model with specified attention mechanism
    """
    if attention_type not in MODEL_CLASSES:
        raise ValueError(f"Unknown attention type: {attention_type}. "
                        f"Available: {list(MODEL_CLASSES.keys())}")
    
    model_class = MODEL_CLASSES[attention_type]
    return model_class(config)


def get_model_class(attention_type: str):
    """Get model class for attention type"""
    if attention_type not in MODEL_CLASSES:
        raise ValueError(f"Unknown attention type: {attention_type}")
    return MODEL_CLASSES[attention_type]