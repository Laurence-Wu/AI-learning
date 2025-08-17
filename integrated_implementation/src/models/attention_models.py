"""
Attention-Specific BERT Models
==============================

Convenience classes for each attention type.
"""

from transformers import BertConfig
from .bert_model import BERTMLMModel


class StandardBERT(BERTMLMModel):
    """BERT with standard attention"""
    def __init__(self, config: BertConfig):
        super().__init__(config, attention_type="standard")


class RoPEBERT(BERTMLMModel):
    """BERT with RoPE attention"""
    def __init__(self, config: BertConfig):
        super().__init__(config, attention_type="rope")


class ExpoSBBERT(BERTMLMModel):
    """BERT with ExpoSB attention"""
    def __init__(self, config: BertConfig):
        super().__init__(config, attention_type="exposb")


class AbsoluteBERT(BERTMLMModel):
    """BERT with absolute positional encoding attention"""
    def __init__(self, config: BertConfig):
        super().__init__(config, attention_type="absolute")