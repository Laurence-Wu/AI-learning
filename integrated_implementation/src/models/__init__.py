"""
Models module for BERT implementations
=====================================

Contains BERT model implementations with different attention mechanisms.
"""

from .bert_models import (
    create_bert_model,
    create_clm_model
)
from .gpt_models import (
    create_gpt_model,
    GPTConfig,
    ModifiedGPTModel
)

__all__ = [
    'create_bert_model',
    'create_clm_model',
    'create_gpt_model',
    'GPTConfig',
    'ModifiedGPTModel'
]
