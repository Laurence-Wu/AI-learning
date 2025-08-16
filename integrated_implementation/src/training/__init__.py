"""
Training System for BERT Attention Comparison
===========================================

Comprehensive training framework supporting:
- Multiple attention mechanisms
- Advanced MLM training patterns
- Experiment tracking and monitoring
- Distributed training support
- Mixed precision training
"""

from .trainer import BERTTrainer, TrainingResults
from .scheduler import get_scheduler
from .optimizer import get_optimizer

__all__ = [
    'BERTTrainer',
    'TrainingResults',
    'get_scheduler',
    'get_optimizer'
]