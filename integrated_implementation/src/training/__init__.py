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
from .scheduler import get_scheduler, SchedulerConfig  
from .optimizer import get_optimizer, OptimizerConfig
from .callbacks import TrainingCallbacks, EvaluationCallback, CheckpointCallback
from .metrics import MetricsTracker, MLMMetrics
from .distributed import setup_distributed, cleanup_distributed

__all__ = [
    'BERTTrainer',
    'TrainingResults',
    'get_scheduler',
    'get_optimizer', 
    'SchedulerConfig',
    'OptimizerConfig',
    'TrainingCallbacks',
    'EvaluationCallback',
    'CheckpointCallback',
    'MetricsTracker',
    'MLMMetrics',
    'setup_distributed',
    'cleanup_distributed'
]