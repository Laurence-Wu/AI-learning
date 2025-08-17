"""
Learning Rate Scheduler Utilities
=================================

Functions for creating learning rate schedulers with warmup and decay strategies.
"""

import torch
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import _LRScheduler
import math
import logging

logger = logging.getLogger(__name__)


class WarmupCosineScheduler(_LRScheduler):
    """Learning Rate Scheduler with Linear Warmup followed by Cosine Annealing"""
    
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, 
                 min_lr_ratio: float = 0.0, last_epoch: int = -1):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self._step_count <= self.warmup_steps:
            # Linear warmup
            warmup_factor = self._step_count / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            cosine_steps = self.total_steps - self.warmup_steps
            current_cosine_step = self._step_count - self.warmup_steps
            cosine_factor = 0.5 * (1 + math.cos(math.pi * current_cosine_step / cosine_steps))
            lr_factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_factor
            return [base_lr * lr_factor for base_lr in self.base_lrs]


def get_scheduler(optimizer, config, num_training_steps):
    """Create learning rate scheduler with warmup and cosine annealing"""
    
    # Calculate warmup steps (10% of total steps)
    warmup_steps = max(50, int(num_training_steps * 0.1))
    
    # Use warmup + cosine annealing (BERT standard)
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=warmup_steps,
        total_steps=num_training_steps,
        min_lr_ratio=0.0
    )
    
    return scheduler


# Simplified scheduler module - merged advanced features into single implementation