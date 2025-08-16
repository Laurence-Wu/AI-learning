"""
Learning Rate Scheduler Utilities
=================================

Functions for creating learning rate schedulers.
"""

import torch
from torch.optim import lr_scheduler
import math


def get_scheduler(optimizer, config, num_training_steps):
    """
    Create learning rate scheduler based on configuration
    
    Args:
        optimizer: PyTorch optimizer
        config: SchedulerConfig object
        num_training_steps: Total number of training steps
    
    Returns:
        Configured scheduler
    """
    scheduler_type = config.type if hasattr(config, 'type') else 'cosine'
    warmup_steps = config.num_warmup_steps if hasattr(config, 'num_warmup_steps') else 0
    
    if scheduler_type.lower() == 'cosine':
        # Cosine annealing with warm restarts
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
            return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_type.lower() == 'linear':
        # Linear decay
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - warmup_steps)))
        
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    elif scheduler_type.lower() == 'constant':
        # Constant learning rate with warmup
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0
        
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")
    
    return scheduler