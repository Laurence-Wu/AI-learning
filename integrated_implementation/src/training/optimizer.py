"""
Optimizer Utilities
==================

Functions for creating and configuring optimizers.
"""

import torch
from torch import optim
from typing import Optional


def get_optimizer(model, config):
    """
    Create optimizer based on configuration
    
    Args:
        model: PyTorch model
        config: OptimizerConfig object
    
    Returns:
        Configured optimizer
    """
    # Parameter groups with weight decay
    no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay)],
            "weight_decay": config.weight_decay if hasattr(config, 'weight_decay') else 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]
    
    # Create optimizer based on type
    optimizer_type = config.type if hasattr(config, 'type') else 'adamw'
    
    if optimizer_type.lower() == 'adamw':
        optimizer = optim.AdamW(
            optimizer_grouped_parameters,
            lr=config.learning_rate if hasattr(config, 'learning_rate') else 5e-5,
            betas=config.betas if hasattr(config, 'betas') else (0.9, 0.999),
            eps=config.eps if hasattr(config, 'eps') else 1e-8
        )
    elif optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(
            optimizer_grouped_parameters,
            lr=config.learning_rate if hasattr(config, 'learning_rate') else 5e-5,
            betas=config.betas if hasattr(config, 'betas') else (0.9, 0.999),
            eps=config.eps if hasattr(config, 'eps') else 1e-8
        )
    elif optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(
            optimizer_grouped_parameters,
            lr=config.learning_rate if hasattr(config, 'learning_rate') else 0.01,
            momentum=0.9
        )
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
    
    return optimizer