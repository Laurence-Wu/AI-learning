"""
Utility Functions and Helpers
============================

Common utilities for BERT training and evaluation:
- Visualization and plotting
- Performance monitoring
- Device management
- File I/O helpers
- Logging utilities
"""

from .visualization import plot_training_curves, plot_attention_comparison, save_plots
from .device import get_device, setup_device, get_memory_info

# Define basic missing functions inline for compatibility
def setup_logging():
    import logging
    logging.basicConfig(level=logging.INFO)
    return logging.getLogger()

def set_seed(seed: int = 42):
    import random
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

__all__ = [
    'plot_training_curves',
    'plot_attention_comparison', 
    'save_plots',
    'get_device',
    'setup_device',
    'get_memory_info',
    'setup_logging',
    'set_seed'
]