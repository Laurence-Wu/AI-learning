"""
Integrated BERT Attention Mechanism Comparison
============================================

A comprehensive framework for comparing different attention mechanisms in BERT models:
- Standard Attention
- RoPE (Rotary Position Embedding)  
- ExpoSB (Exponential Stick Breaking)
- Absolute Positional Encoding

Features:
- Triton-optimized attention implementations
- MLM (Masked Language Modeling) training patterns
- Unified training pipeline
- Comprehensive evaluation metrics
"""

__version__ = "1.0.0"
__author__ = "AI Learning Project"

from .training import *
from .config import *
from .attention import *
from .utils import *
from .data import *
from .models import *