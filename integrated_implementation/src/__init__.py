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

# Lazy imports to avoid loading everything at once
# Individual modules should be imported directly when needed