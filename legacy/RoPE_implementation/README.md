# BERT Attention Comparison: Standard vs RoPE with Triton

## Overview

This project implements a scientific comparison between standard BERT attention (with absolute position embeddings) and RoPE (Rotary Position Embedding) attention, both implemented using Triton kernels for fair performance comparison.

## Project Structure

```
RoPE_implementation/
├── triton_standard_attention.py  # Triton implementation of standard BERT attention
├── triton_rope_attention.py      # Triton implementation of RoPE attention
├── bert_comparison_train.py      # Main training script for comparison
├── BERT_RoPE_Analysis.md        # Detailed analysis of RoPE integration
└── roformer/                     # Original RoFormer repository
```

## Key Features

### 1. **Triton-Accelerated Attention Implementations**

Both attention mechanisms are implemented in Triton for:
- Fair performance comparison (same optimization level)
- GPU acceleration with tensor cores
- Memory-efficient computation using Flash Attention algorithm

### 2. **Standard BERT Attention (`triton_standard_attention.py`)**
- Traditional absolute position embeddings
- Full forward and backward pass in Triton
- Memory-efficient online softmax
- Compatible with existing BERT models

### 3. **RoPE Attention (`triton_rope_attention.py`)**
- Rotary position embeddings applied inline
- No separate position embedding parameters (saves memory)
- Position-aware without explicit position tokens
- Better extrapolation to longer sequences

### 4. **Comprehensive Comparison Framework (`bert_comparison_train.py`)**
- Trains identical BERT models with different attention
- Tracks MLM loss, LM loss, and learning curves
- Generates comparison plots
- Measures parameter efficiency

## Installation

```bash
# Install dependencies
pip install torch triton transformers matplotlib numpy

# Optional: for experiment tracking
pip install wandb
```

## Usage

### Quick Start

```python
# Run the full comparison
python bert_comparison_train.py
```

This will:
1. Train a BERT model with standard attention
2. Train an identical BERT model with RoPE attention
3. Generate comparison plots
4. Save both models for further analysis

### Custom Training

```python
from triton_standard_attention import StandardBERTAttention
from triton_rope_attention import RoPEBERTAttention
from bert_comparison_train import ModifiedBERTModel, TrainingConfig

# Create model with RoPE attention
config = BertConfig(...)
model = ModifiedBERTModel(config, attention_type="rope")

# Train with custom configuration
trainer_config = TrainingConfig(
    model_type="rope",
    batch_size=64,
    learning_rate=5e-5,
    num_epochs=10
)
```

## Key Differences

### Standard Attention
```python
# Uses absolute position embeddings
position_embeddings = self.position_embeddings(position_ids)
hidden_states = hidden_states + position_embeddings

# Attention computation
attention_scores = Q @ K^T / sqrt(d_k)
```

### RoPE Attention
```python
# No position embeddings needed
# Rotation applied directly to Q and K

# Apply rotation based on position
q_rotated = q * cos(pos * freq) - q_shifted * sin(pos * freq)
k_rotated = k * cos(pos * freq) - k_shifted * sin(pos * freq)

# Attention with rotated vectors
attention_scores = Q_rotated @ K_rotated^T / sqrt(d_k)
```

## Performance Metrics

The comparison tracks:

1. **MLM Loss**: Masked Language Modeling performance
2. **LM Loss**: Language Modeling loss
3. **Training Efficiency**: Steps to convergence
4. **Memory Usage**: Parameter count and activation memory
5. **Inference Speed**: Tokens per second

## Expected Results

Based on literature and our Triton optimizations:

### RoPE Advantages
- **Better long-range dependencies**: ~5-10% improvement on long sequences
- **Parameter efficiency**: Saves ~0.5M parameters (no position embeddings)
- **Extrapolation**: Better performance on sequences longer than training
- **Training stability**: More consistent convergence

### Standard Attention Advantages
- **Faster initial convergence**: Simpler optimization landscape
- **Compatibility**: Works with existing BERT checkpoints
- **Well-studied**: Extensive literature and best practices

## Visualization

The training script generates comparison plots showing:

```
┌─────────────────────────┬─────────────────────────┐
│   Training MLM Loss     │   Evaluation MLM Loss   │
│   ━━ Standard           │   ━━ Standard           │
│   ━━ RoPE              │   ━━ RoPE              │
├─────────────────────────┼─────────────────────────┤
│   Training LM Loss      │   Learning Rate         │
│   ━━ Standard           │   Schedule              │
│   ━━ RoPE              │                         │
└─────────────────────────┴─────────────────────────┘
```

## Technical Details

### Triton Kernel Optimizations

1. **Block-wise Computation**: Process attention in SRAM-sized blocks
2. **Online Softmax**: Compute softmax without materializing full attention matrix
3. **Fused Operations**: Combine RoPE rotation with attention computation
4. **Memory Coalescing**: Optimize memory access patterns

### RoPE Mathematics

```
Given position m and dimension i:
θ_i = 10000^(-2i/d)
R_m = [cos(mθ_i), -sin(mθ_i)]
      [sin(mθ_i),  cos(mθ_i)]

Rotated vectors:
q_m = R_m @ q
k_n = R_n @ k

Attention preserves relative position:
q_m^T @ k_n = q^T @ R_{n-m} @ k
```

## Customization

### Modify Attention Parameters

```python
# In triton_rope_attention.py
theta = 10000.0  # Base for frequency computation
BLOCK_M = 64     # Query block size
BLOCK_N = 32     # Key/Value block size
```

### Add Custom Position Encoding

```python
class CustomRoPEAttention(RoPEBERTAttention):
    def compute_freqs(self, seq_len, dim):
        # Custom frequency computation
        return custom_freqs
```

## Troubleshooting

### Out of Memory
- Reduce `batch_size` in TrainingConfig
- Reduce `BLOCK_M` and `BLOCK_N` in kernels
- Enable gradient checkpointing

### Slow Training
- Ensure CUDA and Triton are properly installed
- Check GPU utilization with `nvidia-smi`
- Reduce `num_stages` in kernel configs

### Numerical Instability
- Use FP32 accumulation in attention
- Adjust `sm_scale` parameter
- Check gradient norms

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{rope_bert_comparison,
  title={BERT Attention Comparison: Standard vs RoPE with Triton},
  author={Your Name},
  year={2024},
  howpublished={\url{https://github.com/yourusername/rope-bert-comparison}}
}

@article{su2021roformer,
  title={RoFormer: Enhanced Transformer with Rotary Position Embedding},
  author={Su, Jianlin and Lu, Yu and Pan, Shengfeng and Wen, Bo and Liu, Yunfeng},
  journal={arXiv preprint arXiv:2104.09864},
  year={2021}
}
```

## Future Work

1. **Extended Benchmarks**: Test on longer sequences (>512 tokens)
2. **Other Position Encodings**: ALiBi, T5 relative positions
3. **Downstream Tasks**: Fine-tuning comparison on GLUE
4. **Efficiency Analysis**: FLOPs, memory, latency measurements
5. **Scaling Laws**: How RoPE affects model scaling

## License

MIT License - See LICENSE file for details
