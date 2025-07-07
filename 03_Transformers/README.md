# Transformer Implementations Showcase

🤖 **A comprehensive collection of transformer architectures from basic to cutting-edge**

## Overview

This project provides complete implementations of various transformer architectures, demonstrating the evolution from the original transformer to modern variants used in state-of-the-art AI systems. Each implementation is built from scratch using NumPy to provide clear understanding of the underlying mechanisms.

## Project Structure

```
Transformer/
├── README.md                           # This file
├── transformer_demo.py                 # Master demonstration script
├── basic_transformer/
│   └── transformer.py                  # Original transformer implementation
├── gpt_style/
│   └── gpt_transformer.py             # GPT-style autoregressive transformer
├── bert_style/
│   └── bert_transformer.py            # BERT-style bidirectional transformer
└── modern_variants/
    └── modern_transformers.py          # Modern innovations (ViT, MQA, RoPE, etc.)
```

## Transformer Variants

### 1. Basic Transformer (2017)
**File:** `basic_transformer/transformer.py`

The original "Attention Is All You Need" transformer implementation:
- **Multi-Head Attention**: Standard scaled dot-product attention
- **Position Encoding**: Sinusoidal positional embeddings
- **Feed-Forward**: Simple two-layer MLP with ReLU
- **Normalization**: Layer normalization (post-norm)
- **Use Cases**: General sequence-to-sequence tasks

**Key Features:**
✅ Multi-head self-attention mechanism  
✅ Positional encoding with sine/cosine  
✅ Feed-forward networks  
✅ Layer normalization and residual connections  
✅ Encoder-decoder architecture  

### 2. GPT-Style Transformer (2018+)
**File:** `gpt_style/gpt_transformer.py`

Generative Pre-trained Transformer for autoregressive language modeling:
- **Causal Attention**: Masked self-attention for autoregressive generation
- **Position Encoding**: Learned positional embeddings
- **Activation**: GELU activation function
- **Architecture**: Decoder-only, pre-normalization
- **Use Cases**: Text generation, language modeling

**Key Features:**
✅ Causal (masked) self-attention  
✅ Autoregressive text generation  
✅ GELU activation function  
✅ Pre-normalization architecture  
✅ Learned positional embeddings  
✅ Temperature-controlled sampling  
✅ Top-k sampling strategy  

### 3. BERT-Style Transformer (2018)
**File:** `bert_style/bert_transformer.py`

Bidirectional Encoder Representations from Transformers:
- **Bidirectional Attention**: Full attention without causal masking
- **Special Tokens**: [CLS], [SEP], [MASK], [PAD]
- **Embeddings**: Token + Position + Segment embeddings
- **Training**: Masked Language Modeling (MLM) and Next Sentence Prediction (NSP)
- **Use Cases**: Text understanding, classification, feature extraction

**Key Features:**
✅ Bidirectional self-attention  
✅ Masked Language Modeling (MLM)  
✅ Token + Position + Segment embeddings  
✅ [CLS] token for classification  
✅ [SEP] token for sentence separation  
✅ Post-normalization architecture  
✅ GELU activation function  

### 4. Modern Transformer Variants (2020+)
**File:** `modern_variants/modern_transformers.py`

Cutting-edge transformer innovations used in modern AI systems:

#### Vision Transformer (ViT)
- **Patch Embedding**: Converts images to sequence of patches
- **Class Token**: [CLS] token for image classification
- **Applications**: Image classification, computer vision

#### Modern Language Models
- **Multi-Query Attention (MQA)**: Shared keys/values across heads
- **SwiGLU Activation**: Superior to ReLU/GELU
- **RMSNorm**: Simpler alternative to LayerNorm
- **Rotary Position Embedding (RoPE)**: Better position encoding
- **Applications**: Large language models (LLaMA, PaLM)

**Key Modern Components:**
✅ Multi-Query Attention (MQA) - Memory efficient  
✅ SwiGLU Activation - Better performance  
✅ RMSNorm - Simplified normalization  
✅ Rotary Position Embedding (RoPE) - Improved position encoding  
✅ Patch Embedding - Vision capability  
✅ Vision Transformer (ViT) - Image understanding  

## Quick Start

### Prerequisites
```bash
pip install numpy matplotlib seaborn pandas scikit-learn tqdm
```

### Run Individual Demonstrations

```bash
# Basic Transformer
cd basic_transformer
python transformer.py

# GPT-Style Transformer
cd gpt_style
python gpt_transformer.py

# BERT-Style Transformer
cd bert_style
python bert_transformer.py

# Modern Variants
cd modern_variants
python modern_transformers.py
```

### Run Complete Demonstration
```bash
# From Transformer/ directory
python transformer_demo.py
```

## Architecture Comparison

| Feature | Basic | GPT | BERT | ViT | Modern LLM |
|---------|-------|-----|------|-----|------------|
| **Attention** | Bidirectional | Causal | Bidirectional | Bidirectional | Multi-Query |
| **Position** | Sinusoidal | Learned | Learned | Learned | RoPE |
| **Activation** | ReLU | GELU | GELU | GELU | SwiGLU |
| **Normalization** | LayerNorm | LayerNorm | LayerNorm | LayerNorm | RMSNorm |
| **Architecture** | Post-Norm | Pre-Norm | Post-Norm | Pre-Norm | Pre-Norm |
| **Use Case** | Seq2Seq | Generation | Understanding | Vision | Modern LLM |

## Evolution Timeline

- **2017**: Original Transformer ("Attention Is All You Need")
- **2018**: GPT (Generative Pre-training), BERT (Bidirectional understanding)
- **2019**: GPT-2 (Larger scale), RoBERTa (BERT improvements)
- **2020**: GPT-3 (Massive scale), ViT (Vision Transformer)
- **2021**: Switch Transformer (MoE), PaLM (Pathways)
- **2022**: ChatGPT, InstructGPT (Human feedback)
- **2023**: GPT-4, LLaMA (Efficient architectures), Modern innovations

## Key Innovations by Year

### 2017-2018: Foundation
- Multi-head attention mechanism
- Position encoding techniques
- Transformer architecture

### 2019-2020: Scale and Specialization
- Bidirectional vs. Autoregressive attention
- Pre-training objectives (MLM, NSP, CLM)
- Vision applications

### 2021-2023: Efficiency and Scale
- Multi-Query Attention (memory efficiency)
- Rotary Position Embedding (better position understanding)
- SwiGLU activation (performance improvement)
- RMSNorm (simplified normalization)

## Real-World Applications

### Language Models
- **GPT Family**: Text generation, conversation, coding
- **BERT Family**: Text classification, QA, NLU
- **Modern LLMs**: LLaMA, PaLM, ChatGPT, Claude

### Vision Models
- **ViT**: Image classification
- **CLIP**: Vision-language understanding
- **DALL-E**: Image generation from text

### Multimodal Models
- **GPT-4V**: Vision + language understanding
- **Flamingo**: Few-shot learning across modalities
- **BLIP**: Bootstrapped vision-language pre-training

## Performance Characteristics

### Memory Usage
```
Standard Attention: O(n²d)
Multi-Query Attention: O(nd + n²d_k)  # d_k << d
```

### Training Efficiency
- **RMSNorm**: ~15% faster than LayerNorm
- **SwiGLU**: Better convergence than ReLU/GELU
- **RoPE**: Better length extrapolation

### Model Sizes
- **Basic Transformer**: ~65M parameters (base)
- **GPT-3**: 175B parameters
- **PaLM**: 540B parameters
- **Modern Efficient Models**: Better performance with fewer parameters

## Implementation Details

### Attention Mechanisms
```python
# Standard Multi-Head Attention
Q, K, V = linear_transforms(x)
attention_weights = softmax(QK^T / √d_k)
output = attention_weights @ V

# Multi-Query Attention
Q = multi_head_linear(x)  # Multiple query heads
K, V = single_linear(x)   # Shared key/value
```

### Position Encodings
```python
# Sinusoidal (Original)
PE(pos, 2i) = sin(pos/10000^(2i/d))
PE(pos, 2i+1) = cos(pos/10000^(2i/d))

# Rotary (RoPE)
q' = q * cos(θ) - rotate_half(q) * sin(θ)
k' = k * cos(θ) - rotate_half(k) * sin(θ)
```

### Modern Activations
```python
# SwiGLU
def swiglu(x):
    gate = swish(W_gate @ x)
    up = W_up @ x
    return W_down @ (gate * up)
```

## Visualization Features

Each implementation generates comprehensive visualizations:
- **Attention Patterns**: Heatmaps showing attention weights
- **Training Curves**: Loss and accuracy over time
- **Architecture Diagrams**: Model structure and parameters
- **Performance Metrics**: Speed and efficiency comparisons

## Educational Value

This project is designed for:
- **Students**: Learning transformer architectures step-by-step
- **Researchers**: Understanding implementation details
- **Engineers**: Practical insights for deployment
- **Educators**: Teaching modern AI architectures

## Limitations and Future Work

### Current Limitations
- NumPy-only implementation (not optimized for large scale)
- Simplified training procedures
- No distributed training support
- Limited optimization techniques

### Potential Extensions
1. **PyTorch/JAX Implementation**: For production use
2. **Distributed Training**: Multi-GPU support
3. **Advanced Optimizations**: Flash Attention, Gradient Checkpointing
4. **More Variants**: T5, Switch Transformer, PaLM variants
5. **Real Datasets**: Integration with HuggingFace datasets

## Contributing

Contributions are welcome! Areas for improvement:
- Additional transformer variants
- Performance optimizations
- Better visualizations
- More comprehensive testing
- Documentation improvements

## References

### Foundational Papers
1. Vaswani et al. "Attention Is All You Need" (2017)
2. Radford et al. "Improving Language Understanding by Generative Pre-Training" (2018)
3. Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers" (2018)
4. Dosovitskiy et al. "An Image is Worth 16x16 Words: Transformers for Image Recognition" (2020)

### Modern Innovations
1. Shazeer, "GLU Variants Improve Transformer" (2020)
2. Su et al. "RoFormer: Enhanced Transformer with Rotary Position Embedding" (2021)
3. Zhang & Sennrich, "Root Mean Square Layer Normalization" (2019)
4. Ainslie et al. "GQA: Training Generalized Multi-Query Transformer Models" (2023)

## License

This project is for educational purposes. Please refer to individual model papers for specific licensing terms.

---

**🚀 Ready to explore the evolution of transformers? Start with the basic implementation and work your way up to modern variants!**
