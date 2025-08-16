# Enhanced BERT Attention Mechanism Comparison Framework

A comprehensive, production-ready framework for comparing different attention mechanisms in BERT models with advanced MLM (Masked Language Modeling) training patterns.

## 🚀 Features

### Attention Mechanisms
- **Standard Attention**: Traditional scaled dot-product attention
- **RoPE (Rotary Position Embedding)**: Rotary position embeddings for relative position encoding
- **ExpoSB (Exponential Stick Breaking)**: Advanced position encoding with exponential decay
- **Absolute Positional Encoding**: Enhanced absolute position embeddings

### MLM Training Patterns
- **Standard MLM**: Traditional BERT masking (80% [MASK], 10% random, 10% unchanged)
- **Dynamic MLM**: Variable masking probability per batch
- **Span MLM**: Span-based masking similar to SpanBERT
- **Whole Word MLM**: Mask entire words rather than subword tokens

### Advanced Features
- 🔧 **Modular Architecture**: Clean, extensible codebase with proper separation of concerns
- ⚡ **Triton Optimization**: All attention mechanisms implemented with Triton for maximum performance
- 📊 **Comprehensive Metrics**: Advanced training monitoring and visualization
- 🎛️ **Flexible Configuration**: YAML/JSON/Environment file configuration support
- 🔄 **Reproducible Experiments**: Deterministic training with seed management
- 💾 **Mixed Precision**: FP16 training support for faster training and lower memory usage
- 📈 **Experiment Tracking**: Integration with Weights & Biases and custom metrics
- 🌐 **Distributed Training**: Multi-GPU training support

## 📁 Project Structure

```
integrated_implementation/
├── src/                           # Source code
│   ├── attention/                 # Attention mechanism implementations
│   │   ├── __init__.py
│   │   ├── standard_attention.py  # Standard BERT attention
│   │   ├── rope_attention.py      # RoPE attention
│   │   ├── exposb_attention.py    # ExpoSB attention
│   │   └── absolute_attention.py  # Absolute position attention
│   ├── data/                      # Data processing and MLM patterns
│   │   ├── __init__.py
│   │   ├── dataset.py             # Enhanced BERT dataset
│   │   ├── mlm_patterns.py        # MLM masking strategies
│   │   └── preprocessing.py       # Advanced data preprocessing
│   ├── models/                    # Model implementations
│   │   ├── __init__.py
│   │   ├── bert_model.py          # Modular BERT model
│   │   ├── attention_models.py    # Attention-specific models
│   │   └── model_factory.py       # Model creation utilities
│   ├── training/                  # Training system
│   │   ├── __init__.py
│   │   ├── trainer.py             # Main training loop
│   │   ├── callbacks.py           # Training callbacks
│   │   ├── metrics.py             # Metrics tracking
│   │   └── distributed.py         # Distributed training
│   ├── config/                    # Configuration management
│   │   ├── __init__.py
│   │   ├── experiment_config.py   # Main configuration
│   │   ├── training_config.py     # Training parameters
│   │   ├── model_config.py        # Model parameters
│   │   └── data_config.py         # Data parameters
│   └── utils/                     # Utilities
│       ├── __init__.py
│       ├── visualization.py       # Plotting and visualization
│       ├── device.py              # Device management
│       └── logging.py             # Logging utilities
├── configs/                       # Configuration files
│   ├── default.yaml              # Default configuration
│   ├── quick_test.yaml           # Quick test configuration
│   └── production.yaml           # Production configuration
├── scripts/                       # Utility scripts
│   ├── setup_environment.py      # Environment setup
│   ├── download_data.py          # Data download utilities
│   └── analyze_results.py        # Results analysis
├── tests/                         # Test suite
│   ├── test_attention.py         # Attention mechanism tests
│   ├── test_mlm_patterns.py      # MLM pattern tests
│   └── test_training.py          # Training tests
├── train.py                       # Main training script
├── evaluate.py                   # Evaluation script
├── config.yaml                   # Default configuration
└── README.md                     # This file
```

## 🛠️ Installation

### Prerequisites
- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU training)
- Triton (for optimized attention)

### Setup

1. **Clone the repository**:
```bash
git clone <repository_url>
cd integrated_implementation
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Setup tokenizer** (optional - downloads automatically if not present):
```bash
python scripts/setup_environment.py --setup-tokenizer
```

## 🚀 Quick Start

### Basic Training

Train all attention mechanisms with default settings:

```bash
python train.py
```

### Custom Configuration

Create a custom configuration file:

```yaml
# config.yaml
experiment_name: "my_bert_comparison"
attention_algorithms: ["standard", "rope"]

training:
  num_epochs: 5
  batch_size: 16
  learning_rate: 5e-5

data:
  max_seq_length: 256
  mlm_strategy: "dynamic"
  mlm_probability: 0.15

model:
  hidden_size: 512
  num_hidden_layers: 6
  num_attention_heads: 8
```

Run with custom config:
```bash
python train.py --config config.yaml
```

### Command Line Options

```bash
# Train specific attention mechanisms
python train.py --attention standard rope --epochs 10

# Use different MLM strategy
python train.py --mlm-strategy span --mlm-probability 0.20

# Enable mixed precision and distributed training
python train.py --mixed-precision --distributed

# Debug mode with detailed logging
python train.py --debug --batch-size 4 --epochs 1
```

## 📊 MLM Training Patterns

### Standard MLM (BERT-style)
- 15% of tokens are masked
- 80% replaced with [MASK], 10% random token, 10% unchanged
- Random token-level masking

```python
from src.data import MLMConfig, MLMStrategy

mlm_config = MLMConfig(
    strategy=MLMStrategy.STANDARD,
    mlm_probability=0.15
)
```

### Dynamic MLM
- Variable masking probability per batch (10-20%)
- Increases training robustness
- Better generalization

```python
mlm_config = MLMConfig(
    strategy=MLMStrategy.DYNAMIC,
    min_mlm_prob=0.10,
    max_mlm_prob=0.20
)
```

### Span MLM (SpanBERT-style)
- Masks contiguous spans of tokens
- Better contextual understanding
- Configurable span length distribution

```python
mlm_config = MLMConfig(
    strategy=MLMStrategy.SPAN,
    span_length_mean=3.0,
    max_span_length=10
)
```

### Whole Word MLM
- Masks entire words (not subword tokens)
- More challenging prediction task
- Better word-level understanding

```python
mlm_config = MLMConfig(
    strategy=MLMStrategy.WHOLE_WORD,
    mlm_probability=0.15
)
```

## 🎯 Attention Mechanisms

### Standard Attention
Traditional BERT attention with learned position embeddings:
```python
from src.attention import StandardBERTAttention

attention = StandardBERTAttention(
    hidden_size=768,
    num_heads=12,
    max_position_embeddings=512
)
```

### RoPE Attention
Rotary Position Embedding for relative position encoding:
```python
from src.attention import RoPEBERTAttention

attention = RoPEBERTAttention(
    hidden_size=768,
    num_heads=12
)
# No max_position_embeddings needed - RoPE handles infinite sequences
```

### ExpoSB Attention
Exponential Stick Breaking attention with advanced position encoding:
```python
from src.attention import ExpoSBBERTAttention

attention = ExpoSBBERTAttention(
    hidden_size=768,
    num_heads=12,
    max_position_embeddings=512
)
```

### Absolute Attention
Enhanced absolute positional encoding:
```python
from src.attention import AbsoluteBERTAttention

attention = AbsoluteBERTAttention(
    hidden_size=768,
    num_heads=12,
    max_position_embeddings=512
)
```

## 📈 Configuration Management

### YAML Configuration
```yaml
experiment_name: "bert_attention_study"
description: "Comprehensive attention mechanism comparison"

# Training parameters
training:
  num_epochs: 20
  batch_size: 32
  learning_rate: 5e-5
  warmup_steps: 1000
  weight_decay: 0.01
  gradient_clipping: 1.0
  
  # Optimizer settings
  optimizer:
    type: "adamw"
    betas: [0.9, 0.999]
    eps: 1e-8
  
  # Scheduler settings
  scheduler:
    type: "cosine"
    num_warmup_steps: 1000

# Model parameters
model:
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12
  intermediate_size: 3072
  max_position_embeddings: 512
  attention_dropout: 0.1
  hidden_dropout: 0.1

# Data parameters
data:
  training_data_file: "./training_data/processed_training_data.txt"
  max_seq_length: 512
  mlm_strategy: "standard"
  mlm_probability: 0.15
  
  # Preprocessing
  preprocessing:
    min_chars: 20
    max_chars: 10000
    deduplicate: true

# Attention mechanisms to compare
attention_algorithms: ["standard", "rope", "exposb", "absolute"]

# Output settings
output_dir: "./outputs"
save_config: true
use_wandb: false
```

### Environment File Configuration
```bash
# config.env
EXPERIMENT_NAME=bert_comparison
ATTENTION_ALGORITHMS=standard,rope,exposb,absolute
NUM_EPOCHS=10
BATCH_SIZE=32
LEARNING_RATE=5e-5
OUTPUT_DIR=./outputs
SEED=42
DEVICE=auto
MIXED_PRECISION=true
```

### Programmatic Configuration
```python
from src.config import ExperimentConfig, TrainingConfig, ModelConfig

config = ExperimentConfig(
    experiment_name="custom_experiment",
    attention_algorithms=["standard", "rope"],
    training=TrainingConfig(
        num_epochs=10,
        batch_size=16,
        learning_rate=3e-5
    ),
    model=ModelConfig(
        hidden_size=512,
        num_hidden_layers=6
    )
)
```

## 📊 Results and Visualization

### Training Metrics
The framework tracks comprehensive metrics:
- Training/Validation MLM Loss
- Learning Rate Schedule
- Gradient Norms
- Memory Usage
- Training Speed (tokens/second)

### Automatic Visualization
Training generates comparison plots:
- Loss curves for all attention mechanisms
- Learning rate schedules
- Performance comparisons
- Memory usage analysis

### Results Analysis
```python
from src.utils import analyze_results

# Load and analyze results
results = analyze_results("outputs/experiment_results.json")
results.plot_comparison()
results.print_summary()
```

## 🧪 Testing and Validation

### Run Tests
```bash
# Run all tests
python -m pytest tests/

# Test specific components
python -m pytest tests/test_attention.py
python -m pytest tests/test_mlm_patterns.py

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

### Validation Scripts
```bash
# Validate attention implementations
python scripts/validate_attention.py

# Test MLM patterns
python scripts/test_mlm_patterns.py

# Benchmark performance
python scripts/benchmark_performance.py
```

## 🔧 Advanced Usage

### Custom Attention Mechanism
```python
from src.attention import BaseAttention
import torch.nn as nn

class MyCustomAttention(BaseAttention):
    def __init__(self, hidden_size, num_heads, **kwargs):
        super().__init__()
        # Your implementation here
        
    def forward(self, hidden_states, attention_mask=None, **kwargs):
        # Your attention computation
        return outputs

# Register your attention
from src.attention import ATTENTION_REGISTRY
ATTENTION_REGISTRY['custom'] = MyCustomAttention
```

### Custom MLM Strategy
```python
from src.data.mlm_patterns import MLMCollator

class CustomMLMCollator(MLMCollator):
    def _apply_custom_mlm(self, input_ids):
        # Your custom masking logic
        return labels

# Use in training
collator = CustomMLMCollator(tokenizer, custom_config)
```

### Distributed Training
```bash
# Single node, multiple GPUs
python -m torch.distributed.launch --nproc_per_node=4 train.py --distributed

# Multiple nodes
python -m torch.distributed.launch \
    --nnodes=2 \
    --node_rank=0 \
    --master_addr="master_node_ip" \
    --master_port=29500 \
    --nproc_per_node=4 \
    train.py --distributed
```

## 📋 Requirements

```txt
torch>=2.0.0
triton>=2.0.0
transformers>=4.21.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
pyyaml>=6.0
wandb>=0.13.0  # optional
pytest>=7.0.0  # for testing
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Make your changes and add tests
4. Run tests: `python -m pytest tests/`
5. Submit a pull request

### Code Style
- Follow PEP 8
- Use type hints
- Add docstrings for all functions/classes
- Write tests for new features

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Original BERT paper: Devlin et al., "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
- RoPE paper: Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding"
- SpanBERT paper: Joshi et al., "SpanBERT: Improving Pre-training by Representing and Predicting Spans"
- Triton framework for GPU kernel optimization

## 📞 Support

For questions and support:
- Open an issue on GitHub
- Check the documentation in `docs/`
- Review example configurations in `configs/`

---

**Happy Training! 🚀**