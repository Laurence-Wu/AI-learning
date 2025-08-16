# Comprehensive BERT Training: MLM vs CLM Comparison

## Overview

This training script provides a rigorous comparison between **Masked Language Modeling (MLM)** and **Causal Language Modeling (CLM)** across multiple attention mechanisms with comprehensive statistical analysis.

### Key Features

- 🔬 **Statistical Rigor**: Multiple runs, cross-validation, significance testing
- 📊 **Comprehensive Metrics**: Loss, accuracy, convergence analysis
- 🎯 **Bias Reduction**: Stratified splits, randomized seeds, proper controls
- 📈 **Rich Visualizations**: Comparative plots, heatmaps, statistical analysis
- ⚙️ **Configurable**: YAML-based configuration with command-line overrides

## Quick Start

### 1. Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Setup local tokenizer (optional, for offline training)
python setup_tokenizer.py
```

### 2. Basic Usage

```bash
# Quick test (minimal epochs and data)
python train.py --quick-test

# Full comparison with default settings
python train.py

# Custom configuration
python train.py --config configs/custom.yaml --num-runs 5

# Specific objectives and attention mechanisms
python train.py --objectives mlm clm --attention standard rope
```

### 3. Statistical Analysis

```bash
# Cross-validation for robust results
python train.py --cross-validation --folds 5 --num-runs 3

# Custom statistical parameters
python train.py --significance-level 0.01 --confidence-level 0.99
```

## Configuration

### Default Configuration (`configs/default.yaml`)

```yaml
# Training objectives to compare
training_objectives: ["mlm", "clm"]

# Attention mechanisms to compare  
attention_algorithms: ["standard", "rope", "exposb", "absolute"]

# Training parameters
training:
  num_epochs: 20
  batch_size: 32
  learning_rate: 5e-5
  
# Data configuration
data:
  max_seq_length: 512
  mlm_strategy: "standard"
  clm_strategy: "standard"
  
# Model architecture
model:
  hidden_size: 768
  num_hidden_layers: 12
  num_attention_heads: 12
```

### Command Line Arguments

#### Core Options
- `--config`: Path to configuration file
- `--objectives`: Training objectives (`mlm`, `clm`, `both`)
- `--attention`: Attention mechanisms to compare
- `--num-runs`: Number of independent runs (default: 3)

#### Statistical Rigor
- `--cross-validation`: Enable cross-validation
- `--folds`: Number of CV folds (default: 3)
- `--seeds`: Custom random seeds
- `--significance-level`: Statistical significance level (default: 0.05)

#### Quick Overrides
- `--epochs`: Number of training epochs
- `--batch-size`: Training batch size
- `--learning-rate`: Learning rate
- `--data`: Path to training data file

#### Testing & Debugging
- `--quick-test`: Minimal epochs and data for testing
- `--dry-run`: Configuration validation without training
- `--debug`: Enable debug mode

## Experimental Design

### Training Objectives

#### MLM (Masked Language Modeling)
- **Strategy**: Bidirectional context understanding
- **Masking**: 15% of tokens masked randomly
- **Variants**: Standard, dynamic, span, whole-word masking
- **Metric**: Masked token prediction accuracy

#### CLM (Causal Language Modeling)  
- **Strategy**: Autoregressive next-token prediction
- **Context**: Left-to-right sequential processing
- **Variants**: Standard, packed sequences, sliding window
- **Metric**: Next-token prediction accuracy

### Attention Mechanisms

1. **Standard**: Traditional scaled dot-product attention
2. **RoPE**: Rotary Position Embedding attention
3. **ExpoSB**: Exponential Scaling and Binning attention
4. **Absolute**: Absolute position embedding attention

### Statistical Controls

#### Bias Reduction Measures
- **Multiple Random Seeds**: Ensures reproducible variance
- **Stratified Data Splits**: Balanced train/val/test by text length
- **Cross-Validation**: K-fold validation for robust estimates
- **Proper Controls**: Same data, same architectures across comparisons

#### Statistical Analysis
- **Normality Testing**: Shapiro-Wilk test
- **Significance Testing**: t-test (normal) or Mann-Whitney U (non-normal)
- **Effect Size**: Cohen's d calculation
- **Confidence Intervals**: Bootstrapped confidence intervals

## Output Structure

### Files Generated

```
outputs/
├── experiment_name_raw_results.json          # Raw experimental data
├── experiment_name_summary.json              # Aggregated statistics
├── experiment_name_statistical_analysis.json # Significance testing
├── experiment_name_report.md                 # Human-readable report
├── experiment_name_mlm_vs_clm_comparison.png # Main comparison plot
├── experiment_name_attention_heatmap.png     # Performance heatmap
├── experiment_name_training_curves.png       # Training progression
└── experiment_name_statistical_analysis.png  # Statistical significance
```

### Models Saved

```
outputs/
└── experiments/
    ├── standard_mlm_run0/
    │   └── final_model.pt
    ├── standard_clm_run0/
    │   └── final_model.pt
    └── ...
```

## Example Workflows

### 1. Quick Comparison Test

```bash
# Fast test with minimal resources
python train.py --quick-test --objectives both --attention standard rope
```

**Expected Output**: 2 objectives × 2 attention × 3 runs = 12 experiments

### 2. Full Rigorous Comparison

```bash
# Comprehensive statistical analysis
python train.py --num-runs 5 --cross-validation --folds 3
```

**Expected Output**: 2 objectives × 4 attention × 3 folds × 5 runs = 120 experiments

### 3. Custom Research Question

```bash
# Compare only MLM with specific attention mechanisms
python train.py --objectives mlm --attention rope exposb --epochs 50
```

### 4. Reproduce Results

```bash
# Use specific seeds for reproducibility
python train.py --seeds 42 123 456 --num-runs 3
```

## Statistical Interpretation

### Key Metrics

- **Validation Loss**: Primary comparison metric (lower is better)
- **Validation Accuracy**: Secondary metric (higher is better)
- **Training Time**: Efficiency comparison
- **Convergence Step**: When model stops improving

### Significance Testing

The script automatically performs:

1. **MLM vs CLM comparisons** for each attention mechanism
2. **Attention mechanism comparisons** within each objective
3. **Effect size calculations** (Cohen's d)
4. **Confidence interval estimation**

### Interpreting Results

- **p < 0.05**: Statistically significant difference
- **Effect Size > 0.8**: Large practical difference
- **Confidence Intervals**: Range of likely true performance

## Troubleshooting

### Common Issues

1. **Memory Errors**: Reduce batch size or enable gradient checkpointing
2. **CUDA Errors**: Ensure GPU compatibility or use CPU
3. **Import Errors**: Install all dependencies from requirements.txt
4. **Config Errors**: Validate YAML syntax and required fields

### Performance Optimization

```bash
# Reduce memory usage
python train.py --batch-size 16 --gradient-checkpointing

# Use mixed precision
python train.py --mixed-precision

# CPU-only training
python train.py --device cpu
```

### Debugging

```bash
# Enable detailed logging
python train.py --debug

# Validate configuration without training
python train.py --dry-run

# Test configuration structure
python test_train.py
```

## Research Applications

### Supported Research Questions

1. **Objective Comparison**: "Is MLM or CLM better for specific tasks?"
2. **Attention Analysis**: "Which attention mechanism performs best?"
3. **Architecture Study**: "How do different components interact?"
4. **Efficiency Analysis**: "What's the speed/accuracy trade-off?"

### Extending the Framework

The script is designed for extensibility:

- **Add New Objectives**: Extend training objective types
- **Custom Attention**: Implement new attention mechanisms  
- **Additional Metrics**: Add task-specific evaluation metrics
- **New Architectures**: Compare different model architectures

## Citation

If you use this training framework in your research, please cite:

```bibtex
@software{bert_mlm_clm_comparison,
  title={Comprehensive BERT Training: MLM vs CLM Comparison Framework},
  author={[Your Name]},
  year={2024},
  url={https://github.com/[your-repo]/bert-attention-comparison}
}
```

## License

[Add your license information here]

---

**Note**: This framework is designed for research purposes. For production use, additional optimizations and validations may be required.