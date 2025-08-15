# Integrated Attention Mechanism Comparison

This folder contains an integrated implementation that compares multiple attention mechanisms for BERT:

## Attention Mechanisms Implemented

1. **Standard Attention**: Traditional scaled dot-product attention
2. **RoPE (Rotary Position Embedding)**: Rotary position embeddings that encode relative positions
3. **ExpoSB (Exponential Stick Breaking)**: Advanced position encoding with exponential decay
4. **Absolute Positional Encoding**: Traditional sinusoidal position embeddings

## Folder Structure

```
integrated_implementation/
├── attention_implementation/       # All attention mechanism implementations
│   ├── triton_standard_attention.py
│   ├── triton_rope_attention.py
│   ├── triton_exposb_attention.py
│   └── triton_absolute_attention.py
├── local_tokenizer/               # BERT tokenizer files
├── training_data/                 # Training data
├── bert_config.py                 # Configuration class
├── config.env                     # Environment configuration
├── data_preprocessing.py          # Data loading utilities
├── integrated_bert_comparison.py  # Main comparison script
└── setup_tokenizer.py            # Tokenizer setup

```

## Configuration

The `config.env` file contains all training parameters and the list of attention algorithms to compare:

```env
ATTENTION_ALGORITHMS=standard,rope,exposb,absolute
```

## Running the Comparison

To run the integrated comparison:

```bash
cd integrated_implementation
python integrated_bert_comparison.py
```

This will:
1. Train BERT models with each attention mechanism
2. Generate comparison plots
3. Save training histories and final models
4. Display performance metrics for all methods

## Output

- Training plots: `bert_comparison.png`
- Model checkpoints: `bert_comparison_outputs/`
- Training histories: `training_histories.json`

## Key Features

- All attention mechanisms implemented in Triton for fair comparison
- Unified training pipeline for consistent evaluation
- Automatic performance visualization
- Configurable through environment variables