#!/usr/bin/env python3
"""
Simple BERT Training: MLM vs CLM Comparison
==========================================

Basic training script that compares MLM and CLM objectives across attention mechanisms.
Generates simple loss comparison plots.

Usage:
    python train_simple.py
    python train_simple.py --debug
    python train_simple.py --config custom.env
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List
from dataclasses import dataclass

# Add project root and src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from transformers import BertTokenizer, BertConfig

from configs.experiment_config import load_config
from src.data.mlm_patterns import BERTMLMDataset, get_dataloader
from src.data.mlm_patterns import MLMConfig
from src.data.clm_patterns import CLMDataset, get_clm_dataloader
from src.models.bert_models import create_bert_model, create_clm_model
from src.training.trainer import BERTTrainer
from src.training.optimizer import get_optimizer
from src.training.scheduler import get_scheduler
from src.utils.device import get_device
from src.utils import set_seed


@dataclass
class TrainingResult:
    """Simple container for training results"""
    attention_type: str
    objective: str
    train_losses: List[float]
    val_losses: List[float]
    final_val_loss: float
    training_time: float


def parse_args():
    """Parse essential command line arguments"""
    parser = argparse.ArgumentParser(
        description="Simple BERT Training Comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--config", "-c", 
        type=str, 
        default="config.env",
        help="Path to configuration file"
    )
    
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    
    return parser.parse_args()


def apply_env_overrides(config):
    """Apply environment variable overrides to config"""
    
    # Training objectives and attention mechanisms
    objectives = os.getenv('OBJECTIVES', 'both')
    if objectives == 'both':
        config.training_objectives = ["mlm", "clm"]
    else:
        config.training_objectives = objectives.split(',')
    
    attention_list = os.getenv('ATTENTION_ALGORITHMS', 'standard,rope')
    config.attention_algorithms = attention_list.split(',')
    
    # Basic experiment settings
    config.experiment_name = os.getenv('EXPERIMENT_NAME', 'bert_comparison')
    config.output_dir = os.getenv('OUTPUT_DIR', './outputs')
    config.device = os.getenv('DEVICE', 'auto')
    config.debug = os.getenv('DEBUG', 'false').lower() == 'true'
    
    return config


def load_tokenizer(config):
    """Load appropriate tokenizer"""
    tokenizer_paths = [
        Path("./local_tokenizer"),
        Path("../local_tokenizer")
    ]
    
    for path in tokenizer_paths:
        if path.exists():
            return BertTokenizer.from_pretrained(str(path))
    
    return BertTokenizer.from_pretrained("bert-base-uncased")


def load_and_prepare_data(config):
    """Load and prepare training data"""
    
    # Load training data
    data_file = config.data.training_data_file if config.data else 'training_data.txt'
    if Path(data_file).exists():
        with open(data_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        # Sample data
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming the world of artificial intelligence.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning models require large amounts of data for training."
        ] * 50  # Repeat for more training data
    
    # Simple split
    split_idx = int(0.8 * len(texts))
    train_texts = texts[:split_idx]
    val_texts = texts[split_idx:]
    
    return train_texts, val_texts


def create_bert_config(config):
    """Create BERT configuration from config object"""
    if config.model:
        return BertConfig(**config.model.get_bert_config_dict())
    else:
        # Fallback to default values
        return BertConfig(
            vocab_size=30522,
            hidden_size=384,
            num_hidden_layers=6,
            num_attention_heads=6,
            intermediate_size=1536,
            max_position_embeddings=512,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1
        )


def train_single_model(attention_type: str, objective: str, train_texts: List[str], 
                      val_texts: List[str], tokenizer, config) -> TrainingResult:
    """Train a single model configuration"""
    
    print(f"\n=== Training {attention_type.upper()} + {objective.upper()} ===")
    
    start_time = time.time()
    device = get_device(config.device)
    
    # Create BERT config
    bert_config = create_bert_config(config)
    
    # Create model
    if objective == "mlm":
        model = create_bert_model(bert_config, attention_type)
    else:  # clm
        model = create_clm_model(bert_config, attention_type)
    
    model = model.to(device)
    
    # Create datasets
    if objective == "mlm":
        mlm_config = MLMConfig(mlm_probability=config.mlm_probability)
        train_dataset = BERTMLMDataset(train_texts, tokenizer, 
                                     max_length=config.max_seq_length,
                                     mlm_config=mlm_config)
        val_dataset = BERTMLMDataset(val_texts, tokenizer,
                                   max_length=config.max_seq_length,
                                   mlm_config=mlm_config)
        train_loader = get_dataloader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = get_dataloader(val_dataset, batch_size=config.batch_size, shuffle=False)
    else:  # clm
        train_dataset = CLMDataset(train_texts, tokenizer,
                                 max_length=config.max_seq_length)
        val_dataset = CLMDataset(val_texts, tokenizer,
                               max_length=config.max_seq_length)
        train_loader = get_clm_dataloader(train_dataset, batch_size=config.batch_size, shuffle=True)
        val_loader = get_clm_dataloader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(model, config)
    
    # Calculate training steps for scheduler
    batch_size = config.batch_size
    num_epochs = config.num_epochs
    gradient_accumulation_steps = config.gradient_accumulation_steps
    
    steps_per_epoch = len(train_dataset) // (batch_size * gradient_accumulation_steps)
    num_training_steps = steps_per_epoch * num_epochs
    
    scheduler = get_scheduler(optimizer, config, num_training_steps)
    
    # Create trainer
    trainer = BERTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
        device=device
    )
    
    # Train model
    results = trainer.train()
    
    training_time = time.time() - start_time
    
    print(f"Completed: Val Loss = {results.best_val_loss:.4f}")
    
    return TrainingResult(
        attention_type=attention_type,
        objective=objective,
        train_losses=results.train_losses,
        val_losses=results.val_losses,
        final_val_loss=results.best_val_loss,
        training_time=training_time
    )


def plot_comparison_results(results: List[TrainingResult], output_dir: Path):
    """Create simple comparison plots"""
    import matplotlib.pyplot as plt
    
    # Group results by objective
    mlm_results = [r for r in results if r.objective == "mlm"]
    clm_results = [r for r in results if r.objective == "clm"]
    
    plt.figure(figsize=(12, 8))
    
    # Plot MLM results
    plt.subplot(2, 2, 1)
    for result in mlm_results:
        plt.plot(result.train_losses, label=f"{result.attention_type}")
    plt.title("MLM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(2, 2, 2)
    for result in mlm_results:
        plt.plot(result.val_losses, label=f"{result.attention_type}")
    plt.title("MLM Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    # Plot CLM results
    plt.subplot(2, 2, 3)
    for result in clm_results:
        plt.plot(result.train_losses, label=f"{result.attention_type}")
    plt.title("CLM Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.subplot(2, 2, 4)
    for result in clm_results:
        plt.plot(result.val_losses, label=f"{result.attention_type}")
    plt.title("CLM Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    
    plt.tight_layout()
    plot_path = output_dir / "mlm_clm_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to {plot_path}")


def main():
    """Main training function"""
    args = parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        config = apply_env_overrides(config)
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return
    
    # Set seed for reproducibility
    set_seed(42)  # Default seed
    
    # Setup output directory
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load tokenizer and data
    tokenizer = load_tokenizer(config)
    train_texts, val_texts = load_and_prepare_data(config)
    
    print(f"Loaded {len(train_texts)} training and {len(val_texts)} validation examples")
    
    # Train all combinations
    results = []
    
    for objective in config.training_objectives:
        for attention_type in config.attention_algorithms:
            try:
                result = train_single_model(
                    attention_type, objective, train_texts, val_texts, tokenizer, config
                )
                results.append(result)
            except Exception as e:
                print(f"Training failed for {attention_type} + {objective}: {e}")
                continue
    
    # Generate plots
    if results:
        plot_comparison_results(results, output_dir)
        print(f"\nTraining completed. Results saved to {output_dir}")
    else:
        print("No successful training runs to plot.")


if __name__ == "__main__":
    main()