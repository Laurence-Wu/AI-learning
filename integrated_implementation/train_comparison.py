#!/usr/bin/env python3
"""
Simple BERT Attention Mechanism Comparison
==========================================

Basic training script for quick comparison of attention mechanisms.
Generates simple loss plots without statistical analysis.

Usage:
    python train_comparison_simple.py
    python train_comparison_simple.py --debug
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List
from dataclasses import dataclass

# Add src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

from transformers import BertTokenizer, BertConfig

from configs.experiment_config import load_config
from src.data.dataset import BERTMLMDataset, get_dataloader
from src.data.mlm_patterns import MLMConfig
from src.models.model_factory import create_bert_model
from src.training.trainer import BERTTrainer
from src.training.optimizer import get_optimizer
from src.training.scheduler import get_scheduler
from src.utils.device import get_device
from src.utils import set_seed


@dataclass
class SimpleResult:
    """Simple container for training results"""
    attention_type: str
    train_losses: List[float]
    val_losses: List[float]
    final_val_loss: float


def parse_args():
    """Parse essential command line arguments"""
    parser = argparse.ArgumentParser(
        description="Simple BERT Attention Comparison",
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
    """Apply environment variable overrides"""
    
    # Use only MLM objective for simplicity
    config.training_objectives = ["mlm"]
    
    # Get attention mechanisms from environment
    attention_list = os.getenv('ATTENTION_ALGORITHMS', 'standard,rope')
    config.attention_algorithms = attention_list.split(',')
    
    # Basic settings
    config.experiment_name = os.getenv('EXPERIMENT_NAME', 'attention_comparison')
    config.output_dir = os.getenv('OUTPUT_DIR', './outputs')
    config.device = os.getenv('DEVICE', 'auto')
    
    return config


def load_tokenizer():
    """Load tokenizer"""
    tokenizer_paths = [
        Path("./local_tokenizer"),
        Path("../local_tokenizer")
    ]
    
    for path in tokenizer_paths:
        if path.exists():
            return BertTokenizer.from_pretrained(str(path))
    
    return BertTokenizer.from_pretrained("bert-base-uncased")


def load_data():
    """Load training data"""
    data_file = os.getenv('TRAINING_DATA_FILE', 'training_data.txt')
    if Path(data_file).exists():
        with open(data_file, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
    else:
        # Sample data
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming artificial intelligence.",
            "Natural language processing enables computers to understand text.",
            "Deep learning models require large datasets for training.",
            "Attention mechanisms help models focus on relevant information.",
            "Transformers have revolutionized natural language processing."
        ] * 25  # Repeat for more data
    
    # Simple split
    split_idx = int(0.8 * len(texts))
    return texts[:split_idx], texts[split_idx:]


def create_bert_config():
    """Create a BERT configuration from environment variables"""
    return BertConfig(
        vocab_size=int(os.getenv('VOCAB_SIZE', '30522')),
        hidden_size=int(os.getenv('HIDDEN_SIZE', '384')),
        num_hidden_layers=int(os.getenv('NUM_HIDDEN_LAYERS', '6')),
        num_attention_heads=int(os.getenv('NUM_ATTENTION_HEADS', '6')),
        intermediate_size=int(os.getenv('INTERMEDIATE_SIZE', '1536')),
        max_position_embeddings=int(os.getenv('MAX_POSITION_EMBEDDINGS', '512')),
        hidden_dropout_prob=float(os.getenv('HIDDEN_DROPOUT', '0.1')),
        attention_probs_dropout_prob=float(os.getenv('ATTENTION_DROPOUT', '0.1'))
    )


def train_attention_model(attention_type: str, train_texts: List[str], 
                         val_texts: List[str], tokenizer, config) -> SimpleResult:
    """Train a single attention model"""
    
    print(f"\n=== Training {attention_type.upper()} Attention ===")
    
    device = get_device(config.device)
    
    # Create BERT config and model
    bert_config = create_bert_config()
    model = create_bert_model(bert_config, attention_type)
    model = model.to(device)
    
    # Create datasets
    max_length = int(os.getenv('MAX_SEQ_LENGTH', '256'))
    batch_size = int(os.getenv('BATCH_SIZE', '16'))
    mlm_config = MLMConfig(mlm_probability=float(os.getenv('MLM_PROBABILITY', '0.15')))
    
    train_dataset = BERTMLMDataset(train_texts, tokenizer, max_length, mlm_config=mlm_config)
    val_dataset = BERTMLMDataset(val_texts, tokenizer, max_length, mlm_config=mlm_config)
    
    train_loader = get_dataloader(train_dataset, batch_size, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size, shuffle=False)
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(model, config)
    scheduler = get_scheduler(optimizer, config)
    
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
    
    # Train
    results = trainer.train()
    
    print(f"Final validation loss: {results.best_val_loss:.4f}")
    
    return SimpleResult(
        attention_type=attention_type,
        train_losses=results.train_losses,
        val_losses=results.val_losses,
        final_val_loss=results.best_val_loss
    )


def plot_attention_comparison(results: List[SimpleResult], output_dir: Path):
    """Create attention mechanism comparison plot"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 5))
    
    # Training loss plot
    plt.subplot(1, 2, 1)
    for result in results:
        plt.plot(result.train_losses, label=result.attention_type)
    plt.title("Training Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Validation loss plot
    plt.subplot(1, 2, 2)
    for result in results:
        plt.plot(result.val_losses, label=result.attention_type)
    plt.title("Validation Loss Comparison")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / "attention_comparison.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison plot saved to {plot_path}")


def main():
    """Main function"""
    args = parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        config = apply_env_overrides(config)
    except Exception as e:
        print(f"Failed to load configuration: {e}")
        return
    
    # Set seed
    set_seed(int(os.getenv('SEED', '42')))
    
    # Setup
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tokenizer = load_tokenizer()
    train_texts, val_texts = load_data()
    
    print(f"Loaded {len(train_texts)} training and {len(val_texts)} validation examples")
    
    # Train all attention mechanisms
    results = []
    
    for attention_type in config.attention_algorithms:
        try:
            result = train_attention_model(
                attention_type, train_texts, val_texts, tokenizer, config
            )
            results.append(result)
        except Exception as e:
            print(f"Training failed for {attention_type}: {e}")
            continue
    
    # Generate plot
    if results:
        plot_attention_comparison(results, output_dir)
        print(f"\nTraining completed. Results saved to {output_dir}")
        
        # Print summary
        print("\nFinal Results:")
        for result in results:
            print(f"{result.attention_type}: {result.final_val_loss:.4f}")
    else:
        print("No successful training runs.")


if __name__ == "__main__":
    main()