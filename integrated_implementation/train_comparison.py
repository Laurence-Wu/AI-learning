#!/usr/bin/env python3
"""
Enhanced BERT Attention Mechanism Comparison Training
====================================================

Advanced training script comparing MLM vs CLM objectives across different attention mechanisms:
- MLM (Masked Language Modeling): BERT-style bidirectional training
- CLM (Causal Language Modeling): GPT-style autoregressive training
- Four attention mechanisms: Standard, RoPE, ExpoSB, Absolute

Usage:
    python train_comparison.py --config config.yaml
    python train_comparison.py --objectives mlm clm --attention standard rope
    python train_comparison.py --objectives both --epochs 10
    python train_comparison.py --help
"""

import argparse
import sys
import logging
from pathlib import Path
import itertools

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader

from src.config import load_config, ExperimentConfig
from src.data import (
    BERTMLMDataset, get_dataloader, DataProcessor, MLMConfig, MLMStrategy,
    CLMDataset, get_clm_dataloader, CLMStrategy
)
from src.models import create_bert_model, create_clm_model
from src.training import BERTTrainer, get_optimizer, get_scheduler
from src.utils import setup_logging, get_device, set_seed, plot_attention_comparison
from src.attention import get_attention_class

from transformers import BertTokenizer, BertConfig

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="BERT MLM vs CLM Attention Mechanism Comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        "--config", "-c", 
        type=str, 
        default=None,
        help="Path to configuration file (YAML/JSON/ENV)"
    )
    
    # Training objectives
    parser.add_argument(
        "--objectives", 
        type=str, 
        nargs="+",
        choices=["mlm", "clm", "both"],
        default=None,
        help="Training objectives to compare (mlm, clm, or both)"
    )
    
    # Quick overrides
    parser.add_argument(
        "--attention", 
        type=str, 
        nargs="+",
        default=None,
        help="Attention mechanisms to compare (e.g., standard rope exposb absolute)"
    )
    
    parser.add_argument(
        "--epochs", 
        type=int, 
        default=None,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size", 
        type=int, 
        default=None,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--learning-rate", "--lr",
        type=float, 
        default=None,
        help="Learning rate"
    )
    
    parser.add_argument(
        "--data", 
        type=str, 
        default=None,
        help="Path to training data file"
    )
    
    parser.add_argument(
        "--output-dir", "--output",
        type=str, 
        default=None,
        help="Output directory for results"
    )
    
    # MLM/CLM specific options
    parser.add_argument(
        "--mlm-strategy",
        type=str,
        choices=["standard", "dynamic", "span", "whole_word"],
        default=None,
        help="MLM masking strategy"
    )
    
    parser.add_argument(
        "--clm-strategy",
        type=str,
        choices=["standard", "packed", "sliding"],
        default=None,
        help="CLM processing strategy"
    )
    
    # Device and precision
    parser.add_argument(
        "--device", 
        type=str, 
        choices=["auto", "cpu", "cuda", "mps"],
        default=None,
        help="Device to use for training"
    )
    
    parser.add_argument(
        "--mixed-precision", 
        action="store_true",
        help="Enable mixed precision training"
    )
    
    # Reproducibility
    parser.add_argument(
        "--seed", 
        type=int, 
        default=None,
        help="Random seed for reproducibility"
    )
    
    # Debugging
    parser.add_argument(
        "--debug", 
        action="store_true",
        help="Enable debug mode"
    )
    
    parser.add_argument(
        "--dry-run", 
        action="store_true",
        help="Dry run without actual training"
    )
    
    return parser.parse_args()


def override_config(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """Override configuration with command line arguments"""
    if args.objectives is not None:
        if "both" in args.objectives:
            config.training_objectives = ["mlm", "clm"]
        else:
            config.training_objectives = args.objectives
    
    if args.attention is not None:
        config.attention_algorithms = args.attention
    
    if args.epochs is not None:
        config.training.num_epochs = args.epochs
    
    if args.batch_size is not None:
        config.training.batch_size = args.batch_size
    
    if args.learning_rate is not None:
        config.training.learning_rate = args.learning_rate
    
    if args.data is not None:
        config.data.training_data_file = args.data
    
    if args.output_dir is not None:
        config.output_dir = args.output_dir
    
    if args.mlm_strategy is not None:
        config.data.mlm_strategy = args.mlm_strategy
    
    if args.device is not None:
        config.device = args.device
    
    if hasattr(args, 'mixed_precision') and args.mixed_precision is not None:
        config.mixed_precision = args.mixed_precision
    
    if args.seed is not None:
        config.seed = args.seed
    
    if args.debug:
        config.log_level = "DEBUG"
        config.training.logging_steps = 1
    
    return config


def load_tokenizer(config: ExperimentConfig):
    """Load appropriate tokenizer"""
    tokenizer_paths = [
        Path(config.output_dir).parent / "local_tokenizer",
        Path("./local_tokenizer"),
        Path("../local_tokenizer")
    ]
    
    for path in tokenizer_paths:
        if path.exists():
            logger.info(f"Loading tokenizer from {path}")
            return BertTokenizer.from_pretrained(str(path))
    
    logger.info("Loading tokenizer from Hugging Face")
    return BertTokenizer.from_pretrained("bert-base-uncased")


def load_data_for_objective(config: ExperimentConfig, tokenizer, objective: str):
    """Load and prepare data for specific training objective"""
    logger.info(f"Loading data for {objective.upper()} training...")
    
    # Data processing
    processor = DataProcessor(config.data.get_preprocessing_config())
    
    # Load texts
    if hasattr(config.data, 'training_data_file') and config.data.training_data_file:
        from src.data.preprocessing import load_training_data
        texts = load_training_data(
            config.data.training_data_file,
            config.data.get_preprocessing_config(),
            max_examples=getattr(config.data, 'max_examples', None)
        )
    else:
        # Default sample data
        logger.warning("No training data file specified, using sample data")
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming the world of technology.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning models require large amounts of training data.",
            "Transformer architectures have revolutionized NLP tasks.",
            "BERT uses bidirectional attention to understand context.",
            "Attention mechanisms allow models to focus on relevant information.",
            "Pre-training on large corpora improves downstream task performance.",
            "Causal language modeling predicts the next token in a sequence.",
            "Masked language modeling learns bidirectional representations."
        ] * 50
    
    if not texts:
        raise ValueError("No training data available")
    
    logger.info(f"Loaded {len(texts)} training examples for {objective}")
    
    # Split data
    from src.data.preprocessing import split_dataset
    train_texts, val_texts, _ = split_dataset(
        texts, 
        train_ratio=0.8, 
        val_ratio=0.2, 
        test_ratio=0.0,
        shuffle=True,
        random_seed=config.seed
    )
    
    if objective == "mlm":
        # Create MLM datasets
        mlm_config = MLMConfig(
            strategy=MLMStrategy(config.data.mlm_strategy),
            mlm_probability=config.data.mlm_probability
        )
        
        train_dataset = BERTMLMDataset(
            texts=train_texts,
            tokenizer=tokenizer,
            max_length=config.data.max_seq_length,
            mlm_config=mlm_config
        )
        
        val_dataset = BERTMLMDataset(
            texts=val_texts,
            tokenizer=tokenizer,
            max_length=config.data.max_seq_length,
            mlm_config=mlm_config
        ) if val_texts else None
        
        # Create dataloaders
        train_loader = get_dataloader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
            mlm_config=mlm_config
        )
        
        val_loader = get_dataloader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            mlm_config=mlm_config
        ) if val_dataset else None
    
    elif objective == "clm":
        # Create CLM datasets
        from src.data.clm_patterns import CLMConfig, CLMStrategy
        clm_config = CLMConfig(
            strategy=CLMStrategy(getattr(config.data, 'clm_strategy', 'standard'))
        )
        
        train_dataset = CLMDataset(
            texts=train_texts,
            tokenizer=tokenizer,
            max_length=config.data.max_seq_length,
            clm_config=clm_config
        )
        
        val_dataset = CLMDataset(
            texts=val_texts,
            tokenizer=tokenizer,
            max_length=config.data.max_seq_length,
            clm_config=clm_config
        ) if val_texts else None
        
        # Create dataloaders
        train_loader = get_clm_dataloader(
            train_dataset,
            batch_size=config.training.batch_size,
            shuffle=True,
            num_workers=config.training.num_workers,
            clm_config=clm_config
        )
        
        val_loader = get_clm_dataloader(
            val_dataset,
            batch_size=config.training.batch_size,
            shuffle=False,
            num_workers=config.training.num_workers,
            clm_config=clm_config
        ) if val_dataset else None
    
    else:
        raise ValueError(f"Unknown objective: {objective}")
    
    logger.info(f"Created {objective.upper()} dataloaders: {len(train_loader)} train batches, "
               f"{len(val_loader) if val_loader else 0} val batches")
    
    return train_loader, val_loader


def train_model_variant(
    config: ExperimentConfig,
    attention_type: str,
    objective: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    tokenizer,
    device: torch.device
) -> dict:
    """Train a single model variant (attention + objective combination)"""
    variant_name = f"{attention_type}_{objective}"
    logger.info(f"Training {variant_name}: {attention_type} attention with {objective.upper()} objective")
    
    # Create model configuration for this attention type
    model_config = config.get_model_config_for_attention(attention_type)
    
    # Create BERT configuration
    bert_config = BertConfig(
        vocab_size=model_config.vocab_size,
        hidden_size=model_config.hidden_size,
        num_hidden_layers=model_config.num_hidden_layers,
        num_attention_heads=model_config.num_attention_heads,
        intermediate_size=model_config.intermediate_size,
        max_position_embeddings=model_config.max_position_embeddings,
        attention_probs_dropout_prob=model_config.attention_dropout,
        hidden_dropout_prob=model_config.hidden_dropout
    )
    
    # Create model based on objective
    if objective == "mlm":
        model = create_bert_model(bert_config, attention_type)
    elif objective == "clm":
        model = create_clm_model(bert_config, attention_type)
    else:
        raise ValueError(f"Unknown objective: {objective}")
    
    model = model.to(device)
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(model, config.training.get_optimizer_config())
    scheduler = get_scheduler(optimizer, config.training.get_scheduler_config(), len(train_loader))
    
    # Create trainer
    experiment_dir = config.get_experiment_dir() / variant_name
    trainer = BERTTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config.training,
        device=device,
        experiment_dir=experiment_dir
    )
    
    # Train model
    results = trainer.train()
    
    # Save model
    model_path = experiment_dir / "final_model.pt"
    torch.save(model.state_dict(), model_path)
    logger.info(f"Saved {variant_name} model to {model_path}")
    
    return results


def main():
    """Main training function"""
    global args
    
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    try:
        config = load_config(args.config)
        config = override_config(config, args)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        sys.exit(1)
    
    # Setup experiment
    config.setup_logging()
    config.setup_reproducibility()
    
    if args.dry_run:
        logger.info("DRY RUN MODE - Configuration loaded successfully")
        config.print_config()
        return
    
    # Print configuration
    config.print_config()
    
    # Setup device
    device = get_device(config.device)
    logger.info(f"Using device: {device}")
    
    try:
        # Load tokenizer
        tokenizer = load_tokenizer(config)
        
        # Train models for each combination of attention + objective
        all_results = {}
        
        for objective in config.training_objectives:
            logger.info(f"\n{'='*70}")
            logger.info(f"PREPARING DATA FOR {objective.upper()} TRAINING")
            logger.info(f"{'='*70}")
            
            # Load data for this objective
            train_loader, val_loader = load_data_for_objective(config, tokenizer, objective)
            
            for attention_type in config.attention_algorithms:
                variant_name = f"{attention_type}_{objective}"
                
                logger.info(f"\n{'='*60}")
                logger.info(f"Training: {attention_type.upper()} + {objective.upper()}")
                logger.info(f"{'='*60}")
                
                try:
                    results = train_model_variant(
                        config, attention_type, objective, 
                        train_loader, val_loader, tokenizer, device
                    )
                    all_results[variant_name] = results
                    
                    # Clear GPU memory between models
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                except Exception as e:
                    logger.error(f"Failed to train {variant_name}: {e}")
                    if args.debug:
                        raise
                    continue
        
        # Generate comparison plots
        if len(all_results) > 1:
            logger.info("Generating comprehensive comparison plots...")
            
            # Create separate plots for each objective
            for objective in config.training_objectives:
                objective_results = {
                    name: results for name, results in all_results.items() 
                    if name.endswith(f"_{objective}")
                }
                
                if objective_results:
                    plot_path = Path(config.output_dir) / f"{config.experiment_name}_{objective}_comparison.png"
                    plot_attention_comparison(objective_results, str(plot_path))
                    logger.info(f"{objective.upper()} comparison plots saved to {plot_path}")
            
            # Create combined plot
            plot_path = Path(config.output_dir) / f"{config.experiment_name}_full_comparison.png"
            plot_attention_comparison(all_results, str(plot_path))
            logger.info(f"Full comparison plots saved to {plot_path}")
        
        # Save results
        results_path = Path(config.output_dir) / f"{config.experiment_name}_results.json"
        import json
        with open(results_path, 'w') as f:
            # Convert results to JSON-serializable format
            json_results = {}
            for variant_name, results in all_results.items():
                attention_type, objective = variant_name.split('_', 1)
                if attention_type not in json_results:
                    json_results[attention_type] = {}
                
                json_results[attention_type][objective] = {
                    'final_train_loss': results.train_losses[-1] if results.train_losses else None,
                    'final_val_loss': results.val_losses[-1] if results.val_losses else None,
                    'best_val_loss': min(results.val_losses) if results.val_losses else None,
                    'total_steps': len(results.train_losses),
                    'training_time': getattr(results, 'training_time', None)
                }
            json.dump(json_results, f, indent=2)
        
        logger.info(f"Results saved to {results_path}")
        
        # Print final summary
        logger.info(f"\n{'='*80}")
        logger.info("TRAINING COMPLETE - FINAL RESULTS SUMMARY")
        logger.info(f"{'='*80}")
        
        for objective in config.training_objectives:
            logger.info(f"\n{objective.upper()} Results:")
            logger.info("-" * 50)
            
            for attention_type in config.attention_algorithms:
                variant_name = f"{attention_type}_{objective}"
                if variant_name in all_results:
                    results = all_results[variant_name]
                    logger.info(f"{attention_type.upper()}:")
                    if results.train_losses:
                        logger.info(f"  Final Train Loss: {results.train_losses[-1]:.4f}")
                    if results.val_losses:
                        logger.info(f"  Final Val Loss: {results.val_losses[-1]:.4f}")
                        logger.info(f"  Best Val Loss: {min(results.val_losses):.4f}")
        
        logger.info(f"\nAll outputs saved to: {config.output_dir}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if args.debug:
            raise
        sys.exit(1)


if __name__ == "__main__":
    main()