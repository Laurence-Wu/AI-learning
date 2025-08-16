#!/usr/bin/env python3
"""
Comprehensive BERT Training: MLM vs CLM Comparison Across Attention Mechanisms
==============================================================================

Rigorous training script that compares:
- Training Objectives: MLM (Masked Language Modeling) vs CLM (Causal Language Modeling)
- Attention Mechanisms: Standard, RoPE, ExpoSB, Absolute
- Statistical controls for unbiased comparison with multiple runs and cross-validation

Features:
- Multiple random seeds for statistical significance
- Stratified data splitting
- Cross-validation support
- Comprehensive metrics tracking
- Statistical significance testing
- Memory-efficient training
- Robust error handling

Usage:
    python train.py --config configs/default.yaml
    python train.py --objectives mlm clm --attention standard rope --num-runs 5
    python train.py --objectives both --cross-validation --folds 3
    python train.py --help
"""

import argparse
import sys
import logging
import json
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
import numpy as np
from scipy import stats

# Add project root and src to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import torch.nn.functional as F

from src.config import load_config, ExperimentConfig
from src.data import (
    BERTMLMDataset, get_dataloader, DataProcessor, MLMConfig, MLMStrategy,
    CLMDataset, get_clm_dataloader, CLMStrategy
)
from src.models import create_bert_model, create_clm_model
from src.training import BERTTrainer, get_optimizer, get_scheduler
from src.utils import setup_logging, get_device, set_seed
from src.attention import get_attention_class

from transformers import BertTokenizer, BertConfig

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

logger = logging.getLogger(__name__)


@dataclass
class ExperimentResults:
    """Container for experiment results with statistical measures"""
    attention_type: str
    objective: str
    run_id: int
    fold_id: Optional[int]
    seed: int
    
    # Training metrics
    train_losses: List[float]
    val_losses: List[float]
    train_accuracies: List[float] 
    val_accuracies: List[float]
    
    # Final metrics
    final_train_loss: float
    final_val_loss: float
    best_val_loss: float
    final_train_accuracy: float
    final_val_accuracy: float
    best_val_accuracy: float
    
    # Meta information
    training_time: float
    total_steps: int
    convergence_step: Optional[int]
    model_parameters: int
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization"""
        return asdict(self)


class StatisticalAnalyzer:
    """Statistical analysis for experiment results"""
    
    @staticmethod
    def compute_significance(group1: List[float], group2: List[float], 
                           alpha: float = 0.05) -> Dict[str, Any]:
        """Compute statistical significance between two groups"""
        if len(group1) < 2 or len(group2) < 2:
            return {"significant": False, "reason": "insufficient_data"}
        
        # Shapiro-Wilk test for normality
        _, p1 = stats.shapiro(group1) if len(group1) >= 3 else (0, 0)
        _, p2 = stats.shapiro(group2) if len(group2) >= 3 else (0, 0)
        
        normal_dist = p1 > alpha and p2 > alpha
        
        if normal_dist:
            # t-test for normal distributions
            statistic, p_value = stats.ttest_ind(group1, group2)
            test_type = "t-test"
        else:
            # Mann-Whitney U test for non-normal distributions
            statistic, p_value = stats.mannwhitneyu(group1, group2, alternative='two-sided')
            test_type = "mann-whitney"
        
        return {
            "significant": p_value < alpha,
            "p_value": p_value,
            "statistic": statistic,
            "test_type": test_type,
            "alpha": alpha,
            "effect_size": StatisticalAnalyzer._compute_effect_size(group1, group2)
        }
    
    @staticmethod
    def _compute_effect_size(group1: List[float], group2: List[float]) -> float:
        """Compute Cohen's d effect size"""
        mean1, mean2 = np.mean(group1), np.mean(group2)
        std1, std2 = np.std(group1, ddof=1), np.std(group2, ddof=1)
        
        if std1 == 0 and std2 == 0:
            return 0.0
        
        pooled_std = np.sqrt(((len(group1) - 1) * std1**2 + (len(group2) - 1) * std2**2) / 
                            (len(group1) + len(group2) - 2))
        
        return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
    
    @staticmethod
    def compute_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
        """Compute confidence interval for mean"""
        if len(data) < 2:
            return (np.mean(data), np.mean(data))
        
        mean = np.mean(data)
        sem = stats.sem(data)
        h = sem * stats.t.ppf((1 + confidence) / 2., len(data) - 1)
        
        return (mean - h, mean + h)


def parse_args():
    """Parse command line arguments with comprehensive options"""
    parser = argparse.ArgumentParser(
        description="Comprehensive BERT MLM vs CLM Training Comparison",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        "--config", "-c", 
        type=str, 
        default="configs/default.yaml",
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
    
    # Attention mechanisms
    parser.add_argument(
        "--attention", 
        type=str, 
        nargs="+",
        default=None,
        help="Attention mechanisms to compare (e.g., standard rope exposb absolute)"
    )
    
    # Statistical rigor
    parser.add_argument(
        "--num-runs", 
        type=int, 
        default=3,
        help="Number of independent runs for statistical significance"
    )
    
    parser.add_argument(
        "--cross-validation", 
        action="store_true",
        help="Enable cross-validation training"
    )
    
    parser.add_argument(
        "--folds", 
        type=int, 
        default=3,
        help="Number of cross-validation folds"
    )
    
    parser.add_argument(
        "--seeds", 
        type=int, 
        nargs="+",
        default=None,
        help="Custom random seeds for reproducibility"
    )
    
    # Quick overrides
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
    
    # Experiment tracking
    parser.add_argument(
        "--wandb", 
        action="store_true",
        help="Enable Weights & Biases logging"
    )
    
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for the experiment"
    )
    
    # Debugging and testing
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
    
    parser.add_argument(
        "--quick-test", 
        action="store_true",
        help="Quick test with minimal epochs and data"
    )
    
    # Statistical analysis
    parser.add_argument(
        "--significance-level", 
        type=float, 
        default=0.05,
        help="Statistical significance level (alpha)"
    )
    
    parser.add_argument(
        "--confidence-level", 
        type=float, 
        default=0.95,
        help="Confidence level for intervals"
    )
    
    return parser.parse_args()


def override_config(config: ExperimentConfig, args: argparse.Namespace) -> ExperimentConfig:
    """Override configuration with command line arguments"""
    if args.objectives is not None:
        if "both" in args.objectives:
            config.training_objectives = ["mlm", "clm"]
        else:
            config.training_objectives = args.objectives
    else:
        # Default to both if not specified
        config.training_objectives = ["mlm", "clm"]
    
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
    
    if args.wandb:
        config.use_wandb = True
    
    if args.experiment_name is not None:
        config.experiment_name = args.experiment_name
    
    if args.debug:
        config.log_level = "DEBUG"
        config.training.logging_steps = 1
    
    if args.quick_test:
        config.training.num_epochs = 2
        config.data.max_examples = 1000
        config.training.logging_steps = 10
        config.training.eval_steps = 50
    
    return config


def generate_experiment_seeds(num_runs: int, base_seed: int = 42, 
                            custom_seeds: Optional[List[int]] = None) -> List[int]:
    """Generate reproducible seeds for multiple runs"""
    if custom_seeds:
        if len(custom_seeds) >= num_runs:
            return custom_seeds[:num_runs]
        else:
            logger.warning(f"Not enough custom seeds ({len(custom_seeds)}), "
                         f"generating additional seeds")
            additional_needed = num_runs - len(custom_seeds)
            np.random.seed(base_seed)
            additional_seeds = np.random.randint(0, 2**31, additional_needed).tolist()
            return custom_seeds + additional_seeds
    else:
        np.random.seed(base_seed)
        return np.random.randint(0, 2**31, num_runs).tolist()


def load_tokenizer(config: ExperimentConfig) -> BertTokenizer:
    """Load appropriate tokenizer with fallback options"""
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


def create_stratified_splits(texts: List[str], num_folds: int = 3, 
                           val_ratio: float = 0.1, test_ratio: float = 0.1,
                           random_seed: int = 42) -> List[Tuple[List[str], List[str], List[str]]]:
    """Create stratified splits for cross-validation"""
    np.random.seed(random_seed)
    
    # Simple stratification based on text length
    text_lengths = [len(text) for text in texts]
    length_percentiles = np.percentile(text_lengths, [33, 66])
    
    short_texts = [t for t in texts if len(t) <= length_percentiles[0]]
    medium_texts = [t for t in texts if length_percentiles[0] < len(t) <= length_percentiles[1]]
    long_texts = [t for t in texts if len(t) > length_percentiles[1]]
    
    def split_group(group_texts: List[str]) -> List[Tuple[List[str], List[str], List[str]]]:
        """Split a group into train/val/test for each fold"""
        n = len(group_texts)
        shuffled = group_texts.copy()
        np.random.shuffle(shuffled)
        
        fold_splits = []
        fold_size = n // num_folds
        
        for fold in range(num_folds):
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size if fold < num_folds - 1 else n
            
            test_texts = shuffled[start_idx:end_idx]
            remaining = shuffled[:start_idx] + shuffled[end_idx:]
            
            val_size = int(len(remaining) * val_ratio / (1 - test_ratio))
            val_texts = remaining[:val_size]
            train_texts = remaining[val_size:]
            
            fold_splits.append((train_texts, val_texts, test_texts))
        
        return fold_splits
    
    short_splits = split_group(short_texts)
    medium_splits = split_group(medium_texts)
    long_splits = split_group(long_texts)
    
    # Combine splits from all groups
    combined_splits = []
    for fold in range(num_folds):
        train_combined = short_splits[fold][0] + medium_splits[fold][0] + long_splits[fold][0]
        val_combined = short_splits[fold][1] + medium_splits[fold][1] + long_splits[fold][1]
        test_combined = short_splits[fold][2] + medium_splits[fold][2] + long_splits[fold][2]
        
        # Shuffle combined splits
        np.random.shuffle(train_combined)
        np.random.shuffle(val_combined)
        np.random.shuffle(test_combined)
        
        combined_splits.append((train_combined, val_combined, test_combined))
    
    return combined_splits


def load_and_split_data(config: ExperimentConfig, cross_validation: bool = False,
                       num_folds: int = 3, random_seed: int = 42) -> List[Tuple[List[str], List[str], List[str]]]:
    """Load data and create train/val/test splits"""
    logger.info("Loading and processing training data...")
    
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
        # Enhanced sample data for testing
        logger.warning("No training data file specified, using enhanced sample data")
        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming the world of technology and artificial intelligence.",
            "Natural language processing enables computers to understand human language patterns.",
            "Deep learning models require large amounts of training data to achieve good performance.",
            "Transformer architectures have revolutionized NLP tasks across multiple domains.",
            "BERT uses bidirectional attention to understand context from both directions.",
            "Attention mechanisms allow models to focus on relevant information dynamically.",
            "Pre-training on large corpora improves downstream task performance significantly.",
            "Causal language modeling predicts the next token in a sequence autoregressively.",
            "Masked language modeling learns bidirectional representations for better understanding.",
            "The attention mechanism computes weighted representations of input sequences.",
            "Cross-attention allows models to attend to different sequences simultaneously.",
            "Self-attention mechanisms enable long-range dependency modeling in transformers.",
            "Position embeddings help models understand the order of tokens in sequences.",
            "Layer normalization stabilizes training in deep neural network architectures.",
        ] * 100  # Multiply for sufficient data
    
    if not texts:
        raise ValueError("No training data available")
    
    logger.info(f"Loaded {len(texts)} training examples")
    
    if cross_validation:
        splits = create_stratified_splits(texts, num_folds, random_seed=random_seed)
        logger.info(f"Created {num_folds} stratified cross-validation splits")
    else:
        # Single train/val/test split
        from src.data.preprocessing import split_dataset
        train_texts, val_texts, test_texts = split_dataset(
            texts, 
            train_ratio=0.8, 
            val_ratio=0.1, 
            test_ratio=0.1,
            shuffle=True,
            random_seed=random_seed
        )
        splits = [(train_texts, val_texts, test_texts)]
        logger.info("Created single train/val/test split")
    
    return splits


def create_datasets_and_loaders(texts_split: Tuple[List[str], List[str], List[str]],
                               tokenizer: BertTokenizer, config: ExperimentConfig,
                               objective: str, args: argparse.Namespace) -> Tuple[DataLoader, DataLoader]:
    """Create datasets and dataloaders for specific objective"""
    train_texts, val_texts, _ = texts_split
    
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
        from src.data.clm_patterns import CLMConfig
        clm_config = CLMConfig(
            strategy=CLMStrategy(getattr(args, 'clm_strategy', 'standard')),
            max_length=config.data.max_seq_length
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
    
    return train_loader, val_loader


def train_single_model(config: ExperimentConfig, attention_type: str, objective: str,
                      train_loader: DataLoader, val_loader: DataLoader,
                      tokenizer: BertTokenizer, device: torch.device,
                      run_id: int, fold_id: Optional[int], seed: int) -> ExperimentResults:
    """Train a single model variant with comprehensive tracking"""
    variant_name = f"{attention_type}_{objective}"
    if fold_id is not None:
        variant_name += f"_fold{fold_id}"
    variant_name += f"_run{run_id}"
    
    logger.info(f"Training {variant_name}: {attention_type} + {objective.upper()} (seed={seed})")
    
    # Set seed for this specific training run
    set_seed(seed)
    
    # Create model configuration
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
    start_time = time.time()
    
    if objective == "mlm":
        model = create_bert_model(bert_config, attention_type)
    elif objective == "clm":
        model = create_clm_model(bert_config, attention_type)
    else:
        raise ValueError(f"Unknown objective: {objective}")
    
    model = model.to(device)
    
    # Count parameters
    model_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model has {model_parameters:,} trainable parameters")
    
    # Create optimizer and scheduler
    optimizer = get_optimizer(model, config.training.get_optimizer_config())
    scheduler = get_scheduler(optimizer, config.training.get_scheduler_config(), len(train_loader))
    
    # Create experiment directory
    experiment_dir = config.get_experiment_dir() / variant_name
    experiment_dir.mkdir(parents=True, exist_ok=True)
    
    # Create trainer
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
    try:
        training_results = trainer.train()
        training_time = time.time() - start_time
        
        # Extract metrics
        train_losses = training_results.train_losses
        val_losses = training_results.val_losses
        train_accuracies = getattr(training_results, 'train_accuracies', [])
        val_accuracies = getattr(training_results, 'val_accuracies', [])
        
        # Compute final metrics
        final_train_loss = train_losses[-1] if train_losses else float('inf')
        final_val_loss = val_losses[-1] if val_losses else float('inf')
        best_val_loss = min(val_losses) if val_losses else float('inf')
        
        final_train_accuracy = train_accuracies[-1] if train_accuracies else 0.0
        final_val_accuracy = val_accuracies[-1] if val_accuracies else 0.0
        best_val_accuracy = max(val_accuracies) if val_accuracies else 0.0
        
        # Find convergence step (when validation loss stops improving significantly)
        convergence_step = None
        if len(val_losses) > 5:
            min_loss = min(val_losses)
            for i, loss in enumerate(val_losses):
                if loss <= min_loss * 1.01:  # Within 1% of minimum
                    convergence_step = i
                    break
        
        # Save model
        model_path = experiment_dir / "final_model.pt"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'config': bert_config,
            'attention_type': attention_type,
            'objective': objective,
            'seed': seed,
            'final_metrics': {
                'train_loss': final_train_loss,
                'val_loss': final_val_loss,
                'train_accuracy': final_train_accuracy,
                'val_accuracy': final_val_accuracy
            }
        }, model_path)
        
        logger.info(f"Completed {variant_name}: Val Loss = {final_val_loss:.4f}, "
                   f"Val Acc = {final_val_accuracy:.4f}")
        
        return ExperimentResults(
            attention_type=attention_type,
            objective=objective,
            run_id=run_id,
            fold_id=fold_id,
            seed=seed,
            train_losses=train_losses,
            val_losses=val_losses,
            train_accuracies=train_accuracies,
            val_accuracies=val_accuracies,
            final_train_loss=final_train_loss,
            final_val_loss=final_val_loss,
            best_val_loss=best_val_loss,
            final_train_accuracy=final_train_accuracy,
            final_val_accuracy=final_val_accuracy,
            best_val_accuracy=best_val_accuracy,
            training_time=training_time,
            total_steps=len(train_losses),
            convergence_step=convergence_step,
            model_parameters=model_parameters
        )
        
    except Exception as e:
        logger.error(f"Training failed for {variant_name}: {e}")
        # Return failed result
        return ExperimentResults(
            attention_type=attention_type,
            objective=objective,
            run_id=run_id,
            fold_id=fold_id,
            seed=seed,
            train_losses=[],
            val_losses=[],
            train_accuracies=[],
            val_accuracies=[],
            final_train_loss=float('inf'),
            final_val_loss=float('inf'),
            best_val_loss=float('inf'),
            final_train_accuracy=0.0,
            final_val_accuracy=0.0,
            best_val_accuracy=0.0,
            training_time=0.0,
            total_steps=0,
            convergence_step=None,
            model_parameters=0
        )


def aggregate_results(results: List[ExperimentResults]) -> Dict[str, Any]:
    """Aggregate results by attention type and objective"""
    aggregated = defaultdict(lambda: defaultdict(list))
    
    for result in results:
        key = f"{result.attention_type}_{result.objective}"
        aggregated[key]['final_val_loss'].append(result.final_val_loss)
        aggregated[key]['best_val_loss'].append(result.best_val_loss)
        aggregated[key]['final_val_accuracy'].append(result.final_val_accuracy)
        aggregated[key]['best_val_accuracy'].append(result.best_val_accuracy)
        aggregated[key]['training_time'].append(result.training_time)
        aggregated[key]['convergence_step'].append(result.convergence_step)
        aggregated[key]['results'].append(result)
    
    # Compute statistics
    summary = {}
    for variant, metrics in aggregated.items():
        summary[variant] = {}
        for metric_name, values in metrics.items():
            if metric_name == 'results':
                continue
            
            clean_values = [v for v in values if v is not None and not np.isinf(v)]
            if clean_values:
                summary[variant][metric_name] = {
                    'mean': np.mean(clean_values),
                    'std': np.std(clean_values),
                    'median': np.median(clean_values),
                    'min': np.min(clean_values),
                    'max': np.max(clean_values),
                    'count': len(clean_values)
                }
            else:
                summary[variant][metric_name] = {
                    'mean': float('nan'), 'std': float('nan'), 
                    'median': float('nan'), 'min': float('nan'), 
                    'max': float('nan'), 'count': 0
                }
    
    return summary, dict(aggregated)


def create_comprehensive_visualizations(aggregated_results: Dict[str, Any], 
                                      config: ExperimentConfig,
                                      significance_results: Dict[str, Any]) -> None:
    """Create comprehensive visualizations for MLM vs CLM comparison"""
    
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        from src.utils.mlm_clm_visualization import (
            plot_mlm_clm_comparison, plot_attention_heatmap, 
            plot_training_curves, plot_statistical_comparison
        )
        
        output_dir = Path(config.output_dir)
        
        # Extract data for plotting
        mlm_results = {k: v for k, v in aggregated_results.items() if '_mlm' in k}
        clm_results = {k: v for k, v in aggregated_results.items() if '_clm' in k}
        
        # 1. Main comparison plot
        plot_path = output_dir / f"{config.experiment_name}_mlm_vs_clm_comparison.png"
        plot_mlm_clm_comparison(mlm_results, clm_results, str(plot_path))
        logger.info(f"MLM vs CLM comparison plot saved to {plot_path}")
        
        # 2. Attention mechanism heatmap
        plot_path = output_dir / f"{config.experiment_name}_attention_heatmap.png"
        plot_attention_heatmap(aggregated_results, str(plot_path))
        logger.info(f"Attention heatmap saved to {plot_path}")
        
        # 3. Training curves for each variant
        plot_path = output_dir / f"{config.experiment_name}_training_curves.png"
        plot_training_curves(aggregated_results, str(plot_path))
        logger.info(f"Training curves saved to {plot_path}")
        
        # 4. Statistical comparison plot
        plot_path = output_dir / f"{config.experiment_name}_statistical_analysis.png"
        plot_statistical_comparison(aggregated_results, significance_results, str(plot_path))
        logger.info(f"Statistical analysis plot saved to {plot_path}")
        
    except ImportError as e:
        logger.warning(f"Visualization libraries not available: {e}")
        logger.info("Install matplotlib and seaborn for comprehensive visualizations")


def perform_statistical_analysis(aggregated_results: Dict[str, Any], 
                                config: ExperimentConfig, args: argparse.Namespace) -> Dict[str, Any]:
    """Perform comprehensive statistical analysis"""
    analyzer = StatisticalAnalyzer()
    significance_results = {}
    
    # Compare MLM vs CLM for each attention mechanism
    for attention_type in config.attention_algorithms:
        mlm_key = f"{attention_type}_mlm"
        clm_key = f"{attention_type}_clm"
        
        if mlm_key in aggregated_results and clm_key in aggregated_results:
            mlm_losses = aggregated_results[mlm_key]['final_val_loss']
            clm_losses = aggregated_results[clm_key]['final_val_loss']
            
            # Clean data
            mlm_clean = [x for x in mlm_losses if not np.isinf(x)]
            clm_clean = [x for x in clm_losses if not np.isinf(x)]
            
            if len(mlm_clean) > 1 and len(clm_clean) > 1:
                comparison_key = f"{attention_type}_mlm_vs_clm"
                significance_results[comparison_key] = analyzer.compute_significance(
                    mlm_clean, clm_clean, args.significance_level
                )
                
                # Add confidence intervals
                mlm_ci = analyzer.compute_confidence_interval(mlm_clean, args.confidence_level)
                clm_ci = analyzer.compute_confidence_interval(clm_clean, args.confidence_level)
                
                significance_results[comparison_key]['mlm_ci'] = mlm_ci
                significance_results[comparison_key]['clm_ci'] = clm_ci
    
    # Compare attention mechanisms within each objective
    for objective in config.training_objectives:
        attention_results = {}
        for attention_type in config.attention_algorithms:
            key = f"{attention_type}_{objective}"
            if key in aggregated_results:
                losses = aggregated_results[key]['final_val_loss']
                clean_losses = [x for x in losses if not np.isinf(x)]
                if clean_losses:
                    attention_results[attention_type] = clean_losses
        
        # Pairwise comparisons between attention mechanisms
        attention_types = list(attention_results.keys())
        for i in range(len(attention_types)):
            for j in range(i + 1, len(attention_types)):
                att1, att2 = attention_types[i], attention_types[j]
                comparison_key = f"{objective}_{att1}_vs_{att2}"
                
                significance_results[comparison_key] = analyzer.compute_significance(
                    attention_results[att1], attention_results[att2], args.significance_level
                )
    
    return significance_results


def save_comprehensive_results(all_results: List[ExperimentResults], 
                             summary_results: Dict[str, Any],
                             significance_results: Dict[str, Any],
                             config: ExperimentConfig, args: argparse.Namespace) -> None:
    """Save comprehensive results in multiple formats"""
    output_dir = Path(config.output_dir)
    
    # 1. Raw results (JSON)
    raw_results_path = output_dir / f"{config.experiment_name}_raw_results.json"
    with open(raw_results_path, 'w') as f:
        json.dump([result.to_dict() for result in all_results], f, indent=2)
    
    # 2. Summary statistics (JSON)
    summary_path = output_dir / f"{config.experiment_name}_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary_results, f, indent=2)
    
    # 3. Statistical analysis (JSON)
    stats_path = output_dir / f"{config.experiment_name}_statistical_analysis.json"
    with open(stats_path, 'w') as f:
        json.dump(significance_results, f, indent=2)
    
    # 4. Human-readable report (Markdown)
    report_path = output_dir / f"{config.experiment_name}_report.md"
    with open(report_path, 'w') as f:
        f.write(f"# {config.experiment_name} - Training Report\n\n")
        f.write(f"Generated on: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Experiment configuration
        f.write("## Experiment Configuration\n\n")
        f.write(f"- Objectives: {config.training_objectives}\n")
        f.write(f"- Attention Mechanisms: {config.attention_algorithms}\n")
        f.write(f"- Number of Runs: {args.num_runs}\n")
        f.write(f"- Cross-validation: {args.cross_validation}\n")
        if args.cross_validation:
            f.write(f"- Number of Folds: {args.folds}\n")
        f.write(f"- Epochs: {config.training.num_epochs}\n")
        f.write(f"- Batch Size: {config.training.batch_size}\n")
        f.write(f"- Learning Rate: {config.training.learning_rate}\n\n")
        
        # Results summary
        f.write("## Results Summary\n\n")
        f.write("### Validation Loss (Lower is Better)\n\n")
        f.write("| Variant | Mean ± Std | Median | Min | Max | Count |\n")
        f.write("|---------|------------|--------|-----|-----|-------|\n")
        
        for variant, metrics in summary_results.items():
            if 'final_val_loss' in metrics:
                loss_stats = metrics['final_val_loss']
                f.write(f"| {variant} | {loss_stats['mean']:.4f} ± {loss_stats['std']:.4f} | "
                       f"{loss_stats['median']:.4f} | {loss_stats['min']:.4f} | "
                       f"{loss_stats['max']:.4f} | {loss_stats['count']} |\n")
        
        f.write("\n### Validation Accuracy (Higher is Better)\n\n")
        f.write("| Variant | Mean ± Std | Median | Min | Max | Count |\n")
        f.write("|---------|------------|--------|-----|-----|-------|\n")
        
        for variant, metrics in summary_results.items():
            if 'final_val_accuracy' in metrics:
                acc_stats = metrics['final_val_accuracy']
                f.write(f"| {variant} | {acc_stats['mean']:.4f} ± {acc_stats['std']:.4f} | "
                       f"{acc_stats['median']:.4f} | {acc_stats['min']:.4f} | "
                       f"{acc_stats['max']:.4f} | {acc_stats['count']} |\n")
        
        # Statistical significance
        f.write("\n## Statistical Analysis\n\n")
        f.write("### Significance Tests (α = 0.05)\n\n")
        f.write("| Comparison | Significant | p-value | Effect Size | Test Type |\n")
        f.write("|------------|-------------|---------|-------------|----------|\n")
        
        for comparison, results in significance_results.items():
            if 'p_value' in results:
                sig = "✓" if results['significant'] else "✗"
                f.write(f"| {comparison} | {sig} | {results['p_value']:.4f} | "
                       f"{results['effect_size']:.4f} | {results['test_type']} |\n")
    
    logger.info(f"Comprehensive results saved:")
    logger.info(f"  Raw results: {raw_results_path}")
    logger.info(f"  Summary: {summary_path}")
    logger.info(f"  Statistics: {stats_path}")
    logger.info(f"  Report: {report_path}")


def main():
    """Main training function with comprehensive statistical analysis"""
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
    
    if args.dry_run:
        logger.info("DRY RUN MODE - Configuration loaded successfully")
        config.print_config()
        return
    
    # Print configuration
    config.print_config()
    
    # Setup device
    device = get_device(config.device)
    logger.info(f"Using device: {device}")
    
    # Generate seeds for multiple runs
    seeds = generate_experiment_seeds(args.num_runs, config.seed, args.seeds)
    logger.info(f"Running {args.num_runs} experiments with seeds: {seeds}")
    
    try:
        # Load tokenizer
        tokenizer = load_tokenizer(config)
        
        # Load and split data
        data_splits = load_and_split_data(
            config, args.cross_validation, args.folds, config.seed
        )
        
        # Run comprehensive training
        all_results = []
        total_experiments = len(config.training_objectives) * len(config.attention_algorithms) * len(data_splits) * args.num_runs
        experiment_count = 0
        
        for fold_id, texts_split in enumerate(data_splits):
            fold_name = f"fold_{fold_id}" if args.cross_validation else "single_split"
            logger.info(f"\n{'='*80}")
            logger.info(f"PROCESSING {fold_name.upper()}")
            logger.info(f"{'='*80}")
            
            for objective in config.training_objectives:
                logger.info(f"\n{'='*70}")
                logger.info(f"OBJECTIVE: {objective.upper()}")
                logger.info(f"{'='*70}")
                
                # Create datasets for this objective and split
                train_loader, val_loader = create_datasets_and_loaders(
                    texts_split, tokenizer, config, objective, args
                )
                
                for attention_type in config.attention_algorithms:
                    logger.info(f"\n{'='*60}")
                    logger.info(f"ATTENTION: {attention_type.upper()}")
                    logger.info(f"{'='*60}")
                    
                    for run_id, seed in enumerate(seeds):
                        experiment_count += 1
                        logger.info(f"\nExperiment {experiment_count}/{total_experiments}")
                        
                        try:
                            result = train_single_model(
                                config, attention_type, objective,
                                train_loader, val_loader, tokenizer, device,
                                run_id, fold_id if args.cross_validation else None, seed
                            )
                            all_results.append(result)
                            
                            # Clear GPU memory
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                                
                        except Exception as e:
                            logger.error(f"Failed experiment {experiment_count}: {e}")
                            if args.debug:
                                raise
                            continue
        
        if not all_results:
            logger.error("No successful training runs completed")
            return
        
        logger.info(f"\n{'='*80}")
        logger.info("ANALYSIS AND VISUALIZATION")
        logger.info(f"{'='*80}")
        
        # Aggregate results
        summary_results, aggregated_results = aggregate_results(all_results)
        
        # Perform statistical analysis
        significance_results = perform_statistical_analysis(
            aggregated_results, config, args
        )
        
        # Create visualizations
        create_comprehensive_visualizations(
            aggregated_results, config, significance_results
        )
        
        # Save comprehensive results
        save_comprehensive_results(
            all_results, summary_results, significance_results, config, args
        )
        
        # Print final summary
        logger.info(f"\n{'='*80}")
        logger.info("EXPERIMENT COMPLETE - FINAL SUMMARY")
        logger.info(f"{'='*80}")
        
        logger.info(f"Total experiments completed: {len(all_results)}")
        logger.info(f"Results saved to: {config.output_dir}")
        
        # Print best performers
        best_mlm = min([r for r in all_results if r.objective == "mlm"], 
                      key=lambda x: x.final_val_loss, default=None)
        best_clm = min([r for r in all_results if r.objective == "clm"], 
                      key=lambda x: x.final_val_loss, default=None)
        
        if best_mlm:
            logger.info(f"Best MLM: {best_mlm.attention_type} "
                       f"(Loss: {best_mlm.final_val_loss:.4f}, "
                       f"Acc: {best_mlm.final_val_accuracy:.4f})")
        
        if best_clm:
            logger.info(f"Best CLM: {best_clm.attention_type} "
                       f"(Loss: {best_clm.final_val_loss:.4f}, "
                       f"Acc: {best_clm.final_val_accuracy:.4f})")
        
        # Print significant differences
        sig_count = sum(1 for r in significance_results.values() 
                       if r.get('significant', False))
        logger.info(f"Statistically significant differences found: {sig_count}")
        
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