"""
Visualization Utilities
======================

Functions for plotting training results and comparisons.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Any


def plot_attention_comparison(results: Dict[str, Any], save_path: str):
    """
    Create comprehensive comparison plots for different attention mechanisms
    
    Args:
        results: Dictionary mapping attention type to TrainingResults
        save_path: Path to save the plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    colors = {
        'standard': '#1f77b4',  # Blue
        'rope': '#ff7f0e',       # Orange  
        'exposb': '#2ca02c',     # Green
        'absolute': '#d62728'    # Red
    }
    
    # Plot 1: Training MLM Loss
    ax = axes[0, 0]
    for attention_type, result in results.items():
        if hasattr(result, 'train_losses') and result.train_losses:
            steps = np.arange(len(result.train_losses)) * 100  # Assuming logging every 100 steps
            ax.plot(steps, result.train_losses, 
                   label=f'{attention_type.upper()} (MLM)', 
                   color=colors.get(attention_type, 'gray'),
                   linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('MLM Loss', fontsize=12)
    ax.set_title('Training MLM Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Validation MLM Loss
    ax = axes[0, 1]
    for attention_type, result in results.items():
        if hasattr(result, 'val_losses') and result.val_losses:
            eval_steps = np.arange(len(result.val_losses))
            ax.plot(eval_steps, result.val_losses,
                   label=f'{attention_type.upper()} (MLM)',
                   color=colors.get(attention_type, 'gray'),
                   marker='o', linewidth=2, markersize=6)
    
    ax.set_xlabel('Evaluation Steps', fontsize=12)
    ax.set_ylabel('Validation MLM Loss', fontsize=12)
    ax.set_title('Validation MLM Loss Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate Schedule
    ax = axes[0, 2]
    for attention_type, result in results.items():
        if hasattr(result, 'learning_rates') and result.learning_rates:
            steps = np.arange(len(result.learning_rates)) * 100
            ax.plot(steps, result.learning_rates,
                   label=f'{attention_type.upper()}',
                   color=colors.get(attention_type, 'gray'),
                   linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Learning Rate', fontsize=12)
    ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: MLM Loss Convergence Speed
    ax = axes[1, 0]
    for attention_type, result in results.items():
        if hasattr(result, 'train_losses') and result.train_losses:
            # Normalize losses to show convergence
            losses = np.array(result.train_losses)
            if len(losses) > 0:
                normalized = (losses - losses.min()) / (losses.max() - losses.min())
                steps = np.arange(len(normalized)) * 100
                ax.plot(steps, normalized,
                       label=f'{attention_type.upper()}',
                       color=colors.get(attention_type, 'gray'),
                       linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Training Steps', fontsize=12)
    ax.set_ylabel('Normalized MLM Loss', fontsize=12)
    ax.set_title('MLM Convergence Speed Comparison', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 5: Final Performance Bar Chart
    ax = axes[1, 1]
    final_train_losses = {}
    final_val_losses = {}
    
    for attention_type, result in results.items():
        if hasattr(result, 'train_losses') and result.train_losses:
            final_train_losses[attention_type] = result.train_losses[-1]
        if hasattr(result, 'val_losses') and result.val_losses:
            final_val_losses[attention_type] = result.val_losses[-1]
    
    if final_train_losses:
        x = np.arange(len(final_train_losses))
        width = 0.35
        
        train_bars = ax.bar(x - width/2, list(final_train_losses.values()), width,
                           label='Train MLM Loss', color='steelblue', alpha=0.8)
        
        if final_val_losses:
            val_bars = ax.bar(x + width/2, list(final_val_losses.values()), width,
                             label='Val MLM Loss', color='coral', alpha=0.8)
        
        ax.set_xlabel('Attention Type', fontsize=12)
        ax.set_ylabel('Final MLM Loss', fontsize=12)
        ax.set_title('Final MLM Performance Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([name.upper() for name in final_train_losses.keys()])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bars in [train_bars, val_bars if final_val_losses else []]:
            if bars:
                for bar in bars:
                    height = bar.get_height()
                    ax.annotate(f'{height:.3f}',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom',
                               fontsize=9)
    
    # Plot 6: Training Time Comparison
    ax = axes[1, 2]
    training_times = {}
    for attention_type, result in results.items():
        if hasattr(result, 'training_time'):
            training_times[attention_type] = result.training_time / 60  # Convert to minutes
    
    if training_times:
        bars = ax.bar(range(len(training_times)), list(training_times.values()),
                      color=[colors.get(name, 'gray') for name in training_times.keys()],
                      alpha=0.8)
        ax.set_xlabel('Attention Type', fontsize=12)
        ax.set_ylabel('Training Time (minutes)', fontsize=12)
        ax.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(training_times)))
        ax.set_xticklabels([name.upper() for name in training_times.keys()])
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.1f}m',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=10)
    
    # Main title
    plt.suptitle('BERT Attention Mechanisms: MLM Training Comparison\n' + 
                 'All models trained with Masked Language Modeling objective',
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def plot_training_curves(train_losses: List[float], 
                        val_losses: List[float],
                        title: str = "Training Curves",
                        save_path: str = None):
    """Plot simple training curves"""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    steps = np.arange(len(train_losses))
    ax.plot(steps, train_losses, label='Training Loss', linewidth=2)
    
    if val_losses:
        val_steps = np.linspace(0, len(train_losses), len(val_losses))
        ax.plot(val_steps, val_losses, label='Validation Loss', linewidth=2, marker='o')
    
    ax.set_xlabel('Steps')
    ax.set_ylabel('Loss')
    ax.set_title(title)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    return fig


def save_plots(fig, path: str):
    """Save figure to file"""
    fig.savefig(path, dpi=300, bbox_inches='tight')