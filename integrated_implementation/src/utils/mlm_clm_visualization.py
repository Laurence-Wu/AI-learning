"""
Enhanced Visualization for MLM vs CLM Comparison
===============================================

Advanced plotting functions for comparing MLM and CLM training across different attention mechanisms.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter


def smooth_data(data, method='gaussian', window_size=5, sigma=1.0):
    """
    Apply smoothing to data using various methods
    
    Args:
        data: Input data array
        method: Smoothing method ('gaussian', 'savgol', 'moving_avg', 'ewma')
        window_size: Window size for smoothing
        sigma: Standard deviation for Gaussian filter
    
    Returns:
        Smoothed data array
    """
    if len(data) < window_size:
        return data
    
    if method == 'gaussian':
        return gaussian_filter1d(data, sigma=sigma)
    elif method == 'savgol':
        # Ensure window_size is odd
        window = window_size if window_size % 2 == 1 else window_size + 1
        # Ensure window doesn't exceed data length
        window = min(window, len(data))
        if window < 4:
            return data
        return savgol_filter(data, window, min(3, window - 1))
    elif method == 'moving_avg':
        # Simple moving average
        kernel = np.ones(window_size) / window_size
        # Pad data to handle boundaries
        padded = np.pad(data, (window_size//2, window_size//2), mode='edge')
        return np.convolve(padded, kernel, mode='valid')[:len(data)]
    elif method == 'ewma':
        # Exponential weighted moving average
        df = pd.Series(data)
        return df.ewm(span=window_size, adjust=False).mean().values
    else:
        return data


def plot_mlm_vs_clm_comparison(results: Dict[str, Any], save_path: str, smoothing_config: Optional[Dict] = None):
    """
    Create comprehensive MLM vs CLM comparison plots
    
    Args:
        results: Dictionary mapping variant_name (attention_objective) to TrainingResults
        save_path: Path to save the plot
        smoothing_config: Dictionary with smoothing configuration:
            {
                'method': 'gaussian' | 'savgol' | 'moving_avg' | 'ewma',
                'window_size': int (default 5),
                'sigma': float (for gaussian, default 1.0),
                'apply_to_val': bool (whether to smooth validation data, default True)
            }
        
    Expected format:
    {
        'standard_mlm': TrainingResults(...),
        'standard_clm': TrainingResults(...),
        'rope_mlm': TrainingResults(...),
        'rope_clm': TrainingResults(...),
        ...
    }
    """
    # Parse results into structured format
    structured_results = {}
    objectives = set()
    attention_types = set()
    
    for variant_name, result in results.items():
        if '_' in variant_name:
            attention_type, objective = variant_name.rsplit('_', 1)
            if attention_type not in structured_results:
                structured_results[attention_type] = {}
            structured_results[attention_type][objective] = result
            objectives.add(objective)
            attention_types.add(attention_type)
    
    # Create comprehensive figure layout
    fig = plt.figure(figsize=(20, 16))
    
    # Define colors
    mlm_colors = {
        'standard': '#1f77b4',  # Blue
        'rope': '#ff7f0e',      # Orange  
        'exposb': '#2ca02c',    # Green
        'absolute': '#d62728'   # Red
    }
    
    clm_colors = {
        'standard': '#6495ED',  # CornflowerBlue (lighter blue)
        'rope': '#FF6347',      # Tomato (lighter orange)
        'exposb': '#90EE90',    # LightGreen
        'absolute': '#F08080'   # LightCoral (lighter red)
    }
    
    # Create subplots grid
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Plot 1: Training Loss Comparison (MLM vs CLM)
    ax1 = fig.add_subplot(gs[0, 0:2])
    for attention_type in sorted(attention_types):
        if attention_type in structured_results:
            # MLM training loss
            if 'mlm' in structured_results[attention_type]:
                mlm_result = structured_results[attention_type]['mlm']
                if hasattr(mlm_result, 'train_losses') and mlm_result.train_losses:
                    steps = np.arange(len(mlm_result.train_losses)) * 100
                    ax1.plot(steps, mlm_result.train_losses, 
                            label=f'{attention_type.upper()} MLM', 
                            color=mlm_colors.get(attention_type, 'gray'),
                            linewidth=2.5, alpha=0.9, linestyle='-')
            
            # CLM training loss
            if 'clm' in structured_results[attention_type]:
                clm_result = structured_results[attention_type]['clm']
                if hasattr(clm_result, 'train_losses') and clm_result.train_losses:
                    steps = np.arange(len(clm_result.train_losses)) * 100
                    ax1.plot(steps, clm_result.train_losses,
                            label=f'{attention_type.upper()} CLM',
                            color=clm_colors.get(attention_type, 'lightgray'),
                            linewidth=2.5, alpha=0.9, linestyle='--')
    
    ax1.set_xlabel('Training Steps', fontsize=12)
    ax1.set_ylabel('Training Loss', fontsize=12)
    ax1.set_title('Training Loss: MLM vs CLM Comparison', fontsize=14, fontweight='bold')
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Validation Loss Comparison
    ax2 = fig.add_subplot(gs[0, 2:4])
    
    # Default smoothing config
    if smoothing_config is None:
        smoothing_config = {'method': 'gaussian', 'window_size': 5, 'sigma': 1.5, 'apply_to_val': True}
    
    for attention_type in sorted(attention_types):
        if attention_type in structured_results:
            # MLM validation loss
            if 'mlm' in structured_results[attention_type]:
                mlm_result = structured_results[attention_type]['mlm']
                if hasattr(mlm_result, 'val_losses') and mlm_result.val_losses:
                    eval_steps = np.arange(len(mlm_result.val_losses))
                    val_losses = mlm_result.val_losses
                    
                    # Apply smoothing if configured
                    if smoothing_config.get('apply_to_val', True):
                        val_losses_smooth = smooth_data(
                            val_losses,
                            method=smoothing_config.get('method', 'gaussian'),
                            window_size=smoothing_config.get('window_size', 5),
                            sigma=smoothing_config.get('sigma', 1.5)
                        )
                        # Plot original data with low alpha
                        ax2.plot(eval_steps, val_losses,
                                color=mlm_colors.get(attention_type, 'gray'),
                                alpha=0.2, linewidth=1, linestyle='-')
                        # Plot smoothed data
                        ax2.plot(eval_steps, val_losses_smooth,
                                label=f'{attention_type.upper()} MLM',
                                color=mlm_colors.get(attention_type, 'gray'),
                                linewidth=2.5, linestyle='-')
                    else:
                        ax2.plot(eval_steps, val_losses,
                                label=f'{attention_type.upper()} MLM',
                                color=mlm_colors.get(attention_type, 'gray'),
                                marker='o', linewidth=2, markersize=6, linestyle='-')
            
            # CLM validation loss
            if 'clm' in structured_results[attention_type]:
                clm_result = structured_results[attention_type]['clm']
                if hasattr(clm_result, 'val_losses') and clm_result.val_losses:
                    eval_steps = np.arange(len(clm_result.val_losses))
                    val_losses = clm_result.val_losses
                    
                    # Apply smoothing if configured
                    if smoothing_config.get('apply_to_val', True):
                        val_losses_smooth = smooth_data(
                            val_losses,
                            method=smoothing_config.get('method', 'gaussian'),
                            window_size=smoothing_config.get('window_size', 5),
                            sigma=smoothing_config.get('sigma', 1.5)
                        )
                        # Plot original data with low alpha
                        ax2.plot(eval_steps, val_losses,
                                color=clm_colors.get(attention_type, 'lightgray'),
                                alpha=0.2, linewidth=1, linestyle='--')
                        # Plot smoothed data
                        ax2.plot(eval_steps, val_losses_smooth,
                                label=f'{attention_type.upper()} CLM',
                                color=clm_colors.get(attention_type, 'lightgray'),
                                linewidth=2.5, linestyle='--')
                    else:
                        ax2.plot(eval_steps, val_losses,
                                label=f'{attention_type.upper()} CLM',
                                color=clm_colors.get(attention_type, 'lightgray'),
                                marker='s', linewidth=2, markersize=6, linestyle='--')
    
    ax2.set_xlabel('Evaluation Steps', fontsize=12)
    ax2.set_ylabel('Validation Loss', fontsize=12)
    ax2.set_title('Validation Loss: MLM vs CLM Comparison', fontsize=14, fontweight='bold')
    ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Final Performance Heatmap
    ax3 = fig.add_subplot(gs[1, 0:2])
    
    # Create data for heatmap
    heatmap_data = []
    row_labels = []
    col_labels = ['MLM Train', 'MLM Val', 'CLM Train', 'CLM Val']
    
    for attention_type in sorted(attention_types):
        if attention_type in structured_results:
            row_data = []
            
            # MLM data
            if 'mlm' in structured_results[attention_type]:
                mlm_result = structured_results[attention_type]['mlm']
                mlm_train = mlm_result.train_losses[-1] if mlm_result.train_losses else np.nan
                mlm_val = mlm_result.val_losses[-1] if mlm_result.val_losses else np.nan
            else:
                mlm_train = mlm_val = np.nan
            
            # CLM data
            if 'clm' in structured_results[attention_type]:
                clm_result = structured_results[attention_type]['clm']
                clm_train = clm_result.train_losses[-1] if clm_result.train_losses else np.nan
                clm_val = clm_result.val_losses[-1] if clm_result.val_losses else np.nan
            else:
                clm_train = clm_val = np.nan
            
            row_data = [mlm_train, mlm_val, clm_train, clm_val]
            heatmap_data.append(row_data)
            row_labels.append(attention_type.upper())
    
    if heatmap_data:
        heatmap_array = np.array(heatmap_data)
        im = ax3.imshow(heatmap_array, cmap='RdYlBu_r', aspect='auto')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax3)
        cbar.set_label('Final Loss', rotation=270, labelpad=15)
        
        # Set ticks and labels
        ax3.set_xticks(np.arange(len(col_labels)))
        ax3.set_yticks(np.arange(len(row_labels)))
        ax3.set_xticklabels(col_labels)
        ax3.set_yticklabels(row_labels)
        
        # Add text annotations
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                if not np.isnan(heatmap_array[i, j]):
                    text = ax3.text(j, i, f'{heatmap_array[i, j]:.3f}',
                                   ha="center", va="center", color="black", fontweight='bold')
        
        ax3.set_title('Final Loss Heatmap: All Combinations', fontsize=14, fontweight='bold')
    
    # Plot 4: Convergence Speed Comparison
    ax4 = fig.add_subplot(gs[1, 2:4])
    
    for attention_type in sorted(attention_types):
        if attention_type in structured_results:
            # MLM convergence
            if 'mlm' in structured_results[attention_type]:
                mlm_result = structured_results[attention_type]['mlm']
                if hasattr(mlm_result, 'train_losses') and mlm_result.train_losses:
                    losses = np.array(mlm_result.train_losses)
                    if len(losses) > 0:
                        # Normalize for convergence speed comparison
                        normalized = (losses[0] - losses) / (losses[0] - losses.min()) if losses[0] != losses.min() else losses * 0
                        steps = np.arange(len(normalized)) * 100
                        ax4.plot(steps, normalized * 100,  # Convert to percentage
                                label=f'{attention_type.upper()} MLM',
                                color=mlm_colors.get(attention_type, 'gray'),
                                linewidth=2, alpha=0.9)
            
            # CLM convergence
            if 'clm' in structured_results[attention_type]:
                clm_result = structured_results[attention_type]['clm']
                if hasattr(clm_result, 'train_losses') and clm_result.train_losses:
                    losses = np.array(clm_result.train_losses)
                    if len(losses) > 0:
                        normalized = (losses[0] - losses) / (losses[0] - losses.min()) if losses[0] != losses.min() else losses * 0
                        steps = np.arange(len(normalized)) * 100
                        ax4.plot(steps, normalized * 100,
                                label=f'{attention_type.upper()} CLM',
                                color=clm_colors.get(attention_type, 'lightgray'),
                                linewidth=2, alpha=0.9, linestyle='--')
    
    ax4.set_xlabel('Training Steps', fontsize=12)
    ax4.set_ylabel('Convergence Progress (%)', fontsize=12)
    ax4.set_title('Convergence Speed: MLM vs CLM', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Training Time Comparison
    ax5 = fig.add_subplot(gs[2, 0:2])
    
    training_times = {'MLM': {}, 'CLM': {}}
    for attention_type in sorted(attention_types):
        if attention_type in structured_results:
            if 'mlm' in structured_results[attention_type]:
                mlm_result = structured_results[attention_type]['mlm']
                if hasattr(mlm_result, 'training_time'):
                    training_times['MLM'][attention_type] = mlm_result.training_time / 60  # Convert to minutes
            
            if 'clm' in structured_results[attention_type]:
                clm_result = structured_results[attention_type]['clm']
                if hasattr(clm_result, 'training_time'):
                    training_times['CLM'][attention_type] = clm_result.training_time / 60
    
    if training_times['MLM'] or training_times['CLM']:
        x = np.arange(len(attention_types))
        width = 0.35
        
        mlm_times = [training_times['MLM'].get(att, 0) for att in sorted(attention_types)]
        clm_times = [training_times['CLM'].get(att, 0) for att in sorted(attention_types)]
        
        bars1 = ax5.bar(x - width/2, mlm_times, width, label='MLM Training', 
                       color='steelblue', alpha=0.8)
        bars2 = ax5.bar(x + width/2, clm_times, width, label='CLM Training', 
                       color='coral', alpha=0.8)
        
        ax5.set_xlabel('Attention Type', fontsize=12)
        ax5.set_ylabel('Training Time (minutes)', fontsize=12)
        ax5.set_title('Training Time Comparison: MLM vs CLM', fontsize=14, fontweight='bold')
        ax5.set_xticks(x)
        ax5.set_xticklabels([att.upper() for att in sorted(attention_types)])
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax5.annotate(f'{height:.1f}m',
                               xy=(bar.get_x() + bar.get_width() / 2, height),
                               xytext=(0, 3),
                               textcoords="offset points",
                               ha='center', va='bottom',
                               fontsize=9)
    
    # Plot 6: Performance Delta (MLM vs CLM)
    ax6 = fig.add_subplot(gs[2, 2:4])
    
    deltas = {}
    for attention_type in sorted(attention_types):
        if attention_type in structured_results:
            mlm_final = None
            clm_final = None
            
            if 'mlm' in structured_results[attention_type]:
                mlm_result = structured_results[attention_type]['mlm']
                if mlm_result.train_losses:
                    mlm_final = mlm_result.train_losses[-1]
            
            if 'clm' in structured_results[attention_type]:
                clm_result = structured_results[attention_type]['clm']
                if clm_result.train_losses:
                    clm_final = clm_result.train_losses[-1]
            
            if mlm_final is not None and clm_final is not None:
                # Percentage difference: (CLM - MLM) / MLM * 100
                delta = (clm_final - mlm_final) / mlm_final * 100
                deltas[attention_type] = delta
    
    if deltas:
        bars = ax6.bar(range(len(deltas)), list(deltas.values()),
                      color=['green' if v < 0 else 'red' for v in deltas.values()],
                      alpha=0.7)
        
        ax6.set_xlabel('Attention Type', fontsize=12)
        ax6.set_ylabel('Performance Delta (%)', fontsize=12)
        ax6.set_title('CLM vs MLM Final Loss Difference', fontsize=14, fontweight='bold')
        ax6.set_xticks(range(len(deltas)))
        ax6.set_xticklabels([att.upper() for att in deltas.keys()])
        ax6.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax6.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, (att, delta) in enumerate(deltas.items()):
            ax6.annotate(f'{delta:+.1f}%',
                        xy=(i, delta),
                        xytext=(0, 5 if delta >= 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if delta >= 0 else 'top',
                        fontsize=10, fontweight='bold')
        
        # Add interpretation text
        ax6.text(0.5, 0.95, 'Green: CLM performs better\nRed: MLM performs better',
                transform=ax6.transAxes, ha='center', va='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Main title
    plt.suptitle('Comprehensive BERT Attention Mechanisms Comparison\n' + 
                 'MLM (Masked Language Modeling) vs CLM (Causal Language Modeling)',
                 fontsize=18, fontweight='bold', y=0.98)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return fig


def plot_objective_specific_comparison(results: Dict[str, Any], objective: str, save_path: str):
    """
    Create plots comparing attention mechanisms for a specific objective (MLM or CLM)
    """
    # Filter results for specific objective
    filtered_results = {
        name.replace(f'_{objective}', ''): result 
        for name, result in results.items() 
        if name.endswith(f'_{objective}')
    }
    
    if not filtered_results:
        print(f"No results found for objective: {objective}")
        return None
    
    # Use the original visualization function with filtered results
    from .visualization import plot_attention_comparison
    return plot_attention_comparison(filtered_results, save_path)


# Example usage and testing
if __name__ == "__main__":
    # Mock results for testing
    class MockResults:
        def __init__(self, train_losses, val_losses, training_time):
            self.train_losses = train_losses
            self.val_losses = val_losses
            self.training_time = training_time
    
    # Create mock data
    mock_results = {
        'standard_mlm': MockResults([2.5, 2.0, 1.8, 1.6, 1.5], [2.2, 1.9, 1.7, 1.6], 120),
        'standard_clm': MockResults([3.0, 2.3, 2.0, 1.8, 1.7], [2.5, 2.1, 1.9, 1.8], 130),
        'rope_mlm': MockResults([2.4, 1.9, 1.7, 1.5, 1.4], [2.1, 1.8, 1.6, 1.5], 125),
        'rope_clm': MockResults([2.9, 2.2, 1.9, 1.7, 1.6], [2.4, 2.0, 1.8, 1.7], 135),
        'exposb_mlm': MockResults([2.6, 2.1, 1.9, 1.7, 1.6], [2.3, 2.0, 1.8, 1.7], 140),
        'exposb_clm': MockResults([3.1, 2.4, 2.1, 1.9, 1.8], [2.6, 2.2, 2.0, 1.9], 145),
        'absolute_mlm': MockResults([2.5, 2.0, 1.8, 1.6, 1.5], [2.2, 1.9, 1.7, 1.6], 115),
        'absolute_clm': MockResults([3.0, 2.3, 2.0, 1.8, 1.7], [2.5, 2.1, 1.9, 1.8], 125),
    }
    
    # Test the visualization
    plot_mlm_vs_clm_comparison(mock_results, "test_mlm_clm_comparison.png")
    print("Test visualization created: test_mlm_clm_comparison.png")