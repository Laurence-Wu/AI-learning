#!/usr/bin/env python3
"""
Demonstration of smoothing techniques for MLM validation loss graphs
"""

import numpy as np
import matplotlib.pyplot as plt
from src.utils.mlm_clm_visualization import smooth_data

# Generate sample noisy validation loss data
np.random.seed(42)
epochs = 150
base_loss = 8.0 * np.exp(-0.02 * np.arange(epochs)) + 7.5
noise = np.random.normal(0, 0.3, epochs)
noisy_loss = base_loss + noise

# Apply different smoothing methods
smoothing_methods = {
    'Original': noisy_loss,
    'Gaussian (σ=1.5)': smooth_data(noisy_loss, method='gaussian', sigma=1.5),
    'Gaussian (σ=2.5)': smooth_data(noisy_loss, method='gaussian', sigma=2.5),
    'Savitzky-Golay (window=7)': smooth_data(noisy_loss, method='savgol', window_size=7),
    'Moving Average (window=5)': smooth_data(noisy_loss, method='moving_avg', window_size=5),
    'EWMA (span=10)': smooth_data(noisy_loss, method='ewma', window_size=10)
}

# Create comparison plot
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (method_name, smoothed_data) in enumerate(smoothing_methods.items()):
    ax = axes[idx]
    
    if method_name == 'Original':
        ax.plot(smoothed_data, label=method_name, color='red', alpha=0.6, linewidth=1.5)
    else:
        # Show original in background
        ax.plot(noisy_loss, alpha=0.3, color='gray', linewidth=1, label='Original')
        # Show smoothed
        ax.plot(smoothed_data, label=method_name, color='blue', linewidth=2.5)
    
    ax.set_title(method_name, fontsize=12, fontweight='bold')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Validation Loss')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add statistics
    if method_name != 'Original':
        variance_reduction = (np.var(noisy_loss) - np.var(smoothed_data)) / np.var(noisy_loss) * 100
        ax.text(0.05, 0.95, f'Variance reduction: {variance_reduction:.1f}%',
                transform=ax.transAxes, fontsize=9,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

plt.suptitle('Smoothing Methods for MLM Validation Loss', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig('smoothing_comparison.png', dpi=150, bbox_inches='tight')
plt.show()

print("\nRecommended Smoothing Configurations:")
print("="*50)
print("\n1. For slightly noisy data (your case):")
print("   config = {'method': 'gaussian', 'window_size': 5, 'sigma': 1.5}")
print("\n2. For very noisy data:")
print("   config = {'method': 'savgol', 'window_size': 9}")
print("\n3. For trend analysis:")
print("   config = {'method': 'ewma', 'window_size': 10}")
print("\n4. For minimal smoothing:")
print("   config = {'method': 'moving_avg', 'window_size': 3}")

# Example usage in your training script
print("\n" + "="*50)
print("Usage in your code:")
print("="*50)
print("""
from src.utils.mlm_clm_visualization import plot_mlm_vs_clm_comparison

# Configure smoothing
smoothing_config = {
    'method': 'gaussian',      # Options: 'gaussian', 'savgol', 'moving_avg', 'ewma'
    'window_size': 5,          # Window size for smoothing
    'sigma': 1.5,              # For Gaussian only
    'apply_to_val': True       # Whether to smooth validation data
}

# Generate plots with smoothing
plot_mlm_vs_clm_comparison(results, 'outputs/mlm_clm_comparison.png', smoothing_config)
""")