"""
GPU Memory Monitoring Utilities for 8GB GPU Training
====================================================

Provides memory monitoring and optimization utilities for training
BERT models on GPUs with limited memory (8GB).
"""

import torch
import logging
import gc
from typing import Dict, Optional

logger = logging.getLogger(__name__)


class GPUMemoryMonitor:
    """Monitor and manage GPU memory usage during training"""
    
    def __init__(self, max_memory_gb: float = 8.0, warning_threshold: float = 0.8):
        self.max_memory_gb = max_memory_gb
        self.warning_threshold = warning_threshold
        self.memory_history = []
        
    def get_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory information"""
        if not torch.cuda.is_available():
            return {"available": False}
        
        allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
        reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
        max_allocated = torch.cuda.max_memory_allocated() / (1024**3)  # GB
        
        # Get total GPU memory
        try:
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        except:
            total_memory = self.max_memory_gb
        
        usage_ratio = allocated / total_memory
        
        return {
            "available": True,
            "allocated_gb": allocated,
            "reserved_gb": reserved,
            "max_allocated_gb": max_allocated,
            "total_gb": total_memory,
            "usage_ratio": usage_ratio,
            "free_gb": total_memory - allocated
        }
    
    def log_memory_status(self, stage: str = ""):
        """Log current memory status"""
        info = self.get_memory_info()
        if not info["available"]:
            logger.info("GPU not available for memory monitoring")
            return
        
        stage_prefix = f"[{stage}] " if stage else ""
        logger.info(f"{stage_prefix}GPU Memory: "
                   f"{info['allocated_gb']:.2f}GB allocated, "
                   f"{info['free_gb']:.2f}GB free, "
                   f"{info['usage_ratio']:.1%} used")
        
        # Warning if memory usage is high
        if info['usage_ratio'] > self.warning_threshold:
            logger.warning(f"High GPU memory usage: {info['usage_ratio']:.1%}")
        
        # Store history
        self.memory_history.append({
            "stage": stage,
            "allocated": info['allocated_gb'],
            "usage_ratio": info['usage_ratio']
        })
    
    def clear_cache(self):
        """Clear GPU cache to free memory"""
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()  # Also run Python garbage collection
            logger.debug("Cleared GPU cache and ran garbage collection")
    
    def check_memory_and_clear(self, threshold: float = 0.8) -> bool:
        """Check memory usage and clear cache if needed"""
        info = self.get_memory_info()
        if not info["available"]:
            return False
        
        if info['usage_ratio'] > threshold:
            logger.info(f"Memory usage high ({info['usage_ratio']:.1%}), clearing cache...")
            self.clear_cache()
            
            # Check again after clearing
            new_info = self.get_memory_info()
            logger.info(f"After clearing: {new_info['usage_ratio']:.1%} used")
            return True
        
        return False
    
    def estimate_batch_memory(self, batch_size: int, seq_length: int, 
                            hidden_size: int, vocab_size: int) -> float:
        """Estimate memory usage for a training batch"""
        
        # Rough estimation based on BERT memory requirements
        # Input embeddings: batch_size * seq_length * hidden_size * 4 bytes
        input_memory = batch_size * seq_length * hidden_size * 4
        
        # Model parameters: roughly hidden_size^2 * num_layers * 4 bytes
        # This is very rough - actual depends on model architecture
        model_memory = hidden_size * hidden_size * 6 * 4  # Assume 6 layers
        
        # Gradients (same size as parameters)
        gradient_memory = model_memory
        
        # Optimizer states (Adam: 2x parameters)
        optimizer_memory = model_memory * 2
        
        # Activations during forward/backward (highly variable)
        activation_memory = batch_size * seq_length * hidden_size * 8 * 4  # Rough estimate
        
        total_bytes = input_memory + model_memory + gradient_memory + optimizer_memory + activation_memory
        total_gb = total_bytes / (1024**3)
        
        return total_gb
    
    def suggest_batch_size(self, seq_length: int, hidden_size: int, 
                          vocab_size: int, target_memory_gb: float = 6.0) -> int:
        """Suggest optimal batch size for given memory constraints"""
        
        for batch_size in [1, 2, 4, 8, 16, 32]:
            estimated_memory = self.estimate_batch_memory(
                batch_size, seq_length, hidden_size, vocab_size
            )
            
            if estimated_memory <= target_memory_gb:
                continue
            else:
                # Return previous batch size that fit
                return max(1, batch_size // 2)
        
        return 1  # Fallback to batch size 1
    
    def get_memory_summary(self) -> Dict:
        """Get summary of memory usage throughout training"""
        if not self.memory_history:
            return {"no_data": True}
        
        max_usage = max(h['usage_ratio'] for h in self.memory_history)
        avg_usage = sum(h['usage_ratio'] for h in self.memory_history) / len(self.memory_history)
        peak_memory = max(h['allocated'] for h in self.memory_history)
        
        return {
            "max_usage_ratio": max_usage,
            "avg_usage_ratio": avg_usage,
            "peak_memory_gb": peak_memory,
            "num_measurements": len(self.memory_history)
        }


def optimize_model_for_memory(model, config):
    """Apply memory optimizations to model"""
    optimizations_applied = []
    
    # Enable gradient checkpointing if available
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        optimizations_applied.append("gradient_checkpointing")
    
    # Set model to half precision if enabled
    if getattr(config, 'fp16', False):
        model = model.half()
        optimizations_applied.append("fp16")
    
    # Enable attention memory optimization if available
    if hasattr(model, 'set_mem_eff_attention'):
        model.set_mem_eff_attention(True)
        optimizations_applied.append("memory_efficient_attention")
    
    logger.info(f"Applied memory optimizations: {', '.join(optimizations_applied)}")
    return model


def setup_memory_efficient_training(model, optimizer, config):
    """Setup memory-efficient training configuration"""
    
    # Clear cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Apply model optimizations
    model = optimize_model_for_memory(model, config)
    
    # Setup memory monitor
    monitor = GPUMemoryMonitor(max_memory_gb=8.0)
    monitor.log_memory_status("setup_complete")
    
    # Log memory recommendations
    info = monitor.get_memory_info()
    if info["available"]:
        suggested_batch_size = monitor.suggest_batch_size(
            seq_length=getattr(config, 'max_seq_length', 128),
            hidden_size=getattr(config.model, 'hidden_size', 192) if config.model else 192,
            vocab_size=getattr(config.model, 'vocab_size', 30522) if config.model else 30522,
            target_memory_gb=6.0  # Leave 2GB free
        )
        
        current_batch_size = getattr(config, 'batch_size', 2)
        if suggested_batch_size < current_batch_size:
            logger.warning(f"Current batch size ({current_batch_size}) may be too large. "
                         f"Suggested: {suggested_batch_size}")
    
    return model, monitor