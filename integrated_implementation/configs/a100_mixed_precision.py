"""
A100 Mixed Precision and Optimization Configuration
====================================================

Advanced settings for maximizing A100 performance with mixed precision training.
"""

import torch
from dataclasses import dataclass
from typing import Optional, Dict, Any

@dataclass
class A100OptimizationConfig:
    """Configuration for A100-specific optimizations"""
    
    # Mixed Precision Settings
    use_amp: bool = True                    # Automatic Mixed Precision
    amp_dtype: str = "bfloat16"            # A100 native BF16 support
    amp_opt_level: str = "O2"              # O1=conservative, O2=aggressive
    grad_scaler_enabled: bool = True       # Gradient scaling for FP16
    grad_scaler_init_scale: float = 2**16  # Initial scale factor
    grad_scaler_growth_factor: float = 2.0 # Scale growth factor
    grad_scaler_backoff_factor: float = 0.5 # Scale reduction factor
    grad_scaler_growth_interval: int = 2000 # Steps between scale increases
    
    # Memory Optimization
    gradient_checkpointing: bool = True     # Recompute activations
    checkpoint_segments: int = 4            # Number of checkpointing segments
    activation_checkpointing: bool = True   # Checkpoint attention activations
    cpu_offload: bool = False               # Offload to CPU (not needed for A100)
    
    # A100-Specific Features
    use_tf32: bool = True                   # TensorFloat-32 for matmuls
    use_flash_attention: bool = True        # FlashAttention v2
    use_nested_tensors: bool = True         # Variable-length sequences
    use_cudnn_sdp: bool = True              # cuDNN scaled dot product
    
    # Compilation Settings (PyTorch 2.0+)
    compile_model: bool = True              # torch.compile
    compile_mode: str = "max-autotune"      # reduce-overhead, default, max-autotune
    compile_backend: str = "inductor"       # Compilation backend
    compile_fullgraph: bool = True          # Compile entire graph
    compile_dynamic: bool = False           # Dynamic shapes (slower)
    
    # Performance Settings
    cudnn_benchmark: bool = True            # Auto-tune convolutions
    cudnn_deterministic: bool = False       # Trade reproducibility for speed
    allow_tf32_matmul: bool = True          # TF32 for matrix multiplications
    allow_tf32_cudnn: bool = True           # TF32 for cuDNN operations
    
    # Distributed Training (Multi-A100)
    use_distributed: bool = False           # Enable DDP
    ddp_backend: str = "nccl"              # NVIDIA collective comms
    ddp_find_unused_parameters: bool = False # Overhead reduction
    gradient_as_bucket_view: bool = True    # Memory efficiency
    static_graph: bool = True               # Graph optimization
    
    # Batch Size Optimization
    find_optimal_batch_size: bool = True    # Auto-find max batch size
    batch_size_multiplier: float = 0.9      # Safety margin for batch size
    
    def get_amp_config(self) -> Dict[str, Any]:
        """Get AMP configuration for training"""
        if self.amp_dtype == "bfloat16":
            return {
                "enabled": self.use_amp,
                "dtype": torch.bfloat16,
                "cache_enabled": True,
            }
        else:  # fp16
            return {
                "enabled": self.use_amp,
                "opt_level": self.amp_opt_level,
                "cache_enabled": True,
                "loss_scale": "dynamic" if self.grad_scaler_enabled else 1.0,
            }
    
    def get_scaler_config(self) -> Dict[str, Any]:
        """Get gradient scaler configuration"""
        if not self.grad_scaler_enabled or self.amp_dtype == "bfloat16":
            return {"enabled": False}
        
        return {
            "enabled": True,
            "init_scale": self.grad_scaler_init_scale,
            "growth_factor": self.grad_scaler_growth_factor,
            "backoff_factor": self.grad_scaler_backoff_factor,
            "growth_interval": self.grad_scaler_growth_interval,
        }
    
    def setup_environment(self):
        """Configure environment for A100 optimization"""
        import os
        
        # TF32 settings
        if self.use_tf32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            os.environ["NVIDIA_TF32_OVERRIDE"] = "1"
        
        # cuDNN settings
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = self.cudnn_benchmark
        torch.backends.cudnn.deterministic = self.cudnn_deterministic
        
        # Memory allocation
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"
        
        # NCCL settings for multi-GPU
        if self.use_distributed:
            os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"
            os.environ["NCCL_TREE_THRESHOLD"] = "0"
    
    def optimize_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Apply A100 optimizations to model"""
        
        # Gradient checkpointing
        if self.gradient_checkpointing:
            model.gradient_checkpointing_enable()
        
        # Compile with PyTorch 2.0
        if self.compile_model and hasattr(torch, 'compile'):
            model = torch.compile(
                model,
                mode=self.compile_mode,
                backend=self.compile_backend,
                fullgraph=self.compile_fullgraph,
                dynamic=self.compile_dynamic,
            )
        
        # Flash Attention
        if self.use_flash_attention:
            # Enable SDPA with Flash Attention
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        
        return model


def get_a100_optimizer_config(model_size: str = "large") -> Dict[str, Any]:
    """
    Get optimizer configuration based on model size
    
    Args:
        model_size: "base", "large", or "xlarge"
    """
    configs = {
        "base": {
            "lr": 5e-4,
            "weight_decay": 0.01,
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_epsilon": 1e-6,
            "max_grad_norm": 1.0,
            "use_8bit_adam": False,
        },
        "large": {
            "lr": 3e-4,
            "weight_decay": 0.01,
            "adam_beta1": 0.9,
            "adam_beta2": 0.98,  # Slightly lower for stability
            "adam_epsilon": 1e-6,
            "max_grad_norm": 0.5,  # Tighter clipping
            "use_8bit_adam": True,  # Memory savings
        },
        "xlarge": {
            "lr": 2e-4,
            "weight_decay": 0.01,
            "adam_beta1": 0.9,
            "adam_beta2": 0.95,  # Even lower for very large models
            "adam_epsilon": 1e-6,
            "max_grad_norm": 0.3,
            "use_8bit_adam": True,
            "use_adafactor": True,  # Alternative optimizer
        }
    }
    return configs.get(model_size, configs["large"])


def estimate_batch_size(model: torch.nn.Module, 
                       sequence_length: int = 512,
                       gpu_memory_gb: int = 40) -> int:
    """
    Estimate optimal batch size for A100
    
    Args:
        model: The model to train
        sequence_length: Maximum sequence length
        gpu_memory_gb: GPU memory in GB (40 or 80 for A100)
    
    Returns:
        Estimated optimal batch size
    """
    # Count parameters
    param_count = sum(p.numel() for p in model.parameters())
    
    # Estimate memory per sample (rough approximation)
    # Factors: parameters, gradients, optimizer states, activations
    bytes_per_param = 4 if torch.get_default_dtype() == torch.float32 else 2
    
    # Memory formula (simplified)
    # Parameters + Gradients + Adam states (2x) + Activations
    memory_per_sample_mb = (
        (param_count * bytes_per_param * 4) / (1024 * 1024) +  # Model + optimizer
        (sequence_length * 768 * 12 * bytes_per_param) / (1024 * 1024)  # Activations
    ) * 1.5  # Safety factor
    
    # Available memory (leave 10% free)
    available_memory_mb = gpu_memory_gb * 1024 * 0.9
    
    # Calculate batch size
    estimated_batch_size = int(available_memory_mb / memory_per_sample_mb)
    
    # Round to nearest power of 2 for efficiency
    import math
    batch_size = 2 ** int(math.log2(estimated_batch_size))
    
    return max(1, min(batch_size, 256))  # Cap at reasonable maximum


# Example usage
if __name__ == "__main__":
    config = A100OptimizationConfig()
    config.setup_environment()
    
    print("A100 Optimization Configuration")
    print("=" * 40)
    print(f"Mixed Precision: {config.amp_dtype}")
    print(f"TF32 Enabled: {config.use_tf32}")
    print(f"Flash Attention: {config.use_flash_attention}")
    print(f"Gradient Checkpointing: {config.gradient_checkpointing}")
    print(f"Model Compilation: {config.compile_model}")
    print(f"Compile Mode: {config.compile_mode}")
    
    print("\nAMP Config:", config.get_amp_config())
    print("Scaler Config:", config.get_scaler_config())
    
    # Test batch size estimation
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.transformer = torch.nn.TransformerEncoder(
                torch.nn.TransformerEncoderLayer(768, 12, 3072),
                num_layers=12
            )
    
    model = DummyModel()
    batch_size_40gb = estimate_batch_size(model, 512, 40)
    batch_size_80gb = estimate_batch_size(model, 512, 80)
    
    print(f"\nEstimated Batch Sizes:")
    print(f"A100 40GB: {batch_size_40gb}")
    print(f"A100 80GB: {batch_size_80gb}")