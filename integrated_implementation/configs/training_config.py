"""
Training Configuration
=====================

Configuration for model training parameters.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict
from .base_config import BaseConfig


@dataclass
class OptimizerConfig(BaseConfig):
    """Optimizer configuration"""
    type: str = "adamw"
    betas: List[float] = field(default_factory=lambda: [0.9, 0.999])
    eps: float = 1e-8
    weight_decay: float = 0.01
    amsgrad: bool = False


@dataclass
class SchedulerConfig(BaseConfig):
    """Learning rate scheduler configuration"""
    type: str = "cosine"
    num_warmup_steps: int = 1000
    num_training_steps: Optional[int] = None
    min_lr_ratio: float = 0.01


@dataclass
class TrainingConfig(BaseConfig):
    """Training configuration"""
    # Basic parameters
    num_epochs: int = 20
    batch_size: int = 32
    learning_rate: float = 5e-5
    warmup_steps: int = 1000
    
    # Regularization
    weight_decay: float = 0.01
    gradient_clipping: float = 1.0
    dropout: float = 0.1
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    
    # Logging and checkpointing
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 2000
    
    # Performance
    num_workers: int = 4
    pin_memory: bool = True
    dataloader_drop_last: bool = True
    fp16: bool = False
    
    # Sub-configurations
    optimizer: OptimizerConfig = None
    scheduler: SchedulerConfig = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.optimizer is None:
            self.optimizer = OptimizerConfig()
        if self.scheduler is None:
            self.scheduler = SchedulerConfig(num_warmup_steps=self.warmup_steps)
    
    def get_optimizer_config(self) -> OptimizerConfig:
        """Get optimizer configuration"""
        return self.optimizer
    
    def get_scheduler_config(self) -> SchedulerConfig:
        """Get scheduler configuration"""
        return self.scheduler
    
    @classmethod
    def from_env_dict(cls, env_dict: Dict[str, str]) -> 'TrainingConfig':
        """Create from environment variables"""
        config = {}
        
        # Parse basic fields
        field_mapping = {
            'NUM_EPOCHS': 'num_epochs',
            'BATCH_SIZE': 'batch_size',
            'LEARNING_RATE': 'learning_rate',
            'WARMUP_STEPS': 'warmup_steps',
            'GRADIENT_ACCUMULATION_STEPS': 'gradient_accumulation_steps',
            'LOGGING_STEPS': 'logging_steps',
            'EVAL_STEPS': 'eval_steps',
            'SAVE_STEPS': 'save_steps',
            'WEIGHT_DECAY': 'weight_decay',
            'GRADIENT_CLIPPING': 'gradient_clipping',
            'MAX_GRAD_NORM': 'max_grad_norm',
            'FP16': 'fp16',
            'DATALOADER_NUM_WORKERS': 'num_workers'
        }
        
        for env_key, field_name in field_mapping.items():
            if env_key in env_dict:
                value = env_dict[env_key]
                if field_name in ['learning_rate', 'weight_decay', 'gradient_clipping', 'max_grad_norm']:
                    config[field_name] = float(value)
                elif field_name in ['fp16']:
                    config[field_name] = value.lower() in ('true', '1', 'yes')
                else:
                    config[field_name] = int(value)
        
        return cls(**config)