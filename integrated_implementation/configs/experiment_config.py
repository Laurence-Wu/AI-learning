"""
Comprehensive Experiment Configuration System
===========================================

Complete configuration management for BERT attention mechanism comparison experiments.
"""

import os
import json
import yaml
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, asdict
import logging

from .base_config import BaseConfig
from .training_config import TrainingConfig
from .model_config import ModelConfig  
from .data_config import DataConfig

logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig(BaseConfig):
    """
    Complete experiment configuration combining all sub-configurations
    """
    # Experiment metadata
    experiment_name: str = "bert_attention_comparison"
    description: str = "BERT attention mechanism comparison"
    version: str = "1.0.0"
    tags: List[str] = None
    
    # Sub-configurations
    training: TrainingConfig = None
    model: ModelConfig = None
    data: DataConfig = None
    
    # Attention mechanisms to compare
    attention_algorithms: List[str] = None
    
    # Training objectives to compare
    training_objectives: List[str] = None  # ["mlm", "clm", "both"]
    
    # Output and logging
    output_dir: str = "./outputs"
    log_level: str = "INFO"
    save_config: bool = True
    
    # Reproducibility
    seed: int = 42
    deterministic: bool = True
    
    # Resource management
    device: str = "auto"  # auto, cpu, cuda, mps
    mixed_precision: bool = True
    gradient_checkpointing: bool = False
    memory_efficient_attention: bool = True
    
    # Monitoring
    monitor_gpu: bool = True
    monitor_memory: bool = True
    profile_performance: bool = False
    
    # Experiment tracking
    use_wandb: bool = False
    wandb_project: str = "bert-attention-comparison"
    wandb_entity: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    
    # Evaluation configuration
    evaluation: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        super().__post_init__()
        
        # Initialize sub-configs if not provided
        if self.training is None:
            self.training = TrainingConfig()
        if self.model is None:
            self.model = ModelConfig()
        if self.data is None:
            self.data = DataConfig()
        
        # Set default attention algorithms
        if self.attention_algorithms is None:
            self.attention_algorithms = ["standard", "rope", "exposb", "absolute"]
        
        # Set default training objectives
        if self.training_objectives is None:
            self.training_objectives = ["mlm", "clm"]
        
        # Set default tags
        if self.tags is None:
            self.tags = ["bert", "attention", "comparison"]
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Validate configuration
        self._validate()
    
    def _validate(self):
        """Validate configuration consistency"""
        super()._validate()
        
        # Validate attention algorithms
        valid_algorithms = {"standard", "rope", "exposb", "absolute"}
        if self.attention_algorithms is not None:
            for algo in self.attention_algorithms:
                if algo not in valid_algorithms:
                    raise ValueError(f"Unknown attention algorithm: {algo}. "
                                   f"Valid options: {valid_algorithms}")
        
        # Validate training objectives
        valid_objectives = {"mlm", "clm", "both"}
        if self.training_objectives is not None:
            for obj in self.training_objectives:
                if obj not in valid_objectives:
                    raise ValueError(f"Unknown training objective: {obj}. "
                                   f"Valid options: {valid_objectives}")
        
        # Validate device
        if self.device not in {"auto", "cpu", "cuda", "mps"}:
            raise ValueError(f"Invalid device: {self.device}")
        
        # Validate log level
        valid_levels = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if self.log_level.upper() not in valid_levels:
            raise ValueError(f"Invalid log level: {self.log_level}")
    
    @classmethod
    def from_env_file(cls, env_file: Union[str, Path] = None) -> 'ExperimentConfig':
        """Load configuration from environment file"""
        if env_file is None:
            env_file = Path.cwd() / "config.env"
        
        env_file = Path(env_file)
        if not env_file.exists():
            logger.warning(f"Environment file {env_file} not found, using defaults")
            return cls()
        
        # Load environment variables from file
        env_vars = {}
        with open(env_file, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith('#') and '=' in line:
                    key, value = line.split('=', 1)
                    env_vars[key.strip()] = value.strip()
        
        # Parse configuration from environment variables
        config_dict = {}
        
        # Parse attention algorithms
        if 'ATTENTION_ALGORITHMS' in env_vars:
            config_dict['attention_algorithms'] = [
                algo.strip() for algo in env_vars['ATTENTION_ALGORITHMS'].split(',')
            ]
        
        # Parse simple fields
        simple_fields = {
            'experiment_name': 'EXPERIMENT_NAME',
            'output_dir': 'OUTPUT_DIR', 
            'seed': 'SEED',
            'device': 'DEVICE',
            'log_level': 'LOG_LEVEL'
        }
        
        for field, env_key in simple_fields.items():
            if env_key in env_vars:
                value = env_vars[env_key]
                # Type conversion
                if field == 'seed':
                    value = int(value)
                elif field in ['use_wandb', 'mixed_precision', 'deterministic']:
                    value = value.lower() in ('true', '1', 'yes', 'on')
                config_dict[field] = value
        
        # Create sub-configurations from environment
        config_dict['training'] = TrainingConfig.from_env_dict(env_vars)
        config_dict['model'] = ModelConfig.from_env_dict(env_vars)
        config_dict['data'] = DataConfig.from_env_dict(env_vars)
        
        return cls(**config_dict)
    
    @classmethod
    def from_yaml(cls, yaml_file: Union[str, Path]) -> 'ExperimentConfig':
        """Load configuration from YAML file"""
        yaml_file = Path(yaml_file)
        with open(yaml_file, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_json(cls, json_file: Union[str, Path]) -> 'ExperimentConfig':
        """Load configuration from JSON file"""
        json_file = Path(json_file)
        with open(json_file, 'r') as f:
            config_dict = json.load(f)
        
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'ExperimentConfig':
        """Create configuration from dictionary"""
        # Handle nested configurations
        if 'training' in config_dict and isinstance(config_dict['training'], dict):
            config_dict['training'] = TrainingConfig(**config_dict['training'])
        
        if 'model' in config_dict and isinstance(config_dict['model'], dict):
            config_dict['model'] = ModelConfig(**config_dict['model'])
        
        if 'data' in config_dict and isinstance(config_dict['data'], dict):
            config_dict['data'] = DataConfig(**config_dict['data'])
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        config_dict = asdict(self)
        
        # Convert sub-configs to dicts
        if self.training:
            config_dict['training'] = asdict(self.training)
        if self.model:
            config_dict['model'] = asdict(self.model)  
        if self.data:
            config_dict['data'] = asdict(self.data)
        
        return config_dict
    
    def save_yaml(self, file_path: Union[str, Path]):
        """Save configuration to YAML file"""
        file_path = Path(file_path)
        with open(file_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
        logger.info(f"Configuration saved to {file_path}")
    
    def save_json(self, file_path: Union[str, Path]):
        """Save configuration to JSON file"""
        file_path = Path(file_path)
        with open(file_path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        logger.info(f"Configuration saved to {file_path}")
    
    def get_model_config_for_attention(self, attention_type: str) -> ModelConfig:
        """Get model configuration for specific attention type"""
        model_config = ModelConfig(**asdict(self.model))
        model_config.attention_type = attention_type
        return model_config
    
    def get_experiment_dir(self, attention_type: str = None) -> Path:
        """Get experiment directory for specific attention type"""
        base_dir = Path(self.output_dir) / self.experiment_name
        if attention_type:
            return base_dir / attention_type
        return base_dir
    
    def setup_logging(self):
        """Setup logging configuration"""
        log_level = getattr(logging, self.log_level.upper())
        
        # Create logs directory
        log_dir = Path(self.output_dir) / "logs"
        log_dir.mkdir(exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_dir / f"{self.experiment_name}.log"),
                logging.StreamHandler()
            ]
        )
    
    def setup_reproducibility(self):
        """Setup reproducibility settings"""
        import random
        import numpy as np
        import torch
        
        # Set seeds
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)
            torch.cuda.manual_seed_all(self.seed)
        
        # Set deterministic behavior
        if self.deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            os.environ['PYTHONHASHSEED'] = str(self.seed)
    
    def print_config(self):
        """Print configuration summary"""
        print("=" * 80)
        print(f"Experiment Configuration: {self.experiment_name}")
        print("=" * 80)
        print(f"Description: {self.description}")
        print(f"Version: {self.version}")
        print(f"Tags: {', '.join(self.tags)}")
        print(f"Attention Algorithms: {', '.join(self.attention_algorithms)}")
        print(f"Output Directory: {self.output_dir}")
        print(f"Device: {self.device}")
        print(f"Mixed Precision: {self.mixed_precision}")
        print(f"Seed: {self.seed}")
        print("\nTraining Configuration:")
        print(f"  Epochs: {self.training.num_epochs}")
        print(f"  Batch Size: {self.training.batch_size}")
        print(f"  Learning Rate: {self.training.learning_rate}")
        print(f"  Warmup Steps: {self.training.warmup_steps}")
        print("\nModel Configuration:")
        print(f"  Hidden Size: {self.model.hidden_size}")
        print(f"  Num Layers: {self.model.num_hidden_layers}")
        print(f"  Num Heads: {self.model.num_attention_heads}")
        print(f"  Max Position: {self.model.max_position_embeddings}")
        print("\nData Configuration:")
        print(f"  Max Sequence Length: {self.data.max_seq_length}")
        print(f"  MLM Probability: {self.data.mlm_probability}")
        print(f"  MLM Strategy: {self.data.mlm_strategy}")
        print("=" * 80)


def load_config(config_path: Union[str, Path] = None) -> ExperimentConfig:
    """
    Load configuration from file or environment
    
    Args:
        config_path: Path to config file (YAML/JSON) or environment file
    
    Returns:
        ExperimentConfig instance
    """
    if config_path is None:
        # Try to find config file automatically
        config_files = [
            "config.yaml", "config.yml", "config.json", "config.env"
        ]
        
        for config_file in config_files:
            if Path(config_file).exists():
                config_path = config_file
                break
        
        if config_path is None:
            logger.info("No config file found, using defaults")
            return ExperimentConfig()
    
    config_path = Path(config_path)
    
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        return ExperimentConfig.from_yaml(config_path)
    elif config_path.suffix.lower() == '.json':
        return ExperimentConfig.from_json(config_path)
    elif config_path.suffix.lower() == '.env':
        return ExperimentConfig.from_env_file(config_path)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")


def save_config(config: ExperimentConfig, 
               file_path: Union[str, Path],
               format: str = "auto") -> None:
    """
    Save configuration to file
    
    Args:
        config: Configuration to save
        file_path: Output file path
        format: Output format (auto, yaml, json)
    """
    file_path = Path(file_path)
    
    if format == "auto":
        format = file_path.suffix.lower().lstrip('.')
    
    if format in ['yaml', 'yml']:
        config.save_yaml(file_path)
    elif format == 'json':
        config.save_json(file_path)
    else:
        raise ValueError(f"Unsupported format: {format}")


# Example usage
if __name__ == "__main__":
    # Example 1: Create default configuration
    config = ExperimentConfig()
    config.print_config()
    
    # Example 2: Save configuration
    config.save_yaml("example_config.yaml")
    config.save_json("example_config.json")
    
    # Example 3: Load from environment file
    try:
        env_config = ExperimentConfig.from_env_file("config.env")
        print("\nLoaded from environment file:")
        env_config.print_config()
    except FileNotFoundError:
        print("No config.env file found")
    
    # Example 4: Create custom configuration
    custom_config = ExperimentConfig(
        experiment_name="custom_experiment",
        attention_algorithms=["standard", "rope"],
        output_dir="./custom_outputs"
    )
    
    print("\nCustom configuration:")
    custom_config.print_config()