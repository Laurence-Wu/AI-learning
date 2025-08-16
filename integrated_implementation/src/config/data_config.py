"""
Data Configuration
=================

Configuration for data processing and MLM strategies.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from .base_config import BaseConfig


@dataclass
class DataConfig(BaseConfig):
    """Data processing configuration"""
    # Data files
    training_data_file: str = "./training_data/processed_training_data.txt"
    validation_split: float = 0.1
    max_examples: Optional[int] = None
    
    # Sequence processing
    max_seq_length: int = 512
    truncation: bool = True
    padding: str = "max_length"
    
    # MLM configuration
    mlm_strategy: str = "standard"
    mlm_probability: float = 0.15
    
    # Preprocessing
    min_chars: int = 20
    max_chars: int = 10000
    deduplicate: bool = True
    
    def get_preprocessing_config(self):
        """Get preprocessing configuration object"""
        from ..data.preprocessing import PreprocessingConfig
        return PreprocessingConfig(
            min_chars=self.min_chars,
            max_chars=self.max_chars,
            deduplicate=self.deduplicate
        )
    
    @classmethod
    def from_env_dict(cls, env_dict: Dict[str, str]) -> 'DataConfig':
        """Create from environment variables"""
        config = {}
        
        # Parse data fields
        field_mapping = {
            'MAX_SEQ_LENGTH': 'max_seq_length',
            'MLM_PROBABILITY': 'mlm_probability',
            'MLM_STRATEGY': 'mlm_strategy',
            'TRAINING_DATA_FILE': 'training_data_file'
        }
        
        for env_key, field_name in field_mapping.items():
            if env_key in env_dict:
                value = env_dict[env_key]
                if field_name == 'mlm_probability':
                    config[field_name] = float(value)
                elif field_name == 'max_seq_length':
                    config[field_name] = int(value)
                else:
                    config[field_name] = value
        
        return cls(**config)