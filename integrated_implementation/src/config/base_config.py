"""
Base Configuration Classes
=========================

Foundation for all configuration management in the framework.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import os


class ConfigError(Exception):
    """Configuration related errors"""
    pass


@dataclass
class BaseConfig:
    """Base configuration class with common functionality"""
    
    def __post_init__(self):
        """Post-initialization validation and setup"""
        self._validate()
    
    def _validate(self):
        """Validate configuration values"""
        pass
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]):
        """Create configuration from dictionary"""
        return cls(**config_dict)
    
    @classmethod
    def from_env_dict(cls, env_dict: Dict[str, str]):
        """Create configuration from environment variables"""
        config_dict = {}
        
        # Override with class-specific parsing
        for key, value in env_dict.items():
            # Convert environment variable format to config field
            field_name = key.lower()
            
            # Type conversion based on field
            if value.lower() in ('true', 'false'):
                config_dict[field_name] = value.lower() == 'true'
            elif value.isdigit():
                config_dict[field_name] = int(value)
            elif '.' in value and value.replace('.', '').isdigit():
                config_dict[field_name] = float(value)
            else:
                config_dict[field_name] = value
        
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        from dataclasses import asdict
        return asdict(self)