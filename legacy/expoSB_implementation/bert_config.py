"""
Configuration management for BERT Comparison Training
Uses environment variables loaded from config.env file
"""

import os
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv


def str_to_bool(value: str) -> bool:
    """Convert string to boolean"""
    return value.lower() in ('true', '1', 'yes', 'on')


@dataclass
class BERTComparisonConfig:
    """Configuration class for BERT comparison training"""
    
    # Model Configuration
    vocab_size: int = 30522
    hidden_size: int = 768
    num_hidden_layers: int = 6
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    
    # Training Configuration
    batch_size: int = 32
    learning_rate: float = 5e-5
    num_epochs: int = 50
    max_seq_length: int = 512
    mlm_probability: float = 0.15
    warmup_steps: int = 50
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 1000
    gradient_accumulation_steps: int = 4
    
    # Training Options
    fp16: bool = False
    seed: int = 42
    
    # Data Configuration
    training_data_file: str = "training_data.txt"
    train_split: float = 0.8
    
    # Output Configuration
    output_dir: str = "./bert_comparison_outputs"
    plot_save_path: str = "bert_comparison.png"
    standard_model_save_path: str = "bert_standard_attention.pt"
    exposb_model_save_path: str = "bert_exposb_attention.pt"
    
    # Device Configuration
    device: str = "auto"  # auto, cuda, cpu
    
    @classmethod
    def from_env(cls, env_file: str = "config.env") -> 'BERTComparisonConfig':
        """Load configuration from environment file"""
        # Load environment variables from file
        load_dotenv(env_file)
        
        return cls(
            # Model Configuration
            vocab_size=int(os.getenv('VOCAB_SIZE', 30522)),
            hidden_size=int(os.getenv('HIDDEN_SIZE', 768)),
            num_hidden_layers=int(os.getenv('NUM_HIDDEN_LAYERS', 6)),
            num_attention_heads=int(os.getenv('NUM_ATTENTION_HEADS', 12)),
            intermediate_size=int(os.getenv('INTERMEDIATE_SIZE', 3072)),
            max_position_embeddings=int(os.getenv('MAX_POSITION_EMBEDDINGS', 512)),
            
            # Training Configuration
            batch_size=int(os.getenv('BATCH_SIZE', 32)),
            learning_rate=float(os.getenv('LEARNING_RATE', 5e-5)),
            num_epochs=int(os.getenv('NUM_EPOCHS', 50)),
            max_seq_length=int(os.getenv('MAX_SEQ_LENGTH', 512)),
            mlm_probability=float(os.getenv('MLM_PROBABILITY', 0.15)),
            warmup_steps=int(os.getenv('WARMUP_STEPS', 50)),
            logging_steps=int(os.getenv('LOGGING_STEPS', 10)),
            eval_steps=int(os.getenv('EVAL_STEPS', 50)),
            save_steps=int(os.getenv('SAVE_STEPS', 1000)),
            gradient_accumulation_steps=int(os.getenv('GRADIENT_ACCUMULATION_STEPS', 4)),
            
            # Training Options
            fp16=str_to_bool(os.getenv('FP16', 'false')),
            seed=int(os.getenv('SEED', 42)),
            
            # Data Configuration
            training_data_file=os.getenv('TRAINING_DATA_FILE', 'training_data.txt'),
            train_split=float(os.getenv('TRAIN_SPLIT', 0.8)),
            
            # Output Configuration
            output_dir=os.getenv('OUTPUT_DIR', './bert_comparison_outputs'),
            plot_save_path=os.getenv('PLOT_SAVE_PATH', 'bert_comparison.png'),
            standard_model_save_path=os.getenv('STANDARD_MODEL_SAVE_PATH', 'bert_standard_attention.pt'),
            exposb_model_save_path=os.getenv('EXPOSB_MODEL_SAVE_PATH', 'bert_exposb_attention.pt'),
            
            # Device Configuration
            device=os.getenv('DEVICE', 'auto'),
        )
    
    def get_bert_config_dict(self):
        """Get configuration dictionary for BertConfig"""
        return {
            'vocab_size': self.vocab_size,
            'hidden_size': self.hidden_size,
            'num_hidden_layers': self.num_hidden_layers,
            'num_attention_heads': self.num_attention_heads,
            'intermediate_size': self.intermediate_size,
            'max_position_embeddings': self.max_position_embeddings,
            'attention_probs_dropout_prob': 0.1,
            'hidden_dropout_prob': 0.1,
            'type_vocab_size': 2,
        }
    
    def print_config(self):
        """Print current configuration"""
        print("=" * 60)
        print("BERT Comparison Training Configuration")
        print("=" * 60)
        print(f"Model Configuration:")
        print(f"  Hidden Size: {self.hidden_size}")
        print(f"  Number of Layers: {self.num_hidden_layers}")
        print(f"  Number of Attention Heads: {self.num_attention_heads}")
        print(f"  Max Position Embeddings: {self.max_position_embeddings}")
        print(f"\nTraining Configuration:")
        print(f"  Batch Size: {self.batch_size}")
        print(f"  Learning Rate: {self.learning_rate}")
        print(f"  Number of Epochs: {self.num_epochs}")
        print(f"  Max Sequence Length: {self.max_seq_length}")
        print(f"  MLM Probability: {self.mlm_probability}")
        print(f"  Mixed Precision (FP16): {self.fp16}")
        print(f"\nLogging Configuration:")
        print(f"  Logging Steps: {self.logging_steps}")
        print(f"  Evaluation Steps: {self.eval_steps}")
        print(f"  Save Steps: {self.save_steps}")
        print(f"\nData Configuration:")
        print(f"  Training Data File: {self.training_data_file}")
        print(f"  Train/Eval Split: {self.train_split:.1f}/{1-self.train_split:.1f}")
        print(f"\nOutput Configuration:")
        print(f"  Output Directory: {self.output_dir}")
        print(f"  Plot Save Path: {self.plot_save_path}")
        print("=" * 60)


if __name__ == "__main__":
    # Test configuration loading
    config = BERTComparisonConfig.from_env()
    config.print_config()
