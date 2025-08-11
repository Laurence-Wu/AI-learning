"""
BERT Pretraining Comparison: Standard Attention vs RoPE Attention
Both implemented with Triton for fair comparison
"""

import os
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import BertConfig, BertForMaskedLM, BertTokenizer
from transformers.models.bert.modeling_bert import BertSelfAttention
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple
from dataclasses import dataclass
import wandb  # Optional: for experiment tracking

# Import our configuration and attention implementations
from bert_config import BERTComparisonConfig
from triton_standard_attention import StandardBERTAttention
from triton_rope_attention import RoPEBERTAttention


@dataclass
class TrainingConfig:
    """Legacy training configuration - use BERTComparisonConfig instead"""
    model_type: str  # "standard" or "rope"
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
    fp16: bool = False
    seed: int = 42
    output_dir: str = "./bert_comparison_outputs"


class BERTDataset(Dataset):
    """Simple dataset for BERT pretraining"""
    
    def __init__(self, texts: List[str], tokenizer, max_length: int = 256, mlm_probability: float = 0.15):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        
        # Create MLM labels
        labels = input_ids.clone()
        
        # Create random mask
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        # Create special tokens mask with proper shape handling
        special_tokens_mask = torch.isin(labels, torch.tensor(list(self.tokenizer.all_special_ids)))
        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # Only compute loss on masked tokens
        
        # 80% of time, replace with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        # 10% of time, replace with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        # 10% of time, keep original
        
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


class ModifiedBERTModel(nn.Module):
    """
    Modified BERT model that can use either standard or RoPE attention
    """
    
    def __init__(self, config: BertConfig, attention_type: str = "standard"):
        super().__init__()
        self.config = config
        self.attention_type = attention_type
        
        # Create base BERT model
        self.bert = BertForMaskedLM(config)
        
        # Replace attention layers with our Triton implementations
        self._replace_attention_layers()
        
    def _replace_attention_layers(self):
        """Replace all attention layers with Triton implementations"""
        print(f"Replacing {self.config.num_hidden_layers} attention layers with {self.attention_type} attention...")
        
        for layer_idx in range(self.config.num_hidden_layers):
            # Get the attention layer
            layer = self.bert.bert.encoder.layer[layer_idx]
            original_attention = layer.attention.self
            
            # Choose attention implementation
            if self.attention_type == "rope":
                new_attention = RoPEBERTAttention(
                    hidden_size=self.config.hidden_size,
                    num_heads=self.config.num_attention_heads,
                    max_position_embeddings=self.config.max_position_embeddings,
                    dropout=self.config.attention_probs_dropout_prob
                )
            else:  # standard
                new_attention = StandardBERTAttention(
                    hidden_size=self.config.hidden_size,
                    num_heads=self.config.num_attention_heads,
                    max_position_embeddings=self.config.max_position_embeddings,
                    dropout=self.config.attention_probs_dropout_prob
                )
            
            # Replace the self-attention
            layer.attention.self = new_attention
            print(f"  Layer {layer_idx}: {type(original_attention).__name__} -> {type(new_attention).__name__}")
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.bert(input_ids=input_ids, attention_mask=attention_mask, labels=labels)


class Trainer:
    """Trainer for comparing attention mechanisms"""
    
    def __init__(self, model, train_dataloader, eval_dataloader, config: TrainingConfig):
        self.model = model
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.config = config
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
        
        # Learning rate scheduler
        total_steps = len(train_dataloader) * config.num_epochs // config.gradient_accumulation_steps
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=config.warmup_steps
        )
        
        # Training history
        self.history = {
            "train_loss": [],
            "train_mlm_loss": [],
            "eval_loss": [],
            "eval_mlm_loss": [],
            "learning_rate": [],
            "steps": []
        }
        
        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Clear GPU cache before moving model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        self.model.to(self.device)
        
        # Mixed precision training
        self.scaler = torch.amp.GradScaler('cuda') if config.fp16 else None
        
        self.global_step = 0
        
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(self.train_dataloader):
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            with torch.amp.autocast('cuda', enabled=self.config.fp16):
                outputs = self.model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps
                
            # Check for NaN/inf in loss
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected at step {self.global_step}")
                print(f"Loss value: {loss.item()}")
                print(f"Outputs loss: {outputs.loss.item()}")
                # Skip this batch
                continue
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    # Gradient clipping before optimizer step
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    # Gradient clipping
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    self.history["train_loss"].append(loss.item() * self.config.gradient_accumulation_steps)
                    self.history["train_mlm_loss"].append(loss.item() * self.config.gradient_accumulation_steps)
                    self.history["learning_rate"].append(self.scheduler.get_last_lr()[0])
                    self.history["steps"].append(self.global_step)
                    
                    print(f"Step {self.global_step}: Loss = {loss.item():.4f}, LR = {self.scheduler.get_last_lr()[0]:.6f}")
                
                # Evaluation
                if self.global_step % self.config.eval_steps == 0:
                    eval_loss = self.evaluate()
                    self.history["eval_loss"].append(eval_loss)
                    self.history["eval_mlm_loss"].append(eval_loss)
                    print(f"Step {self.global_step}: Eval Loss = {eval_loss:.4f}")
            
            total_loss += loss.item()
        
        return total_loss / len(self.train_dataloader)
    
    def evaluate(self):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in self.eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                with torch.amp.autocast('cuda', enabled=self.config.fp16):
                    outputs = self.model(**batch)
                    loss = outputs.loss
                
                total_loss += loss.item()
        
        self.model.train()
        return total_loss / len(self.eval_dataloader)
    
    def train(self):
        """Full training loop"""
        print(f"Training {self.config.model_type} attention model...")
        print(f"Total steps: {len(self.train_dataloader) * self.config.num_epochs // self.config.gradient_accumulation_steps}")
        
        for epoch in range(self.config.num_epochs):
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            avg_loss = self.train_epoch()
            print(f"Average training loss: {avg_loss:.4f}")
            
            # Evaluate periodically
            if epoch % max(1, self.config.num_epochs // 5) == 0 or epoch == self.config.num_epochs - 1:
                eval_loss = self.evaluate()
                print(f"Evaluation loss: {eval_loss:.4f}")
        
        return self.history


def smooth_data(data, window=5):
    """Apply moving average smoothing to reduce noise in plots"""
    if len(data) < window:
        return data
    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window // 2)
        end = min(len(data), i + window // 2 + 1)
        smoothed.append(np.mean(data[start:end]))
    return smoothed

def plot_comparison(standard_history: Dict, rope_history: Dict, save_path: str = "bert_comparison.png"):
    """Plot comparison graphs with smoothing for clarity"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training MLM Loss with smoothing
    # Plot raw data with low alpha
    axes[0, 0].plot(standard_history["steps"], standard_history["train_mlm_loss"], 
                    color="blue", alpha=0.2, linewidth=0.5)
    axes[0, 0].plot(rope_history["steps"], rope_history["train_mlm_loss"], 
                    color="red", alpha=0.2, linewidth=0.5)
    
    # Plot smoothed data on top
    smoothed_standard = smooth_data(standard_history["train_mlm_loss"], window=10)
    smoothed_rope = smooth_data(rope_history["train_mlm_loss"], window=10)
    axes[0, 0].plot(standard_history["steps"][:len(smoothed_standard)], smoothed_standard, 
                    label="Standard Attention (smoothed)", color="blue", alpha=0.9, linewidth=2)
    axes[0, 0].plot(rope_history["steps"][:len(smoothed_rope)], smoothed_rope, 
                    label="RoPE Attention (smoothed)", color="red", alpha=0.9, linewidth=2)
    axes[0, 0].set_xlabel("Training Steps (K)", fontsize=12)
    axes[0, 0].set_ylabel("MLM Loss", fontsize=12)
    axes[0, 0].set_title("Training MLM Loss Comparison", fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Evaluation MLM Loss - with safety checks and proper step alignment
    if len(standard_history["eval_mlm_loss"]) > 0 and len(standard_history["steps"]) > 0:
        # Create evaluation steps based on eval_steps interval from config
        num_evals = len(standard_history["eval_mlm_loss"])
        if num_evals > 0:
            # Calculate actual evaluation steps
            total_steps = standard_history["steps"][-1] if standard_history["steps"] else 0
            eval_interval = max(1, total_steps // max(1, num_evals))
            eval_steps_standard = [eval_interval * (i + 1) for i in range(num_evals)]
            # Ensure we don't exceed available data
            eval_steps_standard = eval_steps_standard[:len(standard_history["eval_mlm_loss"])]
            
            axes[0, 1].plot(eval_steps_standard, standard_history["eval_mlm_loss"][:len(eval_steps_standard)], 
                            label="Standard Attention", color="blue", marker="o", alpha=0.7)
    
    if len(rope_history["eval_mlm_loss"]) > 0 and len(rope_history["steps"]) > 0:
        # Create evaluation steps based on eval_steps interval from config
        num_evals = len(rope_history["eval_mlm_loss"])
        if num_evals > 0:
            # Calculate actual evaluation steps
            total_steps = rope_history["steps"][-1] if rope_history["steps"] else 0
            eval_interval = max(1, total_steps // max(1, num_evals))
            eval_steps_rope = [eval_interval * (i + 1) for i in range(num_evals)]
            # Ensure we don't exceed available data
            eval_steps_rope = eval_steps_rope[:len(rope_history["eval_mlm_loss"])]
            
            axes[0, 1].plot(eval_steps_rope, rope_history["eval_mlm_loss"][:len(eval_steps_rope)], 
                            label="RoPE Attention", color="red", marker="s", alpha=0.7)
    
    axes[0, 1].set_xlabel("Training Steps (K)", fontsize=12)
    axes[0, 1].set_ylabel("MLM Loss", fontsize=12)
    axes[0, 1].set_title("Evaluation MLM Loss Comparison", fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Language Modeling Loss (same as MLM for BERT) with smoothing
    # Plot raw data with low alpha
    axes[1, 0].plot(standard_history["steps"], standard_history["train_loss"], 
                    color="blue", alpha=0.2, linewidth=0.5)
    axes[1, 0].plot(rope_history["steps"], rope_history["train_loss"], 
                    color="red", alpha=0.2, linewidth=0.5)
    
    # Plot smoothed data on top
    smoothed_standard_lm = smooth_data(standard_history["train_loss"], window=10)
    smoothed_rope_lm = smooth_data(rope_history["train_loss"], window=10)
    axes[1, 0].plot(standard_history["steps"][:len(smoothed_standard_lm)], smoothed_standard_lm, 
                    label="Standard Attention (smoothed)", color="blue", alpha=0.9, linewidth=2)
    axes[1, 0].plot(rope_history["steps"][:len(smoothed_rope_lm)], smoothed_rope_lm, 
                    label="RoPE Attention (smoothed)", color="red", alpha=0.9, linewidth=2)
    axes[1, 0].set_xlabel("Training Steps (K)", fontsize=12)
    axes[1, 0].set_ylabel("LM Loss", fontsize=12)
    axes[1, 0].set_title("Training LM Loss Comparison", fontsize=14)
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Learning Rate Schedule
    axes[1, 1].plot(standard_history["steps"], standard_history["learning_rate"], 
                    label="Learning Rate", color="green", alpha=0.7)
    axes[1, 1].set_xlabel("Training Steps (K)", fontsize=12)
    axes[1, 1].set_ylabel("Learning Rate", fontsize=12)
    axes[1, 1].set_title("Learning Rate Schedule", fontsize=14)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.suptitle("BERT Pretraining: Standard vs RoPE Attention Comparison", fontsize=16)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"Comparison plot saved to {save_path}")


def load_training_data(file_path: str):
    """Load training texts from file"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            texts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(texts)} training texts from {file_path}")
        return texts
    except FileNotFoundError:
        print(f"Warning: {file_path} not found. Using minimal fallback dataset.")
        return [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is a subset of artificial intelligence.",
            "Transformers have revolutionized natural language processing."
        ] * 100


def main():
    """Main training script with configuration management"""
    # Load configuration from environment file
    config = BERTComparisonConfig.from_env("config.env")
    config.print_config()
    
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load tokenizer (try local first, then remote)
    tokenizer_dir = "./local_tokenizer"
    if os.path.exists(tokenizer_dir):
        print("Loading tokenizer from local directory...")
        tokenizer = BertTokenizer.from_pretrained(tokenizer_dir)
    else:
        print("Local tokenizer not found. Downloading from Hugging Face...")
        print("(Run setup_tokenizer.py first to avoid internet dependency)")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Load training data
    sample_texts = load_training_data(config.training_data_file)
    
    # Create datasets with proper train/eval split
    split_idx = int(config.train_split * len(sample_texts))
    train_texts = sample_texts[:split_idx]
    eval_texts = sample_texts[split_idx:]
    
    print(f"Training on {len(train_texts)} texts, evaluating on {len(eval_texts)} texts")
    
    train_dataset = BERTDataset(train_texts, tokenizer, config.max_seq_length, config.mlm_probability)
    eval_dataset = BERTDataset(eval_texts, tokenizer, config.max_seq_length, config.mlm_probability)
    
    # Create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=config.batch_size, shuffle=False)
    
    # BERT configuration from config
    bert_config = BertConfig(**config.get_bert_config_dict())
    
    # Train standard attention model
    print("=" * 50)
    print("Training Standard Attention BERT")
    print("=" * 50)
    
    # Create training config for standard attention from main config
    standard_config = TrainingConfig(
        model_type="standard", 
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        num_epochs=config.num_epochs,
        max_seq_length=config.max_seq_length,
        mlm_probability=config.mlm_probability,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        fp16=config.fp16,
        seed=config.seed,
        output_dir=config.output_dir
    )
    
    standard_model = ModifiedBERTModel(bert_config, attention_type="standard")
    standard_trainer = Trainer(standard_model, train_dataloader, eval_dataloader, standard_config)
    standard_history = standard_trainer.train()
    
    # Save standard model
    torch.save(standard_model.state_dict(), config.standard_model_save_path)
    
    # Clear GPU memory before training RoPE model
    del standard_model
    del standard_trainer
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print("GPU memory cleared between models")
    
    # Train RoPE attention model
    print("\n" + "=" * 50)
    print("Training RoPE Attention BERT")
    print("=" * 50)
    
    # Create training config for RoPE attention from main config
    rope_config = TrainingConfig(
        model_type="rope",
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        num_epochs=config.num_epochs,
        max_seq_length=config.max_seq_length,
        mlm_probability=config.mlm_probability,
        warmup_steps=config.warmup_steps,
        logging_steps=config.logging_steps,
        eval_steps=config.eval_steps,
        save_steps=config.save_steps,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        fp16=config.fp16,
        seed=config.seed,
        output_dir=config.output_dir
    )
    
    rope_model = ModifiedBERTModel(bert_config, attention_type="rope")
    rope_trainer = Trainer(rope_model, train_dataloader, eval_dataloader, rope_config)
    rope_history = rope_trainer.train()
    
    # Save RoPE model
    torch.save(rope_model.state_dict(), config.rope_model_save_path)
    
    # Plot comparison
    plot_comparison(standard_history, rope_history, config.plot_save_path)
    
    # Print final comparison with safety checks
    print("\n" + "=" * 50)
    print("Final Comparison")
    print("=" * 50)
    
    if len(standard_history['train_loss']) > 0 and len(rope_history['train_loss']) > 0:
        print(f"Standard Attention - Final Train Loss: {standard_history['train_loss'][-1]:.4f}")
        print(f"RoPE Attention - Final Train Loss: {rope_history['train_loss'][-1]:.4f}")
        
        if len(standard_history['eval_loss']) > 0 and len(rope_history['eval_loss']) > 0:
            print(f"Standard Attention - Final Eval Loss: {standard_history['eval_loss'][-1]:.4f}")
            print(f"RoPE Attention - Final Eval Loss: {rope_history['eval_loss'][-1]:.4f}")
            
            # Calculate improvement
            train_improvement = (standard_history['train_loss'][-1] - rope_history['train_loss'][-1]) / standard_history['train_loss'][-1] * 100
            eval_improvement = (standard_history['eval_loss'][-1] - rope_history['eval_loss'][-1]) / standard_history['eval_loss'][-1] * 100
            
            print(f"\nRoPE vs Standard:")
            print(f"Training Loss Improvement: {train_improvement:.2f}%")
            print(f"Evaluation Loss Improvement: {eval_improvement:.2f}%")
        else:
            print("Warning: No evaluation loss data available for comparison")
    else:
        print("Warning: No training data available for comparison")


if __name__ == "__main__":
    main()
