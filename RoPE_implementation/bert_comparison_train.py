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

# Import our Triton attention implementations
from triton_standard_attention import StandardBERTAttention
from triton_rope_attention import RoPEBERTAttention


@dataclass
class TrainingConfig:
    """Training configuration"""
    model_type: str  # "standard" or "rope"
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 10
    max_seq_length: int = 512
    mlm_probability: float = 0.15
    warmup_steps: int = 1000
    logging_steps: int = 100
    eval_steps: int = 500
    save_steps: int = 1000
    gradient_accumulation_steps: int = 4
    fp16: bool = True
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
        for layer_idx in range(self.config.num_hidden_layers):
            # Get the attention layer
            layer = self.bert.bert.encoder.layer[layer_idx]
            
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
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
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
        
        return self.history


def plot_comparison(standard_history: Dict, rope_history: Dict, save_path: str = "bert_comparison.png"):
    """Plot comparison graphs"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training MLM Loss
    axes[0, 0].plot(standard_history["steps"], standard_history["train_mlm_loss"], 
                    label="Standard Attention", color="blue", alpha=0.7)
    axes[0, 0].plot(rope_history["steps"], rope_history["train_mlm_loss"], 
                    label="RoPE Attention", color="red", alpha=0.7)
    axes[0, 0].set_xlabel("Training Steps (K)", fontsize=12)
    axes[0, 0].set_ylabel("MLM Loss", fontsize=12)
    axes[0, 0].set_title("Training MLM Loss Comparison", fontsize=14)
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Evaluation MLM Loss
    eval_steps_standard = [s for i, s in enumerate(standard_history["steps"]) 
                          if i % (standard_history["steps"][-1] // len(standard_history["eval_mlm_loss"])) == 0][:len(standard_history["eval_mlm_loss"])]
    eval_steps_rope = [s for i, s in enumerate(rope_history["steps"]) 
                       if i % (rope_history["steps"][-1] // len(rope_history["eval_mlm_loss"])) == 0][:len(rope_history["eval_mlm_loss"])]
    
    axes[0, 1].plot(eval_steps_standard, standard_history["eval_mlm_loss"], 
                    label="Standard Attention", color="blue", marker="o", alpha=0.7)
    axes[0, 1].plot(eval_steps_rope, rope_history["eval_mlm_loss"], 
                    label="RoPE Attention", color="red", marker="s", alpha=0.7)
    axes[0, 1].set_xlabel("Training Steps (K)", fontsize=12)
    axes[0, 1].set_ylabel("MLM Loss", fontsize=12)
    axes[0, 1].set_title("Evaluation MLM Loss Comparison", fontsize=14)
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Language Modeling Loss (same as MLM for BERT)
    axes[1, 0].plot(standard_history["steps"], standard_history["train_loss"], 
                    label="Standard Attention", color="blue", alpha=0.7)
    axes[1, 0].plot(rope_history["steps"], rope_history["train_loss"], 
                    label="RoPE Attention", color="red", alpha=0.7)
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


def main():
    """Main training script"""
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Load tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    
    # Create sample dataset (replace with your actual data)
    sample_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Transformers have revolutionized natural language processing.",
        # Add more texts here
    ] * 1000  # Replicate for demonstration
    
    # Create datasets
    train_dataset = BERTDataset(sample_texts[:800], tokenizer)
    eval_dataset = BERTDataset(sample_texts[800:], tokenizer)
    
    # Create dataloaders with smaller batch size for 8GB GPU
    train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=4, shuffle=False)
    
    # BERT configuration - smaller model for 8GB GPU
    bert_config = BertConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=384,  # Reduced from 768
        num_hidden_layers=6,  # Reduced from 12
        num_attention_heads=6,  # Reduced from 12
        intermediate_size=1536,  # Reduced from 3072
        max_position_embeddings=256,  # Reduced from 512
    )
    
    # Train standard attention model
    print("=" * 50)
    print("Training Standard Attention BERT")
    print("=" * 50)
    
    standard_config = TrainingConfig(model_type="standard", num_epochs=5)
    standard_model = ModifiedBERTModel(bert_config, attention_type="standard")
    standard_trainer = Trainer(standard_model, train_dataloader, eval_dataloader, standard_config)
    standard_history = standard_trainer.train()
    
    # Save standard model
    torch.save(standard_model.state_dict(), "bert_standard_attention.pt")
    
    # Train RoPE attention model
    print("\n" + "=" * 50)
    print("Training RoPE Attention BERT")
    print("=" * 50)
    
    rope_config = TrainingConfig(model_type="rope", num_epochs=5)
    rope_model = ModifiedBERTModel(bert_config, attention_type="rope")
    rope_trainer = Trainer(rope_model, train_dataloader, eval_dataloader, rope_config)
    rope_history = rope_trainer.train()
    
    # Save RoPE model
    torch.save(rope_model.state_dict(), "bert_rope_attention.pt")
    
    # Plot comparison
    plot_comparison(standard_history, rope_history)
    
    # Print final comparison
    print("\n" + "=" * 50)
    print("Final Comparison")
    print("=" * 50)
    print(f"Standard Attention - Final Train Loss: {standard_history['train_loss'][-1]:.4f}")
    print(f"Standard Attention - Final Eval Loss: {standard_history['eval_loss'][-1]:.4f}")
    print(f"RoPE Attention - Final Train Loss: {rope_history['train_loss'][-1]:.4f}")
    print(f"RoPE Attention - Final Eval Loss: {rope_history['eval_loss'][-1]:.4f}")
    
    # Calculate improvement
    train_improvement = (standard_history['train_loss'][-1] - rope_history['train_loss'][-1]) / standard_history['train_loss'][-1] * 100
    eval_improvement = (standard_history['eval_loss'][-1] - rope_history['eval_loss'][-1]) / standard_history['eval_loss'][-1] * 100
    
    print(f"\nRoPE vs Standard:")
    print(f"Training Loss Improvement: {train_improvement:.2f}%")
    print(f"Evaluation Loss Improvement: {eval_improvement:.2f}%")


if __name__ == "__main__":
    main()
