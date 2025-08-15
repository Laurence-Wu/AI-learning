"""
Integrated BERT Pretraining Comparison: Multiple Attention Mechanisms
Compares Standard, RoPE, ExpoSB, and Absolute Positional Encoding
All implemented with Triton for fair comparison
"""

import os
import json
import time
import math
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
from attention_implementation.triton_standard_attention import StandardBERTAttention
from attention_implementation.triton_rope_attention import RoPEBERTAttention
from attention_implementation.triton_exposb_attention import ExpoSBBERTAttention
from attention_implementation.triton_absolute_attention import AbsoluteBERTAttention
from data_preprocessing import load_training_data


class BERTDataset(Dataset):
    """Simple dataset for BERT pretraining"""
    
    def __init__(self, data, tokenizer, max_length=512, mlm_probability=0.15):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_probability = mlm_probability
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        text = self.data[idx]
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        input_ids = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        # Create MLM labels
        labels = input_ids.clone()
        
        # Create mask for MLM
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        special_tokens_mask = self.tokenizer.get_special_tokens_mask(
            labels.tolist(), already_has_special_tokens=True
        )
        probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)
        
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens
        
        # 80% of the time, replace masked input tokens with [MASK]
        indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.convert_tokens_to_ids('[MASK]')
        
        # 10% of the time, replace with random word
        indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(len(self.tokenizer), labels.shape, dtype=torch.long)
        input_ids[indices_random] = random_words[indices_random]
        
        # The rest of the time (10%), keep the masked input tokens unchanged
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }


class ModifiedBERTModel(BertForMaskedLM):
    """Modified BERT model that can use different attention mechanisms"""
    
    def __init__(self, config, attention_type="standard"):
        super().__init__(config)
        self.attention_type = attention_type
        
        # Replace all attention layers with our custom implementation
        for layer in self.bert.encoder.layer:
            if attention_type == "standard":
                layer.attention.self = StandardBERTAttention(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    max_position_embeddings=config.max_position_embeddings,
                    dropout=config.attention_probs_dropout_prob
                )
            elif attention_type == "rope":
                layer.attention.self = RoPEBERTAttention(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    max_position_embeddings=config.max_position_embeddings,
                    dropout=config.attention_probs_dropout_prob
                )
            elif attention_type == "exposb":
                layer.attention.self = ExpoSBBERTAttention(
                    hidden_size=config.hidden_size,
                    num_heads=config.num_attention_heads,
                    max_position_embeddings=config.max_position_embeddings,
                    dropout=config.attention_probs_dropout_prob
                )
            elif attention_type == "absolute":
                layer.attention.self = AbsoluteBERTAttention(config)
            else:
                raise ValueError(f"Unknown attention type: {attention_type}")


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    config: BERTComparisonConfig,
    attention_type: str,
    device: torch.device
) -> Dict[str, List[float]]:
    """Train a single model and return training history"""
    
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate)
    
    # Learning rate scheduler
    total_steps = len(train_loader) * config.num_epochs
    warmup_steps = config.warmup_steps
    
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler() if config.fp16 and torch.cuda.is_available() else None
    
    history = {
        'train_loss': [],
        'val_loss': [],
        'learning_rate': []
    }
    
    print(f"\nTraining {attention_type} model...")
    global_step = 0
    
    for epoch in range(config.num_epochs):
        # Training phase
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Forward pass with mixed precision
            if scaler:
                with torch.cuda.amp.autocast():
                    outputs = model(**batch)
                    loss = outputs.loss / config.gradient_accumulation_steps
                
                scaler.scale(loss).backward()
                
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    scheduler.step()
            else:
                outputs = model(**batch)
                loss = outputs.loss / config.gradient_accumulation_steps
                loss.backward()
                
                if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
            
            epoch_loss += loss.item() * config.gradient_accumulation_steps
            batch_count += 1
            global_step += 1
            
            # Logging
            if global_step % config.logging_steps == 0:
                avg_loss = epoch_loss / batch_count
                lr = scheduler.get_last_lr()[0]
                print(f"  [{attention_type}] Epoch {epoch+1}/{config.num_epochs}, "
                      f"Step {global_step}, Loss: {avg_loss:.4f}, LR: {lr:.6f}")
        
        # Validation phase
        if val_loader and (epoch + 1) % config.eval_steps == 0:
            model.eval()
            val_loss = 0
            val_count = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    outputs = model(**batch)
                    val_loss += outputs.loss.item()
                    val_count += 1
            
            avg_val_loss = val_loss / val_count if val_count > 0 else 0
            history['val_loss'].append(avg_val_loss)
            print(f"  [{attention_type}] Validation Loss: {avg_val_loss:.4f}")
        
        # Record history
        avg_train_loss = epoch_loss / batch_count if batch_count > 0 else 0
        history['train_loss'].append(avg_train_loss)
        history['learning_rate'].append(scheduler.get_last_lr()[0])
        
        # Save checkpoint
        if (epoch + 1) % config.save_steps == 0:
            save_path = os.path.join(config.output_dir, f'{attention_type}_checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': avg_train_loss,
            }, save_path)
            print(f"  [{attention_type}] Checkpoint saved to {save_path}")
    
    return history


def plot_comparison(histories: Dict[str, Dict[str, List[float]]], config: BERTComparisonConfig):
    """Plot training comparison between different attention mechanisms"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Training Loss
    ax = axes[0, 0]
    for name, history in histories.items():
        ax.plot(history['train_loss'], label=f'{name.upper()} Train Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Validation Loss
    ax = axes[0, 1]
    for name, history in histories.items():
        if history['val_loss']:
            ax.plot(history['val_loss'], label=f'{name.upper()} Val Loss', linewidth=2)
    ax.set_xlabel('Evaluation Step')
    ax.set_ylabel('Loss')
    ax.set_title('Validation Loss Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Learning Rate Schedule
    ax = axes[1, 0]
    for name, history in histories.items():
        ax.plot(history['learning_rate'], label=f'{name.upper()} LR', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Final Loss Comparison (Bar Chart)
    ax = axes[1, 1]
    final_train_losses = {name: history['train_loss'][-1] for name, history in histories.items()}
    final_val_losses = {name: history['val_loss'][-1] if history['val_loss'] else 0 
                       for name, history in histories.items()}
    
    x = np.arange(len(histories))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, list(final_train_losses.values()), width, label='Final Train Loss')
    bars2 = ax.bar(x + width/2, list(final_val_losses.values()), width, label='Final Val Loss')
    
    ax.set_xlabel('Attention Type')
    ax.set_ylabel('Loss')
    ax.set_title('Final Loss Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([name.upper() for name in histories.keys()])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.4f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=8)
    
    plt.tight_layout()
    plt.savefig(config.plot_save_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\nComparison plot saved to {config.plot_save_path}")


def main():
    """Main training loop for integrated comparison"""
    
    # Load configuration
    config = BERTComparisonConfig.from_env()
    
    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    
    # Set device
    if config.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.device)
    
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(config.output_dir, exist_ok=True)
    
    # Load tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = BertTokenizer.from_pretrained('./local_tokenizer/')
    except:
        print("Local tokenizer not found. Using default BERT tokenizer.")
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    # Load data
    print("Loading training data...")
    all_texts = load_training_data(config.training_data_file)
    
    # Split into train and validation sets (80-20 split)
    split_idx = int(len(all_texts) * 0.8)
    train_texts = all_texts[:split_idx]
    val_texts = all_texts[split_idx:]
    
    if not train_texts:
        print("No training data found. Creating sample data...")
        train_texts = [
            "The quick brown fox jumps over the lazy dog.",
            "Machine learning is transforming the world of technology.",
            "Natural language processing enables computers to understand human language.",
        ] * 100
        val_texts = train_texts[:20]
    
    # Create datasets
    train_dataset = BERTDataset(
        train_texts, 
        tokenizer, 
        max_length=config.max_seq_length,
        mlm_probability=config.mlm_probability
    )
    
    val_dataset = BERTDataset(
        val_texts,
        tokenizer,
        max_length=config.max_seq_length,
        mlm_probability=config.mlm_probability
    ) if val_texts else None
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=0
    ) if val_dataset else None
    
    # Create BERT configuration
    bert_config = BertConfig(
        vocab_size=config.vocab_size,
        hidden_size=config.hidden_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        intermediate_size=config.intermediate_size,
        max_position_embeddings=config.max_position_embeddings,
        attention_probs_dropout_prob=0.1,
        hidden_dropout_prob=0.1,
    )
    
    # Parse attention algorithms from config
    attention_algorithms = [algo.strip() for algo in config.attention_algorithms.split(',')]
    
    # Train models with different attention mechanisms
    histories = {}
    
    for attention_type in attention_algorithms:
        print(f"\n{'='*60}")
        print(f"Training {attention_type.upper()} Attention Model")
        print(f"{'='*60}")
        
        # Create model
        model = ModifiedBERTModel(bert_config, attention_type=attention_type)
        
        # Train model
        history = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            attention_type=attention_type,
            device=device
        )
        
        histories[attention_type] = history
        
        # Save final model
        model_save_path = os.path.join(config.output_dir, f'bert_{attention_type}_final.pt')
        torch.save(model.state_dict(), model_save_path)
        print(f"Final {attention_type} model saved to {model_save_path}")
        
        # Clear memory
        del model
        torch.cuda.empty_cache()
    
    # Plot comparison
    print("\n" + "="*60)
    print("Generating Comparison Plots")
    print("="*60)
    plot_comparison(histories, config)
    
    # Save training histories
    histories_path = os.path.join(config.output_dir, 'training_histories.json')
    with open(histories_path, 'w') as f:
        json.dump(histories, f, indent=2)
    print(f"Training histories saved to {histories_path}")
    
    # Print final comparison
    print("\n" + "="*60)
    print("FINAL RESULTS COMPARISON")
    print("="*60)
    
    for attention_type in attention_algorithms:
        history = histories[attention_type]
        final_train_loss = history['train_loss'][-1]
        final_val_loss = history['val_loss'][-1] if history['val_loss'] else None
        
        print(f"\n{attention_type.upper()} Attention:")
        print(f"  Final Training Loss: {final_train_loss:.4f}")
        if final_val_loss:
            print(f"  Final Validation Loss: {final_val_loss:.4f}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)


if __name__ == "__main__":
    main()