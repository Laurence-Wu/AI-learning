"""
BERT Trainer Implementation
==========================

Main training loop for BERT models with different attention mechanisms.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
import logging
from typing import Optional, Dict, Any
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


@dataclass
class TrainingResults:
    """Container for training results"""
    train_losses: list
    val_losses: list
    learning_rates: list
    training_time: float
    best_val_loss: float
    best_epoch: int


class BERTTrainer:
    """
    Trainer for BERT models with MLM objective
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler],
        config: Any,
        device: torch.device,
        experiment_dir: Path = None
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.device = device
        self.experiment_dir = Path(experiment_dir) if experiment_dir else Path("./outputs")
        
        # Create experiment directory
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Training state
        self.global_step = 0
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        
        # Results tracking
        self.train_losses = []
        self.val_losses = []
        self.learning_rates = []
        
        # Mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if hasattr(config, 'fp16') and config.fp16 else None
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        num_batches = 0
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            if self.scaler:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    loss = outputs.loss / self.config.gradient_accumulation_steps
            else:
                outputs = self.model(**batch)
                loss = outputs.loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.scaler:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                # Gradient clipping
                if hasattr(self.config, 'max_grad_norm'):
                    if self.scaler:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # Optimizer step
                if self.scaler:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                if self.scheduler:
                    self.scheduler.step()
                
                self.optimizer.zero_grad()
                self.global_step += 1
                
                # Logging
                if self.global_step % self.config.logging_steps == 0:
                    current_lr = self.scheduler.get_last_lr()[0] if self.scheduler else self.config.learning_rate
                    # Minimal logging - only epoch progress
                    self.train_losses.append(loss.item() * self.config.gradient_accumulation_steps)
                    self.learning_rates.append(current_lr)
            
            epoch_loss += loss.item()
            num_batches += 1
        
        return epoch_loss / num_batches if num_batches > 0 else 0
    
    def evaluate(self):
        """Evaluate the model"""
        if not self.val_loader:
            return None
        
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                if self.scaler:
                    with torch.cuda.amp.autocast():
                        outputs = self.model(**batch)
                else:
                    outputs = self.model(**batch)
                
                total_loss += outputs.loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        
        # Save best model
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_checkpoint("best_model.pt")
        
        self.model.train()
        return avg_loss
    
    def save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint_path = self.experiment_dir / filename
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'global_step': self.global_step,
            'epoch': self.current_epoch,
            'best_val_loss': self.best_val_loss
        }, checkpoint_path)
        # Checkpoint saved silently
    
    def train(self) -> TrainingResults:
        """Full training loop"""
        print(f"Starting training for {self.config.num_epochs} epochs")
        start_time = time.time()
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            print(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Train
            avg_train_loss = self.train_epoch()
            print(f"Training loss: {avg_train_loss:.4f}")
            
            # Evaluate
            if self.val_loader and (epoch + 1) % max(1, self.config.eval_steps // len(self.train_loader)) == 0:
                val_loss = self.evaluate()
                self.val_losses.append(val_loss)
                print(f"Validation loss: {val_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % max(1, self.config.save_steps // len(self.train_loader)) == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}.pt")
        
        training_time = time.time() - start_time
        
        return TrainingResults(
            train_losses=self.train_losses,
            val_losses=self.val_losses,
            learning_rates=self.learning_rates,
            training_time=training_time,
            best_val_loss=self.best_val_loss,
            best_epoch=self.current_epoch
        )