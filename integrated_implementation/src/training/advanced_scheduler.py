"""
Advanced Learning Rate Schedulers for BERT Training
===================================================

Implements sophisticated learning rate scheduling strategies including:
- Linear warmup followed by cosine annealing
- Custom warmup strategies
- Learning rate range tests
- Plateau detection and adjustment
"""

import torch
import math
import logging
from torch.optim.lr_scheduler import _LRScheduler
from typing import List, Optional, Union

logger = logging.getLogger(__name__)


class WarmupCosineScheduler(_LRScheduler):
    """
    Learning Rate Scheduler with Linear Warmup followed by Cosine Annealing
    
    This is the standard scheduler used in BERT and most transformer models:
    1. Linear warmup from 0 to max_lr over warmup_steps
    2. Cosine annealing from max_lr to min_lr over remaining steps
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        
        if warmup_steps >= total_steps:
            raise ValueError(f"warmup_steps ({warmup_steps}) should be < total_steps ({total_steps})")
        
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step"""
        if self._step_count <= self.warmup_steps:
            # Linear warmup phase
            warmup_factor = self._step_count / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing phase
            cosine_steps = self.total_steps - self.warmup_steps
            current_cosine_step = self._step_count - self.warmup_steps
            
            cosine_factor = 0.5 * (1 + math.cos(math.pi * current_cosine_step / cosine_steps))
            
            # Scale between min_lr_ratio and 1.0
            lr_factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_factor
            
            return [base_lr * lr_factor for base_lr in self.base_lrs]


class WarmupLinearScheduler(_LRScheduler):
    """
    Learning Rate Scheduler with Linear Warmup followed by Linear Decay
    
    Alternative to cosine annealing - uses linear decay instead
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step"""
        if self._step_count <= self.warmup_steps:
            # Linear warmup phase
            warmup_factor = self._step_count / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Linear decay phase
            decay_steps = self.total_steps - self.warmup_steps
            current_decay_step = self._step_count - self.warmup_steps
            
            # Linear interpolation from 1.0 to min_lr_ratio
            progress = current_decay_step / decay_steps
            lr_factor = 1.0 - progress * (1.0 - self.min_lr_ratio)
            lr_factor = max(lr_factor, self.min_lr_ratio)  # Clamp to minimum
            
            return [base_lr * lr_factor for base_lr in self.base_lrs]


class WarmupConstantScheduler(_LRScheduler):
    """
    Learning Rate Scheduler with Linear Warmup followed by Constant Rate
    
    Useful for fine-tuning where you want stable learning after warmup
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step"""
        if self._step_count <= self.warmup_steps:
            # Linear warmup phase
            warmup_factor = self._step_count / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Constant phase
            return self.base_lrs


class WarmupPolynomialScheduler(_LRScheduler):
    """
    Learning Rate Scheduler with Linear Warmup followed by Polynomial Decay
    
    Uses polynomial decay which can be tuned between linear (power=1) and 
    more aggressive decay (power>1)
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        power: float = 2.0,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1
    ):
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.power = power
        self.min_lr_ratio = min_lr_ratio
        super().__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate for current step"""
        if self._step_count <= self.warmup_steps:
            # Linear warmup phase
            warmup_factor = self._step_count / self.warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Polynomial decay phase
            decay_steps = self.total_steps - self.warmup_steps
            current_decay_step = self._step_count - self.warmup_steps
            
            # Polynomial decay
            progress = current_decay_step / decay_steps
            lr_factor = (1.0 - progress) ** self.power
            lr_factor = lr_factor * (1.0 - self.min_lr_ratio) + self.min_lr_ratio
            
            return [base_lr * lr_factor for base_lr in self.base_lrs]


class AdaptiveWarmupScheduler(_LRScheduler):
    """
    Adaptive Learning Rate Scheduler with Dynamic Warmup
    
    Adjusts warmup duration based on training progress and validation metrics
    """
    
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        initial_warmup_steps: int,
        total_steps: int,
        scheduler_type: str = "cosine",
        patience: int = 5,
        factor: float = 0.5,
        min_lr_ratio: float = 0.0,
        last_epoch: int = -1
    ):
        self.initial_warmup_steps = initial_warmup_steps
        self.current_warmup_steps = initial_warmup_steps
        self.total_steps = total_steps
        self.scheduler_type = scheduler_type
        self.patience = patience
        self.factor = factor
        self.min_lr_ratio = min_lr_ratio
        
        # Tracking for adaptation
        self.val_losses = []
        self.plateau_count = 0
        self.best_val_loss = float('inf')
        
        super().__init__(optimizer, last_epoch)
    
    def step(self, val_loss: Optional[float] = None):
        """Step scheduler and optionally adapt based on validation loss"""
        if val_loss is not None:
            self.val_losses.append(val_loss)
            
            # Check for plateau
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.plateau_count = 0
            else:
                self.plateau_count += 1
            
            # Adapt warmup if plateau detected
            if self.plateau_count >= self.patience and self._step_count < self.current_warmup_steps:
                old_warmup = self.current_warmup_steps
                self.current_warmup_steps = min(
                    int(self.current_warmup_steps * (1 + self.factor)),
                    self.total_steps // 3  # Cap at 1/3 of total steps
                )
                logger.info(f"Extended warmup from {old_warmup} to {self.current_warmup_steps} steps")
                self.plateau_count = 0
        
        super().step()
    
    def get_lr(self) -> List[float]:
        """Calculate learning rate using selected scheduler type"""
        if self._step_count <= self.current_warmup_steps:
            # Linear warmup phase
            warmup_factor = self._step_count / self.current_warmup_steps
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Decay phase based on scheduler type
            if self.scheduler_type == "cosine":
                return self._cosine_decay()
            elif self.scheduler_type == "linear":
                return self._linear_decay()
            else:
                return self.base_lrs  # Constant
    
    def _cosine_decay(self) -> List[float]:
        """Cosine annealing decay"""
        cosine_steps = self.total_steps - self.current_warmup_steps
        current_cosine_step = self._step_count - self.current_warmup_steps
        
        cosine_factor = 0.5 * (1 + math.cos(math.pi * current_cosine_step / cosine_steps))
        lr_factor = self.min_lr_ratio + (1.0 - self.min_lr_ratio) * cosine_factor
        
        return [base_lr * lr_factor for base_lr in self.base_lrs]
    
    def _linear_decay(self) -> List[float]:
        """Linear decay"""
        decay_steps = self.total_steps - self.current_warmup_steps
        current_decay_step = self._step_count - self.current_warmup_steps
        
        progress = current_decay_step / decay_steps
        lr_factor = 1.0 - progress * (1.0 - self.min_lr_ratio)
        lr_factor = max(lr_factor, self.min_lr_ratio)
        
        return [base_lr * lr_factor for base_lr in self.base_lrs]


def get_advanced_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str,
    warmup_steps: int,
    total_steps: int,
    **kwargs
) -> _LRScheduler:
    """
    Factory function to create advanced learning rate schedulers
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ("warmup_cosine", "warmup_linear", etc.)
        warmup_steps: Number of warmup steps
        total_steps: Total training steps
        **kwargs: Additional scheduler-specific parameters
    
    Returns:
        Configured scheduler
    """
    
    min_lr_ratio = kwargs.get('min_lr_ratio', 0.0)
    
    if scheduler_type == "warmup_cosine":
        return WarmupCosineScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio
        )
    
    elif scheduler_type == "warmup_linear":
        return WarmupLinearScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            min_lr_ratio=min_lr_ratio
        )
    
    elif scheduler_type == "warmup_constant":
        return WarmupConstantScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps
        )
    
    elif scheduler_type == "warmup_polynomial":
        power = kwargs.get('power', 2.0)
        return WarmupPolynomialScheduler(
            optimizer=optimizer,
            warmup_steps=warmup_steps,
            total_steps=total_steps,
            power=power,
            min_lr_ratio=min_lr_ratio
        )
    
    elif scheduler_type == "adaptive_warmup":
        return AdaptiveWarmupScheduler(
            optimizer=optimizer,
            initial_warmup_steps=warmup_steps,
            total_steps=total_steps,
            scheduler_type=kwargs.get('decay_type', 'cosine'),
            patience=kwargs.get('patience', 5),
            factor=kwargs.get('factor', 0.5),
            min_lr_ratio=min_lr_ratio
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def calculate_optimal_warmup_steps(
    total_steps: int,
    warmup_ratio: float = 0.1,
    min_warmup: int = 100,
    max_warmup: int = 10000
) -> int:
    """
    Calculate optimal warmup steps based on total training steps
    
    Args:
        total_steps: Total number of training steps
        warmup_ratio: Fraction of total steps to use for warmup
        min_warmup: Minimum warmup steps
        max_warmup: Maximum warmup steps
    
    Returns:
        Optimal warmup steps
    """
    warmup_steps = int(total_steps * warmup_ratio)
    warmup_steps = max(min_warmup, min(warmup_steps, max_warmup))
    
    logger.info(f"Calculated warmup steps: {warmup_steps} (ratio: {warmup_steps/total_steps:.3f})")
    return warmup_steps


def learning_rate_range_test(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    optimizer_class: type,
    start_lr: float = 1e-7,
    end_lr: float = 1e-1,
    num_iterations: int = 100,
    device: str = "cuda"
) -> List[tuple]:
    """
    Perform learning rate range test to find optimal learning rate
    
    Based on Leslie Smith's method from "Cyclical Learning Rates for Training Neural Networks"
    
    Returns:
        List of (learning_rate, loss) tuples
    """
    model.train()
    results = []
    
    # Create fresh optimizer for test
    optimizer = optimizer_class(model.parameters(), lr=start_lr)
    
    # Calculate multiplication factor
    mult_factor = (end_lr / start_lr) ** (1 / num_iterations)
    
    current_lr = start_lr
    dataloader_iter = iter(dataloader)
    
    for i in range(num_iterations):
        try:
            batch = next(dataloader_iter)
        except StopIteration:
            # Reset iterator if we run out of data
            dataloader_iter = iter(dataloader)
            batch = next(dataloader_iter)
        
        # Move batch to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        # Record result
        results.append((current_lr, loss.item()))
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Update learning rate
        current_lr *= mult_factor
        for param_group in optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # Stop if loss explodes
        if loss.item() > 10.0:
            logger.warning(f"Loss exploded at lr={current_lr:.2e}, stopping LR range test")
            break
    
    return results


# Example usage and configuration
def create_bert_scheduler(optimizer, config, num_training_steps):
    """
    Create BERT-style learning rate scheduler
    
    Standard configuration:
    - Linear warmup for 10% of total steps
    - Cosine annealing for remaining 90%
    - Minimum LR is 0% of max LR
    """
    
    # Calculate warmup steps
    warmup_steps = calculate_optimal_warmup_steps(
        total_steps=num_training_steps,
        warmup_ratio=0.1  # 10% warmup
    )
    
    # Create scheduler
    scheduler = get_advanced_scheduler(
        optimizer=optimizer,
        scheduler_type="warmup_cosine",
        warmup_steps=warmup_steps,
        total_steps=num_training_steps,
        min_lr_ratio=0.0  # Decay to 0
    )
    
    logger.info(f"Created BERT scheduler: {warmup_steps} warmup steps, {num_training_steps} total steps")
    return scheduler