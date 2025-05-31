import torch
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from typing import Optional
from transformers import (
    get_linear_schedule_with_warmup, 
    get_cosine_schedule_with_warmup, 
    get_cosine_with_hard_restarts_schedule_with_warmup
)

from utils.config.heartwise_config import HeartWiseConfig

def get_scheduler(
    scheduler_name: str, 
    optimizer: Optimizer, 
    num_epochs: int,
    train_dataloader: DataLoader,
    gamma: Optional[float] = 0.3,
    step_size: Optional[int] = 15,
    gradient_accumulation_steps: int = 1,
    num_warmup_percent: Optional[float] = 0.1,
    num_hard_restarts_cycles: Optional[float] = 1.0,          
    warm_restart_tmult: Optional[int] = 2,
    num_restarts: int = 10  # New parameter for cosine_warm_restart
) -> LRScheduler:
    """
    Configures and returns a learning rate scheduler based on the specified name.

    Args:
        scheduler_name (str): Name of the scheduler ('cosine', 'step', 'cosine_warm_restart',
                              'linear_warmup', 'cosine_with_warmup', 
                              'cosine_with_hard_restarts_with_warmup').
        optimizer (torch.optim.Optimizer): Optimizer to attach the scheduler to.
        num_epochs (int): Number of training epochs.
        train_dataloader (torch.utils.data.DataLoader): Training data loader.
        gamma (float, optional): Factor by which to reduce LR for StepLR. Defaults to 0.3.
        step_size (int, optional): Number of epochs between LR reductions for StepLR. Defaults to 15.
        gradient_accumulation_steps (int, optional): Number of gradient accumulation steps. Defaults to 1.
        num_warmup_percent (float, optional): Percentage of total steps for warmup. Defaults to 0.1.
        num_hard_restarts_cycles (float, optional): Number of cycles for cosine with hard restarts. Defaults to 1.0.
        warm_restart_tmult (int, optional): T_mult factor for cosine warm restarts. Defaults to 2.
        num_restarts (int, optional): Number of desired restarts for cosine_warm_restart. Defaults to 10.

    Returns:
        torch.optim.lr_scheduler.LRScheduler: Configured learning rate scheduler.
    """
    # Handle None values by using defaults
    gamma = gamma if gamma is not None else 0.3
    step_size = step_size if step_size is not None else 15
    num_warmup_percent = num_warmup_percent if num_warmup_percent is not None else 0.1
    num_hard_restarts_cycles = num_hard_restarts_cycles if num_hard_restarts_cycles is not None else 1.0
    warm_restart_tmult = warm_restart_tmult if warm_restart_tmult is not None else 2
    
    # Compute total number of optimizer update steps
    t_total = (len(train_dataloader) * num_epochs) // gradient_accumulation_steps

    if scheduler_name == "cosine":
        # CosineAnnealingLR: LR follows a cosine annealing schedule over t_total steps
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, 
            T_max=t_total  # One full cosine cycle across all training steps
        )

    elif scheduler_name == "step":
        # StepLR: Reduces LR by factor every step_size epochs
        return torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=step_size, 
            gamma=gamma
        )

    elif scheduler_name == 'cosine_warm_restart':
        # CosineAnnealingWarmRestarts: Cosine annealing with periodic restarts
        # Compute T_0 (steps until first restart) based on desired number of restarts
        T_0 = max(1, t_total // num_restarts) if t_total > num_restarts else t_total
        print(f"[cosine_warm_restart] t_total={t_total}, T_0={T_0}, T_mult={warm_restart_tmult}")
        return torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=T_0,  # Steps until first restart
            T_mult=warm_restart_tmult,  # Multiplier for subsequent restart periods
            eta_min=0,
            last_epoch=-1
        )

    elif scheduler_name == 'linear_warmup':
        # Linear warmup followed by linear decay
        num_warmup_steps = int(t_total * num_warmup_percent)
        print(f"[linear_warmup] t_total={t_total}, num_warmup_steps={num_warmup_steps}")
        return get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=t_total
        )

    elif scheduler_name == 'cosine_with_warmup':
        # Linear warmup followed by cosine decay
        num_warmup_steps = int(t_total * num_warmup_percent)
        print(f"[cosine_with_warmup] t_total={t_total}, num_warmup_steps={num_warmup_steps}")
        return get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=t_total
        )

    elif scheduler_name == 'cosine_with_hard_restarts_with_warmup':
        # Linear warmup followed by cosine annealing with hard restarts
        num_warmup_steps = int(t_total * num_warmup_percent)
        print(f"[cosine_with_hard_restarts] t_total={t_total}, num_warmup_steps={num_warmup_steps}, "
              f"num_hard_restarts_cycles={num_hard_restarts_cycles}")
        return get_cosine_with_hard_restarts_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=t_total,
            num_cycles=int(num_hard_restarts_cycles)  # Number of restart cycles
        )

    else:
        raise ValueError(f"Scheduler {scheduler_name} not found")

def scheduler_is_per_iteration(config: HeartWiseConfig) -> bool:
    """
    Returns True if the scheduler expects a call to .step() per iteration (batch).
    Returns False for schedulers that should be updated per epoch.
    
    Most modern schedulers work best with per-iteration updates, with StepLR 
    being the main exception that works on an epoch basis.
    """
    sched_name = getattr(config, "scheduler_name", "").lower()
    
    # Only StepLR should be updated per-epoch in our implementation
    PER_EPOCH_SCHEDULERS = ["step"]
    
    # All other scheduler types should be updated per-iteration (batch)
    return sched_name not in PER_EPOCH_SCHEDULERS