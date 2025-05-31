import os
import torch
from typing import Dict, Callable
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer

from utils.enums import RunMode
from utils.wandb_wrapper import WandbWrapper
from utils.config.heartwise_config import HeartWiseConfig


class BaseRunner(ABC):
    """Abstract base class for all runners providing common functionality."""
    
    def __init__(
        self,
        config: HeartWiseConfig,
        wandb_wrapper: WandbWrapper | None = None,
    ):
        self.config = config
        self.wandb_wrapper = wandb_wrapper
    
    def execute(
        self, 
        mode: RunMode
    ):
        """
        Execute the runner in the specified mode.
        
        Args:
            mode: The execution mode (TRAIN, INFERENCE, VALIDATE, EXTRACT_EMBEDDINGS)
        """
        if mode == RunMode.TRAIN:
            self.train()
        elif mode == RunMode.INFERENCE:
            self.inference()
        elif mode == RunMode.VALIDATE:
            self.validate()
        elif mode == RunMode.EXTRACT_EMBEDDINGS:
            self.extract_embeddings()
        else:
            raise ValueError(f"Invalid mode: {mode}")
    
    @abstractmethod
    def _run_epoch(
        self,
        mode: RunMode,
        epoch: int,
        dataloader: DataLoader,
        step_fn: Callable,
    ) -> dict[str, float]:
        """Run an epoch of training or validation."""
        pass
    
    @abstractmethod
    def train(self):
        """Execute training logic."""
        pass
    
    @abstractmethod
    def inference(self):
        """Execute inference logic."""
        pass
    
    def validate(self):
        """Execute validation logic. Default implementation raises NotImplementedError."""
        raise NotImplementedError("Validation not implemented for this runner")
    
    def extract_embeddings(self):
        """Execute embedding extraction logic. Default implementation raises NotImplementedError.""" 
        raise NotImplementedError("Embedding extraction not implemented for this runner")
        
    def _save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: Optimizer,
        epoch: int,
        loss: float,
        checkpoint_path: str,
        **additional_data
    ):
        """
        Save a model checkpoint with common structure.
        
        Args:
            model: The model to save
            optimizer: The optimizer to save
            epoch: Current epoch
            loss: Current loss value
            checkpoint_path: Path to save the checkpoint
            **additional_data: Any additional data to save
        """
        if not self.config.is_ref_device:
            return
            
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        
        checkpoint_data = {
            "model_state_dict": model.module.state_dict() if hasattr(model, 'module') else model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "config": self.config.__dict__ if hasattr(self.config, '__dict__') else None,
            **additional_data
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        print(f"[{self.__class__.__name__}] Saved checkpoint: {checkpoint_path}")
    
    def _log_metrics(self, metrics: Dict[str, float]):
        """Log metrics to wandb if available and on reference device."""
        if self.wandb_wrapper and self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
            self.wandb_wrapper.log(metrics) 