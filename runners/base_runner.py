import os
import torch
from abc import ABC, abstractmethod
from typing import Any, Dict, Union
from tqdm import tqdm
from torch.utils.data import DataLoader

from utils.enums import RunMode
from utils.ddp import DistributedUtils
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
    
    def _run_epoch(
        self,
        mode: RunMode,
        epoch: int,
        dataloader: DataLoader,
        step_fn: callable,
    ) -> Dict[str, float]:
        """
        Common epoch running logic that can be used by subclasses.
        
        Args:
            mode: The run mode (TRAIN/VALIDATE)
            epoch: Current epoch number
            dataloader: DataLoader to iterate over
            step_fn: Function to call for each batch
            
        Returns:
            Dictionary of metrics for the epoch
        """
        # Create progress bar
        data_iter = tqdm(
            dataloader,
            desc=f"[GPU {self.config.device}]: {mode} epoch {epoch}/{self.config.num_epochs}",
            leave=True,
            disable=not self.config.is_ref_device
        )
        
        # Sync before starting batch iterations
        DistributedUtils.sync_process_group(
            world_size=self.config.world_size,
            device_ids=self.config.device
        )
        
        total_loss = 0.0
        num_batches = 0
        
        for batch_idx, batch in enumerate(data_iter):
            # Execute the step function
            outputs = step_fn(batch, batch_idx)
            
            if isinstance(outputs, dict) and 'loss' in outputs:
                total_loss += outputs['loss']
                num_batches += 1
        
        # Calculate average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        return {f"{mode}/loss": avg_loss}
    
    def _sync_process_group(self):
        """Synchronize the distributed process group."""
        DistributedUtils.sync_process_group(
            world_size=self.config.world_size,
            device_ids=self.config.device
        )
    
    def _save_checkpoint(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
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
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
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