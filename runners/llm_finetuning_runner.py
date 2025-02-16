import os
import torch
from torch.optim import AdamW
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

from utils.enums import RunMode
from utils.ddp import DistributedUtils
from utils.registry import RunnerRegistry
from utils.config import LLMFinetuningConfig
from utils.wandb_wrapper import WandbWrapper
from models.gpt2_with_embeddings import GPT2WithEmbedding

from tqdm import tqdm
from typing import Any
@RunnerRegistry.register("LLM_finetuning_runner")
class LLMFinetuningRunner:
    def __init__(
        self, 
        config: LLMFinetuningConfig, 
        wandb_wrapper: WandbWrapper,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        optimizer: AdamW,
        scheduler: LRScheduler,
        scaler: GradScaler,
        model: GPT2WithEmbedding,
        loss_fn: torch.nn.Module,
    ):
        self.config: LLMFinetuningConfig = config
        self.wandb_wrapper: WandbWrapper = wandb_wrapper
        self.train_dataloader: DataLoader = train_dataloader
        self.val_dataloader: DataLoader = val_dataloader
        self.optimizer: AdamW = optimizer
        self.scheduler: LRScheduler = scheduler
        self.scaler: GradScaler = scaler
        self.model: GPT2WithEmbedding = model
        self.loss_fn: torch.nn.Module = loss_fn
        
    def train(self):
        best_val_loss: float = float('inf')
        
        for epoch in range(self.config.num_epochs):
            # Sync before starting each epoch
            DistributedUtils.sync_process_group(
                world_size=self.config.world_size,
                device_ids=self.config.device
            )
            
            mean_train_loss: float = self._run_epoch(
                RunMode.TRAIN,
                epoch
            )
                        
            if self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
                self.wandb_wrapper.log({
                    "train/loss": mean_train_loss,
                    "train/step": epoch
                })
            
            # Sync the process group
            DistributedUtils.sync_process_group(
                world_size=self.config.world_size,
                device_ids=self.config.device
            )
            
            mean_val_loss: float = self._run_epoch(
                RunMode.VALIDATION,
                epoch
            )
            
            # Save best model (only on reference device)
            if self.config.is_ref_device:
                if mean_val_loss < best_val_loss:
                    best_val_loss = mean_val_loss
                    self._save_model(
                        epoch=epoch,
                        loss=mean_val_loss,
                        is_best=True
                    )
                
                # Also save regular checkpoint
                self._save_model(
                    epoch=epoch,
                    loss=mean_val_loss,
                    is_best=False
                )
            
            # Sync after validation epoch, before next epoch            
            if self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
                self.wandb_wrapper.log({
                    "val/loss": mean_val_loss,
                    "val/step": epoch,
                    "val/best_loss": best_val_loss
                })
                
            # Sync the process group
            DistributedUtils.sync_process_group(
                world_size=self.config.world_size,
                device_ids=self.config.device
            )
                
    def _run_epoch(
        self,
        mode: RunMode,
        epoch: int
    )->float:
        assert mode in [RunMode.TRAIN, RunMode.VALIDATION]
        
        # Set the model to training or evaluation mode
        self.model.train(mode == RunMode.TRAIN)
        
        # Get the dataloader and step function
        dataloader: DataLoader = self.train_dataloader if mode == RunMode.TRAIN else self.val_dataloader
        step_fn: callable = self._train_step if mode == RunMode.TRAIN else self._val_step
        
        # Create a progress bar for the epoch
        data_iter: tqdm = tqdm(dataloader, desc=f"{mode} epoch {epoch+1}/{self.config.num_epochs}", leave=True)
        
        # Initialize the total loss
        total_loss: float = 0.0
        
        # Sync before starting batch iterations
        DistributedUtils.sync_process_group(
            world_size=self.config.world_size,
            device_ids=self.config.device
        )
        
        # Iterate over the dataloader
        for batch_idx, batch in enumerate(data_iter):
            # Preprocess the batch
            embeddings: torch.Tensor = batch['embedding'].to(self.config.device)
            input_ids: torch.Tensor = batch['input_ids'].to(self.config.device)
            attention_mask: torch.Tensor = batch['attention_mask'].to(self.config.device)
            labels: torch.Tensor = input_ids.clone()
            
            # Run the step function
            metrics: dict[str, torch.Tensor] = step_fn(
                embeddings=embeddings, 
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            
            # Gather and average loss across all GPUs
            gathered_loss: float = DistributedUtils.gather_loss(
                [metrics['loss'].item()], 
                self.config.device
            )
            
            # Update the total loss with gathered loss
            total_loss += gathered_loss
            mean_loss: float = total_loss / (batch_idx + 1)
            
            # Log the loss to wandb
            if self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
                self.wandb_wrapper.log({
                    f"{mode}/loss": gathered_loss,  # Log the gathered loss for current batch
                    f"{mode}/mean_loss": mean_loss,  # Log the running mean loss
                    f"{mode}/step": batch_idx + (epoch * len(dataloader))
                })
            
            # Update progress bar with gathered losses
            data_iter.set_postfix({
                'batch_loss': f'{gathered_loss:.4f}',
                'mean_loss': f'{mean_loss:.4f}'
            })
            
            # Sync after logging
            DistributedUtils.sync_process_group(
                world_size=self.config.world_size,
                device_ids=self.config.device
            )
                
        # Return the mean loss for the epoch
        return total_loss / len(dataloader)

    def _train_step(
        self, 
        embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        # Clear gradients
        self.optimizer.zero_grad()
        
        # Forward pass with autocast for mixed precision
        with torch.amp.autocast(
            device_type='cuda',
            dtype=torch.bfloat16
        ):
            outputs: dict[str, torch.Tensor] = self.model(
                ecg_embeddings=embeddings,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            loss: torch.Tensor = outputs.loss

        # Backward pass with gradient scaling
        self.scaler.scale(loss).backward()
        
        # Sync gradients across processes before optimizer step
        DistributedUtils.sync_process_group(
            world_size=self.config.world_size,
            device_ids=self.config.device
        )
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        return {
            "loss": loss
        }

    def _val_step(
        self, 
        embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            outputs: dict[str, torch.Tensor] = self.model(
                ecg_embeddings=embeddings,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            if hasattr(self.model, 'module'):
                generated_ids: torch.Tensor = self.model.module.generate_report(embeddings)
            else:
                generated_ids: torch.Tensor = self.model.generate_report(embeddings)
           
            return {
                "loss": outputs.loss,
                "generated_ids": generated_ids
            }

    def validate(self):
        raise NotImplementedError("Validation not implemented")

    def _save_model(
        self,
        epoch: int,
        loss: float,
        is_best: bool = False
    ):
        """Save model checkpoint and optionally mark as best model."""
        save_dir: str = self.config.checkpoint_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Prepare checkpoint - get the underlying model's state dict for DDP models
        checkpoint: dict[str, Any] = {
            'epoch': epoch,
            'model_state_dict': self.model.module.state_dict() if hasattr(self.model, 'module') else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict(),
            'loss': loss,
            'config': self.config
        }
        
        # Save regular checkpoint for current epoch
        checkpoint_path: str = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Delete the checkpoint from the previous epoch if it exists
        if epoch > 0:
            prev_checkpoint_path: str = os.path.join(save_dir, f'checkpoint_epoch_{epoch - 1}.pt')
            if os.path.exists(prev_checkpoint_path):
                os.remove(prev_checkpoint_path)
                print(f"Deleted old checkpoint: {prev_checkpoint_path}")
        
        # If this is the best model, save it separately
        if is_best:
            best_model_path: str = os.path.join(save_dir, 'best_model.pt')
            torch.save(checkpoint, best_model_path)
            
        if self.wandb_wrapper.is_initialized():
            self.wandb_wrapper.log({
                "checkpoint/epoch": epoch,
                "checkpoint/loss": loss,
                "checkpoint/is_best": is_best
            })