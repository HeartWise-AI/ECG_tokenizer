import os
import torch
from torch.optim import AdamW
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

from utils.enums import RunMode
from utils.ddp import DistributedUtils
from utils.registry import (
    RunnerRegistry,
    MetricRegistry
)
from utils.config import LLMFinetuningConfig
from utils.wandb_wrapper import WandbWrapper
from models.gpt2_with_embeddings import GPT2WithEmbedding
from utils.metrics.llm_metrics import (
    RougeMetric,
    BleuMetric,
    MeteorMetric
)

from tqdm import tqdm
from typing import Any, Union

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
    ):
        self.config: LLMFinetuningConfig = config
        self.wandb_wrapper: WandbWrapper = wandb_wrapper
        self.train_dataloader: DataLoader = train_dataloader
        self.val_dataloader: DataLoader = val_dataloader
        self.optimizer: AdamW = optimizer
        self.scheduler: LRScheduler = scheduler
        self.scaler: GradScaler = scaler
        self.model: GPT2WithEmbedding = model
        
    def train(self):
        best_val_loss: float = float('inf')
        
        for epoch in range(1, self.config.num_epochs + 1):
            # Sync before starting each epoch
            DistributedUtils.sync_process_group(
                world_size=self.config.world_size,
                device_ids=self.config.device
            )
            
            epoch_metrics: dict[str, float] = self._run_epoch(
                RunMode.TRAIN,
                epoch
            )
                        
            if self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
                self.wandb_wrapper.log(
                    epoch_metrics
                )
            
            # Sync the process group
            DistributedUtils.sync_process_group(
                world_size=self.config.world_size,
                device_ids=self.config.device
            )
            
            epoch_metrics: dict[str, float] = self._run_epoch(
                RunMode.VALIDATION,
                epoch
            )
            
            # Save best model (only on reference device)
            if self.config.is_ref_device:
                if epoch_metrics[f'{RunMode.VALIDATION}/loss'] < best_val_loss:
                    best_val_loss = epoch_metrics[f'{RunMode.VALIDATION}/loss']
                    self._save_model(
                        epoch=epoch,
                        loss=epoch_metrics[f'{RunMode.VALIDATION}/loss'],
                        is_best=True
                    )
                
                # Also save regular checkpoint
                self._save_model(
                    epoch=epoch,
                    loss=epoch_metrics[f'{RunMode.VALIDATION}/loss'],
                    is_best=False
                )
            
            # Sync after validation epoch, before next epoch            
            if self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
                self.wandb_wrapper.log({
                    **epoch_metrics,
                    f"{RunMode.VALIDATION}/best_loss": best_val_loss
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
    )->dict[str, float]:
        assert mode in [RunMode.TRAIN, RunMode.VALIDATION]
        
        # Set the model to training or evaluation mode
        self.model.train(mode == RunMode.TRAIN)
        
        # Get the dataloader and step function
        dataloader: DataLoader = self.train_dataloader if mode == RunMode.TRAIN else self.val_dataloader
        step_fn: callable = self._train_step if mode == RunMode.TRAIN else self._val_step
        
        # Create a progress bar for the epoch
        data_iter: tqdm = tqdm(dataloader, desc=f"{mode} epoch {epoch}/{self.config.num_epochs}", leave=True)
        
        # Initialize the total loss
        total_loss: float = 0.0
        
        # Sync before starting batch iterations
        DistributedUtils.sync_process_group(
            world_size=self.config.world_size,
            device_ids=self.config.device
        )
        
        # Iterate over the dataloader
        epoch_metrics: dict[str, float] = {}
        total_elements: int = 0
        for batch_idx, batch in enumerate(data_iter):            
            # Preprocess the batch
            embeddings: torch.Tensor = batch['embedding'].to(self.config.device)
            input_ids: torch.Tensor = batch['input_ids'].to(self.config.device)
            attention_mask: torch.Tensor = batch['attention_mask'].to(self.config.device)
            labels: torch.Tensor = input_ids.clone()
            
            # Run the step function
            outputs: dict[str, torch.Tensor] = step_fn(
                embeddings=embeddings, 
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            
            # initialize metrics
            metrics: dict[str, float] = {}
            metrics['loss'] = outputs['loss'].item()
            
            # Compute rouge score, bleu score, and meteor score
            if mode == RunMode.VALIDATION:
                for metric in self.config.metrics:
                    registered_metrics: Union[
                        RougeMetric, 
                        BleuMetric, 
                        MeteorMetric
                    ] = MetricRegistry.get(metric)
                    metrics.update(
                        registered_metrics.compute_score(
                            outputs['generated_ids'], 
                            labels, 
                            dataloader.dataset.tokenizer
                        )
                    )
                    
            
            # Gather and average loss across all GPUs
            gathered_metrics: dict[str, float] = {}
            for k in metrics:
                gathered_metrics[f"{mode}/{k}"] = DistributedUtils.gather_loss(
                    [metrics[k]], 
                    self.config.device
                )
                            
            # Update the epoch metrics
            for k, v in gathered_metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + float(v)
            
            # Update the total loss with gathered loss
            total_loss += gathered_metrics[f'{mode}/loss']
            mean_loss: float = total_loss / (batch_idx + 1)
            
            # Log the loss to wandb
            if mode == RunMode.TRAIN:
                if self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
                    self.wandb_wrapper.log({
                        f"{mode}/loss": gathered_metrics[f'{mode}/loss'],  # Log the gathered loss for current batch
                        f"{mode}/mean_loss": mean_loss,  # Log the running mean loss
                    })
            
            # Sync after logging
            DistributedUtils.sync_process_group(
                world_size=self.config.world_size,
                device_ids=self.config.device
            )
            
            # Update progress bar with gathered losses
            data_iter.set_postfix({
                f"{mode}/loss": f'{gathered_metrics[f"{mode}/loss"]:.4f}',
                f"{mode}/mean_loss": f'{mean_loss:.4f}'
            })
                
        # Normalize the epoch metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= len(dataloader)
        
        # Return the epoch metrics
        return epoch_metrics

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
                generated_ids: torch.Tensor = self.model.module.generate_report(
                    ecg_embeddings=embeddings, 
                    max_token_length=self.config.max_token_length
                )
            else:
                generated_ids: torch.Tensor = self.model.generate_report(
                    ecg_embeddings=embeddings, 
                    max_token_length=self.config.max_token_length
                )

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
        save_dir: str = self.config.output_dir
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
            })