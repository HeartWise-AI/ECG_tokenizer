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
        for epoch in range(self.config.num_epochs):
            # Sync before starting each epoch
            DistributedUtils.sync_process_group(
                world_size=self.config.world_size,
                device_ids=self.config.device
            )
            
            train_loss: float = self._run_epoch(
                RunMode.TRAIN,
                epoch
            )
                        
            if self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
                self.wandb_wrapper.log({
                    "train/loss": train_loss,
                    "train/step": epoch
                })
            
            # Sync the process group
            DistributedUtils.sync_process_group(
                world_size=self.config.world_size,
                device_ids=self.config.device
            )
            
            val_loss: float = self._run_epoch(
                RunMode.VALIDATION,
                epoch
            )
            
            # Sync after validation epoch, before next epoch            
            if self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
                self.wandb_wrapper.log({
                    "val/loss": val_loss,
                    "val/step": epoch
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
            loss: torch.Tensor = step_fn(
                embeddings=embeddings, 
                input_ids=input_ids, 
                attention_mask=attention_mask, 
                labels=labels
            )
            
            # Sync the process group
            DistributedUtils.sync_process_group(
                world_size=self.config.world_size,
                device_ids=self.config.device
            )
            
            # Update the total loss
            total_loss += loss.item()
            mean_loss: float = total_loss / (batch_idx + 1)
            
            # Log the loss to wandb
            if self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
                self.wandb_wrapper.log({
                    f"{mode}/loss": mean_loss,
                    f"{mode}/step": batch_idx + (epoch * len(dataloader))
                })
                
            # Sync the process group
            DistributedUtils.sync_process_group(
                world_size=self.config.world_size,
                device_ids=self.config.device
            )
            
            # Update progress bar
            data_iter.set_postfix({
                'loss': f'{loss.item():.4f}',
                'mean_loss': f'{mean_loss:.4f}'
            })
        
        # Sync at the end of epoch
        DistributedUtils.sync_process_group(
            world_size=self.config.world_size,
            device_ids=self.config.device
        )
        
        # Return the total loss
        return total_loss

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
            outputs = self.model(
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
        
        return loss

    def _val_step(
        self, 
        embeddings: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        with torch.no_grad():
            outputs = self.model(
                ecg_embeddings=embeddings,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            generated_ids: torch.Tensor = self.model.generate_report(embeddings)
            
            return {
                "loss": outputs.loss,
                "generated_ids": generated_ids
            }

    def validate(self):
        raise NotImplementedError("Validation not implemented")

    def save_model(self):
        raise NotImplementedError("Saving model not implemented")