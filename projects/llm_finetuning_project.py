import os
import torch
import numpy as np
from torch.amp import GradScaler
from torch.optim import AdamW, RAdam
from torch.optim.lr_scheduler import LRScheduler

from transformers import GPT2Tokenizer

from utils.registry import (
    ModelRegistry,
    RunnerRegistry,
    ProjectRegistry 
)
from utils.ddp import DistributedUtils
from utils.config import LLMFinetuningConfig
from utils.wandb_wrapper import WandbWrapper
from models.gpt2_with_embeddings import GPT2WithEmbedding
from runners.llm_finetuning_runner import LLMFinetuningRunner
from data.ecg_clinical_report_dataset import get_distributed_clinical_report_dataloader

from typing import Any

@ProjectRegistry.register("ECG_tokenizer_LLM_finetuning")
class LLMFinetuningProject:
    def __init__(
        self,
        config: LLMFinetuningConfig,
        wandb_wrapper: WandbWrapper
    ):
        self.config: LLMFinetuningConfig = config
        self.wandb_wrapper: WandbWrapper = wandb_wrapper

    def _setup_training_objects(self)->dict[str, Any]:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        # Get the dataloaders
        training_dataloader = get_distributed_clinical_report_dataloader(
            reports_path=self.config.train_dataset_path,
            embeddings_path=self.config.train_embeddings_path,
            tokenizer=tokenizer,
            max_token_length=self.config.max_token_length,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            num_replicas=self.config.world_size,
            rank=self.config.device,
            shuffle=True, 
            pin_memory=True
        )
        
        validation_dataloader = get_distributed_clinical_report_dataloader(
            reports_path=self.config.validation_dataset_path,
            embeddings_path=self.config.validation_embeddings_path,
            tokenizer=tokenizer,
            max_token_length=self.config.max_token_length,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            num_replicas=self.config.world_size,
            rank=self.config.device,
            shuffle=False, 
            pin_memory=True
        )

        # Get the model
        ecg_embedding_size: tuple[int, ...] = self._get_embedding_size(self.config.train_embeddings_path)
        model: GPT2WithEmbedding = ModelRegistry.get(self.config.trainable_model_name)(
            gpt2_model_name=self.config.huggingface_model_name, 
            gpt2_embedding_size=self.config.gpt2_embedding_size, 
            ecg_embedding_size=ecg_embedding_size,
            reducer_name=self.config.embedding_reducer_name,
            reducer_dropout=self.config.reducer_dropout
        ).to(self.config.device)

        # Wrap the model in DDP
        model = DistributedUtils.DDP(
            model,
            device_ids=[self.config.device]
        )

        # Get the optimizer
        if self.config.optimizer == "AdamW":
            optimizer: AdamW = torch.optim.AdamW(model.parameters(), lr=self.config.lr)
        elif self.config.optimizer == "RAdam":
            optimizer: RAdam = torch.optim.RAdam(model.parameters(), lr=self.config.lr)

        # Get the scheduler
        if self.config.scheduler_type == "step":
            scheduler: LRScheduler = torch.optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=self.config.step_size, gamma=self.config.gamma)
        elif self.config.scheduler_type == "cosine":
            scheduler: LRScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.config.num_epochs)

        # Get the scaler
        scaler: GradScaler = torch.amp.GradScaler()
                
        return {
            "train_dataloader": training_dataloader,
            "val_dataloader": validation_dataloader,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "model": model,
        }
    
    def _setup_inference_objects(self)->dict[str, Any]:
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token
        
        validation_dataloader = get_distributed_clinical_report_dataloader(
            reports_path=self.config.validation_dataset_path,
            embeddings_path=self.config.embeddings_path,
            tokenizer=tokenizer,
            max_token_length=self.config.max_token_length,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            num_replicas=self.config.world_size,
            rank=self.config.device,
            shuffle=False, 
            pin_memory=True
        )        
        
        # Get the model
        model: GPT2WithEmbedding = ModelRegistry.get(self.config.trainable_model_name)(
            gpt2_model_name=self.config.huggingface_model_name, 
            embedding_size=self.config.embedding_size, 
            reducer_name=self.config.embedding_reducer_name
        ).to(self.config.device)

        # Wrap the model in DDP
        model = DistributedUtils.DDP(
            model,
            device_ids=[self.config.device]
        )
        
        # Load the checkpoint
        checkpoint = self._load_checkpoint(self.config.checkpoint_dir)
        
        # Load the model state dict
        model.module.load_state_dict(checkpoint["model_state_dict"])
        
        return {
            "model": model,
            "val_dataloader": validation_dataloader
        }
    
    def run(self):
        runner_args = {
            "config": self.config,
            "wandb_wrapper": self.wandb_wrapper
        }
        if self.config.run_mode == "train":
            training_objects: dict[str, Any] = self._setup_training_objects()
            runner_args.update(training_objects)
        elif self.config.run_mode == "inference":
            inference_objects: dict[str, Any] = self._setup_inference_objects()
            runner_args.update(inference_objects)

        runner: LLMFinetuningRunner = RunnerRegistry.get(self.config.runner_name)(**runner_args)
        runner.execute(mode=self.config.run_mode)
        
        
    def _load_checkpoint(
        self, 
        checkpoint_path: str
    )->dict[str, Any]:
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint file does not exist: {checkpoint_path}")
        
        print(
            f"[LLMFinetuningProject] Loading checkpoint: {checkpoint_path}"
        )
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        return checkpoint

    def _get_embedding_size(self, embeddings_dir: str) -> tuple[int, ...]:
        for fname in os.listdir(embeddings_dir):
            full_path = os.path.join(embeddings_dir, fname)
            try:
                embedding = np.load(full_path)
                return embedding.shape
            except Exception as e:
                print(f"Warning: could not load {full_path} due to {e}")
        raise ValueError(f"No valid embedding file found in directory: {embeddings_dir}")