import os
import torch
import numpy as np
from torch.amp import GradScaler
from torch.optim.lr_scheduler import LRScheduler

from transformers import GPT2Tokenizer

from utils.registry import (
    ModelRegistry,
    ProjectRegistry 
)
from utils.ddp import DistributedUtils
from utils.config import LLMFinetuningConfig
from utils.wandb_wrapper import WandbWrapper
from utils.schedulers import get_scheduler
from projects.base_project import BaseProject
from models.gpt2_with_embeddings import GPT2WithEmbedding
from data.ecg_clinical_report_dataset import get_distributed_clinical_report_dataloader

from typing import Any

@ProjectRegistry.register("ECG_tokenizer_LLM_finetuning")
class LLMFinetuningProject(BaseProject):
    def __init__(
        self,
        config: LLMFinetuningConfig,
        wandb_wrapper: WandbWrapper
    ):
        super().__init__(config, wandb_wrapper)

    def run(self):
        super().run()

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
            pin_memory=True,
            subset_fraction=1.0  # Use only 10% of training data
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
            pin_memory=True,
            subset_fraction=1.0  # Use only 10% of validation data
        )

        # Get the model
        print("Getting embedding size...")
        ecg_embedding_size: tuple[int, ...] = self._get_embedding_size(self.config.train_embeddings_path)
        
        # Get model class and parameters based on model type
        model_class_name = self.config.get_model_class_name()
        embedding_size = self.config.get_embedding_size()
        
        # Create model with appropriate parameters based on model type
        model_kwargs = {
            "ecg_embedding_size": ecg_embedding_size,
            "reducer_name": self.config.embedding_reducer_name or self.config.get_default_reducer_name(),
            "reducer_dropout": self.config.reducer_dropout
        }
        
        if self.config.model_type == "gpt2":
            model_kwargs.update({
                "gpt2_model_name": self.config.huggingface_model_name,
                "gpt2_embedding_size": embedding_size
            })
        elif self.config.model_type == "bloom":
            model_kwargs.update({
                "bloom_model_name": self.config.huggingface_model_name,
                "bloom_embedding_size": embedding_size
            })
        elif self.config.model_type == "opt":
            model_kwargs.update({
                "opt_model_name": self.config.huggingface_model_name,
                "opt_embedding_size": embedding_size
            })
        elif self.config.model_type == "mistral":
            model_kwargs.update({
                "mistral_model_name": self.config.huggingface_model_name,
                "mistral_embedding_size": embedding_size
            })
        elif self.config.model_type == "gptneo":
            model_kwargs.update({
                "gptneo_model_name": self.config.huggingface_model_name,
                "gptneo_embedding_size": embedding_size
            })
        elif self.config.model_type == "gptj":
            model_kwargs.update({
                "gptj_model_name": self.config.huggingface_model_name,
                "gptj_embedding_size": embedding_size
            })
        
        model = ModelRegistry.get(model_class_name)(**model_kwargs).to(self.config.device)

        # Get the base model for parameter grouping (different models have different attribute names)
        if hasattr(model, 'gpt2'):
            base_model = model.gpt2
        elif hasattr(model, 'bloom'):
            base_model = model.bloom
        elif hasattr(model, 'opt'):
            base_model = model.opt
        elif hasattr(model, 'mistral'):
            base_model = model.mistral
        elif hasattr(model, 'gptneo'):
            base_model = model.gptneo
        elif hasattr(model, 'gptj'):
            base_model = model.gptj
        else:
            raise ValueError(f"Unknown model type: {self.config.model_type}")

        param_groups = [
            {
                "params": base_model.parameters(),
                "lr": self.config.llm_lr,
                "weight_decay": self.config.llm_weight_decay,
                "name": "llm"
            },
            {
                "params": model.embedding_reducer.parameters(),
                "lr": self.config.embedding_reducer_lr,
                "weight_decay": self.config.embedding_reducer_weight_decay,
                "name": "embedding_reducer"
            }
        ]
        
        # Wrap the model in DDP
        model = DistributedUtils.DDP(
            model,
            device_ids=[self.config.device]
        )

        # Get the optimizer
        optimizer_class: torch.optim.Optimizer = getattr(torch.optim, self.config.optimizer)
        optimizer: torch.optim.Optimizer = optimizer_class(param_groups)

        # Get the scheduler
        scheduler: LRScheduler = get_scheduler(
            scheduler_name=self.config.scheduler_type,
            optimizer=optimizer,
            num_epochs=self.config.num_epochs,
            train_dataloader=training_dataloader,
            gamma=self.config.gamma if hasattr(self.config, 'gamma') else None,
            step_size=self.config.step_size if hasattr(self.config, 'step_size') else None,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps if hasattr(self.config, 'gradient_accumulation_steps') else 1,
            num_warmup_percent=self.config.num_warmup_percent if hasattr(self.config, 'num_warmup_percent') else None,
            num_hard_restarts_cycles=self.config.num_hard_restarts_cycles if hasattr(self.config, 'num_hard_restarts_cycles') else None,
            warm_restart_tmult=self.config.warm_restart_tmult if hasattr(self.config, 'warm_restart_tmult') else None
        )

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
            reports_path=self.config.inference_dataset_path,
            embeddings_path=self.config.validation_embeddings_path,
            tokenizer=tokenizer,
            max_token_length=self.config.max_token_length,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            num_replicas=self.config.world_size,
            rank=self.config.device,
            shuffle=False, 
            pin_memory=True,
            subset_fraction=0.1  # Use only 10% of inference data
        )        
        
        # Get the model
        print("Getting embedding size...")
        ecg_embedding_size: tuple[int, ...] = self._get_embedding_size(self.config.validation_embeddings_path)
        
        # Get model class and parameters based on model type
        model_class_name = self.config.get_model_class_name()
        embedding_size = self.config.get_embedding_size()
        
        # Create model with appropriate parameters based on model type
        model_kwargs = {
            "ecg_embedding_size": ecg_embedding_size,
            "reducer_name": self.config.embedding_reducer_name or self.config.get_default_reducer_name(),
            "reducer_dropout": self.config.reducer_dropout
        }
        
        if self.config.model_type == "gpt2":
            model_kwargs.update({
                "gpt2_model_name": self.config.huggingface_model_name,
                "gpt2_embedding_size": embedding_size
            })
        elif self.config.model_type == "bloom":
            model_kwargs.update({
                "bloom_model_name": self.config.huggingface_model_name,
                "bloom_embedding_size": embedding_size
            })
        elif self.config.model_type == "opt":
            model_kwargs.update({
                "opt_model_name": self.config.huggingface_model_name,
                "opt_embedding_size": embedding_size
            })
        elif self.config.model_type == "mistral":
            model_kwargs.update({
                "mistral_model_name": self.config.huggingface_model_name,
                "mistral_embedding_size": embedding_size
            })
        elif self.config.model_type == "gptneo":
            model_kwargs.update({
                "gptneo_model_name": self.config.huggingface_model_name,
                "gptneo_embedding_size": embedding_size
            })
        elif self.config.model_type == "gptj":
            model_kwargs.update({
                "gptj_model_name": self.config.huggingface_model_name,
                "gptj_embedding_size": embedding_size
            })
        
        model = ModelRegistry.get(model_class_name)(**model_kwargs).to(self.config.device)

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
    
    def _setup_extraction_objects(self)->dict[str, Any]:
        """
        Extraction is not supported for LLM finetuning projects.
        LLM finetuning uses pre-extracted embeddings passed via config.
        """
        raise NotImplementedError(
            "Extraction is not supported for LLM finetuning projects. "
            "Use ECGTokenizerTrainingProject for embedding extraction."
        )
        
    def _get_embedding_size(self, embeddings_dir: str) -> tuple[int, ...]:
        for fname in os.listdir(embeddings_dir):
            full_path = os.path.join(embeddings_dir, fname)
            try:
                embedding = np.load(full_path)
                return embedding.shape
            except Exception as e:
                print(f"Warning: could not load {full_path} due to {e}")
        raise ValueError(f"No valid embedding file found in directory: {embeddings_dir}")
    
    def _setup_extraction_objects(self)->dict[str, Any]:
        raise NotImplementedError("Extraction is not implemented for this project")