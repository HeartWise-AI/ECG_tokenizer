
import torch
from torch.optim import AdamW
from torch.amp import GradScaler
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
from models.embedding_reducer import EmbeddingReducer
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
            embeddings_path=self.config.embeddings_path,
            tokenizer=tokenizer,
            max_token_length=self.config.max_token_length,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            num_replicas=self.config.world_size,
            rank=self.config.device
        )
        
        validation_dataloader = get_distributed_clinical_report_dataloader(
            reports_path=self.config.validation_dataset_path,
            embeddings_path=self.config.embeddings_path,
            tokenizer=tokenizer,
            max_token_length=self.config.max_token_length,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            num_replicas=self.config.world_size,
            rank=self.config.device
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
            device_ids=[self.config.device],
            find_unused_parameters=True
        )

        # Get the optimizer
        optimizer: AdamW = torch.optim.AdamW(model.parameters(), lr=self.config.lr)

        # Get the scheduler
        scheduler: LRScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.config.num_epochs)

        # Get the scaler
        scaler: GradScaler = torch.amp.GradScaler()
                
        return {
            "training_dataloader": training_dataloader,
            "validation_dataloader": validation_dataloader,
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "model": model,
        }
    
    def run(self):
        training_objects: dict[str, Any] = self._setup_training_objects()

        runner: LLMFinetuningRunner = RunnerRegistry.get(self.config.runner_name)(
            config=self.config,
            wandb_wrapper=self.wandb_wrapper,
            train_dataloader=training_objects["training_dataloader"],
            val_dataloader=training_objects["validation_dataloader"],
            optimizer=training_objects["optimizer"],
            scheduler=training_objects["scheduler"],
            scaler=training_objects["scaler"],
            model=training_objects["model"]
        )

        runner.train()