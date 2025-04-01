from typing import Any

import torch
from torch.optim import AdamW, RAdam
from torch.cuda.amp import GradScaler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LRScheduler

from utils.registry import (
    ProjectRegistry, 
    ModelRegistry,
    RunnerRegistry
)
from utils.ddp import DistributedUtils
from utils.wandb_wrapper import WandbWrapper
from utils.config import ECGTokenizerTrainingConfig
from models.tokenizer import ECG_Tokenizer_Wrapper
from runners.tokenizer_runner import ECGTokenizerRunner
from data.ecg_dataset import get_distributed_ecg_dataloader


@ProjectRegistry.register("ECG_Tokenizer_Training")
class ECGTokenizerTrainingProject:
    def __init__(
        self, 
        config: ECGTokenizerTrainingConfig,
        wandb_wrapper: WandbWrapper
    ):
        self.config = config
        self.wandb_wrapper = wandb_wrapper
        
    def _setup_training_objects(self)->dict[str, Any]:
        
        training_dataloader: DataLoader = get_distributed_ecg_dataloader(
            parquet_file=self.config.train_dataset_path,
            expected_waveform_length=self.config.waveform_length,
            num_leads=self.config.num_leads,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )
        
        validation_dataloader: DataLoader = get_distributed_ecg_dataloader(
            parquet_file=self.config.validation_dataset_path,
            expected_waveform_length=self.config.waveform_length,
            num_leads=self.config.num_leads,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
        )
        
        ecg_tokenizer: ECG_Tokenizer_Wrapper = ModelRegistry.get(self.config.pipeline_project)(
            encoder_name=self.config.encoder_name,
            quantizer_name=self.config.quantizer_name,
            decoder_name=self.config.decoder_name,
            num_quantizers=self.config.num_quantizers,
            codebook_size=self.config.codebook_size
        ).to(self.config.device)

        ecg_tokenizer = DistributedUtils.DDP(
            ecg_tokenizer,
            device_ids=[self.config.device]
        )
        
        # Get the optimizer
        if self.config.optimizer == "AdamW":
            optimizer: AdamW = torch.optim.AdamW(
                ecg_tokenizer.parameters(), 
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "RAdam":
            optimizer: RAdam = torch.optim.RAdam(
                ecg_tokenizer.parameters(), 
                lr=self.config.lr,
                weight_decay=self.config.weight_decay
            )
        
        # Get the scheduler
        if self.config.scheduler_name == "step":
            scheduler: LRScheduler = torch.optim.lr_scheduler.StepLR(
                optimizer=optimizer, 
                step_size=self.config.step_size, 
                gamma=self.config.gamma
            )
        elif self.config.scheduler_name == "cosine":
            scheduler: LRScheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer=optimizer, 
                T_max=self.config.num_epochs
            )
        
        # Get the scaler
        scaler: GradScaler = torch.amp.GradScaler()
        
        return {
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "ecg_tokenizer": ecg_tokenizer,
            "training_dataloader": training_dataloader,
            "validation_dataloader": validation_dataloader
        }
    
    def _setup_inference_objects(self)->dict[str, Any]:
        raise NotImplementedError("Subclasses must implement this method")
    
    def run(self):
        runner_args = {
            "config": self.config,
            "wandb_wrapper": self.wandb_wrapper
        }
        if self.config.run_mode == "train":
            training_objects: dict[str, Any] = self._setup_training_objects()
            runner_args.update(training_objects)
        elif self.config.run_mode == "inference":
            raise NotImplementedError("Inference is not implemented")
        
        runner: ECGTokenizerRunner = RunnerRegistry.get(self.config.pipeline_project)(**runner_args)
        runner.execute(mode=self.config.run_mode)