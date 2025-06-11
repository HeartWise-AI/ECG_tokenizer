import torch

from typing import Any
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import LRScheduler

from utils.ddp import DistributedUtils
from utils.schedulers import get_scheduler
from utils.registry import (
    ModelRegistry,
    ProjectRegistry
)
from utils.wandb_wrapper import WandbWrapper
from utils.enums import ProjectName
from utils.config import ECGTokenizerLinearProbingConfig
from projects.base_project import BaseProject
from models.tokenizer import ECG_Tokenizer_Wrapper
from data.ecg_tokenizer_classifier_dataset import get_distributed_ecg_tokenizer_classifier_dataloader


@ProjectRegistry.register(ProjectName.ECG_TOKENIZER_LINEAR_PROBING)
class ECGTokenizerLinearProbing(BaseProject):
    def __init__(
        self, 
        config: ECGTokenizerLinearProbingConfig, 
        wandb_wrapper: WandbWrapper
    ):
        super().__init__(config, wandb_wrapper)
        self.config: ECGTokenizerLinearProbingConfig = config # cast to ECGTokenizerLinearProbingConfig to avoid type errors
        
    def run(self):
        super().run()
        
    def _setup_training_objects(self)->dict[str, Any]:
        
        # Load the pretrained tokenizer
        state_dict = self._load_checkpoint(self.config.pretrained_tokenizer_path)
        # Get the config from the pretrained tokenizer
        pretrained_config = state_dict['config']
        
        # Initialize the tokenizer with the appropriate configuration
        ecg_tokenizer: ECG_Tokenizer_Wrapper = ModelRegistry.get(self.config.pipeline_project)(
            encoder_name=pretrained_config.encoder_name, 
            quantizer_name=pretrained_config.quantizer_name,
            decoder_name=self.config.decoder_name, # use the decoder from the current config
            num_quantizers=pretrained_config.num_quantizers,
            codebook_size=pretrained_config.codebook_size,
            decoder_mode=self.config.decoder_mode, # use the decoder mode from the current config
            num_classes=self.config.num_classes, # use the number of classes from the current config
        ).to(self.config.device)
        # Set the codebook size to the pretrained codebook size
        self.config.codebook_size = pretrained_config.codebook_size # required to compute % of active codebook during training
        
        # Load the pretrained state dict
        pretrained_state_dict = state_dict['model_state_dict']
        ecg_tokenizer._load_pretrained_weights(pretrained_state_dict, freeze_pretrained_components=True)
        
        # Print training configuration
        self._print_training_config(ecg_tokenizer)
               
        # Get the train dataloader
        train_dataloader: DataLoader = get_distributed_ecg_tokenizer_classifier_dataloader(
            parquet_file=self.config.train_dataset_path,
            expected_waveform_length=self.config.waveform_length,
            num_leads=self.config.num_leads,
            normalize_waveforms=self.config.normalize_waveforms,
            lead_stats=self.config.lead_stats,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            num_replicas=self.config.world_size,
            rank=self.config.device,
            shuffle=True,
            pin_memory=True
        )
        # Get the validation dataloader
        validation_dataloader: DataLoader = get_distributed_ecg_tokenizer_classifier_dataloader(
            parquet_file=self.config.validation_dataset_path,
            expected_waveform_length=self.config.waveform_length,
            num_leads=self.config.num_leads,
            normalize_waveforms=self.config.normalize_waveforms,
            lead_stats=self.config.lead_stats,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            num_replicas=self.config.world_size,
            rank=self.config.device,
            shuffle=False,
            pin_memory=True
        )

        # Wrap the model in DDP
        ecg_tokenizer = DistributedUtils.DDP(
            ecg_tokenizer,
            device_ids=[self.config.device]
        )
        
        # Get the parameter groups
        param_groups = [
            {
                "params": ecg_tokenizer.module.parameters(),
                "lr": self.config.lr,
                "weight_decay": self.config.weight_decay,
                "name": "ecg_tokenizer"
            }
        ]
        
        # Get the optimizer
        optimizer_class = getattr(torch.optim, self.config.optimizer)
        optimizer: Optimizer = optimizer_class(param_groups)

        # Get the scheduler
        scheduler: LRScheduler = get_scheduler(
            scheduler_name=self.config.scheduler_type,
            optimizer=optimizer,
            num_epochs=self.config.num_epochs,
            train_dataloader=train_dataloader,
            gamma=self.config.gamma if hasattr(self.config, 'gamma') else None,
            step_size=self.config.step_size if hasattr(self.config, 'step_size') else None,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps if hasattr(self.config, 'gradient_accumulation_steps') else 1,
            num_warmup_percent=self.config.num_warmup_percent if hasattr(self.config, 'num_warmup_percent') else None,
            num_hard_restarts_cycles=self.config.num_hard_restarts_cycles if hasattr(self.config, 'num_hard_restarts_cycles') else None,
            warm_restart_tmult=self.config.warm_restart_tmult if hasattr(self.config, 'warm_restart_tmult') else None
        )
                
        # Get the scaler
        scaler: GradScaler = GradScaler()
        
        return {
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "ecg_tokenizer": ecg_tokenizer,
            "train_dataloader": train_dataloader,
            "validation_dataloader": validation_dataloader
        }
    
    def _print_training_config(self, model: ECG_Tokenizer_Wrapper):
        """Print the current training configuration."""
        print("\n" + "="*60)
        print("LINEAR PROBING TRAINING CONFIGURATION")
        print("="*60)
        
        training_info = model.get_training_info()
        
        print(f"Total parameters: {training_info['total_params']:,}")
        print(f"Trainable parameters: {training_info['trainable_params']:,}")
        print(f"Frozen parameters: {training_info['frozen_params']:,}")
        print(f"Trainable ratio: {training_info['trainable_ratio']:.2f}%")
        
        print(f"\nDecoder: {model.decoder_name} ({model.decoder_mode.value} mode)")
        print(f"Target classes: {getattr(self.config, 'num_classes', 'N/A')}")
        
        print("\nComponent-wise breakdown:")
        print(f"  Encoder: {training_info['encoder']['trainable']:,}/{training_info['encoder']['total']:,} trainable")
        print(f"  Quantizer: {training_info['quantizer']['trainable']:,}/{training_info['quantizer']['total']:,} trainable")
        print(f"    └─ MLPs: {training_info['quantizer_mlps']['trainable']:,}/{training_info['quantizer_mlps']['total']:,} trainable")
        print(f"  Decoder: {training_info['decoder']['trainable']:,}/{training_info['decoder']['total']:,} trainable")
        
        print("="*60 + "\n")
    
    def _setup_inference_objects(self)->dict[str, Any]:
        raise NotImplementedError("Inference is not implemented for this project")
    
    def _setup_extraction_objects(self)->dict[str, Any]:
        raise NotImplementedError("Extraction is not implemented for this project")