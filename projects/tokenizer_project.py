import torch

from typing import Any
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import LRScheduler

from utils.registry import (
    ProjectRegistry, 
    ModelRegistry
)
from utils.ddp import DistributedUtils
from utils.schedulers import get_scheduler
from utils.wandb_wrapper import WandbWrapper
from utils.enums import DecoderMode, ProjectName
from utils.config import ECGTokenizerTrainingConfig
from projects.base_project import BaseProject
from models.tokenizer import ECG_Tokenizer_Wrapper
from data.ecg_dataset import get_distributed_ecg_dataloader
from data.ecg_tokenizer_classifier_dataset import get_distributed_ecg_tokenizer_classifier_dataloader


@ProjectRegistry.register(ProjectName.ECG_TOKENIZER_TRAINING)
class ECGTokenizerTrainingProject(BaseProject):
    def __init__(
        self, 
        config: ECGTokenizerTrainingConfig,
        wandb_wrapper: WandbWrapper
    ):
        super().__init__(config, wandb_wrapper)
        self.config: ECGTokenizerTrainingConfig = config # cast to ECGTokenizerTrainingConfig to avoid type errors
    
    def run(self):
        super().run()
                    
    def _setup_training_objects(self)->dict[str, Any]:
        # Check if we're in classification mode using DecoderMode enum
        decoder_mode = getattr(self.config, 'decoder_mode', DecoderMode.RECONSTRUCTION)
        decoder_mode = decoder_mode if isinstance(decoder_mode, DecoderMode) else DecoderMode(decoder_mode)
        is_classification = decoder_mode == DecoderMode.CLASSIFICATION
        
        if is_classification:
            # For classification, we need a dataset that provides both signals and labels
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
        else:
            # For reconstruction, use the original ECG dataset
            train_dataloader: DataLoader = get_distributed_ecg_dataloader(
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
            
            validation_dataloader: DataLoader = get_distributed_ecg_dataloader(
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
        
        # Initialize the tokenizer with the appropriate configuration
        ecg_tokenizer: ECG_Tokenizer_Wrapper = ModelRegistry.get(self.config.pipeline_project)(
            encoder_name=self.config.encoder_name,
            quantizer_name=self.config.quantizer_name,
            decoder_name=self.config.decoder_name,
            num_quantizers=self.config.num_quantizers,
            codebook_size=self.config.codebook_size,
            decoder_mode=decoder_mode,
            num_classes=getattr(self.config, 'num_classes', 77)
        ).to(self.config.device)

        ecg_tokenizer = DistributedUtils.DDP(
            ecg_tokenizer,
            device_ids=[self.config.device]
        )
        
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
        
    def _setup_extraction_objects(self)->dict[str, Any]:
        embedding_extraction_dataloader: DataLoader = get_distributed_ecg_dataloader(
            parquet_file=self.config.embedding_extraction_dataset_path,
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
        
        # Initialize the tokenizer for embedding extraction
        decoder_mode = getattr(self.config, 'decoder_mode', DecoderMode.RECONSTRUCTION)
        decoder_mode = decoder_mode if isinstance(decoder_mode, DecoderMode) else DecoderMode(decoder_mode)
        
        ecg_tokenizer: ECG_Tokenizer_Wrapper = ModelRegistry.get(self.config.pipeline_project)(
            encoder_name=self.config.encoder_name,
            quantizer_name=self.config.quantizer_name,
            decoder_name=self.config.decoder_name,
            num_quantizers=self.config.num_quantizers,
            codebook_size=self.config.codebook_size,
            decoder_mode=decoder_mode,
            num_classes=getattr(self.config, 'num_classes', 77)
        ).to(self.config.device)

        # Wrap the model in DDP for extraction
        ecg_tokenizer = DistributedUtils.DDP(
            ecg_tokenizer,
            device_ids=[self.config.device]
        )
        
        # Load the checkpoint if provided
        if hasattr(self.config, 'checkpoint_dir') and self.config.checkpoint_dir:
            checkpoint = self._load_checkpoint(self.config.checkpoint_dir)
            ecg_tokenizer.module.load_state_dict(checkpoint["model_state_dict"])
            print(f"[ECGTokenizerTrainingProject] Loaded checkpoint from {self.config.checkpoint_dir}")
        
        return {
            "ecg_tokenizer": ecg_tokenizer,
            "embedding_extraction_dataloader": embedding_extraction_dataloader
        }
    
    def _setup_inference_objects(self)->dict[str, Any]:
        """Setup objects needed for inference mode."""
        raise NotImplementedError("Inference is not implemented for ECG Tokenizer project")