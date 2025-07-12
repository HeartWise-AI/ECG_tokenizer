import torch

from typing import Any
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import LRScheduler

from transformers import GPT2Tokenizer

from utils.ddp import DistributedUtils
from utils.schedulers import get_scheduler
from utils.registry import (
    ModelRegistry,
    ProjectRegistry
)
from utils.enums import ProjectName
from utils.wandb_wrapper import WandbWrapper
from utils.config import LLMFinetuningConfig, ECGTokenizerTrainingConfig
from projects.base_project import BaseProject
from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper
from data.ecg_clinical_report_dataset import get_distributed_clinical_report_dataloader

# Add the config to the safe globals
torch.serialization.add_safe_globals([LLMFinetuningConfig])
torch.serialization.add_safe_globals([ECGTokenizerTrainingConfig])

@ProjectRegistry.register(ProjectName.ECG_TOKENIZER_LLM_FINETUNING)
class LLMFinetuningProject(BaseProject):
    """LLM finetuning project for ECG tokenizer models.
    
    Implements finetuning of a pretrained LLM on ECG tokenizer features.
    The decoder/classifier head is trained while encoder and quantizer remain frozen.
    """    
    def __init__(
        self, 
        config: LLMFinetuningConfig, 
        wandb_wrapper: WandbWrapper
    ):
        """Initialize LLM finetuning project.
        
        Args:
            config: LLM finetuning configuration
            wandb_wrapper: Weights & Biases logging wrapper
        """        
        super().__init__(config, wandb_wrapper)
        self.config: LLMFinetuningConfig = config # cast to ECGTokenizerLinearProbingConfig to avoid type errors
        
    def run(self):
        """Execute the LLM finetuning workflow."""
        super().run()
        
    def _setup_training_objects(self)->dict[str, Any]: 
        """Setup objects required for LLM finetuning training.
        
        Loads pretrained tokenizer, freezes encoder/quantizer components,
        and prepares training infrastructure including data loaders,
        optimizer, scheduler, and gradient scaler.
        
        Returns:
            Dictionary containing training objects: optimizer, scheduler, 
            scaler, model, and data loaders
        """        
        # Load the pretrained tokenizer
        state_dict = self._load_checkpoint(self.config.pretrained_tokenizer_path)
        
        # Get the config from the pretrained tokenizer
        pretrained_config = state_dict['config']
        if self.config.is_ref_device:
            print(f"Pretrained config: {pretrained_config}")                
        
        # Set encoder_name to the pretrained encoder_name -> otherwise the encoder_name is not saved in the checkpoint
        self.config.encoder_name = pretrained_config.encoder_name
        # Set quantizer_name to the pretrained quantizer_name -> otherwise the quantizer_name is not saved in the checkpoint
        self.config.quantizer_name = pretrained_config.quantizer_name
        # Set num_quantizers to the pretrained num_quantizers -> otherwise the num_quantizers is not saved in the checkpoint
        self.config.num_quantizers = pretrained_config.num_quantizers
        # Set codebook_size to the pretrained codebook_size -> otherwise the codebook_size is not saved in the checkpoint
        self.config.codebook_size = pretrained_config.codebook_size
        
        # Initialize the tokenizer with the appropriate configuration
        ecg_tokenizer: ECG_Tokenizer_Wrapper = ModelRegistry.get(self.config.model_name)(
            encoder_name=pretrained_config.encoder_name, 
            quantizer_name=pretrained_config.quantizer_name,
            decoder_name=self.config.decoder_name, # use the decoder from the current config
            num_quantizers=pretrained_config.num_quantizers,
            codebook_size=pretrained_config.codebook_size,
            decoder_mode=self.config.decoder_mode, # use the decoder mode from the current config
            adapter_name=self.config.adapter_name,
            gpt2_model_name=self.config.gpt2_model_name,
            llm_input_embedding_size=self.config.llm_input_embedding_size,
        ).to(self.config.device)
        # Set the codebook size to the pretrained codebook size
        self.config.codebook_size = pretrained_config.codebook_size # required to compute % of active codebook during training
        
        # Load the pretrained state dict
        pretrained_state_dict = state_dict['model_state_dict']
        ecg_tokenizer._load_pretrained_weights(pretrained_state_dict, freeze_pretrained_components=True)
        
        # Print training configuration
        self._print_training_config(ecg_tokenizer)
               
        # Load the tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained(self.config.tokenizer_name)
        tokenizer.pad_token = tokenizer.eos_token
               
        # Get the dataloaders
        train_dataloader: DataLoader = get_distributed_clinical_report_dataloader(
            dataset_path=self.config.train_dataset_path,
            signal_path_column=self.config.signal_path_column,
            ecg_waveform_length=self.config.ecg_waveform_length,
            ecg_num_leads=self.config.ecg_num_leads,
            tokenizer=tokenizer,
            max_token_length=self.config.max_token_length,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            num_replicas=self.config.world_size,
            rank=self.config.device,
            shuffle=True, 
            pin_memory=True
        )
        
        validation_dataloader: DataLoader = get_distributed_clinical_report_dataloader(
            dataset_path=self.config.validation_dataset_path,
            signal_path_column=self.config.signal_path_column,
            ecg_waveform_length=self.config.ecg_waveform_length,
            ecg_num_leads=self.config.ecg_num_leads,
            tokenizer=tokenizer,
            max_token_length=self.config.max_token_length,
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
                "params": ecg_tokenizer.module.decoder.gpt2.parameters(),
                "lr": self.config.llm_lr,
                "weight_decay": self.config.llm_weight_decay,
                "name": "llm"
            },
            {
                "params": ecg_tokenizer.module.decoder.adapter.parameters(),
                "lr": self.config.adapter_lr,
                "weight_decay": self.config.adapter_weight_decay,
                "name": "adapter"
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
            "model": ecg_tokenizer,
            "train_dataloader": train_dataloader,
            "validation_dataloader": validation_dataloader
        }
    
    def _print_training_config(self, model: ECG_Tokenizer_Wrapper):
        """Print detailed training configuration and model statistics.
        
        Displays parameter counts, trainable ratios, and component-wise
        breakdown of the model architecture for LLM finetuning setup.
        
        Args:
            model: ECG tokenizer wrapper model to analyze
        """
        print("\n" + "="*60)
        print("LLM FINETUNING CONFIGURATION")
        print("="*60)
        
        training_info = model.get_training_info()
        
        print(f"Total parameters: {training_info['total_params']:,}")
        print(f"Trainable parameters: {training_info['trainable_params']:,}")
        print(f"Frozen parameters: {training_info['frozen_params']:,}")
        print(f"Trainable ratio: {training_info['trainable_ratio']:.2f}%")

        print(f"\nAdapter: {model.decoder.adapter_name}")
        print(f"Decoder: {model.decoder_name} ({model.decoder_mode.value} mode)")
        
        print("\nComponent-wise breakdown:")
        print(f"  Encoder: {training_info['encoder']['trainable']:,}/{training_info['encoder']['total']:,} trainable")
        print(f"  Quantizer: {training_info['quantizer']['trainable']:,}/{training_info['quantizer']['total']:,} trainable")
        print(f"    └─ MLPs: {training_info['quantizer_mlps']['trainable']:,}/{training_info['quantizer_mlps']['total']:,} trainable")
        print(f"  Decoder: {training_info['decoder']['trainable']:,}/{training_info['decoder']['total']:,} trainable")
        
        print("="*60 + "\n")
    
    def _setup_inference_objects(self)->dict[str, Any]:
        """Setup objects for inference mode.
        
        Loads pretrained tokenizer, initializes tokenizer,
        and prepares data loader for inference on clinical reports.
        
        Returns:
            Dictionary containing validation data loader and model for inference
        """        
        # Load the pretrained tokenizer
        state_dict = self._load_checkpoint(self.config.pretrained_tokenizer_path)
        
        # Get the config from the pretrained tokenizer
        pretrained_config = state_dict['config']
        if self.config.is_ref_device:
            print(f"Pretrained config: {pretrained_config}")              
        
        # Initialize the tokenizer with the appropriate configuration
        # Use the pretrained config to initialize the ecg_tokenizer_wrapper class
        ecg_tokenizer: ECG_Tokenizer_Wrapper = ModelRegistry.get(self.config.pipeline_project)(
            encoder_name=pretrained_config.encoder_name,
            quantizer_name=pretrained_config.quantizer_name,
            decoder_name=pretrained_config.decoder_name, 
            num_quantizers=pretrained_config.num_quantizers,
            codebook_size=pretrained_config.codebook_size,
            decoder_mode=pretrained_config.decoder_mode,
            adapter_name=pretrained_config.adapter_name,
            gpt2_model_name=pretrained_config.gpt2_model_name if hasattr(pretrained_config, 'gpt2_model_name') else self.config.gpt2_model_name,
            llm_input_embedding_size=pretrained_config.llm_input_embedding_size if hasattr(pretrained_config, 'llm_input_embedding_size') else self.config.llm_input_embedding_size,
        ).to(self.config.device)
        # Set the codebook size to the pretrained codebook size
        self.config.codebook_size = pretrained_config.codebook_size # required to compute % of active codebook during training
        
        # Load the pretrained state dict
        pretrained_state_dict = state_dict['model_state_dict']
        ecg_tokenizer._load_state_dict(pretrained_state_dict, strict=True)
        ecg_tokenizer.eval()
        
        # Load the tokenizer
        tokenizer: GPT2Tokenizer = GPT2Tokenizer.from_pretrained(self.config.tokenizer_name)
        tokenizer.pad_token = tokenizer.eos_token
        
        # Get the dataloaders
        validation_dataloader: DataLoader = get_distributed_clinical_report_dataloader(
            dataset_path=self.config.validation_dataset_path,
            signal_path_column=self.config.signal_path_column,
            ecg_waveform_length=self.config.ecg_waveform_length,
            ecg_num_leads=self.config.ecg_num_leads,
            tokenizer=tokenizer,
            max_token_length=self.config.max_token_length,
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
        
        return {
            "model": ecg_tokenizer,
            "validation_dataloader": validation_dataloader
        }
    
    def _setup_extraction_objects(self)->dict[str, Any]:
        """Setup objects for extraction mode.
        
        Raises:
            NotImplementedError: Extraction not implemented for LLM finetuning
        """        
        raise NotImplementedError("Extraction is not implemented for this project")