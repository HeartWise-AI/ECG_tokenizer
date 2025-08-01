import torch

from typing import Any
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from torch.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import LRScheduler

from transformers import AutoTokenizer

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
        
    def _load_and_setup_model(self, checkpoint_path: str, model_registry_key: str = None, for_training: bool = False) -> tuple[ECG_Tokenizer_Wrapper, dict]:
        """Load checkpoint and setup model with pretrained weights.
        
        Args:
            checkpoint_path: Path to model checkpoint
            model_registry_key: Key for model registry (defaults to self.config.model_name)
            for_training: Whether setting up for training (affects weight loading)
            
        Returns:
            Tuple of (model, pretrained_config)
        """
        # Load checkpoint
        state_dict = self._load_checkpoint(checkpoint_path)
        pretrained_config = state_dict['config']
        
        if self.config.is_ref_device:
            print(f"Loading model from: {checkpoint_path}")
            print(f"Model config: {pretrained_config}")
        
        # Use provided registry key or default to model_name
        registry_key = model_registry_key or self.config.model_name
        
        # Initialize model - training uses current config for some params, inference uses pretrained config
        if for_training:
            ecg_tokenizer = ModelRegistry.get(registry_key)(
                encoder_name=pretrained_config.encoder_name, 
                quantizer_name=pretrained_config.quantizer_name,
                decoder_name=self.config.decoder_name,  # use current config
                num_quantizers=pretrained_config.num_quantizers,
                codebook_size=pretrained_config.codebook_size,
                decoder_mode=self.config.decoder_mode,  # use current config
                adapter_name=self.config.adapter_name,
                huggingface_model_name=self.config.huggingface_model_name,
                llm_input_embedding_size=self.config.llm_input_embedding_size,
            ).to(self.config.device)
        else:
            ecg_tokenizer = ModelRegistry.get(registry_key)(
                encoder_name=pretrained_config.encoder_name,
                quantizer_name=pretrained_config.quantizer_name,
                decoder_name=pretrained_config.decoder_name,  # use pretrained config
                num_quantizers=pretrained_config.num_quantizers,
                codebook_size=pretrained_config.codebook_size,
                decoder_mode=pretrained_config.decoder_mode,  # use pretrained config
                adapter_name=pretrained_config.adapter_name,
                huggingface_model_name=pretrained_config.huggingface_model_name if hasattr(pretrained_config, 'huggingface_model_name') else self.config.huggingface_model_name,
                llm_input_embedding_size=pretrained_config.llm_input_embedding_size if hasattr(pretrained_config, 'llm_input_embedding_size') else self.config.llm_input_embedding_size,
            ).to(self.config.device)
        
        # Set codebook size in current config
        self.config.codebook_size = pretrained_config.codebook_size
        
        # Load weights
        pretrained_state_dict = state_dict['model_state_dict']
        if for_training:
            ecg_tokenizer._load_pretrained_weights(pretrained_state_dict, freeze_pretrained_components=True)
        else:
            ecg_tokenizer._load_state_dict(pretrained_state_dict, strict=True)
            ecg_tokenizer.eval()
        
        return ecg_tokenizer, pretrained_config

    def _create_validation_dataloader(self, tokenizer, shuffle: bool = False) -> DataLoader:
        """Create validation dataloader with common parameters.
        
        Args:
            tokenizer: Tokenizer for the dataset
            shuffle: Whether to shuffle the data
            
        Returns:
            Validation DataLoader
        """
        return get_distributed_clinical_report_dataloader(
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
            shuffle=shuffle,
            pin_memory=True
        )

    def _create_test_dataloader(self, tokenizer, shuffle: bool = False) -> DataLoader:
        """Create test dataloader with common parameters.
        
        Args:
            tokenizer: Tokenizer for the dataset
            shuffle: Whether to shuffle the data
            
        Returns:
            Test DataLoader
        """
        return get_distributed_clinical_report_dataloader(
            dataset_path=self.config.test_dataset_path,
            signal_path_column=self.config.signal_path_column,
            ecg_waveform_length=self.config.ecg_waveform_length,
            ecg_num_leads=self.config.ecg_num_leads,
            tokenizer=tokenizer,
            max_token_length=self.config.max_token_length,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            num_replicas=self.config.world_size,
            rank=self.config.device,
            shuffle=shuffle,
            pin_memory=True
        )

    def _wrap_model_for_distributed(self, model: ECG_Tokenizer_Wrapper) -> ECG_Tokenizer_Wrapper:
        """Wrap model in DDP for distributed training/inference.
        
        Args:
            model: ECG tokenizer wrapper model
            
        Returns:
            DDP-wrapped model
        """
        return DistributedUtils.DDP(
            model,
            device_ids=[self.config.device]
        )

    def _setup_training_objects(self)->dict[str, Any]: 
        """Setup objects required for LLM finetuning training."""        
        # Load model and config
        ecg_tokenizer, pretrained_config = self._load_and_setup_model(
            self.config.pretrained_tokenizer_path, 
            for_training=True
        )
        
        # Copy pretrained config values to current config
        self.config.encoder_name = pretrained_config.encoder_name
        self.config.quantizer_name = pretrained_config.quantizer_name
        self.config.num_quantizers = pretrained_config.num_quantizers
        
        # Print training configuration
        self._print_training_config(ecg_tokenizer)
        
        # Load tokenizer
        tokenizer = self._get_tokenizer(self.config.tokenizer_name)
        
        # Create dataloaders
        train_dataloader = get_distributed_clinical_report_dataloader(
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
        
        validation_dataloader = self._create_validation_dataloader(tokenizer, shuffle=False)

        # Wrap model in DDP
        ecg_tokenizer = self._wrap_model_for_distributed(ecg_tokenizer)
        
        # Setup optimizer, scheduler, scaler (training-specific code)
        param_groups = [
            {
                "params": self._get_llm_parameters(ecg_tokenizer.module.decoder),
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
        
        optimizer_class = getattr(torch.optim, self.config.optimizer)
        optimizer: Optimizer = optimizer_class(param_groups)

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
        """Setup objects for inference mode."""        
        # Load model and config
        ecg_tokenizer, _ = self._load_and_setup_model(
            self.config.inference_model_path,
            model_registry_key=self.config.pipeline_project,
            for_training=False
        )
        
        # Load tokenizer and create dataloader
        tokenizer = self._get_tokenizer(self.config.tokenizer_name)
        validation_dataloader = self._create_validation_dataloader(tokenizer, shuffle=False)

        # Wrap model in DDP
        ecg_tokenizer = self._wrap_model_for_distributed(ecg_tokenizer)
        
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
    
    def _setup_validation_objects(self)->dict[str, Any]:
        """Setup objects for standalone validation mode."""               
        # Load model and config
        ecg_tokenizer, _ = self._load_and_setup_model(
            self.config.inference_model_path,
            for_training=False
        )
        
        # Load tokenizer and create dataloader
        tokenizer = self._get_tokenizer(self.config.tokenizer_name)
        validation_dataloader = self._create_validation_dataloader(tokenizer, shuffle=False)

        # Wrap model in DDP
        ecg_tokenizer = self._wrap_model_for_distributed(ecg_tokenizer)
        
        return {
            "model": ecg_tokenizer,
            "validation_dataloader": validation_dataloader
        }

    def _setup_test_objects(self)->dict[str, Any]:
        """Setup objects for standalone test mode.
        
        Loads a trained model checkpoint and prepares test data loader
        for standalone test evaluation. Similar to validation setup but
        uses test dataset instead.
        
        Returns:
            Dictionary containing test data loader and model for testing
        """               
        # Load model and config
        ecg_tokenizer, _ = self._load_and_setup_model(
            self.config.inference_model_path,
            for_training=False
        )
        
        # Load tokenizer and create dataloader
        tokenizer = self._get_tokenizer(self.config.tokenizer_name)
        test_dataloader = self._create_test_dataloader(tokenizer, shuffle=False)

        # Wrap model in DDP
        ecg_tokenizer = self._wrap_model_for_distributed(ecg_tokenizer)
        
        return {
            "model": ecg_tokenizer,
            "test_dataloader": test_dataloader
        }
    
    def _get_tokenizer(self, tokenizer_name: str):
        """Get the appropriate tokenizer using AutoTokenizer for all models."""
        # Use AutoTokenizer which works for all Hugging Face models
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        # Ensure pad token is set - use eos_token if no pad_token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            
        return tokenizer
    
    def _get_llm_parameters(self, decoder):
        """Get LLM parameters from the decoder's LLM model."""
        if hasattr(decoder, 'llm_model'):
            return decoder.llm_model.parameters()
        else:
            # Fallback to look for any transformer model attribute
            for attr_name in ['transformer', 'model', 'llm']:
                if hasattr(decoder, attr_name):
                    return getattr(decoder, attr_name).parameters()
            raise AttributeError(f"Decoder {type(decoder).__name__} doesn't have a recognizable LLM model attribute")