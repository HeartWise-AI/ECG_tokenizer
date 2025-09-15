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
        
        # Prepare LoRA config if enabled
        lora_config = None
        if self.config.use_lora:
            lora_config = {
                'r': self.config.lora_r,
                'alpha': self.config.lora_alpha,
                'dropout': self.config.lora_dropout,
                'target_modules': self.config.lora_target_modules,
                'bias': self.config.lora_bias
            }

        # Load the tokenizer first (may add special tokens)
        tokenizer_name = self.config.tokenizer_name
        # No need to append -Instruct since we're using the correct model name directly
        tokenizer = self._get_tokenizer(tokenizer_name)
        
        # Initialize the tokenizer with the appropriate configuration
        ecg_tokenizer: ECG_Tokenizer_Wrapper = ModelRegistry.get(self.config.model_name)(
            encoder_name=pretrained_config.encoder_name, 
            quantizer_name=pretrained_config.quantizer_name,
            decoder_name=self.config.decoder_name, # use the decoder from the current config
            num_quantizers=pretrained_config.num_quantizers,
            codebook_size=pretrained_config.codebook_size,
            decoder_mode=self.config.decoder_mode, # use the decoder mode from the current config
            adapter_name=self.config.adapter_name,
            huggingface_model_name=self.config.huggingface_model_name,
            llm_input_embedding_size=self.config.llm_input_embedding_size,
            tokenizer=tokenizer,
            use_lora=self.config.use_lora,
            lora_config=lora_config
        ).to(self.config.device)
        
        # Resize model embeddings if new tokens were added
        if getattr(self.config, 'instruct_mode', False):
            ecg_tokenizer.decoder.llm_model.resize_token_embeddings(len(tokenizer))
        # Set the codebook size to the pretrained codebook size
        self.config.codebook_size = pretrained_config.codebook_size # required to compute % of active codebook during training
        
        # Load the pretrained state dict
        pretrained_state_dict = state_dict['model_state_dict']
        ecg_tokenizer._load_pretrained_weights(pretrained_state_dict, freeze_pretrained_components=True)
        
        # Print training configuration
        self._print_training_config(ecg_tokenizer)
               
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
            pin_memory=True,
            instruct_mode=getattr(self.config, 'instruct_mode', False),
            num_ecg_tokens=getattr(self.config, 'num_ecg_tokens', 128),
            ecg_token_start_id=getattr(self.config, 'ecg_token_start_id', None)
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
            pin_memory=True,
            instruct_mode=getattr(self.config, 'instruct_mode', False),
            num_ecg_tokens=getattr(self.config, 'num_ecg_tokens', 128),
            ecg_token_start_id=getattr(self.config, 'ecg_token_start_id', None)
        )

        # Wrap the model in DDP
        ecg_tokenizer = DistributedUtils.DDP(
            ecg_tokenizer,
            device_ids=[self.config.device]
        )
        
        # Get the parameter groups
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
        
        # Print LoRA configuration if enabled
        if self.config.use_lora:
            print(f"\nLoRA Configuration:")
            print(f"  Rank (r): {self.config.lora_r}")
            print(f"  Alpha: {self.config.lora_alpha}")
            print(f"  Dropout: {self.config.lora_dropout}")
            print(f"  Target modules: {self.config.lora_target_modules}")
            print(f"  Bias: {self.config.lora_bias}")
            
            # Print LoRA-specific parameter counts if available
            llm_model = None
            if hasattr(model.decoder, 'llm_model'):
                llm_model = model.decoder.llm_model
            elif hasattr(model.decoder, 'llm'):
                llm_model = model.decoder.llm
            
            if llm_model and hasattr(llm_model, 'peft_config'):
                total_llm_params = sum(p.numel() for p in llm_model.parameters())
                trainable_llm_params = sum(p.numel() for p in llm_model.parameters() if p.requires_grad)
                print(f"  LoRA trainable parameters: {trainable_llm_params:,}/{total_llm_params:,} ({100 * trainable_llm_params / total_llm_params:.2f}%)")
        
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
        # For inference, we need to check if the checkpoint has LoRA weights
        checkpoint_has_lora = any('lora_A' in key or 'lora_B' in key or 'base_layer' in key for key in state_dict['model_state_dict'].keys())
        
        # If checkpoint has LoRA weights, we should load with LoRA enabled
        # If checkpoint doesn't have LoRA weights, we should load without LoRA
        use_lora_for_inference = checkpoint_has_lora and (hasattr(pretrained_config, 'use_lora') and pretrained_config.use_lora)
        
        if self.config.is_ref_device:
            print(f"Checkpoint has LoRA weights: {checkpoint_has_lora}")
            print(f"Using LoRA for inference: {use_lora_for_inference}")
        
        ecg_tokenizer: ECG_Tokenizer_Wrapper = ModelRegistry.get(self.config.pipeline_project)(
            encoder_name=pretrained_config.encoder_name,
            quantizer_name=pretrained_config.quantizer_name,
            decoder_name=pretrained_config.decoder_name, 
            num_quantizers=pretrained_config.num_quantizers,
            codebook_size=pretrained_config.codebook_size,
            decoder_mode=pretrained_config.decoder_mode,
            adapter_name=pretrained_config.adapter_name,
            huggingface_model_name=pretrained_config.huggingface_model_name if hasattr(pretrained_config, 'huggingface_model_name') else self.config.huggingface_model_name,
            llm_input_embedding_size=pretrained_config.llm_input_embedding_size if hasattr(pretrained_config, 'llm_input_embedding_size') else self.config.llm_input_embedding_size,
            tokenizer=self._get_tokenizer(self.config.tokenizer_name),
            use_lora=use_lora_for_inference,
            lora_config={
                'r': pretrained_config.lora_r if hasattr(pretrained_config, 'lora_r') else 16,
                'alpha': pretrained_config.lora_alpha if hasattr(pretrained_config, 'lora_alpha') else 32,
                'dropout': pretrained_config.lora_dropout if hasattr(pretrained_config, 'lora_dropout') else 0.1,
                'target_modules': pretrained_config.lora_target_modules if hasattr(pretrained_config, 'lora_target_modules') else None,
                'bias': pretrained_config.lora_bias if hasattr(pretrained_config, 'lora_bias') else 'none'
            } if use_lora_for_inference else None
        ).to(self.config.device)
        # Set the codebook size to the pretrained codebook size
        self.config.codebook_size = pretrained_config.codebook_size # required to compute % of active codebook during training
        
        # Load the pretrained state dict
        pretrained_state_dict = state_dict['model_state_dict']
        ecg_tokenizer._load_state_dict(pretrained_state_dict, strict=True)
        
        # Set LoRA to inference mode if using LoRA
        if use_lora_for_inference:
            ecg_tokenizer.set_lora_inference_mode(True)
            
        ecg_tokenizer.eval()
        
        # Load the tokenizer
        tokenizer = self._get_tokenizer(self.config.tokenizer_name)
        
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
            pin_memory=True,
            instruct_mode=getattr(self.config, 'instruct_mode', False),
            num_ecg_tokens=getattr(self.config, 'num_ecg_tokens', 128),
            ecg_token_start_id=getattr(self.config, 'ecg_token_start_id', None)
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
    
    def _get_tokenizer(self, tokenizer_name: str):
        """Get the appropriate tokenizer using AutoTokenizer for all models."""
        # Use AutoTokenizer which works for all Hugging Face models
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if getattr(self.config, 'instruct_mode', False):
        # Override chat template to remove system additions
            # custom_template = """<|begin_of_text|>{% for message in messages %}{% if message['role'] == 'system' %}<|start_header_id|>system<|end_header_id|>

            #     {{ message['content'] }}<|eot_id|>{% elif message['role'] == 'user' %}<|start_header_id|>user<|end_header_id|>

            #     {{ message['content'] }}<|eot_id|>{% elif message['role'] == 'assistant' %}<|start_header_id|>assistant<|end_header_id|>

            #     {{ message['content'] }}<|eot_id|>{% endif %}{% endfor %}{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>

            #     {% endif %}"""

            custom_template = (
                "<|begin_of_text|>"
                "{% for message in messages %}"
                    "{% if message['role'] == 'system' %}"
                        "<|start_header_id|>system<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
                    "{% elif message['role'] == 'user' %}"
                        "<|start_header_id|>user<|end_header_id|>\n\n"
                        # Add placeholders for the ECG tokens here
                        "<|start_ecg|>" + "".join([f"<|ecg_pos_{i}|>" for i in range(self.config.num_ecg_tokens)]) + "<|end_ecg|>\n"
                        "{{ message['content'] }}<|eot_id|>"
                    "{% elif message['role'] == 'assistant' %}"
                        "<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
                    "{% endif %}"
                "{% endfor %}"
                "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
            )
            tokenizer.chat_template = custom_template
        
        # Ensure pad token is set - use eos_token if no pad_token exists
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            
        # Ensure pad_token_id is a single integer (not a list)
        if hasattr(tokenizer, 'pad_token_id') and isinstance(tokenizer.pad_token_id, list):
            tokenizer.pad_token_id = tokenizer.pad_token_id[0]
            
        # Add ECG special tokens for instruction mode
        if getattr(self.config, 'instruct_mode', False):
            special_tokens_dict = {
                'additional_special_tokens': ['<|start_ecg|>', '<|end_ecg|>']
            }
            num_added_tokens = tokenizer.add_special_tokens(special_tokens_dict)
            if num_added_tokens > 0 and self.config.is_ref_device:
                print(f"Added {num_added_tokens} ECG special tokens to tokenizer")
        
        # Add 128 position-specific ECG tokens and record their start id
        if getattr(self.config, 'instruct_mode', False):
            # Determine if tokens already exist
            ecg_tokens = [f"<|ecg_pos_{i}|>" for i in range(getattr(self.config, 'num_ecg_tokens', 128))]
            existing_id = tokenizer.convert_tokens_to_ids(ecg_tokens[0])
            if existing_id is None or existing_id == -1:
                original_vocab_size = len(tokenizer)
                tokenizer.add_tokens(ecg_tokens, special_tokens=True)
                self.config.ecg_token_start_id = original_vocab_size
                if self.config.is_ref_device:
                    print(f"Added {len(ecg_tokens)} ECG position tokens starting at id {self.config.ecg_token_start_id}")
            else:
                self.config.ecg_token_start_id = int(existing_id)
                if self.config.is_ref_device:
                    print(f"ECG position tokens already present starting at id {self.config.ecg_token_start_id}")
        
        # Ensure chat template exists for instruction tuning
        if getattr(self.config, 'instruct_mode', False):
            chat_tmpl = getattr(tokenizer, 'chat_template', None)
            if not chat_tmpl and 'llama' in tokenizer_name.lower():
                tokenizer.chat_template = (
                    "{{ bos_token }}"
                    "{% for message in messages %}"
                    "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n"
                    "{{ message['content'] }}<|eot_id|>"
                    "{% endfor %}"
                    "{% if add_generation_prompt %}"
                    "<|start_header_id|>assistant<|end_header_id|>\n\n"
                    "{% endif %}"
                )
            
        return tokenizer
    
    def _get_llm_parameters(self, decoder):
        """Get LLM parameters from the decoder's LLM model."""
        # First try to get the LLM model
        llm_model = None
        if hasattr(decoder, 'llm_model'):
            llm_model = decoder.llm_model
        else:
            # Fallback to look for any transformer model attribute
            for attr_name in ['transformer', 'model', 'llm']:
                if hasattr(decoder, attr_name):
                    llm_model = getattr(decoder, attr_name)
                    break
        
        if llm_model is None:
            raise AttributeError(f"Decoder {type(decoder).__name__} doesn't have a recognizable LLM model attribute")
        
        # For LoRA, only return trainable parameters
        if self.config.use_lora and hasattr(llm_model, 'peft_config'):
            return [p for p in llm_model.parameters() if p.requires_grad]
        else:
            # Original implementation for full finetuning
            return llm_model.parameters()