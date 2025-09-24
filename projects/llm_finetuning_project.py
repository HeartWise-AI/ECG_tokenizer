import torch

from typing import Any, Optional
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
try:
    from torch.amp import GradScaler as _TorchGradScaler  # type: ignore
    _GRAD_SCALER_ARGS = ('cuda',)
except (ImportError, AttributeError):  # pragma: no cover - fallback for older torch
    from torch.cuda.amp import GradScaler as _TorchGradScaler  # type: ignore
    _GRAD_SCALER_ARGS = ()
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


def _create_grad_scaler() -> _TorchGradScaler:
    """Instantiate AMP GradScaler with forward-compatible API."""
    try:
        return _TorchGradScaler(*_GRAD_SCALER_ARGS)
    except TypeError:  # Older torch versions without device arg
        return _TorchGradScaler()

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
        resume_checkpoint_path = getattr(self.config, 'resume_checkpoint_path', None)
        checkpoint_path = resume_checkpoint_path or self.config.pretrained_tokenizer_path

        # Load the checkpoint (pretrained or resume)
        state_dict = self._load_checkpoint(checkpoint_path)
        
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
        
        # Determine whether any training phase requests LoRA even if globally disabled
        training_phases = getattr(self.config, 'training_phases', {}) or {}
        phase_requests_lora = any(
            isinstance(phase_cfg, dict) and phase_cfg.get('use_lora', False)
            for phase_cfg in training_phases.values()
        )

        # Prepare LoRA config if enabled globally or requested by a training phase
        lora_config = None
        if self.config.use_lora or phase_requests_lora:
            self.config.use_lora = True
            lora_config = {
                'r': self.config.lora_r,
                'lora_alpha': self.config.lora_alpha,
                'lora_dropout': self.config.lora_dropout,
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
        
        resume_epoch = 0
        if resume_checkpoint_path:
            resume_epoch = int(state_dict.get('epoch', 0))
            model_state = state_dict.get('model_state_dict', {})
            ecg_tokenizer._load_state_dict(model_state, strict=False)
            # Maintain frozen encoder/quantizer blocks as in initial training
            ecg_tokenizer._freeze_pretrained_components()
        else:
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
            ecg_token_start_id=getattr(self.config, 'ecg_token_start_id', None),
            prompt_column=self.config.prompt_column,
            answer_column=self.config.answer_column,
            category_column=self.config.category_column
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
            ecg_token_start_id=getattr(self.config, 'ecg_token_start_id', None),
            prompt_column=self.config.prompt_column,
            answer_column=self.config.answer_column,
            category_column=self.config.category_column
        )

        # Wrap the model in DDP
        ecg_tokenizer = DistributedUtils.DDP(
            ecg_tokenizer,
            device_ids=[self.config.device],
            find_unused_parameters=True
        )

        decoder_module = ecg_tokenizer.module.decoder

        training_phases = getattr(self.config, 'training_phases', {}) or {}
        phase1_cfg = training_phases.get('phase1_alignment', {}) or {}
        phase2_cfg = training_phases.get('phase2_finetuning', {}) or {}
        phase1_epochs = int(phase1_cfg.get('epochs', 0))

        # Determine which phase produced the loaded checkpoint so optimizer groups align with state dict
        if resume_checkpoint_path and resume_epoch >= phase1_epochs and phase2_cfg:
            initial_phase_cfg = phase2_cfg
        else:
            initial_phase_cfg = phase1_cfg
        self._apply_model_phase_settings(ecg_tokenizer.module, initial_phase_cfg)

        param_groups = self._build_optimizer_param_groups(
            decoder_module=decoder_module,
            phase_overrides=initial_phase_cfg
        )


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
        scaler = _create_grad_scaler()

        start_epoch = resume_epoch + 1 if resume_checkpoint_path else 1

        if resume_checkpoint_path:
            optimizer_state = state_dict.get('optimizer_state_dict')
            if optimizer_state is not None:
                self._load_optimizer_state_dict(optimizer, optimizer_state)
            scheduler_state = state_dict.get('scheduler_state_dict')
            if scheduler_state is not None and scheduler is not None:
                try:
                    scheduler.load_state_dict(scheduler_state)
                except Exception as exc:
                    print(f"⚠️ Could not load scheduler state from checkpoint: {exc}")
            scaler_state = state_dict.get('scaler_state_dict')
            if scaler_state is not None and scaler is not None:
                try:
                    scaler.load_state_dict(scaler_state)
                except Exception as exc:
                    print(f"⚠️ Could not load scaler state from checkpoint: {exc}")

        return {
            "optimizer": optimizer,
            "scheduler": scheduler,
            "scaler": scaler,
            "model": ecg_tokenizer,
            "train_dataloader": train_dataloader,
            "validation_dataloader": validation_dataloader,
            "start_epoch": max(1, start_epoch)
        }

    def _apply_model_phase_settings(self, model: ECG_Tokenizer_Wrapper, phase_config: dict | None):
        """Apply freeze/unfreeze toggles before building optimizer param groups."""
        if not phase_config:
            return

        decoder = getattr(model, 'decoder', None)
        if decoder is None:
            return

        llm_module = self._get_llm_module_from_decoder(decoder)
        freeze_llm = phase_config.get('freeze_llm')
        if freeze_llm is True and llm_module is not None:
            if hasattr(decoder, 'freeze_llm_parameters'):
                decoder.freeze_llm_parameters()
            else:
                for param in llm_module.parameters():
                    param.requires_grad = False
        elif freeze_llm is False and llm_module is not None:
            if hasattr(decoder, 'unfreeze_llm_parameters'):
                decoder.unfreeze_llm_parameters()
            else:
                for param in llm_module.parameters():
                    param.requires_grad = True

        # LoRA enable/disable is managed in the runner when phases change

    def _build_optimizer_param_groups(self, decoder_module, phase_overrides: dict | None):
        """Construct optimizer parameter groups respecting phase-specific overrides."""
        phase_overrides = phase_overrides or {}

        adapter_module = decoder_module.adapter

        llm_params = [p for p in self._get_llm_parameters(decoder_module) if p.requires_grad]

        embedding_param = None
        if hasattr(decoder_module, 'llm_model'):
            embedding_param = decoder_module.llm_model.get_input_embeddings().weight
            if embedding_param in llm_params:
                llm_params = [p for p in llm_params if p is not embedding_param]
            if embedding_param is not None and not embedding_param.requires_grad:
                embedding_param = None

        adapter_params = [p for p in adapter_module.parameters() if p.requires_grad]
        cross_attention_params = []
        if hasattr(adapter_module, 'cross_attention_layers'):
            for layer in adapter_module.cross_attention_layers:
                cross_attention_params.extend([p for p in layer.parameters() if p.requires_grad])
        elif hasattr(adapter_module, 'cross_attention'):
            cross_attention_params.extend([p for p in adapter_module.cross_attention.parameters() if p.requires_grad])
            if hasattr(adapter_module, 'attention_norm'):
                cross_attention_params.extend([p for p in adapter_module.attention_norm.parameters() if p.requires_grad])
            if hasattr(adapter_module, 'attention_dropout'):
                cross_attention_params.extend([p for p in adapter_module.attention_dropout.parameters() if p.requires_grad])

        seen_ids: set[int] = set()
        unique_cross = []
        for param in cross_attention_params:
            pid = id(param)
            if pid not in seen_ids:
                unique_cross.append(param)
                seen_ids.add(pid)
        cross_attention_params = unique_cross
        cross_param_ids = {id(p) for p in cross_attention_params}
        adapter_core_params = [p for p in adapter_params if id(p) not in cross_param_ids]

        llm_lr = float(phase_overrides.get('llm_lr', self.config.llm_lr))
        adapter_lr = float(phase_overrides.get('adapter_lr', self.config.adapter_lr))
        cross_attention_lr = float(phase_overrides.get('cross_attention_lr', adapter_lr))
        ecg_embedding_lr = float(phase_overrides.get('ecg_embedding_lr', llm_lr))

        llm_weight_decay = float(phase_overrides.get('llm_weight_decay', self.config.llm_weight_decay))
        adapter_weight_decay = float(phase_overrides.get('adapter_weight_decay', self.config.adapter_weight_decay))
        cross_attention_weight_decay = float(phase_overrides.get('cross_attention_weight_decay', adapter_weight_decay))
        ecg_embedding_weight_decay = float(phase_overrides.get('ecg_embedding_weight_decay', llm_weight_decay))

        param_groups = []
        if llm_params:
            param_groups.append({
                "params": llm_params,
                "lr": llm_lr,
                "weight_decay": llm_weight_decay,
                "name": "llm"
            })
        if embedding_param is not None:
            param_groups.append({
                "params": [embedding_param],
                "lr": ecg_embedding_lr,
                "weight_decay": ecg_embedding_weight_decay,
                "name": "ecg_embeddings"
            })
        if adapter_core_params:
            param_groups.append({
                "params": adapter_core_params,
                "lr": adapter_lr,
                "weight_decay": adapter_weight_decay,
                "name": "adapter"
            })
        if cross_attention_params:
            param_groups.append({
                "params": cross_attention_params,
                "lr": cross_attention_lr,
                "weight_decay": cross_attention_weight_decay,
                "name": "cross_attention"
            })

        return param_groups

    def _load_optimizer_state_dict(self, optimizer: Optimizer, saved_state: dict[str, Any]):
        """Load optimizer state with graceful fallback when parameter groups change."""
        try:
            optimizer.load_state_dict(saved_state)
            return
        except ValueError as exc:
            print(f"⚠️ Could not load optimizer state from checkpoint: {exc}. Remapping to current parameter groups...")
        except RuntimeError as exc:
            print(f"⚠️ Could not load optimizer state from checkpoint: {exc}. Remapping to current parameter groups...")

        remapped_state = self._remap_optimizer_state(saved_state, optimizer)
        if remapped_state is None:
            print("⚠️ Optimizer state remap failed; proceeding with freshly initialized optimizer state.")
            return
        try:
            optimizer.load_state_dict(remapped_state)
        except Exception as final_exc:  # pragma: no cover - defensive
            print(f"⚠️ Optimizer state load failed after remap: {final_exc}. Using fresh optimizer state.")

    def _remap_optimizer_state(self, saved_state: dict[str, Any], optimizer: Optimizer) -> Optional[dict[str, Any]]:
        """Adapt a saved optimizer state to the current optimizer parameter order.

        Returns a new state dict aligned with the optimizer's param groups, or None if remap
        is not possible.
        """
        if not saved_state:
            return None

        saved_groups = saved_state.get('param_groups') or []
        saved_state_map = saved_state.get('state') or {}

        # Flatten saved states in group order for sequential reassignment
        saved_param_states = []
        for group in saved_groups:
            for param_idx in group.get('params', []):
                saved_param_states.append(saved_state_map.get(param_idx, {}))

        # Build new param_groups mirroring the optimizer's current layout
        new_state: dict[int, Any] = {}
        new_param_groups: list[dict[str, Any]] = []
        saved_iter_idx = 0

        for group in optimizer.param_groups:
            new_group = {k: v for k, v in group.items() if k != 'params'}
            param_indices: list[int] = []
            for param in group['params']:
                state = {}
                if saved_iter_idx < len(saved_param_states):
                    state = saved_param_states[saved_iter_idx]
                # Use incremental integer keys to follow PyTorch optimizer convention
                param_indices.append(saved_iter_idx)
                new_state[saved_iter_idx] = state
                saved_iter_idx += 1
            new_group['params'] = param_indices
            new_param_groups.append(new_group)

        return {
            'state': new_state,
            'param_groups': new_param_groups,
        }

    def _get_llm_module_from_decoder(self, decoder):
        if hasattr(decoder, 'llm_model'):
            return decoder.llm_model
        for attr_name in ['transformer', 'model', 'llm']:
            if hasattr(decoder, attr_name):
                return getattr(decoder, attr_name)
        return None
    
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
            ecg_token_start_id=getattr(self.config, 'ecg_token_start_id', None),
            prompt_column=self.config.prompt_column,
            answer_column=self.config.answer_column,
            category_column=self.config.category_column
        )

        # Wrap the model in DDP
        ecg_tokenizer = DistributedUtils.DDP(
            ecg_tokenizer,
            device_ids=[self.config.device],
            find_unused_parameters=True
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
                        # Add placeholders for the ECG prefix (actual embeddings supplied externally)
                        "<|start_ecg|><|end_ecg|>\n"
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
        
        # Prefix-tuning path: rely on per-example embeddings instead of dedicated vocab rows
        if getattr(self.config, 'instruct_mode', False):
            self.config.ecg_token_start_id = None
        
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
        llm_model = self._get_llm_module_from_decoder(decoder)

        if llm_model is None:
            raise AttributeError(f"Decoder {type(decoder).__name__} doesn't have a recognizable LLM model attribute")
        
        # For LoRA, only return trainable parameters
        if self.config.use_lora and hasattr(llm_model, 'peft_config'):
            return [p for p in llm_model.parameters() if p.requires_grad]
        else:
            # Original implementation for full finetuning
            return llm_model.parameters()
