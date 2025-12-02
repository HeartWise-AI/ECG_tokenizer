import torch

from typing import Any, Optional, Tuple
from types import SimpleNamespace
from collections import deque
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
try:
    from torch.amp import GradScaler as _TorchGradScaler  # type: ignore
    _GRAD_SCALER_ARGS = ('cuda',)
except (ImportError, AttributeError):  # pragma: no cover - fallback for older torch
    from torch.cuda.amp import GradScaler as _TorchGradScaler  # type: ignore
    _GRAD_SCALER_ARGS = ()
from torch.optim.lr_scheduler import LRScheduler

from transformers import AutoProcessor, AutoTokenizer

from utils.ddp import DistributedUtils
from utils.schedulers import get_scheduler
from utils.registry import (
    ModelRegistry,
    ProjectRegistry
)
from utils.enums import ProjectName, ModelName
from utils.wandb_wrapper import WandbWrapper
from utils.config import LLMFinetuningConfig, ECGTokenizerTrainingConfig
from projects.base_project import BaseProject
from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper
from data.ecg_clinical_report_dataset import get_distributed_clinical_report_dataloader

# Add the config to the safe globals
torch.serialization.add_safe_globals([LLMFinetuningConfig])
torch.serialization.add_safe_globals([ECGTokenizerTrainingConfig])


def _coerce_config(config_obj: Any) -> Any:
    """Convert raw checkpoint config dictionaries into attribute-friendly objects."""
    if isinstance(config_obj, SimpleNamespace):
        return config_obj
    if isinstance(config_obj, dict):
        return SimpleNamespace(**{key: _coerce_config(value) for key, value in config_obj.items()})
    if isinstance(config_obj, list):
        return [_coerce_config(item) for item in config_obj]
    if isinstance(config_obj, tuple):
        return tuple(_coerce_config(item) for item in config_obj)
    return config_obj


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

    def _uses_qformer_bridge(self) -> bool:
        """Return True if the configured bridge is a Q-Former style bridge."""
        name = str(getattr(self.config, 'bridge_name', '')).lower()
        return 'qformer' in name

    def _resolve_checkpoint_structure(
        self,
        checkpoint_config: Any,
        checkpoint_path: str
    ) -> Tuple[Any, str, str, int, int]:
        """Ensure required structural fields are available for downstream initialization."""
        config_obj = _coerce_config(checkpoint_config)

        encoder_name = getattr(config_obj, 'encoder_name', None)
        quantizer_name = getattr(config_obj, 'quantizer_name', None)
        num_quantizers = getattr(config_obj, 'num_quantizers', None)
        codebook_size = getattr(config_obj, 'codebook_size', None)

        missing_fields = [
            field for field, value in (
                ('encoder_name', encoder_name),
                ('quantizer_name', quantizer_name),
                ('num_quantizers', num_quantizers),
                ('codebook_size', codebook_size),
            ) if value is None
        ]

        fallback_checkpoint_path = getattr(config_obj, 'pretrained_encoder_checkpoint', None)
        if missing_fields and fallback_checkpoint_path and str(fallback_checkpoint_path) != str(checkpoint_path):
            try:
                fallback_state = self._load_checkpoint(str(fallback_checkpoint_path))
                fallback_config = _coerce_config(fallback_state.get('config', {}))
            except Exception as exc:  # pragma: no cover - diagnostic logging only
                fallback_config = None
                if self.config.is_ref_device:
                    print(
                        f"[LLMFinetuningProject] Warning: Unable to load fallback checkpoint "
                        f"{fallback_checkpoint_path} to recover config fields ({missing_fields}). Error: {exc}"
                    )
            else:
                encoder_name = encoder_name or getattr(fallback_config, 'encoder_name', None)
                quantizer_name = quantizer_name or getattr(fallback_config, 'quantizer_name', None)
                num_quantizers = num_quantizers or getattr(fallback_config, 'num_quantizers', None)
                codebook_size = codebook_size or getattr(fallback_config, 'codebook_size', None)

        unresolved_fields = [
            field for field, value in (
                ('encoder_name', encoder_name),
                ('quantizer_name', quantizer_name),
                ('num_quantizers', num_quantizers),
                ('codebook_size', codebook_size),
            ) if value is None
        ]
        if unresolved_fields:
            raise ValueError(
                f"Pretrained checkpoint '{checkpoint_path}' does not provide required fields "
                f"{unresolved_fields}. Ensure the checkpoint includes these values or update "
                f"the configuration."
            )

        try:
            num_quantizers_int = int(num_quantizers)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid 'num_quantizers' value ({num_quantizers}) found in checkpoint '{checkpoint_path}'."
            ) from exc

        try:
            codebook_size_int = int(codebook_size)  # type: ignore[arg-type]
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid 'codebook_size' value ({codebook_size}) found in checkpoint '{checkpoint_path}'."
            ) from exc

        return (
            config_obj,
            str(encoder_name),
            str(quantizer_name),
            num_quantizers_int,
            codebook_size_int,
        )
        
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
        pretrained_config_raw = state_dict['config']
        (
            pretrained_config,
            encoder_name,
            quantizer_name,
            num_quantizers,
            codebook_size,
        ) = self._resolve_checkpoint_structure(pretrained_config_raw, checkpoint_path)
        if self.config.is_ref_device:
            print(f"Pretrained config: {pretrained_config}")                
        
        # Ensure config has the resolved structural attributes available for later use
        self.config.encoder_name = encoder_name  # type: ignore[attr-defined]
        self.config.quantizer_name = quantizer_name  # type: ignore[attr-defined]
        self.config.num_quantizers = num_quantizers  # type: ignore[attr-defined]
        self.config.codebook_size = codebook_size  # type: ignore[attr-defined]
        
        # Determine whether any training phase requests LoRA even if globally disabled
        training_phases = getattr(self.config, 'training_phases', {}) or {}
        # Allow sweep override to freeze/unfreeze phase2 LLM backbone
        phase2_override = getattr(self.config, 'phase2_freeze_llm', None)
        if phase2_override is not None:
            phase2_cfg = training_phases.get('phase2_finetuning', {}) or {}
            phase2_cfg['freeze_llm'] = bool(phase2_override)
            training_phases['phase2_finetuning'] = phase2_cfg
            # Persist override for downstream usage
            self.config.training_phases = training_phases  # type: ignore[attr-defined]
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
            if self.config.lora_top_k_layers is not None:
                lora_config['top_k_layers'] = self.config.lora_top_k_layers

        # Load the tokenizer first (may add special tokens)
        tokenizer_name = self.config.tokenizer_name
        tokenizer, processor = self._get_tokenizer(tokenizer_name)
        self.config.tokenizer = tokenizer  # type: ignore[attr-defined]
        if processor is not None:
            self.config.processor = processor  # type: ignore[attr-defined]
        
        # Initialize the tokenizer with the appropriate configuration
        ecg_tokenizer: ECG_Tokenizer_Wrapper = ModelRegistry.get(self.config.model_name)(
            encoder_name=encoder_name, 
            quantizer_name=quantizer_name,
            decoder_name=self.config.decoder_name, # use the decoder from the current config
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            decoder_mode=self.config.decoder_mode, # use the decoder mode from the current config
            bridge_name=self.config.bridge_name,
            huggingface_model_name=self.config.huggingface_model_name,
            llm_input_embedding_size=self.config.llm_input_embedding_size,
            tokenizer=tokenizer,
            processor=processor,
            num_visual_tokens=(getattr(self.config, 'num_query_tokens', None)
                               or self.config.bridge_num_visual_tokens
                               or self.config.num_ecg_tokens),
            bridge_mid_dim=self.config.bridge_mid_dim,
            bridge_num_heads=self.config.bridge_num_heads,
            bridge_dropout=self.config.bridge_dropout,
            bridge_num_special_tokens=self.config.bridge_num_special_tokens,
            bridge_qformer_layers=getattr(self.config, 'bridge_qformer_layers', None),
            bridge_text_hidden_size=getattr(self.config, 'bridge_text_hidden_size', None),
            bridge_bias_last_codebook=getattr(self.config, 'bridge_bias_last_codebook', None),
            bridge_codebook_dropout=getattr(self.config, 'bridge_codebook_dropout', None),
            bridge_cross_every=getattr(self.config, 'bridge_cross_every', None),
            instruction_dropout=getattr(self.config, 'instruction_dropout', 0.0),
            ecg_waveform_length=self.config.ecg_waveform_length,
            ecg_num_leads=self.config.ecg_num_leads,
            ecg_projection_config=getattr(self.config, 'ecg_projection_config', None),
            ecg_token_start_id=getattr(self.config, 'ecg_token_start_id', None),
            prefix_tuning=getattr(self.config, 'prefix_tuning', False),
            default_generation_kwargs=getattr(self.config, 'default_generation_kwargs', None),
            # Projection-bridge knobs
            bridge_use_sinusoidal_pos_emb=getattr(self.config, 'bridge_use_sinusoidal_pos_emb', None),
            bridge_pos_embedding_max_len=getattr(self.config, 'bridge_pos_embedding_max_len', None),
            bridge_softmax_temp=getattr(self.config, 'bridge_softmax_temp', None),
            bridge_mix_residual=getattr(self.config, 'bridge_mix_residual', None),
            bridge_add_modality_embed=getattr(self.config, 'bridge_add_modality_embed', None),
            bridge_add_cls_token=getattr(self.config, 'bridge_add_cls_token', None),
            use_lora=self.config.use_lora,
            lora_config=lora_config,
            stage1_checkpoint_path=getattr(self.config, 'stage1_checkpoint_path', None),
            pattern_loss_weight=getattr(self.config, 'pattern_loss_weight', None),
            pattern_label_count=(len(self.config.pattern_label_columns) if getattr(self.config, 'pattern_label_columns', None) else None),
        ).to(self.config.device)
        
        # Resize model embeddings if new tokens were added
        if getattr(self.config, 'instruct_mode', False) and hasattr(ecg_tokenizer.decoder, 'llm_model'):
            ecg_tokenizer.decoder.llm_model.resize_token_embeddings(len(tokenizer))
        # Set the codebook size to the pretrained codebook size
        self.config.codebook_size = codebook_size # required to compute % of active codebook during training
        
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
        
        training_phases = getattr(self.config, 'training_phases', {}) or {}
        phase1_cfg = training_phases.get('phase1_alignment', {}) or {}
        phase2_cfg = training_phases.get('phase2_finetuning', {}) or {}
        phase1_epochs = int(phase1_cfg.get('epochs', 0))

        if resume_checkpoint_path and resume_epoch >= phase1_epochs and phase2_cfg:
            initial_phase_cfg = phase2_cfg
            # Check if we're transitioning from phase 1 (no LoRA) to phase 2 (with LoRA)
            # The checkpoint from phase 1 won't have LoRA weights, but phase 2 needs them
            phase2_needs_lora = phase2_cfg.get('use_lora', False)
            if phase2_needs_lora and self.config.use_lora and lora_config:
                # Re-apply LoRA after loading checkpoint since phase 1 didn't have it
                if not hasattr(ecg_tokenizer, '_lora_applied') or not ecg_tokenizer._lora_applied:
                    if self.config.is_ref_device:
                        print("📝 Transitioning to phase 2: Initializing LoRA adapters...")
                    ecg_tokenizer._apply_lora(lora_config)
                    ecg_tokenizer._lora_applied = True
        else:
            initial_phase_cfg = phase1_cfg

        self._apply_model_phase_settings(ecg_tokenizer, initial_phase_cfg)

        # Print training configuration using the pre-DDP model so parameter counts reflect freezing
        self._print_training_config(ecg_tokenizer)

        # Resolve dataset mode-specific columns and instruct toggle
        mode = str(getattr(self.config, 'data_mode', 'qa')).lower()
        if mode not in ("qa", "cf"):
            mode = "qa"
        if mode == "cf":
            instruct_flag = bool(getattr(self.config, 'instruct_mode', False))
            signal_col = "signal_path"
            prompt_col = "question"
            answer_col = "ground_truth_answer"
            category_col = "category"
            # Auto-enable CF eval if not set
            if not getattr(self.config, 'use_cf_eval', False):
                setattr(self.config, 'use_cf_eval', True)
            if not getattr(self.config, 'cf_eval_dataset_path', None):
                setattr(self.config, 'cf_eval_dataset_path', str(self.config.validation_dataset_path))
        else:
            instruct_flag = bool(getattr(self.config, 'instruct_mode', False))
            signal_col = self.config.signal_path_column
            prompt_col = self.config.prompt_column
            answer_col = self.config.answer_column
            category_col = self.config.category_column
        # Use MedGemma-style prompts only when explicitly requested or when base model is MedGemma
        name_blob = f"{getattr(self.config, 'tokenizer_name', '')} {getattr(self.config, 'huggingface_model_name', '')}".lower()
        medgemma_prompt_style = bool(
            getattr(self.config, 'medgemma_prompt_style', False)
            or "medgemma" in name_blob
        )
        debug_print_example = bool(getattr(self.config, 'debug_print_example', False))
        prefix_tuning_enabled = getattr(self.config, 'prefix_tuning', False)
        num_ecg_tokens = getattr(self.config, 'num_ecg_tokens', 128)
        if self._uses_qformer_bridge() or prefix_tuning_enabled:
            num_ecg_tokens = 0

        # Get the dataloaders
        train_dataloader: DataLoader = get_distributed_clinical_report_dataloader(
            dataset_path=self.config.train_dataset_path,
            signal_path_column=signal_col,
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
            instruct_mode=instruct_flag,
            # Use 0 placeholders when using Q-Former (or prefix tuning).
            num_ecg_tokens=num_ecg_tokens,
            ecg_token_start_id=getattr(self.config, 'ecg_token_start_id', None),
            prompt_column=prompt_col,
            answer_column=answer_col,
            category_column=category_col,
            prefix_tuning=getattr(self.config, 'prefix_tuning', False),
            pattern_columns=getattr(self.config, 'pattern_label_columns', None),
            subset_size=None,
            balance_categories=False,
            sampling_seed=None,
            medgemma_prompt_style=medgemma_prompt_style,
            debug_print_example=debug_print_example,
        )
        
        validation_dataloader: DataLoader = get_distributed_clinical_report_dataloader(
            dataset_path=self.config.validation_dataset_path,
            signal_path_column=signal_col,
            ecg_waveform_length=self.config.ecg_waveform_length,
            ecg_num_leads=self.config.ecg_num_leads,
            tokenizer=tokenizer,
            max_token_length=self.config.max_token_length,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            num_replicas=self.config.world_size,
            rank=self.config.device,
            shuffle=getattr(self.config, "validation_shuffle", False), 
            pin_memory=True,
            instruct_mode=instruct_flag,
            # Use 0 placeholders when using Q-Former (or prefix tuning).
            num_ecg_tokens=num_ecg_tokens,
            ecg_token_start_id=getattr(self.config, 'ecg_token_start_id', None),
            prompt_column=prompt_col,
            answer_column=answer_col,
            category_column=category_col,
            prefix_tuning=getattr(self.config, 'prefix_tuning', False),
            pattern_columns=getattr(self.config, 'pattern_label_columns', None),
            subset_size=getattr(self.config, "validation_subset_size", None),
            balance_categories=getattr(self.config, "validation_balance_prompt_categories", False),
            sampling_seed=getattr(self.config, "validation_sampling_seed", None),
            medgemma_prompt_style=medgemma_prompt_style,
            debug_print_example=debug_print_example,
        )

        # Wrap the model in DDP
        ecg_tokenizer = DistributedUtils.DDP(
            ecg_tokenizer,
            device_ids=[self.config.device],
            find_unused_parameters=True
        )

        decoder_module = ecg_tokenizer.module.decoder

        # Ensure freeze settings also apply to the DDP-wrapped module
        self._apply_model_phase_settings(ecg_tokenizer.module, initial_phase_cfg)

        param_groups = self._build_optimizer_param_groups(
            decoder_module=decoder_module,
            phase_overrides=initial_phase_cfg
        )

        projection_module = None  # getattr(ecg_tokenizer.module, 'ecg_image_projection', None)  # Disabled
        if projection_module is not None:
            projection_params = [p for p in projection_module.parameters() if p.requires_grad]
            if projection_params:
                ecg_proj_lr = float(initial_phase_cfg.get('ecg_embedding_lr', self.config.llm_lr))
                ecg_proj_wd = float(initial_phase_cfg.get('ecg_embedding_weight_decay', self.config.llm_weight_decay))
                param_groups.append({
                    "params": projection_params,
                    "lr": ecg_proj_lr,
                    "weight_decay": ecg_proj_wd,
                    "name": "ecg_projection"
                })

        
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

        # For mid-epoch resume (resume_global_step is set), stay in the same epoch
        # For full-epoch resume (no resume_global_step), advance to next epoch
        resume_global_step = getattr(self.config, 'resume_global_step', None)
        if resume_checkpoint_path and resume_global_step is not None:
            # Mid-epoch resume: continue from the same epoch
            start_epoch = max(1, resume_epoch)
        elif resume_checkpoint_path:
            # Full-epoch resume: start from next epoch
            start_epoch = resume_epoch + 1
        else:
            start_epoch = 1

        optimizer_state: Optional[dict[str, Any]] = None
        if resume_checkpoint_path:
            optimizer_state = state_dict.get('optimizer_state_dict')
            if isinstance(optimizer_state, dict):
                self._load_optimizer_state_dict(optimizer, optimizer_state)
            scheduler_state = state_dict.get('scheduler_state_dict')
            if scheduler_state is not None and scheduler is not None:
                self._load_scheduler_state_dict(
                    scheduler,
                    scheduler_state,
                    saved_optimizer_state=optimizer_state
                )
            scaler_state = state_dict.get('scaler_state_dict')
            if scaler_state is not None and scaler is not None:
                scaler.load_state_dict(scaler_state)

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

        adapter_module = getattr(decoder_module, 'adapter', None)
        bridge_module = getattr(decoder_module, 'bridge', None)
        core_adapter_module = adapter_module if adapter_module is not None else bridge_module

        llm_params = [p for p in self._get_llm_parameters(decoder_module) if p.requires_grad]

        embedding_param = None
        if hasattr(decoder_module, 'llm_model'):
            embedding_param = decoder_module.llm_model.get_input_embeddings().weight
            if embedding_param is not None:
                if any(p is embedding_param for p in llm_params):
                    llm_params = [p for p in llm_params if p is not embedding_param]
                # Never optimize embeddings for Q-Former or prefix-tuning paths
                if (not embedding_param.requires_grad) or self._uses_qformer_bridge() or getattr(decoder_module, 'prefix_tuning', False):
                    embedding_param = None
        if getattr(decoder_module, 'prefix_tuning', False):
            embedding_param = None

        adapter_params = []
        if core_adapter_module is not None:
            adapter_params = [p for p in core_adapter_module.parameters() if p.requires_grad]
        cross_attention_params = []
        if adapter_module is not None:
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

    def _load_optimizer_state_dict(self, optimizer: Optimizer, saved_state: dict[str, Any]) -> None:
        """Load optimizer state with graceful fallback when parameter groups change."""
        if not saved_state:
            return

        try:
            optimizer.load_state_dict(saved_state)
            return
        except (ValueError, RuntimeError):
            if getattr(self.config, 'is_ref_device', True):
                print("⚠️ Optimizer param groups changed; attempting to remap saved state.")

        remapped_state = self._remap_optimizer_state(saved_state, optimizer)
        if remapped_state is None:
            if getattr(self.config, 'is_ref_device', True):
                print("⚠️ Optimizer state remap failed; proceeding with freshly initialized optimizer state.")
            return

        try:
            optimizer.load_state_dict(remapped_state)
            if getattr(self.config, 'is_ref_device', True):
                print("Info: Loaded optimizer state using remapped parameter groups.")
        except Exception as final_exc:  # pragma: no cover - defensive
            if getattr(self.config, 'is_ref_device', True):
                print(f"⚠️ Optimizer state load failed after remap: {final_exc}. Using fresh optimizer state.")

    def _load_scheduler_state_dict(
        self,
        scheduler: LRScheduler,
        saved_state: dict[str, Any],
        saved_optimizer_state: dict[str, Any] | None = None
    ) -> None:
        """Safely load scheduler state, remapping values when parameter groups change."""
        if not saved_state:
            return

        optimizer = getattr(scheduler, 'optimizer', None)
        if optimizer is None:
            return

        state_to_load = dict(saved_state)

        saved_groups = []
        if isinstance(saved_optimizer_state, dict):
            saved_groups = saved_optimizer_state.get('param_groups') or []

        if saved_groups:
            for key in ('base_lrs', 'last_lr'):
                remapped = self._remap_scheduler_values(
                    state_to_load.get(key),
                    saved_groups,
                    optimizer.param_groups
                )
                if remapped is not None:
                    state_to_load[key] = remapped

        if 'lr_lambdas' in state_to_load and hasattr(scheduler, 'lr_lambdas'):
            saved_lambdas = state_to_load.get('lr_lambdas')
            current_lambdas = getattr(scheduler, 'lr_lambdas') or []
            if isinstance(saved_lambdas, list) and isinstance(current_lambdas, list):
                if len(saved_lambdas) != len(current_lambdas):
                    trimmed = list(saved_lambdas[:len(current_lambdas)])
                    if len(trimmed) < len(current_lambdas):
                        trimmed.extend([None] * (len(current_lambdas) - len(trimmed)))
                    state_to_load['lr_lambdas'] = trimmed

        try:
            scheduler.load_state_dict(state_to_load)
        except (IndexError, KeyError, ValueError) as exc:
            if getattr(self.config, 'is_ref_device', True):
                print(f"⚠️ Could not load scheduler state from checkpoint: {exc}. Resetting scheduler state.")
        except Exception as exc:  # pragma: no cover - defensive
            if getattr(self.config, 'is_ref_device', True):
                print(f"⚠️ Unexpected scheduler state load failure: {exc}. Resetting scheduler state.")

    def _remap_optimizer_state(self, saved_state: dict[str, Any], optimizer: Optimizer) -> Optional[dict[str, Any]]:
        """Adapt a saved optimizer state to the current optimizer parameter order.

        Returns a new state dict aligned with the optimizer's param groups, or None if remap
        is not possible.
        """
        if not saved_state:
            return None

        saved_groups = saved_state.get('param_groups') or []
        saved_state_map = saved_state.get('state') or {}
        if not saved_groups:
            return None

        saved_by_name: dict[Any, list[deque[dict[str, Any]]]] = {}
        unnamed_groups: list[deque[dict[str, Any]]] = []

        for group in saved_groups:
            name = group.get('name')
            param_indices = group.get('params', [])
            param_states = deque([saved_state_map.get(idx, {}) for idx in param_indices])
            if name is not None:
                saved_by_name.setdefault(name, []).append(param_states)
            else:
                unnamed_groups.append(param_states)

        new_state: dict[int, Any] = {}
        new_param_groups: list[dict[str, Any]] = []
        param_counter = 0

        for group in optimizer.param_groups:
            name = group.get('name')
            state_deque: deque[dict[str, Any]] | None = None

            if name is not None and name in saved_by_name and saved_by_name[name]:
                state_deque = saved_by_name[name].pop(0)
            elif unnamed_groups:
                state_deque = unnamed_groups.pop(0)
            else:
                state_deque = deque()

            new_group = {k: v for k, v in group.items() if k != 'params'}
            new_param_indices: list[int] = []

            for _ in group.get('params', []):
                state = state_deque.popleft() if state_deque else {}
                new_state[param_counter] = state
                new_param_indices.append(param_counter)
                param_counter += 1

            new_group['params'] = new_param_indices
            new_param_groups.append(new_group)

        return {
            'state': new_state,
            'param_groups': new_param_groups,
        }

    def _remap_scheduler_values(
        self,
        saved_values: Any,
        saved_groups: list[dict[str, Any]],
        current_groups: list[dict[str, Any]]
    ) -> Optional[list[float]]:
        """Remap scheduler value lists (like base_lrs) to current optimizer groups."""
        if not isinstance(saved_values, list):
            return None

        saved_by_name: dict[Any, deque[float]] = {}
        unnamed_values: deque[float] = deque()

        for group, value in zip(saved_groups, saved_values):
            name = group.get('name')
            if name is not None:
                saved_by_name.setdefault(name, deque()).append(value)
            else:
                unnamed_values.append(value)

        remapped: list[float] = []

        for group in current_groups:
            name = group.get('name')
            if name is not None and name in saved_by_name and saved_by_name[name]:
                remapped.append(saved_by_name[name].popleft())
            elif unnamed_values:
                remapped.append(unnamed_values.popleft())
            else:
                remapped.append(float(group.get('lr', 0.0)))

        return remapped

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

        print(f"\nBridge: {getattr(model.decoder, 'bridge_name', 'unknown')}")
        print(f"Decoder: {model.decoder_name} ({model.decoder_mode.value} mode)")

        bridge_config = getattr(model.decoder, 'bridge_config', None)
        if bridge_config:
            print(f"\nBridge configuration:")
            for key, value in bridge_config.items():
                print(f"  {key}: {value}")
        
        bridge_module = getattr(model.decoder, 'bridge', None)
        if bridge_module is not None:
            bridge_total = sum(p.numel() for p in bridge_module.parameters())
            bridge_trainable = sum(p.numel() for p in bridge_module.parameters() if p.requires_grad)
            print(f"\nBridge parameters: {bridge_trainable:,}/{bridge_total:,} trainable")

            bridge_state_count = len(bridge_module.state_dict())
            bridge_load_info = getattr(model.decoder, "bridge_load_info", None)
            if isinstance(bridge_load_info, dict):
                missing = len(bridge_load_info.get("missing_keys") or [])
                reinit = len(bridge_load_info.get("reinitialized_keys") or [])
                partial = len(bridge_load_info.get("partially_loaded_keys") or [])
                stage1_layers = bridge_load_info.get("stage1_block_count")
                model_layers = bridge_load_info.get("model_block_count")
                stage1_instr = bridge_load_info.get("stage1_instruction_block_count")
                model_instr = bridge_load_info.get("model_instruction_block_count")

                loaded_full = max(bridge_state_count - missing - reinit, 0)
                print("  Stage-1 checkpoint load:")
                print(f"    tensors loaded: {loaded_full}/{bridge_state_count}")
                if partial:
                    print(f"    partially loaded: {partial}")
                if reinit:
                    print(f"    reinitialised: {reinit}")
                if missing:
                    print(f"    missing: {missing}")
                if isinstance(stage1_layers, int) and isinstance(model_layers, int):
                    diff = model_layers - stage1_layers
                    if diff > 0:
                        print(f"    new Q-Former layers: {diff}")
                if isinstance(stage1_instr, int) and isinstance(model_instr, int):
                    diff = model_instr - stage1_instr
                    if diff > 0:
                        print(f"    new instruction layers: {diff}")
                unused = len(bridge_load_info.get("unused_checkpoint_keys") or [])
                if unused:
                    print(f"    unused checkpoint tensors: {unused}")
        
        # Print LoRA configuration if enabled
        if self.config.use_lora:
            print(f"\nLoRA Configuration:")
            print(f"  Rank (r): {self.config.lora_r}")
            print(f"  Alpha: {self.config.lora_alpha}")
            print(f"  Dropout: {self.config.lora_dropout}")
            print(f"  Target modules: {self.config.lora_target_modules}")
            print(f"  Bias: {self.config.lora_bias}")
            if self.config.lora_top_k_layers is not None:
                print(f"  Top-k layers: {self.config.lora_top_k_layers}")
            
            # Print LoRA-specific parameter counts if available
            llm_model = None
            if hasattr(model.decoder, 'llm_model'):
                llm_model = model.decoder.llm_model
            elif hasattr(model.decoder, 'llm'):
                llm_model = model.decoder.llm
            
            if llm_model:
                # Check if this is a PEFT model (has peft_config or is a PeftModel)
                is_peft_model = hasattr(llm_model, 'peft_config') or hasattr(llm_model, 'base_model')
                if is_peft_model:
                    total_llm_params = sum(p.numel() for p in llm_model.parameters())
                    trainable_llm_params = sum(p.numel() for p in llm_model.parameters() if p.requires_grad)
                    lora_params = trainable_llm_params  # In PEFT models, trainable params are LoRA params
                    base_model_params = total_llm_params - lora_params
                    if total_llm_params > 0:
                        print(f"  LoRA trainable parameters: {lora_params:,}/{total_llm_params:,} ({100 * lora_params / total_llm_params:.2f}%)")
                else:
                    print(f"  LoRA not detected in model")
        
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
        pretrained_config_raw = state_dict['config']
        (
            pretrained_config,
            encoder_name,
            quantizer_name,
            num_quantizers,
            codebook_size,
        ) = self._resolve_checkpoint_structure(
            pretrained_config_raw,
            self.config.pretrained_tokenizer_path
        )
        if self.config.is_ref_device:
            print(f"Pretrained config: {pretrained_config}")              
        
        # Initialize the tokenizer with the appropriate configuration
        # Use the pretrained config to initialize the ecg_tokenizer_wrapper class
        # For inference, we need to check if the checkpoint has LoRA weights
        checkpoint_has_lora = any('lora_A' in key or 'lora_B' in key or 'base_layer' in key for key in state_dict['model_state_dict'].keys())
        
        # If checkpoint has LoRA weights, we should load with LoRA enabled
        # If checkpoint doesn't have LoRA weights, we should load without LoRA
        use_lora_for_inference = checkpoint_has_lora and bool(getattr(pretrained_config, 'use_lora', False))
        
        if self.config.is_ref_device:
            print(f"Checkpoint has LoRA weights: {checkpoint_has_lora}")
            print(f"Using LoRA for inference: {use_lora_for_inference}")
        
        infer_tokenizer, infer_processor = self._get_tokenizer(self.config.tokenizer_name)

        decoder_name = getattr(pretrained_config, 'decoder_name', self.config.decoder_name)
        decoder_mode = getattr(pretrained_config, 'decoder_mode', self.config.decoder_mode)
        bridge_name_override = getattr(
            pretrained_config,
            'bridge_name',
            getattr(pretrained_config, 'adapter_name', self.config.bridge_name)
        )
        huggingface_model_name = getattr(pretrained_config, 'huggingface_model_name', self.config.huggingface_model_name)
        llm_input_embedding_size = getattr(pretrained_config, 'llm_input_embedding_size', self.config.llm_input_embedding_size)

        ecg_tokenizer: ECG_Tokenizer_Wrapper = ModelRegistry.get(self.config.pipeline_project)(
            encoder_name=encoder_name,
            quantizer_name=quantizer_name,
            decoder_name=decoder_name, 
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            decoder_mode=decoder_mode,
            bridge_name=bridge_name_override,
            huggingface_model_name=huggingface_model_name,
            llm_input_embedding_size=llm_input_embedding_size,
            tokenizer=infer_tokenizer,
            ecg_token_start_id=getattr(self.config, 'ecg_token_start_id', None),
            prefix_tuning=getattr(pretrained_config, 'prefix_tuning', getattr(self.config, 'prefix_tuning', False)),
            bridge_qformer_layers=getattr(pretrained_config, 'bridge_qformer_layers', getattr(self.config, 'bridge_qformer_layers', None)),
            bridge_text_hidden_size=getattr(pretrained_config, 'bridge_text_hidden_size', getattr(self.config, 'bridge_text_hidden_size', None)),
            bridge_bias_last_codebook=getattr(pretrained_config, 'bridge_bias_last_codebook', getattr(self.config, 'bridge_bias_last_codebook', None)),
            bridge_codebook_dropout=getattr(pretrained_config, 'bridge_codebook_dropout', getattr(self.config, 'bridge_codebook_dropout', None)),
            bridge_cross_every=getattr(pretrained_config, 'bridge_cross_every', getattr(self.config, 'bridge_cross_every', None)),
            instruction_dropout=getattr(pretrained_config, 'instruction_dropout', getattr(self.config, 'instruction_dropout', 0.0)),
            use_lora=use_lora_for_inference,
            lora_config={
                'r': getattr(pretrained_config, 'lora_r', 16),
                'alpha': getattr(pretrained_config, 'lora_alpha', 32),
                'dropout': getattr(pretrained_config, 'lora_dropout', 0.1),
                'target_modules': getattr(pretrained_config, 'lora_target_modules', None),
                'bias': getattr(pretrained_config, 'lora_bias', 'none')
            } if use_lora_for_inference else None,
            stage1_checkpoint_path=getattr(self.config, 'stage1_checkpoint_path', None),
        ).to(self.config.device)
        # Set the codebook size to the pretrained codebook size
        self.config.codebook_size = codebook_size # required to compute % of active codebook during training
        
        # Load the pretrained state dict
        pretrained_state_dict = state_dict['model_state_dict']
        ecg_tokenizer._load_state_dict(pretrained_state_dict, strict=True)
        
        # Set LoRA to inference mode if using LoRA
        if use_lora_for_inference:
            ecg_tokenizer.set_lora_inference_mode(True)
            
        ecg_tokenizer.eval()
        
        # Load the tokenizer
        tokenizer, processor = self._get_tokenizer(self.config.tokenizer_name)
        self.config.tokenizer = tokenizer  # type: ignore[attr-defined]
        if processor is not None:
            self.config.processor = processor  # type: ignore[attr-defined]
        
        # Get the dataloaders
        medgemma_prompt_style = bool(getattr(self.config, 'medgemma_prompt_style', False))
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
            shuffle=getattr(self.config, "validation_shuffle", False), 
            pin_memory=True,
            instruct_mode=getattr(self.config, 'instruct_mode', False),
            # Use 0 placeholders when using Q-Former (or prefix tuning).
            num_ecg_tokens=(
                0 if (self._uses_qformer_bridge() or getattr(self.config, 'prefix_tuning', False))
                else getattr(self.config, 'num_ecg_tokens', 128)
            ),
            ecg_token_start_id=getattr(self.config, 'ecg_token_start_id', None),
            prompt_column=self.config.prompt_column,
            answer_column=self.config.answer_column,
            category_column=self.config.category_column,
            prefix_tuning=getattr(self.config, 'prefix_tuning', False),
            pattern_columns=getattr(self.config, 'pattern_label_columns', None),
            medgemma_prompt_style=medgemma_prompt_style,
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
    
    def _get_tokenizer(self, tokenizer_name: str) -> tuple[Any, Optional[Any]]:
        """Return the text tokenizer and optional processor based on configuration."""

        use_processor = getattr(self.config, 'use_auto_processor', False)
        tokenizer_name_lower = tokenizer_name.lower()
        model_name_lower = str(getattr(self.config, "huggingface_model_name", "")).lower()
        medgemma_like = bool(
            getattr(self.config, "medgemma_prompt_style", False)
            or "medgemma" in tokenizer_name_lower
            or "medgemma" in model_name_lower
        )

        llama_chat_template = (
            "<|begin_of_text|>"
            "{% for message in messages %}"
            "{% if message['role'] == 'system' %}"
            "<|start_header_id|>system<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
            "{% elif message['role'] == 'user' %}"
            "<|start_header_id|>user<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
            "{% elif message['role'] == 'assistant' %}"
            "<|start_header_id|>assistant<|end_header_id|>\n\n{{ message['content'] }}<|eot_id|>"
            "{% endif %}"
            "{% endfor %}"
            "{% if add_generation_prompt %}<|start_header_id|>assistant<|end_header_id|>\n\n{% endif %}"
        )
        medgemma_chat_template = (
            "{{ bos_token }}"
            "{% for message in messages %}"
            "{% set role = message['role'] %}"
            "{% if role == 'assistant' %}{% set role = 'model' %}{% endif %}"
            "<start_of_turn>{{ role }}\n{{ message['content'] }}<end_of_turn>\n"
            "{% endfor %}"
            "{% if add_generation_prompt %}<start_of_turn>model\n{% endif %}"
        )

        processor = None
        if use_processor:
            processor_name = getattr(self.config, 'processor_name', None) or tokenizer_name
            processor = AutoProcessor.from_pretrained(processor_name, trust_remote_code=True)
            tokenizer = getattr(processor, 'tokenizer', None)
            if tokenizer is None:
                tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        else:
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

        if getattr(self.config, 'instruct_mode', False) and processor is None:
            # Do not include any ECG delimiters in chat template; text only
            tokenizer.chat_template = medgemma_chat_template if medgemma_like else llama_chat_template

        if hasattr(tokenizer, 'pad_token') and tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        if hasattr(tokenizer, 'pad_token_id') and isinstance(tokenizer.pad_token_id, list):
            tokenizer.pad_token_id = tokenizer.pad_token_id[0]

        if getattr(self.config, 'instruct_mode', False) and processor is None:
            # Do not add ECG-specific special tokens; rely on bridge-only conditioning
            self.config.ecg_token_start_id = None

        # Ensure chat template exists for instruction tuning
        if getattr(self.config, 'instruct_mode', False):
            chat_tmpl = getattr(tokenizer, 'chat_template', None)
            if not chat_tmpl:
                if 'llama' in tokenizer_name_lower:
                    tokenizer.chat_template = llama_chat_template
                elif medgemma_like:
                    tokenizer.chat_template = medgemma_chat_template

        return tokenizer, processor if use_processor else None

    
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
