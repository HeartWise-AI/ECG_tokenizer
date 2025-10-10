import os
import json
import re
import time
import traceback
import torch
import pandas as pd
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - visualization is optional
    matplotlib = None
    plt = None
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import LRScheduler

from utils.enums import RunMode, RunnerName
from utils.ddp import DistributedUtils
from utils.registry import (
    RunnerRegistry,
    MetricRegistry
)
from utils.config import LLMFinetuningConfig
from utils.wandb_wrapper import WandbWrapper
from utils.schedulers import scheduler_is_per_iteration, get_scheduler
from utils.metrics.llm_metrics import (
    RougeMetric,
    SacreBleuMetric as BleuMetric,  # Using SacreBLEU implementation
    MeteorMetric,
    BertScoreMetric,
    update_best_metric,
    update_worst_metric,
    update_random_batch_metric,
    decode_assistant_only_text
)
from utils.metrics.category_metrics import CategoryMetricsCalculator
from runners.base_runner import BaseRunner
from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper
from utils.plot_validation_ecgs import (
    DEFAULT_PARQUET_PATH as VALIDATION_PLOT_PARQUET_PATH,
    load_parquet_mapping,
    compute_ecg_score,
    select_tiered_ecg_samples,
    build_parquet_lookup,
    resolve_ecg_source,
    determine_plotter_params,
    clean_chat_artifacts,
)

import random
from tqdm import tqdm
from typing import (
    Any, 
    Union, 
    Callable,
    Optional,
    Dict,
    List,
    Tuple
)
from pathlib import Path
import numpy as np
import sys

# Add DeepECG_Preprocess to path for ECG plotting
sys.path.append('/volume/ECG_tokenizer/DeepECG_Preprocess')
sys.path.append('/volume/ECG_tokenizer/DeepECG_Preprocess/ecg_plotter')


@RunnerRegistry.register(RunnerName.LLM_FINETUNING)
class LLMFinetuningRunner(BaseRunner):
    """
    Runner for LLM finetuning.
    """
    
    def __init__(
        self, 
        model: ECG_Tokenizer_Wrapper,
        config: LLMFinetuningConfig, 
        validation_dataloader: DataLoader,
        wandb_wrapper: WandbWrapper | None = None,
        train_dataloader: DataLoader | None = None,
        optimizer: AdamW | None = None,
        scheduler: LRScheduler | None = None,
        scaler: GradScaler | None = None,
        start_epoch: int = 1,
    ):
        """
        Args:
            model: ECG tokenizer wrapper
            config: Configuration for the runner
            validation_dataloader: DataLoader for validation
            wandb_wrapper: WandbWrapper for logging
            train_dataloader: DataLoader for training
            optimizer: Optimizer for the model
            scheduler: Scheduler for the optimizer
            scaler: Scaler for the optimizer
        """
        self.model: ECG_Tokenizer_Wrapper = model
        self.config: LLMFinetuningConfig = config
        self.wandb_wrapper: WandbWrapper | None = wandb_wrapper
        self.train_dataloader: DataLoader | None = train_dataloader
        self.validation_dataloader: DataLoader | None = validation_dataloader
        self.optimizer: AdamW | None = optimizer
        self.scheduler: LRScheduler | None = scheduler
        self.scaler: GradScaler | None = scaler
        self.scheduler_per_iteration: bool = scheduler_is_per_iteration(self.config)
        
        # Phase tracking
        self.current_phase = None
        self.phase1_epochs = getattr(self.config, 'training_phases', {}).get('phase1_alignment', {}).get('epochs', 2)
        self.start_epoch = max(1, int(start_epoch))
        self.total_epochs_completed = self.start_epoch - 1
        
        # Initialize category metrics calculator if enabled
        self.category_metrics_calculator = None
        if getattr(self.config, 'compute_category_metrics', False):
            self.category_metrics_calculator = CategoryMetricsCalculator(
                metric_names=getattr(self.config, 'category_metrics', ['rouge', 'bleu', 'meteor']),
                device=self.config.device
            )
        
        # Control expensive metric evaluation (e.g., BERTScore) across validation batches
        self.bertscore_max_batches: Optional[int] = getattr(self.config, 'bertscore_max_batches', None)
        self._bertscore_skip_logged: bool = False

        # Persist the original debug configuration so phase overrides can be merged cleanly.
        base_debug_config = getattr(self.config, 'debug_config', None) or {}
        self._base_debug_config: dict[str, Any] = dict(base_debug_config)
        
    def execute(
        self, 
        mode: RunMode
    ):
        """
        Execute the runner in the specified mode.
        
        Args:
            mode: The execution mode (TRAIN, INFERENCE, VALIDATE, EXTRACT_EMBEDDINGS)
        """
        super().execute(mode)
        
    def _configure_training_phase(self, epoch: int):
        """
        Configure training based on current phase (logging only - no optimizer changes).
        
        Args:
            epoch: Current epoch number (1-indexed)
        """
        phase_config = getattr(self.config, 'training_phases', {})
        phase1 = phase_config.get('phase1_alignment', {})
        phase2 = phase_config.get('phase2_finetuning', {})
        
        phase1_epochs = phase1.get('epochs', 2)
        
        # Determine current phase and apply phase-specific settings
        if epoch <= phase1_epochs:
            if self.current_phase != 'phase1':
                self.current_phase = 'phase1'
                self._apply_phase_settings('phase1', phase1)
                if self.config.is_ref_device:
                    print("\n" + "="*80)
                    print(f"🔄 ENTERING PHASE 1: ALIGNMENT TRAINING (Epochs 1-{phase1_epochs})")
                    print("   - Using config-based learning rates for ECG alignment")
                    print("   - Enhanced debugging enabled")
                    print("="*80 + "\n")
        else:
            if self.current_phase != 'phase2':
                self.current_phase = 'phase2'
                self._apply_phase_settings('phase2', phase2)
                if self.config.is_ref_device:
                    print("\n" + "="*80)
                    print(f"🚀 ENTERING PHASE 2: FINE-TUNING (Epochs {phase1_epochs+1}-{self.config.num_epochs})")
                    print("   - Continuing with existing optimizer configuration")
                    print("   - Monitoring for improved performance")
                    print("="*80 + "\n")
    
    def _apply_phase_settings(self, phase_name: str, phase_config: dict | None):
        """Apply freezing, LoRA, and learning-rate tweaks for the active phase."""
        phase_config = phase_config or {}
        model = self.model.module if hasattr(self.model, 'module') else self.model

        freeze_llm = phase_config.get('freeze_llm')
        llm_module = self._get_llm_module(model)
        if freeze_llm is True and llm_module is not None:
            if hasattr(model.decoder, 'freeze_llm_parameters'):
                model.decoder.freeze_llm_parameters()
            else:
                for param in llm_module.parameters():
                    param.requires_grad = False
        elif freeze_llm is False and llm_module is not None:
            if hasattr(model.decoder, 'unfreeze_llm_parameters'):
                model.decoder.unfreeze_llm_parameters()
            else:
                for param in llm_module.parameters():
                    param.requires_grad = True

        if 'use_lora' in phase_config:
            self._set_lora_training_state(model, bool(phase_config['use_lora']))

        rebuilt = False
        if self.optimizer is not None:
            rebuilt = self._reset_optimizer_for_phase(phase_config)
            if not rebuilt:
                if 'llm_lr' in phase_config:
                    self._set_param_group_lr('llm', float(phase_config['llm_lr']))
                elif freeze_llm is True:
                    self._set_param_group_lr('llm', 0.0)
                if 'adapter_lr' in phase_config:
                    self._set_param_group_lr('adapter', float(phase_config['adapter_lr']))
                if 'cross_attention_lr' in phase_config:
                    self._set_param_group_lr('cross_attention', float(phase_config['cross_attention_lr']))
                if 'ecg_embedding_lr' in phase_config:
                    self._set_param_group_lr('ecg_embeddings', float(phase_config['ecg_embedding_lr']))
                elif freeze_llm is True:
                    self._set_param_group_lr('ecg_embeddings', 0.0)

        # Merge phase-specific debug overrides (e.g., gradient logging) on top of the base config.
        debug_overrides = phase_config.get('debug_config')
        if debug_overrides is not None:
            merged_debug_config = dict(self._base_debug_config)
            merged_debug_config.update(debug_overrides)
            self.config.debug_config = merged_debug_config
        else:
            self.config.debug_config = dict(self._base_debug_config) if self._base_debug_config else None

        if self.config.is_ref_device:
            log_bits = []
            if freeze_llm is True:
                log_bits.append('LLM frozen')
            elif freeze_llm is False:
                log_bits.append('LLM unfrozen')
            if 'use_lora' in phase_config:
                log_bits.append(f"LoRA {'enabled' if phase_config['use_lora'] else 'disabled'}")
            if 'llm_lr' in phase_config:
                log_bits.append(f"llm_lr={float(phase_config['llm_lr']):.2e}")
            if 'adapter_lr' in phase_config:
                log_bits.append(f"adapter_lr={float(phase_config['adapter_lr']):.2e}")
            if 'cross_attention_lr' in phase_config:
                log_bits.append(f"cross_attention_lr={float(phase_config['cross_attention_lr']):.2e}")
            if 'ecg_embedding_lr' in phase_config:
                log_bits.append(f"ecg_embedding_lr={float(phase_config['ecg_embedding_lr']):.2e}")
            if log_bits:
                print(f"   - Phase settings ({phase_name}): " + ', '.join(log_bits))

    def _get_llm_module(self, model):
        """Return the underlying LLM module if available."""
        decoder = getattr(model, 'decoder', None)
        if decoder is None:
            return None
        if hasattr(decoder, 'llm_model'):
            return decoder.llm_model
        if hasattr(decoder, 'llm'):
            return decoder.llm
        return None

    def _get_decoder_module(self):
        model = self.model
        if model is None:
            return None
        if hasattr(model, 'module'):
            model = model.module
        return getattr(model, 'decoder', None)

    def _infer_prefix_offset(self, generated_ids: torch.Tensor, label_ids: torch.Tensor) -> int:
        """
        Calculate how many leading tokens belong to the ECG prefix.
        
        Note: Most decoders strip prefix tokens after generation, so this typically returns 0.
        This method is kept for compatibility with decoders that don't strip prefix tokens.
        """
        decoder = self._get_decoder_module()
        if decoder is None:
            return 0
        
        # If prefix_tuning is enabled, the decoder typically strips prefix tokens after generation
        # So we return 0 (most common case for MedGemma, Llama, etc.)
        if getattr(decoder, 'prefix_tuning', False):
            return 0
        
        # For non-prefix-tuning modes where ECG tokens are prepended as regular tokens
        # we need to calculate the offset
        configured_prefix = getattr(decoder, 'num_ecg_tokens', None)
        if configured_prefix is None:
            configured_prefix = getattr(self.config, 'num_ecg_tokens', 0)
        try:
            prefix_tokens = int(configured_prefix)
        except (TypeError, ValueError):
            prefix_tokens = int(getattr(self.config, 'num_ecg_tokens', 0))

        if prefix_tokens <= 0:
            return 0

        gen_len = int(generated_ids.size(-1))
        label_len = int(label_ids.size(-1)) if label_ids.dim() > 0 else 0
        if gen_len <= label_len:
            return 0

        diff = gen_len - label_len
        if diff <= 0:
            return 0

        return min(prefix_tokens, diff)

    def _set_param_group_lr(self, group_name: str, lr: float):
        """Update the learning rate for a named optimizer group, if present."""
        if self.optimizer is None:
            return
        for group in self.optimizer.param_groups:
            if group.get('name') == group_name:
                group['lr'] = lr

    def _reset_optimizer_for_phase(self, phase_config: dict | None) -> bool:
        """Rebuild optimizer (and scheduler) with phase-specific parameter groups."""
        if self.optimizer is None:
            return False

        model = self.model.module if hasattr(self.model, 'module') else self.model
        param_groups = self._build_optimizer_param_groups(model, phase_config)

        if not param_groups:
            return False

        optimizer_class = type(self.optimizer)
        self.optimizer = optimizer_class(param_groups)
        self.optimizer.zero_grad(set_to_none=True)

        if self.train_dataloader is not None:
            self.scheduler = get_scheduler(
                scheduler_name=self.config.scheduler_type,
                optimizer=self.optimizer,
                num_epochs=self.config.num_epochs,
                train_dataloader=self.train_dataloader,
                gamma=getattr(self.config, 'gamma', None),
                step_size=getattr(self.config, 'step_size', None),
                gradient_accumulation_steps=getattr(self.config, 'gradient_accumulation_steps', 1),
                num_warmup_percent=getattr(self.config, 'num_warmup_percent', None),
                num_hard_restarts_cycles=getattr(self.config, 'num_hard_restarts_cycles', None),
                warm_restart_tmult=getattr(self.config, 'warm_restart_tmult', None)
            )

        return True

    def _build_optimizer_param_groups(self, model: ECG_Tokenizer_Wrapper, phase_config: dict | None):
        phase_config = phase_config or {}

        decoder = getattr(model, 'decoder', None)
        if decoder is None:
            return []

        adapter_module = getattr(decoder, 'adapter', None)
        bridge_module = getattr(decoder, 'bridge', None)
        core_adapter_module = adapter_module if adapter_module is not None else bridge_module

        llm_module = self._get_llm_module(model)
        llm_params = []
        if llm_module is not None:
            llm_params = [p for p in llm_module.parameters() if p.requires_grad]

        embedding_param = None
        if hasattr(decoder, 'llm_model'):
            embedding_param = decoder.llm_model.get_input_embeddings().weight
            if embedding_param is not None:
                if any(p is embedding_param for p in llm_params):
                    llm_params = [p for p in llm_params if p is not embedding_param]
                if not embedding_param.requires_grad:
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

        llm_lr = float(phase_config.get('llm_lr', self.config.llm_lr))
        adapter_lr = float(phase_config.get('adapter_lr', self.config.adapter_lr))
        cross_attention_lr = float(phase_config.get('cross_attention_lr', adapter_lr))
        ecg_embedding_lr = float(phase_config.get('ecg_embedding_lr', llm_lr))

        llm_weight_decay = float(phase_config.get('llm_weight_decay', self.config.llm_weight_decay))
        adapter_weight_decay = float(phase_config.get('adapter_weight_decay', self.config.adapter_weight_decay))
        cross_attention_weight_decay = float(phase_config.get('cross_attention_weight_decay', adapter_weight_decay))
        ecg_embedding_weight_decay = float(phase_config.get('ecg_embedding_weight_decay', llm_weight_decay))

        param_groups = []
        if llm_params:
            param_groups.append({
                'params': llm_params,
                'lr': llm_lr,
                'weight_decay': llm_weight_decay,
                'name': 'llm'
            })
        if embedding_param is not None:
            param_groups.append({
                'params': [embedding_param],
                'lr': ecg_embedding_lr,
                'weight_decay': ecg_embedding_weight_decay,
                'name': 'ecg_embeddings'
            })
        if adapter_core_params:
            param_groups.append({
                'params': adapter_core_params,
                'lr': adapter_lr,
                'weight_decay': adapter_weight_decay,
                'name': 'adapter'
            })
        if cross_attention_params:
            param_groups.append({
                'params': cross_attention_params,
                'lr': cross_attention_lr,
                'weight_decay': cross_attention_weight_decay,
                'name': 'cross_attention'
            })

        return param_groups

    def _set_lora_training_state(self, model, enable: bool):
        """Enable or disable LoRA adapter training parameters."""
        if not getattr(model, 'use_lora', False):
            return
        llm_module = self._get_llm_module(model)
        if llm_module is None:
            return

        toggle_pairs = [
            ('enable_adapter_layers', 'disable_adapter_layers'),
            ('enable_adapters', 'disable_adapters'),
            ('enable_adapter', 'disable_adapter'),
        ]
        for enable_name, disable_name in toggle_pairs:
            enable_fn = getattr(llm_module, enable_name, None)
            disable_fn = getattr(llm_module, disable_name, None)
            if enable and callable(enable_fn):
                enable_fn()
                break
            if not enable and callable(disable_fn):
                disable_fn()
                break

        for name, param in llm_module.named_parameters():
            if 'lora' in name.lower():
                param.requires_grad = enable

    def _setup_phase1_training(self, phase1_config: dict):
        """Setup phase 1: Alignment training with very low LLM LR instead of freezing."""
        # Get the actual model (unwrap from DDP if necessary)
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Instead of freezing, use extremely low LR for LLM (effectively frozen)
        # This prevents DDP unused parameter issues
        
        # Reconfigure optimizer with phase 1 learning rates
        param_groups = []
        
        # LLM parameters with extremely low LR (effectively frozen)
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'llm'):
            param_groups.append({
                'params': model.decoder.llm.parameters(),
                'lr': 1e-10,  # Extremely low LR - effectively frozen
                'name': 'llm_frozen'
            })
        
        # ECG token embeddings (part of LLM embedding layer)
        # These are handled by the LLM parameters above
        
        # Cross-attention (part of adapter for SequenceTokenAdapter)
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'adapter'):
            adapter = model.decoder.adapter
            if hasattr(adapter, 'cross_attention_layers'):
                param_groups.append({
                    'params': adapter.cross_attention_layers.parameters(),
                    'lr': phase1_config.get('cross_attention_lr', 1e-3),
                    'name': 'cross_attention'
                })
        
        # Adapter with high LR
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'adapter'):
            param_groups.append({
                'params': model.decoder.adapter.parameters(),
                'lr': phase1_config.get('adapter_lr', 1e-3),
                'name': 'adapter'
            })
        
        # NOTE: Optimizer reconfiguration disabled to prevent DDP issues
        # The original optimizer from project setup will be used throughout
        
        # Log parameter counts for debugging
        if self.config.is_ref_device:
            model = self.model.module if hasattr(self.model, 'module') else self.model
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Phase 1 trainable parameters: {total_params:,}")
    
    def _setup_phase2_training(self, phase2_config: dict):
        """Setup phase 2: LoRA fine-tuning with normal learning rates."""
        # Get the actual model (unwrap from DDP if necessary)
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Enable LoRA if available
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'llm'):
            if hasattr(model.decoder.llm, 'enable_adapters'):
                model.decoder.llm.enable_adapters()
        
        # Reconfigure optimizer with phase 2 learning rates
        param_groups = []
        
        # LLM (LoRA) parameters with normal LR
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'llm'):
            param_groups.append({
                'params': model.decoder.llm.parameters(),
                'lr': phase2_config.get('llm_lr', 1e-6),
                'name': 'llm_lora'
            })
        
        # ECG token embeddings (part of LLM embedding layer)
        # These are handled by the LLM parameters above
        
        # Cross-attention (lower LR) - part of adapter
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'adapter'):
            adapter = model.decoder.adapter
            if hasattr(adapter, 'cross_attention_layers'):
                param_groups.append({
                    'params': adapter.cross_attention_layers.parameters(),
                    'lr': phase2_config.get('cross_attention_lr', 1e-5),
                    'name': 'cross_attention'
                })
        
        # Adapter (lower LR)
        if hasattr(model, 'decoder') and hasattr(model.decoder, 'adapter'):
            param_groups.append({
                'params': model.decoder.adapter.parameters(),
                'lr': phase2_config.get('adapter_lr', 1e-5),
                'name': 'adapter'
            })
        
        # NOTE: Optimizer reconfiguration disabled to prevent DDP issues
        # The original optimizer from project setup will be used throughout
        
        # Log parameter counts for debugging
        if self.config.is_ref_device:
            model = self.model.module if hasattr(self.model, 'module') else self.model
            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"Phase 2 trainable parameters: {total_params:,}")
    
    def train(self):
        """
        Train the LLM finetuning model with two-stage approach.
        """
        if self.optimizer is None:
            raise ValueError("Optimizer cannot be None")
        # Note: Scaler is not required for bfloat16 training
        
        best_val_loss: float = float('inf')
        
        for epoch in range(self.start_epoch, self.config.num_epochs + 1):
            # Configure training phase
            self._configure_training_phase(epoch)
            
            # Sync before starting each epoch
            DistributedUtils.sync_process_group(
                world_size=self.config.world_size,
                device_ids=self.config.device
            )
            
            epoch_train_metrics: dict[str, float] = self._run_epoch(
                RunMode.TRAIN,
                epoch
            )
            
            # Log phase-specific metrics
            if self.config.is_ref_device:
                epoch_train_metrics[f'{RunMode.TRAIN}/current_phase'] = 1 if self.current_phase == 'phase1' else 2
                        
            if self.wandb_wrapper is not None and self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
                self.wandb_wrapper.log(epoch_train_metrics)
            
            # Step the scheduler if it should be updated per-epoch
            if self.scheduler and (not self.scheduler_per_iteration):
                self.scheduler.step()
            
            # Sync the process group
            DistributedUtils.sync_process_group(
                world_size=self.config.world_size,
                device_ids=self.config.device
            )
            
            epoch_metrics: dict[str, float] = self._run_epoch(
                RunMode.VALIDATE,
                epoch
            )
            
            # Save best model (only on reference device)
            if self.config.is_ref_device:
                if epoch_metrics[f'{RunMode.VALIDATE}/loss'] < best_val_loss:
                    best_val_loss = epoch_metrics[f'{RunMode.VALIDATE}/loss']
                    self._save_model(
                        epoch=epoch,
                        loss=epoch_metrics[f'{RunMode.VALIDATE}/loss'],
                        is_best=True
                    )
                
                # Also save regular checkpoint
                self._save_model(
                    epoch=epoch,
                    loss=epoch_metrics[f'{RunMode.VALIDATE}/loss'],
                    is_best=False
                )
            
            # Sync after validation epoch, before next epoch            
            if self.wandb_wrapper is not None and self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
                # Add learning rate metrics from training epoch metrics
                lr_metrics = {}
                for key, value in epoch_metrics.items():
                    if "lr_" in key:
                        lr_metrics[key] = value
                
                self.wandb_wrapper.log({
                    **epoch_metrics,
                    **lr_metrics,
                    f"{RunMode.VALIDATE}/best_loss": best_val_loss
                })
                
            # Sync the process group
            DistributedUtils.sync_process_group(
                world_size=self.config.world_size,
                device_ids=self.config.device
            )
                
    def _run_epoch(
        self,
        mode: RunMode,
        epoch: int
    )->dict[str, float]:
        """
        Run an epoch of training or validation.
        
        Args:
            mode: The execution mode (TRAIN, VALIDATE)
            epoch: The current epoch
            
        Returns:
            dict[str, float]: Dictionary containing the metrics for the epoch
        """
        assert mode in [RunMode.TRAIN, RunMode.VALIDATE]
        
        # Set the model to training or evaluation mode
        self.model.train(mode == RunMode.TRAIN)
        
        if self.train_dataloader is None or self.validation_dataloader is None:
            raise ValueError("Train or validation dataloader is not set")
        
        # Get the dataloader and step function
        dataloader: DataLoader = self.train_dataloader if mode == RunMode.TRAIN else self.validation_dataloader
        step_fn: Callable | None = self._train_step if mode == RunMode.TRAIN else self._val_step
        
        # Create a progress bar for the epoch
        phase_str = f"[Phase {1 if self.current_phase == 'phase1' else 2}]"
        data_iter: tqdm = tqdm(
            dataloader, 
            desc=f"{phase_str} {mode} epoch {epoch}/{self.config.num_epochs}",
            leave=True,
            disable=not self.config.is_ref_device
        )
        
        # Initialize the total loss
        total_loss: float = 0.0
        
        # Sync before starting batch iterations
        DistributedUtils.sync_process_group(
            world_size=self.config.world_size,
            device_ids=self.config.device
        )
        
        # Iterate over the dataloader
        epoch_metrics: dict[str, float] = {}
        
        if mode == RunMode.VALIDATE:
            self._bertscore_skip_logged = False
            worst_batch_metrics, best_batch_metrics, random_batch_metrics, random_batch_idx = self._init_validation_metrics(dataloader)
            # Initialize JSON file for incremental writing
            json_path = self._get_val_generation_json_path(epoch)
            os.makedirs(os.path.dirname(json_path), exist_ok=True)
            # Initialize with empty dict
            if self.config.is_ref_device:
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump({}, f)
            
            # Reset category metrics calculator for this validation epoch
            if self.category_metrics_calculator is not None:
                self.category_metrics_calculator.reset()
        
        for batch_idx, batch in enumerate(data_iter):            
            # Preprocess the batch
            ecg_signal: torch.Tensor = batch['signal'].to(self.config.device)
            input_ids: torch.Tensor = batch['input_ids'].to(self.config.device)
            attention_mask: torch.Tensor = batch['attention_mask'].to(self.config.device)
            labels: torch.Tensor = batch['labels'].to(self.config.device) if 'labels' in batch else input_ids.clone()
            
            # Run the step function
            if getattr(self.config, 'instruct_mode', False) and 'prompt_input_ids' in batch:
                # Pass prompt_input_ids for both train and validate to prevent answer leakage
                outputs = step_fn(
                    ecg_signal=ecg_signal,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    prompt_input_ids=batch['prompt_input_ids'].to(self.config.device),
                    prompt_attention_mask=batch['prompt_attention_mask'].to(self.config.device) if 'prompt_attention_mask' in batch else None
                )
            else:
                outputs = step_fn(
                    ecg_signal=ecg_signal, 
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels
                )
            
            # initialize metrics
            metrics: dict[str, float] = {}
            metrics['loss'] = outputs['loss'].item()  # type: ignore[index]
            
            # Extract learning rate metrics
            for key, value in outputs.items():  # type: ignore[attr-defined]
                if key.startswith('lr_'):
                    metrics[key] = float(value) if isinstance(value, torch.Tensor) else float(value)
            
            # Compute rouge score, bleu score, and meteor score
            if mode == RunMode.VALIDATE:
                # Process and append to JSON immediately across all devices (write occurs on ref device)
                self._append_batch_to_json(
                    outputs['generated_ids'],
                    labels,
                    batch,
                    json_path
                )
                
                # Metrics Rouge, Bleu, and Meteor are computed on the reference device but aggregated across all GPUs later
                batch_metrics = self._compute_metrics( # this function returns mean metrics for the current batch
                    outputs,
                    labels,
                    dataloader,
                    best_batch_metrics, # parsed and updated by reference object - not returned
                    worst_batch_metrics, # parsed and updated by reference object - not returned
                    random_batch_metrics, # parsed and updated by reference object - not returned
                    random_batch=random_batch_idx == batch_idx,
                    batch=batch,  # Pass batch for category information
                    batch_idx=batch_idx
                )
                metrics.update(batch_metrics)                  
            
            # Gather and average loss across all GPUs
            gathered_metrics: dict[str, float] = {}
            for k in metrics:
                gathered_metrics[f"{mode}/{k}"] = DistributedUtils.gather_loss(
                    [metrics[k]], 
                    self.config.device
                )
                            
            # Update the epoch metrics
            for k, v in gathered_metrics.items():
                epoch_metrics[k] = epoch_metrics.get(k, 0.0) + float(v)
            
            # Update the total loss with gathered loss
            total_loss += gathered_metrics[f'{mode}/loss']
            mean_loss: float = total_loss / (batch_idx + 1)
            
            # Log the loss to wandb
            log_dict: dict[str, float] | None = None
            if mode == RunMode.TRAIN:
                if self.wandb_wrapper is not None and self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
                    log_dict = {
                        f"{mode}/loss": gathered_metrics[f'{mode}/loss'],  # Log the gathered loss for current batch
                        f"{mode}/mean_loss": mean_loss,  # Log the running mean loss
                    }

                    # Add learning rate metrics to log_dict
                    for key, value in gathered_metrics.items():
                        if f"{mode}/lr_" in key:
                            log_dict[key] = value
            
            # Sync after logging
            DistributedUtils.sync_process_group(
                world_size=self.config.world_size,
                device_ids=self.config.device
            )
            
            # Enhanced progress bar with debugging info
            postfix_dict = {
                f"loss": f'{gathered_metrics[f"{mode}/loss"]:.4f}',
                f"mean": f'{mean_loss:.4f}'
            }
            
            # Add gradient norms if debugging
            debug_config = getattr(self.config, 'debug_config', None) or {}
            if debug_config.get('log_gradient_norms', False) and mode == RunMode.TRAIN:
                model = self.model.module if hasattr(self.model, 'module') else self.model

                # Calculate gradient norms for different components
                grad_norms = {}
                if hasattr(model, 'decoder'):
                    adapter_module = getattr(model.decoder, 'adapter', None)
                    bridge_module = getattr(model.decoder, 'bridge', None)
                    core_module = adapter_module if adapter_module is not None else bridge_module

                    if core_module is not None:
                        core_params = [p for p in core_module.parameters() if p.requires_grad and p.grad is not None]
                        if core_params:
                            adapter_grad = torch.nn.utils.clip_grad_norm_(
                                core_params,
                                max_norm=float('inf')
                            )
                            grad_norms['adapter_grad_norm'] = adapter_grad.item() if torch.is_tensor(adapter_grad) else adapter_grad
                            postfix_dict['adapter_grad'] = f'{adapter_grad:.2e}'

                if hasattr(model, 'decoder') and hasattr(model.decoder, 'cross_attention_layers'):
                    cross_params = [p for layer in model.decoder.cross_attention_layers for p in layer.parameters() if p.requires_grad and p.grad is not None]
                    if cross_params:
                        cross_attn_grad = torch.nn.utils.clip_grad_norm_(
                            cross_params,
                            max_norm=float('inf')
                        )
                        grad_norms['cross_attn_grad_norm'] = cross_attn_grad.item() if torch.is_tensor(cross_attn_grad) else cross_attn_grad
                        postfix_dict['cross_grad'] = f'{cross_attn_grad:.2e}'
                
                # Add to epoch metrics for wandb logging
                for key, value in grad_norms.items():
                    epoch_metrics[f"{mode}/{key}"] = value

                if log_dict is not None:
                    for key, value in grad_norms.items():
                        log_dict[f"{mode}/{key}"] = float(value)

            if log_dict is not None and self.wandb_wrapper is not None and self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
                self.wandb_wrapper.log(log_dict)
            
            data_iter.set_postfix(postfix_dict)
            
            # Show sample outputs periodically if debugging
            if debug_config.get('show_sample_outputs', False) and mode == RunMode.VALIDATE:
                log_frequency = debug_config.get('log_frequency', 50)
                if batch_idx % log_frequency == 0 and self.config.is_ref_device:
                    self._log_sample_generation(
                        outputs=outputs,
                        labels=labels,
                        epoch=epoch,
                        batch_idx=batch_idx
                    )
        
        # Show epoch summary with debugging info
        debug_config = getattr(self.config, 'debug_config', None) or {}
        if debug_config.get('verbose_loss_logging', False) and self.config.is_ref_device:
            self._log_epoch_summary(mode, epoch, epoch_metrics, total_loss, len(dataloader))
        
        # === New Block: Log best and worst metrics as HTML to wandb ===
        if mode == RunMode.VALIDATE and self.wandb_wrapper is not None and self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
            # helper function to create an HTML table given the metrics dictionary
            def create_html_table(metrics_dict, table_title):
                html = f"<h3>{table_title}</h3>"
                html += "<table border='1' cellspacing='0' cellpadding='5'>"
                html += "<tr><th>Metric</th><th>Score</th><th>Prompt(s)</th><th>Prediction(s)</th><th>Reference(s)</th></tr>"
                for metric, records in metrics_dict.items():
                    if records:
                        # pick the first record (since K=1)
                        record = records[0]
                        # Assuming record is a dict with keys: 'score', 'predictions', 'references', and 'prompts'
                        score = record['score']
                        # if they are lists, join them with a line break
                        prompts = "<br>".join([f"{i + 1}: {p}" for i, p in enumerate(record.get('prompts', []))])
                        predictions = "<br>".join([f"{i + 1}: {p}" for i, p in enumerate(record['predictions'])])
                        references = "<br>".join([f"{i + 1}: {r}" for i, r in enumerate(record['references'])])
                    else:
                        score, prompts, predictions, references = "", "", "", ""
                    html += f"<tr><td>{metric}</td><td>{score}</td><td>{prompts}</td><td>{predictions}</td><td>{references}</td></tr>"
                html += "</table>"
                return html

            import wandb
            best_html = create_html_table(best_batch_metrics, "Best Metrics")
            worst_html = create_html_table(worst_batch_metrics, "Worst Metrics")
            random_html = create_html_table(random_batch_metrics, "Random Metrics")
            
            # Compute and log category metrics
            category_log_dict = {}
            if self.category_metrics_calculator is not None:
                try:
                    category_results = self.category_metrics_calculator.compute_category_metrics()
                    overall_results = self.category_metrics_calculator.compute_overall_metrics()
                    
                    # Format for logging
                    category_log_dict = self.category_metrics_calculator.format_results_for_logging(
                        category_results, overall_results, "val"
                    )
                    
                    # Print category statistics
                    stats = self.category_metrics_calculator.get_category_statistics()
                    print(f"\n=== Category Metrics Summary (Epoch {epoch}) ===")
                    for category, category_stats in stats.items():
                        print(f"{category}: {category_stats['n_samples']} samples")
                    
                    print(f"\nOverall Metrics:")
                    for metric, score in overall_results.items():
                        print(f"  {metric}: {score:.4f}")
                    
                    print(f"\nPer-Category Metrics (Top 3 categories):")
                    sorted_categories = sorted(category_results.items(), 
                                             key=lambda x: stats[x[0]]['n_samples'], 
                                             reverse=True)[:3]
                    for category, metrics in sorted_categories:
                        print(f"  {category} ({stats[category]['n_samples']} samples):")
                        for metric, score in metrics.items():
                            print(f"    {metric}: {score:.4f}")
                    
                except Exception as e:
                    print(f"Warning: Failed to compute category metrics: {e}")
                    traceback.print_exc()
            
            # Log everything to wandb
            self.wandb_wrapper.log({
                "val/best_metrics_html": wandb.Html(best_html),
                "val/worst_metrics_html": wandb.Html(worst_html),
                "val/random_metrics_html": wandb.Html(random_html),
                **category_log_dict
            })
            
            # JSON export is done incrementally during validation
            
            # Plot ECG waveforms with Q&A annotations if configured
            json_path = self._get_val_generation_json_path(epoch)
            if os.path.exists(json_path):
                self._plot_validation_ecgs(epoch, json_path)
        # === End new block ===
                
        # Normalize the epoch metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= len(dataloader)

        # Create per-epoch validation metric plots on the reference device
        if mode == RunMode.VALIDATE and self.config.is_ref_device:
            metric_key_map = {
                "ROUGE-1": f"{RunMode.VALIDATE}/rouge1",
                "ROUGE-L": f"{RunMode.VALIDATE}/rougeL",
                "BLEU-4": f"{RunMode.VALIDATE}/bleu4",
                "METEOR": f"{RunMode.VALIDATE}/meteor",
            }
            metric_values = {}
            for label, key in metric_key_map.items():
                value = epoch_metrics.get(key)
                if isinstance(value, (int, float)):
                    metric_values[label] = float(value)

            if metric_values and plt is not None:
                plot_dir = os.path.join(self._get_val_generations_dir(), "plots")
                os.makedirs(plot_dir, exist_ok=True)
                plot_path = os.path.join(plot_dir, f"epoch_{epoch:03d}_metrics.png")

                labels = list(metric_values.keys())
                scores = [metric_values[label] for label in labels]

                fig, ax = plt.subplots(figsize=(6, 4))
                bar_container = ax.bar(labels, scores, color="#4C72B0")
                upper_ylim = min(1.1, max(scores) + 0.05)
                upper_ylim = max(upper_ylim, 0.2)
                ax.set_ylim(0.0, upper_ylim)
                ax.set_ylabel("Score")
                ax.set_title(f"Validation Metrics - Epoch {epoch}")
                ax.tick_params(axis='x', rotation=20)
                for rect, score in zip(bar_container, scores):
                    text_y = min(score + 0.02, upper_ylim - 0.02)
                    ax.text(
                        rect.get_x() + rect.get_width() / 2,
                        text_y,
                        f"{score:.3f}",
                        ha='center',
                        va='bottom',
                        fontsize=9
                    )

                fig.tight_layout()
                fig.savefig(plot_path, dpi=200)
                plt.close(fig)

                if self.wandb_wrapper is not None and self.wandb_wrapper.is_initialized():
                    try:
                        import wandb
                        self.wandb_wrapper.log({
                            f"val/metrics_plot_epoch_{epoch}": wandb.Image(plot_path)
                        })
                    except Exception as exc:
                        print(f"Warning: Failed to log validation metrics plot to Weights & Biases: {exc}")

                print(f"Saved validation metric plot: {plot_path}")
            elif metric_values:
                print("matplotlib is unavailable; skipping validation metric plot generation.")

        # Return the epoch metrics
        return epoch_metrics
    
    def _extract_assistant_text(self, tokenizer, generated_ids: torch.Tensor, label_ids: torch.Tensor) -> str:
        """Return only the assistant portion of the generated text."""
        if generated_ids is None or generated_ids.numel() == 0:
            return ""

        # Flatten tensors for easier processing
        generated_flat = generated_ids.view(-1)
        label_flat = label_ids.view(-1)
        
        # Find where the actual answer starts in label_ids (first non -100 token)
        mask = label_flat != -100
        valid_positions = torch.nonzero(mask, as_tuple=False).flatten()
        
        if valid_positions.numel() == 0:
            # No valid labels - decode the entire generated sequence
            return tokenizer.decode(generated_flat.tolist(), skip_special_tokens=True)
        
        # The prompt length is the position of the first valid label
        prompt_len = int(valid_positions[0])
        
        # Account for prefix offset (ECG tokens) if present
        prefix_offset = self._infer_prefix_offset(generated_ids, label_ids)
        
        # Calculate the actual start position in generated_ids
        # generated_ids structure: [prefix_tokens (if any)] + [prompt_tokens] + [generated_answer]
        # We need to skip: prefix_offset + prompt_len
        start_idx = prefix_offset + prompt_len
        
        # Ensure start_idx is within bounds
        if start_idx >= generated_flat.size(0):
            # No generated tokens after the prompt
            return ""
        
        # Extract tokens from start_idx to the end
        assistant_tokens = generated_flat[start_idx:]

        if assistant_tokens.numel() == 0:
            return ""

        return tokenizer.decode(assistant_tokens.tolist(), skip_special_tokens=True)

    @staticmethod
    def _sanitize_chat_text(text: str, max_length: int = 512) -> str:
        """Trim special chat tokens, collapse whitespace, and optionally truncate."""
        if not text:
            return ""

        cleaned = str(text)

        # Remove any known chat delimiters and everything after the first end-of-turn marker
        for delimiter in ("<|eot_id|>", "|eot_id|>"):
            if delimiter in cleaned:
                cleaned = cleaned.split(delimiter, 1)[0]
        # Strip remaining special header markers
        special_tokens = (
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|assistant|>",
            "<|user|>",
            "<|system|>",
            "<|start_of_turn|>",
            "<|end_of_turn|>",
        )
        for token in special_tokens:
            cleaned = cleaned.replace(token, "")

        # Strip common HTML fragments the model sometimes emits
        cleaned = re.sub(r"<\s*br\s*/?>", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"</?p[^>]*>", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"</?div[^>]*>", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"</?span[^>]*>", " ", cleaned, flags=re.IGNORECASE)
        # Remove any remaining generic tags
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)

        cleaned = cleaned.strip()
        if cleaned:
            cleaned = " ".join(cleaned.split())

        if max_length is not None and len(cleaned) > max_length:
            cleaned = cleaned[: max_length - 1].rstrip() + "…"

        return cleaned

    def _decode_prompt_from_ids(
        self,
        tokenizer,
        prompt_ids: torch.Tensor,
        prompt_mask: Optional[torch.Tensor] = None
    ) -> str:
        ids_cpu = prompt_ids.detach().cpu()
        if prompt_mask is not None:
            mask_cpu = prompt_mask.detach().cpu().bool()
            if mask_cpu.shape == ids_cpu.shape:
                ids_cpu = ids_cpu[mask_cpu]
        else:
            pad_id = getattr(tokenizer, 'pad_token_id', None)
            if pad_id is not None:
                ids_cpu = ids_cpu[ids_cpu != pad_id]
        decoded = tokenizer.decode(ids_cpu.tolist(), skip_special_tokens=True)
        return self._sanitize_chat_text(decoded)

    def _extract_prompt_text(
        self,
        tokenizer,
        batch: dict[str, Any],
        index: int
    ) -> str:
        prompt_candidates: list[Any] = []
        for key in ('prompt_text', 'prompt', 'question'):
            value = batch.get(key)
            if isinstance(value, (list, tuple)) and index < len(value):
                prompt_candidates.append(value[index])
            elif value is not None and not isinstance(value, (list, tuple)) and index == 0:
                prompt_candidates.append(value)

        for candidate in prompt_candidates:
            if candidate is None:
                continue
            cleaned = self._sanitize_chat_text(str(candidate))
            if cleaned:
                return cleaned

        if 'prompt_input_ids' in batch:
            prompt_ids = batch['prompt_input_ids'][index]
            prompt_mask = None
            if 'prompt_attention_mask' in batch:
                prompt_mask = batch['prompt_attention_mask'][index]
            decoded = self._decode_prompt_from_ids(tokenizer, prompt_ids, prompt_mask)
            if decoded:
                return decoded

        return ""

    def _log_sample_generation(self, outputs: dict, labels: torch.Tensor, epoch: int, batch_idx: int):
        """Log sample generations for debugging."""
        try:
            model = self.model.module if hasattr(self.model, 'module') else self.model
            tokenizer = model.decoder.tokenizer

            generated_ids = outputs['generated_ids'][0] if 'generated_ids' in outputs else None
            label_ids = labels[0]

            if generated_ids is not None:
                raw_pred, raw_ref = 
                (tokenizer, generated_ids.cpu(), label_ids.cpu())
                generated_text = self._sanitize_chat_text(raw_pred)
                label_text = self._sanitize_chat_text(raw_ref)

                print("\n" + "="*60)
                print(f"Sample Generation (Epoch {epoch}, Batch {batch_idx})")
                print("-"*60)
                print(f"Generated: {generated_text[:200]}..." if len(generated_text) > 200 else f"Generated: {generated_text}")
                print("-"*60)
                print(f"Expected:  {label_text[:200]}..." if len(label_text) > 200 else f"Expected:  {label_text}")
                print("="*60 + "\n")
        except Exception as e:
            print(f"Error logging sample: {e}")
    
    def _log_epoch_summary(self, mode: RunMode, epoch: int, metrics: dict, total_loss: float, num_batches: int):
        """Log detailed epoch summary for debugging."""
        print("\n" + "="*80)
        print(f"{mode.value.upper()} EPOCH {epoch} SUMMARY - Phase {1 if self.current_phase == 'phase1' else 2}")
        print("-"*80)
        print(f"Average Loss: {total_loss/num_batches:.4f}")
        
        # Log metrics
        for key, value in metrics.items():
            if 'rouge' in key.lower() or 'bleu' in key.lower() or 'meteor' in key.lower():
                print(f"{key}: {value/num_batches:.4f}")
        
        # Log learning rates if available
        if self.optimizer:
            print("\nLearning Rates:")
            for group in self.optimizer.param_groups:
                if 'name' in group:
                    print(f"  {group['name']}: {group['lr']:.2e}")
        
        print("="*80 + "\n")

    def _train_step(
        self, 
        ecg_signal: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        prompt_input_ids: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        """
        Train a single step of the model.
        
        Args:
            ecg_signal: Input tensor of shape (batch_size, 12, length)
            input_ids: Input IDs for the LLM
            attention_mask: Attention mask for the LLM
            labels: Labels for the LLM
            
        Returns:
            dict[str, torch.Tensor]: Output from the model
        """
        # Clear gradients
        assert self.optimizer is not None
        
        self.optimizer.zero_grad()
        
        # Forward pass with autocast for mixed precision using bfloat16
        with autocast('cuda', dtype=torch.bfloat16):
            outputs: dict[str, torch.Tensor] = self.model(
                ecg_signal=ecg_signal,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                prompt_input_ids=prompt_input_ids,  # Pass for cross-attention
                prompt_attention_mask=prompt_attention_mask
            )
            loss: torch.Tensor = outputs['loss']

        # Backward pass - bfloat16 doesn't need gradient scaling
        loss.backward()
        
        # Sync gradients across processes before optimizer step
        DistributedUtils.sync_process_group(
            world_size=self.config.world_size,
            device_ids=self.config.device
        )

        max_grad_norm = getattr(self.config, 'max_grad_norm', 1.0)
        if max_grad_norm is not None and max_grad_norm > 0:
            clip_params: list[torch.nn.Parameter] = []
            for group in self.optimizer.param_groups:
                for param in group['params']:
                    if param is None:
                        continue
                    grad = getattr(param, 'grad', None)
                    if grad is not None:
                        clip_params.append(param)
            if clip_params:
                torch.nn.utils.clip_grad_norm_(clip_params, max_grad_norm)
        
        # Step optimizer directly without gradient scaling for bfloat16
        self.optimizer.step()
                
        # Get learning rate metrics
        lr_metrics = {}
        for pg in self.optimizer.param_groups if self.optimizer else []:
            if "name" in pg:
                lr_metrics[f"lr_{pg['name']}"] = pg["lr"]
        
        # Step the scheduler if it should be updated per-iteration
        if self.scheduler and self.scheduler_per_iteration:
            self.scheduler.step()
        
        return {
            "loss": loss,
            **lr_metrics
        }

    def _val_step(
        self, 
        ecg_signal: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        prompt_input_ids: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Validate a single step of the model.
        
        Args:
            ecg_signal: Input tensor of shape (batch_size, 12, length)
            input_ids: Input IDs for the LLM
            attention_mask: Attention mask for the LLM
            labels: Labels for the LLM
            
        Returns:
            dict[str, torch.Tensor]: Output from the model
        """
        with torch.no_grad():
            outputs: dict[str, torch.Tensor] = self.model(
                ecg_signal=ecg_signal,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                prompt_input_ids=prompt_input_ids,  # Pass for cross-attention
                prompt_attention_mask=prompt_attention_mask
            )
            loss: torch.Tensor = outputs['loss']
            
            if getattr(self.config, 'instruct_mode', False) and prompt_input_ids is not None:
                model_for_generation = self.model.module if hasattr(self.model, 'module') else self.model
                model_for_generation.eval()
                generated_ids = model_for_generation.generate_report_with_question(
                    ecg_signal,
                    prompt_input_ids=prompt_input_ids,
                    prompt_attention_mask=prompt_attention_mask,
                    max_token_length=self.config.max_token_length
                )
            else:
                model_for_generation = self.model.module if hasattr(self.model, 'module') else self.model
                model_for_generation.eval()
                generated_ids = model_for_generation.generate_report(
                    ecg_signal,
                    max_token_length=self.config.max_token_length
                )

            # Get learning rate metrics
            lr_metrics = {}
            for pg in self.optimizer.param_groups if self.optimizer else []:
                if "name" in pg:
                    lr_metrics[f"lr_{pg['name']}"] = pg["lr"]

            return {
                "loss": loss,
                "generated_ids": generated_ids,
                **lr_metrics
            }

    def _inference_step(
        self,
        ecg_signal: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        prompt_input_ids: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Inference step that generates text using the model's decoder."""

        with torch.no_grad():
            outputs: dict[str, torch.Tensor] = self.model(
                ecg_signal=ecg_signal,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask
            )
            loss: torch.Tensor = outputs['loss']

            if getattr(self.config, 'instruct_mode', False) and prompt_input_ids is not None:
                model_for_generation = self.model.module if hasattr(self.model, 'module') else self.model
                model_for_generation.eval()
                generated_ids = model_for_generation.generate_report_with_question(
                    ecg_signal,
                    prompt_input_ids=prompt_input_ids,
                    prompt_attention_mask=prompt_attention_mask,
                    max_token_length=self.config.max_token_length
                )
            else:
                model_for_generation = self.model.module if hasattr(self.model, 'module') else self.model
                model_for_generation.eval()
                generated_ids = model_for_generation.generate_report(
                    ecg_signal,
                    max_token_length=self.config.max_token_length
                )

            return {
                "loss": loss,
                "generated_ids": generated_ids
            }

    def inference(self):
        """
        Inference the model.
        """
        if self.validation_dataloader is None:
            raise ValueError("Validation dataloader is not set")
            
        self.model.train(False)
        
        predicted_reports: list[str] = []
        reference_reports: list[str] = []
        waveform_names: list[str] = []   
        tokenizer = self.validation_dataloader.dataset.tokenizer  # type: ignore
        
        # Create a progress bar
        data_iter: tqdm = tqdm(
            self.validation_dataloader, 
            desc="Inference",
            leave=True,
            disable=not self.config.is_ref_device
        )
        
        running_loss: float = 0.0
        for batch_idx, batch in enumerate(data_iter):
            ecg_signal: torch.Tensor = batch['signal'].to(self.config.device)
            input_ids: torch.Tensor = batch['input_ids'].to(self.config.device)
            attention_mask: torch.Tensor = batch['attention_mask'].to(self.config.device)
            labels: torch.Tensor = batch['labels'].to(self.config.device) if 'labels' in batch else input_ids.clone()
            
            outputs: dict[str, torch.Tensor] = self._inference_step(
                ecg_signal=ecg_signal,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                prompt_input_ids=(batch['prompt_input_ids'].to(self.config.device) if getattr(self.config, 'instruct_mode', False) and 'prompt_input_ids' in batch else None),
                prompt_attention_mask=(batch['prompt_attention_mask'].to(self.config.device) if getattr(self.config, 'instruct_mode', False) and 'prompt_input_ids' in batch else None)
            )
            generated_ids: torch.Tensor = outputs['generated_ids']
            loss = outputs['loss'].item()
            
            gathered_loss = DistributedUtils.gather_loss(
                [loss], 
                self.config.device
            )
            
            running_loss += gathered_loss
            
            data_iter.set_postfix({
                f"inference/mean_loss": f'{running_loss / (batch_idx + 1):.4f}'
            })
            batch_waveform_names: list[str] = batch['waveform_name']
            for idx in range(len(batch_waveform_names)):
                gen = generated_ids[idx]
                lab = labels[idx]
                filename = batch_waveform_names[idx]

                decoded_prediction, decoded_reference = decode_assistant_only_text(
                    tokenizer,
                    gen,
                    lab
                )
                predicted_reports.append(decoded_prediction)
                reference_reports.append(decoded_reference)
                waveform_names.append(filename)
                
        # --- Distributed Gathering using DistributedUtils ---
        results = {
            "waveform_names": waveform_names,
            "predicted_reports": predicted_reports,
            "reference_reports": reference_reports,
        }
        
        # Get the world size from DistributedUtils (using the config's world size)
        gathered_results = [None for _ in range(self.config.world_size)]
        DistributedUtils.all_gather_object(gathered_results, results)
        
        # Only the reference device (as determined by self.config.is_ref_device) consolidates and writes the output CSV.
        if self.config.is_ref_device:
            combined_waveform_names = []
            combined_predicted_reports = []
            combined_reference_reports = []
            for res in gathered_results:
                combined_waveform_names.extend(res["waveform_names"] if res is not None else [])
                combined_predicted_reports.extend(res["predicted_reports"] if res is not None else [])
                combined_reference_reports.extend(res["reference_reports"] if res is not None else [])
                    
            df = pd.DataFrame({
                'waveform_name': combined_waveform_names,
                'predicted_report': combined_predicted_reports,
                'reference_report': combined_reference_reports
            })
            os.makedirs(self.config.output_dir, exist_ok=True)
            csv_path = os.path.join(self.config.output_dir, 'inference.csv')
            df.to_csv(csv_path, index=False)

    def validate(self):
        raise NotImplementedError("Validate not implemented")

    def _save_model(
        self,
        epoch: int,
        loss: float,
        is_best: bool = False
    ):
        """
        Save model checkpoint and optionally mark as best model.
        
        Args:
            epoch: The current epoch
            loss: The loss of the model
            is_best: Whether the model is the best model

        Returns:
            None
        """
        save_dir: str = self.config.output_dir
        os.makedirs(save_dir, exist_ok=True)
        
        # Get the underlying model for DDP
        model = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Handle LoRA model saving
        model_state_dict = model.state_dict()
        if self.config.use_lora:
            # Check if any part of the model has LoRA
            llm_model = None
            if hasattr(model.decoder, 'llm_model'):
                llm_model = model.decoder.llm_model
            elif hasattr(model.decoder, 'llm'):
                llm_model = model.decoder.llm
            
            if llm_model and hasattr(llm_model, 'peft_config'):
                # Save only LoRA weights for smaller checkpoints
                try:
                    from peft import get_peft_model_state_dict
                    lora_state_dict = get_peft_model_state_dict(llm_model)
                    # Combine with other components (remove original LLM weights, add LoRA weights)
                    model_state_dict = {
                        **{k: v for k, v in model_state_dict.items() if not k.startswith('decoder.llm_model.') and not k.startswith('decoder.llm.')},
                        **{f'decoder.{"llm_model" if hasattr(model.decoder, "llm_model") else "llm"}.{k}': v for k, v in lora_state_dict.items()}
                    }
                except ImportError:
                    print("Warning: peft not available, saving full model state dict")
        
        # Prepare checkpoint
        checkpoint: dict[str, Any] = {
            'epoch': epoch,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'loss': loss,
            'config': self.config,
            'use_lora': self.config.use_lora  # Store LoRA flag for loading
        }
        
        # Save regular checkpoint for current epoch
        checkpoint_path: str = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        # Delete the checkpoint from the previous epoch if it exists
        if epoch > 0:
            prev_checkpoint_path: str = os.path.join(save_dir, f'checkpoint_epoch_{epoch - 1}.pt')
            if os.path.exists(prev_checkpoint_path):
                os.remove(prev_checkpoint_path)
                print(f"Deleted old checkpoint: {prev_checkpoint_path}")
        
        # If this is the best model, save it separately
        if is_best:
            best_model_path: str = os.path.join(save_dir, 'best_model.pt')
            torch.save(checkpoint, best_model_path)
            
        if self.wandb_wrapper is not None and self.wandb_wrapper.is_initialized():
            # Get current learning rates
            lr_metrics = {}
            if self.optimizer is not None:
                for pg in self.optimizer.param_groups:
                    if "name" in pg:
                        lr_metrics[f"checkpoint/lr_{pg['name']}"] = pg["lr"]
                else:
                    # Fallback for any unnamed groups
                    lr_metrics[f"checkpoint/lr_group_{id(pg) % 1000}"] = pg["lr"]
                
            self.wandb_wrapper.log({
                "checkpoint/epoch": epoch,
                "checkpoint/loss": loss,
                **lr_metrics
            })

    def _compute_metrics(
        self,
        outputs: dict[str, torch.Tensor],
        labels: torch.Tensor,
        dataloader: DataLoader,
        best_batch_metrics: dict[str, list[dict[str, Union[float, list[str]]]]],
        worst_batch_metrics: dict[str, list[dict[str, Union[float, list[str]]]]],
        random_batch_metrics: dict[str, list[dict[str, Union[float, list[str]]]]],
        random_batch: bool = False,
        batch: dict = None,
        batch_idx: int = 0,
    ) -> dict[str, float]:
        """
        Compute metrics for validation and update best/worst batch metrics.
        
        Args:
            outputs: Outputs from the model
            labels: Labels for the model
            dataloader: DataLoader for the model
            best_batch_metrics: Dictionary containing the best batch metrics
            worst_batch_metrics: Dictionary containing the worst batch metrics
            random_batch_metrics: Dictionary containing the random batch metrics
            random_batch: Whether the batch is random
            batch_idx: Index of the current batch within the epoch
            
        Returns:
            dict[str, float]: Dictionary containing the metrics for the epoch
        
        """
        computed_metrics: dict[str, float] = {}
        # Keep raw labels (with -100 prompt mask) so downstream metrics can drop prompt tokens
        tokenizer = dataloader.dataset.tokenizer  # type: ignore
        labels_for_metrics = labels

        # Decode predictions, references and prompts for logging/metrics enrichment
        batch_predictions: list[str] = []
        batch_references: list[str] = []
        batch_categories: list[str] = []
        batch_prompts: list[str] = []

        if batch is not None and 'generated_ids' in outputs:
            generated_ids = outputs['generated_ids']

            for i in range(generated_ids.size(0)):
                gen_tensor = generated_ids[i].cpu()
                label_tensor = labels[i].cpu()

                raw_prediction, raw_reference = decode_assistant_only_text(
                    tokenizer,
                    gen_tensor,
                    label_tensor
                )
                prediction = self._sanitize_chat_text(raw_prediction)
                reference = self._sanitize_chat_text(raw_reference)

                category = ""
                if 'prompt_category' in batch and i < len(batch['prompt_category']):
                    category_value = batch['prompt_category'][i]
                    category = category_value if category_value is not None else ""

                prompt = self._extract_prompt_text(tokenizer, batch, i)

                batch_predictions.append(prediction)
                batch_references.append(reference)
                batch_categories.append(category)
                batch_prompts.append(prompt)

            if self.category_metrics_calculator is not None:
                self.category_metrics_calculator.add_batch(
                    predictions=batch_predictions,
                    references=batch_references,
                    categories=batch_categories
                )

        for metric in self.config.metrics:
            metric_name_lower = metric.lower()
            if metric_name_lower == 'bertscore':
                limit = self.bertscore_max_batches
                if limit is not None and batch_idx >= limit:
                    if self.config.is_ref_device and not self._bertscore_skip_logged:
                        print(f"Skipping BERTScore for batches >= {limit}; current batch {batch_idx} exceeds limit.")
                        self._bertscore_skip_logged = True
                    continue

            registered_metrics: Union[
                RougeMetric, 
                BleuMetric, 
                MeteorMetric,
                BertScoreMetric
            ] = MetricRegistry.get(metric)
            
            # Extract input_ids from batch if available for better prompt trimming
            input_ids = None
            if batch is not None and 'input_ids' in batch:
                input_ids = batch['input_ids'].to(self.config.device)
            
            LLM_metrics: dict[str, Union[float, list[str]]] = registered_metrics.compute_score(
                outputs['generated_ids'],
                labels_for_metrics,
                tokenizer,  # type: ignore
                input_ids=input_ids
            )
            # Add prompts to metrics if available
            if batch_prompts:
                LLM_metrics['prompts'] = batch_prompts

            if 'predictions' in LLM_metrics:
                LLM_metrics['predictions'] = [self._sanitize_chat_text(str(p)) for p in LLM_metrics['predictions']]
            if 'references' in LLM_metrics:
                LLM_metrics['references'] = [self._sanitize_chat_text(str(r)) for r in LLM_metrics['references']]
            if 'prompts' in LLM_metrics:
                LLM_metrics['prompts'] = [self._sanitize_chat_text(str(p)) for p in LLM_metrics['prompts']]
            for metric_name, metric_value in LLM_metrics.items():
                if metric_name not in ('predictions', 'references', 'prompts'):
                    # Update the best metric
                    update_best_metric(
                        metric_name=metric_name,
                        llm_metrics=LLM_metrics,
                        best_metrics=best_batch_metrics,
                        K=1  # TODO: k > 1 implemented but haven't been tested
                    )
                    # Update the worst metric
                    update_worst_metric(
                        metric_name=metric_name,
                        llm_metrics=LLM_metrics,
                        worst_metrics=worst_batch_metrics,
                        K=1  # TODO: k > 1 implemented but haven't been tested with k > 1
                    )
                    
                    if random_batch:
                        # Update the random batch metric
                        update_random_batch_metric(
                            metric_name=metric_name,
                            llm_metrics=LLM_metrics,
                            random_metrics=random_batch_metrics,
                            k=1  # TODO: k > 1 implemented but haven't been tested with k > 1
                        )
                    
                    metric_value = float(metric_value) if isinstance(metric_value, torch.Tensor) else metric_value
                    if isinstance(metric_value, float):
                        computed_metrics[metric_name] = float(metric_value)
                    
        return computed_metrics
    
    def _init_validation_metrics(
            self,
            dataloader: DataLoader
        ) -> tuple[
            dict[str, list[dict[str, Union[float, list[str]]]]],
            dict[str, list[dict[str, Union[float, list[str]]]]],
            dict[str, list[dict[str, Union[float, list[str]]]]],
            int
        ]:
            """
            Initialize dictionaries for best, worst, and random batch metrics.
            
            Args:
                dataloader: DataLoader for the model
                
            Returns:
                tuple[
                    dict[str, list[dict[str, Union[float, list[str]]]]],
                    dict[str, list[dict[str, Union[float, list[str]]]]],
                    dict[str, list[dict[str, Union[float, list[str]]]]],
                    int
                ]:
                    Dictionary containing the best, worst, and random batch metrics
                    and the random batch index
            """
            # Initialize dictionaries for best, worst, and random batch metrics.
            worst_batch_metrics = {}
            best_batch_metrics = {}
            random_batch_metrics = {}
            # Select a random batch index
            random_batch_idx = random.randint(0, len(dataloader) - 1)
            return worst_batch_metrics, best_batch_metrics, random_batch_metrics, random_batch_idx

    def _get_val_generations_dir(self) -> str:
        """Return the directory where validation generations should be written."""
        if not hasattr(self, "_val_generations_dir"):
            output_dir = getattr(self.config, "output_dir", "")
            if isinstance(output_dir, str):
                output_dir = output_dir.strip()

            if output_dir:
                output_dir = os.path.expanduser(output_dir)
                self._val_generations_dir = os.path.normpath(output_dir)
            else:
                run_identifier = ""

                config_run_id = getattr(self.config, "run_id", None)
                if isinstance(config_run_id, str) and config_run_id.strip():
                    run_identifier = config_run_id.strip()

                if not run_identifier and self.wandb_wrapper is not None and self.wandb_wrapper.is_initialized():
                    try:
                        run_identifier = str(self.wandb_wrapper.get_run_id())
                    except Exception:
                        run_identifier = ""

                if not run_identifier:
                    run_identifier = time.strftime("%Y%m%d-%H%M%S_no_run_id")

                run_identifier = re.sub(r"[^A-Za-z0-9._-]", "_", run_identifier)
                if not run_identifier:
                    run_identifier = "run"

                self._val_generations_dir = os.path.join(
                    "ECG_tokenizer",
                    "val_generations",
                    run_identifier
                )

        return self._val_generations_dir

    def _get_val_generation_json_path(self, epoch: int) -> str:
        """Return the epoch-specific JSON path for validation generations."""
        return os.path.join(
            self._get_val_generations_dir(),
            f"val_generations_epoch_{epoch}.json"
        )

    def _append_batch_to_json(
        self,
        generated_ids: torch.Tensor,
        labels: torch.Tensor,
        batch: dict[str, Any],
        json_path: str
    ) -> None:
        """
        Process a batch and append results to JSON file immediately.
        
        Args:
            generated_ids: Generated token tensor for current batch (B, L)
            labels: Label tensor for current batch (B, L)
            batch: Original batch dictionary (for waveform names and prompts)
            json_path: Path to JSON file to append to
        """
        try:
            tokenizer = self.validation_dataloader.dataset.tokenizer  # type: ignore

            waveform_names = batch.get('waveform_name', [])

            if isinstance(waveform_names, torch.Tensor):
                waveform_names = waveform_names.tolist()
            if not isinstance(waveform_names, (list, tuple)):
                waveform_names = [waveform_names] * generated_ids.size(0)

            batch_data: dict[str, dict[str, Any]] = {}

            for i in range(generated_ids.size(0)):
                gen_tensor = generated_ids[i].detach().cpu()
                label_tensor = labels[i].detach().cpu()

                raw_generation, raw_reference = decode_assistant_only_text(
                    tokenizer,
                    gen_tensor,
                    label_tensor
                )
                generation = self._sanitize_chat_text(raw_generation)
                ground_truth = self._sanitize_chat_text(raw_reference)

                question = self._extract_prompt_text(tokenizer, batch, i)
                if not question:
                    try:
                        ds = self.validation_dataloader.dataset  # type: ignore
                        df = getattr(ds, 'df', None)
                        if df is not None and 'waveform_name' in df.columns:
                            wf_name = waveform_names[i] if isinstance(waveform_names, (list, tuple)) else waveform_names
                            question_row = df[df['waveform_name'] == wf_name]
                            if not question_row.empty:
                                for col in ['question', 'prompt_text', 'prompt']:
                                    if col in question_row.columns:
                                        q_val = question_row[col].iloc[0]
                                        if not pd.isna(q_val):
                                            question = self._sanitize_chat_text(str(q_val))
                                            break
                    except Exception:
                        pass

                patient_id = waveform_names[i] if isinstance(waveform_names, (list, tuple)) else waveform_names
                patient_id = str(patient_id)
                batch_data[patient_id] = {
                    'Metrics': {},  # Empty for speed - can compute later if needed
                    'Question': question,
                    'Generation': generation,
                    'Ground truth': ground_truth
                }
        except Exception as e:
            print(f"❌ Failed to prepare validation batch for JSON: {e}")
            traceback.print_exc()
            return

        world_size = max(1, int(getattr(self.config, 'world_size', 1)))
        gathered_batches: list[dict[str, dict[str, Any]] | None] = [None for _ in range(world_size)]

        try:
            DistributedUtils.all_gather_object(gathered_batches, batch_data)
        except Exception as gather_err:
            print(f"❌ Failed to gather validation batches across devices: {gather_err}")
            traceback.print_exc()
            gathered_batches = [batch_data]
            if not self.config.is_ref_device:
                return

        if not self.config.is_ref_device:
            return

        try:
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = {}

            merged_batch_data: dict[str, dict[str, Any]] = {}
            for device_batch in gathered_batches:
                if device_batch:
                    merged_batch_data.update(device_batch)

            if not merged_batch_data:
                return

            existing_data.update(merged_batch_data)

            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)

        except Exception as e:
            print(f"❌ Failed to append gathered batches to JSON: {e}")
            traceback.print_exc()
    
    def _plot_validation_ecgs(self, epoch: int, json_path: str) -> None:
        """Plot ECG waveforms with Q&A annotations for validation results."""
        if not self.config.plot_validation_ecgs or not self.config.is_ref_device:
            return

        try:
            from ecg_plotter.core import NPYECGPlotter
            import wandb

            with open(json_path, 'r', encoding='utf-8') as f:
                val_data = json.load(f)

            if not val_data:
                print("No validation data to plot")
                return

            parquet_df = load_parquet_mapping(VALIDATION_PLOT_PARQUET_PATH)
            parquet_lookup = build_parquet_lookup(parquet_df) if parquet_df is not None else {}

            scored_ecgs: list[dict[str, Any]] = []
            missing_lookup: list[str] = []
            for ecg_name, ecg_info in val_data.items():
                score, metric_name = compute_ecg_score(ecg_info)
                ecg_path, dataset_label = resolve_ecg_source(
                    ecg_name,
                    ecg_info,
                    parquet_lookup
                )

                if not ecg_path:
                    missing_lookup.append(ecg_name)
                    continue

                scored_ecgs.append({
                    "name": ecg_name,
                    "score": score,
                    "metric": metric_name,
                    "data": ecg_info,
                    "path": ecg_path,
                    "dataset": dataset_label,
                })

            if missing_lookup:
                print(
                    f"Skipping {len(missing_lookup)} ECGs without matching parquet entries"
                )

            if not scored_ecgs:
                print("No scored ECGs with resolvable paths available for plotting")
                return

            total_requested = min(self.config.num_validation_plots, len(scored_ecgs))
            if total_requested <= 0:
                return

            selected_ecgs = select_tiered_ecg_samples(scored_ecgs, total_requested)

            plot_images = []
            plot_dir = os.path.join(
                self._get_val_generations_dir(),
                f"ecg_plots_epoch_{epoch}"
            )
            os.makedirs(plot_dir, exist_ok=True)

            for idx, item in enumerate(selected_ecgs):
                ecg_name = item["name"]
                score = item["score"]
                metric_name = item.get("metric")
                tier_label = item.get("tier")
                ecg_info = item["data"]

                ecg_path = item.get("path")
                dataset_label = item.get("dataset")

                if not ecg_path or not os.path.exists(ecg_path):
                    print(f"ECG file not found for {ecg_name}: {ecg_path}")
                    continue

                plot_dataset, fft_normalized, normalized_dataset = determine_plotter_params(dataset_label)

                plotter = NPYECGPlotter(
                    npy_path=ecg_path,
                    dataset=plot_dataset,
                    out_dir=plot_dir,
                    width=2500,
                    fft_normalized=fft_normalized
                )

                question = clean_chat_artifacts(ecg_info.get('Question', 'N/A'))[:100]
                generated = clean_chat_artifacts(ecg_info.get('Generation', 'N/A'))[:150]
                ground_truth = clean_chat_artifacts(ecg_info.get('Ground truth', 'N/A'))[:150]

                if len(generated) > 150:
                    generated = generated[:147] + "..."
                if len(ground_truth) > 150:
                    ground_truth = ground_truth[:147] + "..."

                metadata_bits: list[str] = []
                if tier_label:
                    metadata_bits.append(tier_label.upper())
                metadata_bits.append(f"score={score:.3f}")
                if metric_name:
                    metadata_bits.append(metric_name)
                if normalized_dataset:
                    metadata_bits.append(f"dataset={normalized_dataset}")

                heading = " | ".join(metadata_bits)
                title = (
                    f"Epoch {epoch} - [{heading}] ECG: {ecg_name}\n"
                    f"Q: {question}\n"
                    f"Generated: {generated}\n"
                    f"Ground Truth: {ground_truth}"
                )

                try:
                    img, _ = plotter.plot_ecg(
                        title=title,
                        save=False,
                        anonymize=True,
                        show_diagnosis=False
                    )
                except Exception as plot_err:
                    print(f"Failed to plot ECG {ecg_name}: {plot_err}")
                    continue

                if not img:
                    continue

                safe_tier = tier_label or "sample"
                save_path = os.path.join(
                    plot_dir,
                    f"{safe_tier}_{idx:02d}_{ecg_name.replace('.npy', '')}.png"
                )
                img.save(save_path, dpi=(240, 240))
                caption = f"{safe_tier}: {ecg_name} ({score:.3f})"
                plot_images.append(wandb.Image(img, caption=caption))
                print(f"✓ Plotted ECG {idx + 1}/{len(selected_ecgs)}: {save_path}")

            if plot_images and self.wandb_wrapper is not None and self.wandb_wrapper.is_initialized():
                self.wandb_wrapper.log({
                    f"val/ecg_plots_epoch_{epoch}": plot_images
                })
                print(f"✓ Logged {len(plot_images)} ECG plots to WandB")

        except Exception as e:
            print(f"Error in ECG plotting: {e}")
            traceback.print_exc()
