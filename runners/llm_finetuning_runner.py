import os
import json
import torch
import pandas as pd
from torch.optim.adamw import AdamW
from torch.utils.data import DataLoader
from torch.amp.autocast_mode import autocast
from torch.cuda.amp.grad_scaler import GradScaler
from torch.optim.lr_scheduler import LRScheduler
from transformers import GPT2Tokenizer

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
    BleuMetric,
    MeteorMetric,
    BertScoreMetric,
    update_best_metric,
    update_worst_metric,
    update_random_batch_metric
)
from utils.metrics.category_metrics import CategoryMetricsCalculator
from runners.base_runner import BaseRunner
from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper

import random
from tqdm import tqdm
from typing import (
    Any, 
    Union, 
    Callable,
    Optional
)


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
            json_path = f"./ECG_tokenizer/val_generations/val_generations_epoch_{epoch}.json"
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
                # Process and append to JSON immediately (only on reference device)
                if self.config.is_ref_device:
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
                            
                    self.wandb_wrapper.log(log_dict)
            
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
            debug_config = getattr(self.config, 'debug_config', {})
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
        debug_config = getattr(self.config, 'debug_config', {})
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
                    import traceback
                    traceback.print_exc()
            
            # Log everything to wandb
            self.wandb_wrapper.log({
                "val/best_metrics_html": wandb.Html(best_html),
                "val/worst_metrics_html": wandb.Html(worst_html),
                "val/random_metrics_html": wandb.Html(random_html),
                **category_log_dict
            })
            
            # JSON export is done incrementally during validation
        # === End new block ===
                
        # Normalize the epoch metrics
        for k in epoch_metrics:
            epoch_metrics[k] /= len(dataloader)
        
        # Return the epoch metrics
        return epoch_metrics
    
    def _extract_assistant_text(self, tokenizer, generated_ids: torch.Tensor, label_ids: torch.Tensor) -> str:
        """Return only the assistant portion of the generated text."""
        if generated_ids is None:
            return ""

        # Determine where assistant labels begin so we can drop the prompt portion
        first_assistant_idx = 0
        non_ignored = (label_ids != -100).nonzero(as_tuple=False)
        if non_ignored.numel() > 0:
            first_assistant_idx = int(non_ignored[0].item())

        trimmed_ids = generated_ids[first_assistant_idx:]
        return tokenizer.decode(trimmed_ids.tolist(), skip_special_tokens=True)

    def _log_sample_generation(self, outputs: dict, labels: torch.Tensor, epoch: int, batch_idx: int):
        """Log sample generations for debugging."""
        try:
            model = self.model.module if hasattr(self.model, 'module') else self.model
            tokenizer = model.decoder.tokenizer

            generated_ids = outputs['generated_ids'][0] if 'generated_ids' in outputs else None
            label_ids = labels[0]

            if generated_ids is not None:
                generated_text = self._extract_assistant_text(tokenizer, generated_ids.cpu(), label_ids.cpu())
                label_text = tokenizer.decode(label_ids[label_ids != -100].tolist(), skip_special_tokens=True)

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
            
            # Time the generate_report function
            if getattr(self.config, 'instruct_mode', False) and prompt_input_ids is not None:
                # Build begin_suppress_tokens to prevent leaking headers like 'assistant' at start
                tokenizer = self.validation_dataloader.dataset.tokenizer  # type: ignore
                # begin_suppress_tokens: list[int] = []
                # try:
                #     # Special header tokens
                #     for tok in ["<|start_header_id|>", "<|end_header_id|>"]:
                #         tok_id = tokenizer.convert_tokens_to_ids(tok)
                #         if tok_id is not None and tok_id != -1:
                #             begin_suppress_tokens.append(int(tok_id))
                #     # Note: do not suppress 'assistant' token forms for now
                # except Exception:
                #     pass
                
                # Ensure model is in eval mode for generation
                self.model.eval()
                
                if hasattr(self.model, 'module'):
                    gen_ids_q = self.model.module.generate_report_with_question(
                        ecg_signal,
                        prompt_input_ids=prompt_input_ids,
                        prompt_attention_mask=prompt_attention_mask,
                        max_token_length=self.config.max_token_length
                        # begin_suppress_tokens=begin_suppress_tokens if len(begin_suppress_tokens) > 0 else None
                    )
                else:
                    gen_ids_q = self.model.generate_report_with_question(
                        ecg_signal,
                        prompt_input_ids=prompt_input_ids,
                        prompt_attention_mask=prompt_attention_mask,
                        max_token_length=self.config.max_token_length
                        # begin_suppress_tokens=begin_suppress_tokens if len(begin_suppress_tokens) > 0 else None
                    )
                generated_ids = gen_ids_q
            else:
                if hasattr(self.model, 'module'):
                    gen_ids = self.model.module.generate_report(
                        ecg_signal, 
                        max_token_length=self.config.max_token_length
                    )
                else:
                    gen_ids = self.model.generate_report(
                        ecg_signal, 
                        max_token_length=self.config.max_token_length
                    )
                generated_ids = gen_ids

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
        """
        Inference a single step of the model.
        
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
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask
            )
            loss: torch.Tensor = outputs['loss']
            
            # Time the generate_report function
            if getattr(self.config, 'instruct_mode', False) and prompt_input_ids is not None:
                if hasattr(self.model, 'module'):
                    gen_ids_q = self.model.module.generate_report_with_question(
                        ecg_signal,
                        prompt_input_ids=prompt_input_ids,
                        prompt_attention_mask=prompt_attention_mask,
                        max_token_length=self.config.max_token_length
                    )
                else:
                    gen_ids_q = self.model.generate_report_with_question(
                        ecg_signal,
                        prompt_input_ids=prompt_input_ids,
                        prompt_attention_mask=prompt_attention_mask,
                        max_token_length=self.config.max_token_length
                    )  
                generated_ids = gen_ids_q
            else:
                if hasattr(self.model, 'module'):
                    gen_ids = self.model.module.generate_report(
                        ecg_signal, 
                        max_token_length=self.config.max_token_length
                    )
                else:
                    gen_ids = self.model.generate_report(
                        ecg_signal, 
                        max_token_length=self.config.max_token_length
                    )  
                generated_ids = gen_ids
            
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
        tokenizer: GPT2Tokenizer = self.validation_dataloader.dataset.tokenizer  # type: ignore
        
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

                # Decode only newly generated tokens (skip input portion)
                gen_list = gen.tolist()

                # In instruction mode, skip ECG token + prompt; in normal mode, skip just ECG token
                if getattr(self.config, 'instruct_mode', False) and 'prompt_input_ids' in batch:
                    prompt_row = batch['prompt_input_ids'][idx]
                    if hasattr(tokenizer, 'pad_token_id') and tokenizer.pad_token_id is not None:
                        prompt_len = (prompt_row != tokenizer.pad_token_id).sum().item()
                    else:
                        prompt_len = int((prompt_row != 0).sum().item())
                    skip_tokens = prompt_len + 1  # +1 for ECG token
                else:
                    skip_tokens = 1  # Just skip ECG token

                # Slice from the end of input to get only newly generated tokens
                if len(gen_list) > skip_tokens:
                    gen_list = gen_list[skip_tokens:]
                else:
                    gen_list = []  # No new tokens generated

                decoded_prediction = tokenizer.decode(gen_list, skip_special_tokens=True)
                decoded_reference  = tokenizer.decode(lab.tolist(), skip_special_tokens=True)
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

        # Decode predictions, references and prompts for category metrics
        batch_predictions = []
        batch_references = []
        batch_categories = []
        batch_prompts = []
        
        if self.category_metrics_calculator is not None and batch is not None:
            generated_ids = outputs['generated_ids']
            
            for i in range(generated_ids.size(0)):
                gen_tensor = generated_ids[i].cpu()
                label_tensor = labels[i].cpu()

                prediction = self._extract_assistant_text(tokenizer, gen_tensor, label_tensor)
                reference = tokenizer.decode(label_tensor[label_tensor != -100].tolist(), skip_special_tokens=True)
                
                # Get category if available
                category = ""
                if 'prompt_category' in batch and i < len(batch['prompt_category']):
                    category = batch['prompt_category'][i] if batch['prompt_category'][i] is not None else ""
                
                # Use original prompt text from batch
                prompt = ""
                if 'prompt_text' in batch and i < len(batch['prompt_text']):
                    prompt = batch['prompt_text'][i] if batch['prompt_text'][i] is not None else ""
                
                batch_predictions.append(prediction)
                batch_references.append(reference)
                batch_categories.append(category)
                batch_prompts.append(prompt)
            
            # Add to category metrics calculator
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
            LLM_metrics: dict[str, Union[float, list[str]]] = registered_metrics.compute_score(
                outputs['generated_ids'],
                labels_for_metrics,
                tokenizer  # type: ignore
            )
            # Add prompts to metrics if available
            if batch_prompts:
                LLM_metrics['prompts'] = batch_prompts
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
            prompt_texts = batch.get('prompt_text')
            question_fallback = batch.get('question')

            if isinstance(waveform_names, torch.Tensor):
                waveform_names = waveform_names.tolist()
            if not isinstance(waveform_names, (list, tuple)):
                waveform_names = [waveform_names] * generated_ids.size(0)

            batch_data: dict[str, dict[str, Any]] = {}

            for i in range(generated_ids.size(0)):
                gen_tensor = generated_ids[i].detach().cpu()
                label_tensor = labels[i].detach().cpu()

                generation = self._extract_assistant_text(tokenizer, gen_tensor, label_tensor).strip()
                reference_tokens = label_tensor[label_tensor != -100].tolist()
                ground_truth = tokenizer.decode(reference_tokens, skip_special_tokens=True).strip()

                question = ""
                if isinstance(prompt_texts, (list, tuple)) and i < len(prompt_texts):
                    raw_question = prompt_texts[i]
                    if raw_question is not None:
                        question = str(raw_question).strip()
                elif isinstance(question_fallback, (list, tuple)) and i < len(question_fallback):
                    raw_question = question_fallback[i]
                    if raw_question is not None:
                        question = str(raw_question).strip()
                else:
                    try:
                        ds = self.validation_dataloader.dataset  # type: ignore
                        df = getattr(ds, 'df', None)
                        if df is not None and 'waveform_name' in df.columns:
                            wf_name = waveform_names[i] if isinstance(waveform_names, (list, tuple)) else waveform_names
                            question_row = df[df['waveform_name'] == wf_name]
                            if not question_row.empty:
                                q_val = None
                                for col in ['question', 'prompt_text', 'prompt']:
                                    if col in question_row.columns:
                                        q_val = question_row[col].iloc[0]
                                        if not pd.isna(q_val):
                                            break
                                if q_val is not None and not pd.isna(q_val):
                                    question = str(q_val).strip()
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
            
            # Read existing JSON and append
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except (FileNotFoundError, json.JSONDecodeError):
                existing_data = {}
            
            # Merge batch data
            existing_data.update(batch_data)
            
            # Write back to file
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(existing_data, f, ensure_ascii=False, indent=2)
                
        except Exception as e:
            print(f"❌ Failed to append batch to JSON: {e}")
            import traceback
            traceback.print_exc()
