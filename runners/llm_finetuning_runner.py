import os
import json
import re
import time
import traceback
import math
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
from utils.constants import ECG_PATTERNS
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
from sklearn.metrics import roc_auc_score

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
        phase1_train_dataloader: DataLoader | None = None,
        optimizer: AdamW | None = None,
        scheduler: LRScheduler | None = None,
        scaler: GradScaler | None = None,
        start_epoch: int = 1,
        test_dataloader: DataLoader | None = None,
    ):
        """
        Args:
            model: ECG tokenizer wrapper
            config: Configuration for the runner
            validation_dataloader: DataLoader for validation
            wandb_wrapper: WandbWrapper for logging
            train_dataloader: DataLoader for training
            phase1_train_dataloader: Optional DataLoader for phase1 subset training
            optimizer: Optimizer for the model
            scheduler: Scheduler for the optimizer
            scaler: Scaler for the optimizer
            test_dataloader: DataLoader for testing
        """
        self.model: ECG_Tokenizer_Wrapper = model
        self.config: LLMFinetuningConfig = config
        self.wandb_wrapper: WandbWrapper | None = wandb_wrapper
        self.full_train_dataloader: DataLoader | None = train_dataloader
        self.phase1_train_dataloader: DataLoader | None = phase1_train_dataloader
        self.train_dataloader: DataLoader | None = train_dataloader
        self.validation_dataloader: DataLoader | None = validation_dataloader
        self.test_dataloader: DataLoader | None = test_dataloader
        self.optimizer: AdamW | None = optimizer
        self.scheduler: LRScheduler | None = scheduler
        self.scaler: GradScaler | None = scaler
        self.scheduler_per_iteration: bool = scheduler_is_per_iteration(self.config)
        # Gradient accumulation configuration
        self.grad_accum: int = max(1, int(getattr(self.config, 'gradient_accumulation_steps', 1) or 1))
        self._grad_accum_counter: int = 0
        
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

        # Global step tracking and snapshot bookkeeping
        # Support resuming from a specific global step (for mid-epoch resume)
        resume_step = getattr(self.config, 'resume_global_step', None)
        self.global_step: int = int(resume_step) if resume_step is not None else 0
        # If resuming, set last validation snapshot step to prevent immediate validation
        # The next validation will occur at the next interval multiple after resume_step
        if resume_step is not None and resume_step > 0:
            interval = getattr(self.config, 'validation_step_interval', 500) or 500
            # Set to resume_step so validation won't trigger until next interval
            self._last_validation_snapshot_step: int = int(resume_step)
            if self.config.is_ref_device:
                print(f"[Resume] Starting from global_step={self.global_step}, next validation at step {((self.global_step // interval) + 1) * interval}")
        else:
            self._last_validation_snapshot_step: int = -1  # Track last step where validation snapshot ran
        self._last_train_snapshot_step: int = -1  # Track last step where train snapshot ran
        self._train_metric_remaining: int = 0
        self._train_metric_accumulator: dict[str, float] = {}
        self._train_metric_batches_collected: int = 0
        # Support resuming best_val_loss from checkpoint
        resume_best_val_loss = getattr(self.config, 'resume_best_val_loss', None)
        self.best_val_loss: float = float(resume_best_val_loss) if resume_best_val_loss is not None else float("inf")
        self._last_snapshot_checkpoint: Optional[str] = None
        
        # Pre-download NLTK resources at init to avoid repeated downloads during metrics
        self._ensure_nltk_resources()

        # Initialize CF evaluator if enabled
        self.cf_evaluator = None
        try:
            if getattr(self.config, 'use_cf_eval', False) and getattr(self.config, 'cf_eval_dataset_path', None):
                from utils.metrics.cf_evaluator import CFEvaluator
                # Device string for evaluator (use same CUDA index if available)
                dev_str = f"cuda:{self.config.device}" if torch.cuda.is_available() else "cpu"
                self.cf_evaluator = CFEvaluator(
                    cf_dataset_path=str(self.config.cf_eval_dataset_path),
                    device=dev_str,
                    use_letter_space=bool(getattr(self.config, "medgemma_prompt_style", False)),
                )
                if self.config.is_ref_device:
                    print(f"[CF Eval] Initialized with dataset: {self.config.cf_eval_dataset_path}")
        except Exception as exc:
            self.cf_evaluator = None
            if self.config.is_ref_device:
                print(f"[CF Eval] Disabled due to initialization error: {exc}")
        
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

        self._set_phase_train_dataloader(self.current_phase)

    def _set_phase_train_dataloader(self, phase: str | None) -> None:
        if self.full_train_dataloader is None:
            return
        if phase == 'phase1' and self.phase1_train_dataloader is not None:
            self.train_dataloader = self.phase1_train_dataloader
        else:
            self.train_dataloader = self.full_train_dataloader

    def _build_generation_kwargs(self, tokenizer=None) -> dict:
        """Build generation kwargs for validation generations."""
        generation_kwargs = dict(getattr(self.config, "default_generation_kwargs", {}) or {})
        generation_kwargs.setdefault("max_new_tokens", 96)
        # Use stochastic sampling (default HF behavior) - produces better quality outputs
        generation_kwargs.setdefault("no_repeat_ngram_size", 5)
        generation_kwargs.setdefault("repetition_penalty", 1.1)
        
        # Get tokenizer for EOS token IDs
        if tokenizer is None:
            try:
                tokenizer = self.validation_dataloader.dataset.tokenizer  # type: ignore
            except Exception:
                tokenizer = None
        
        if tokenizer is not None:
            generation_kwargs.setdefault("pad_token_id", tokenizer.pad_token_id)
            # MedGemma should stop at <end_of_turn> (106) or <eos> (1)
            eos_ids = [tokenizer.eos_token_id]
            try:
                end_of_turn_id = tokenizer.convert_tokens_to_ids("<end_of_turn>")
                if isinstance(end_of_turn_id, int) and end_of_turn_id > 0:
                    eos_ids.append(end_of_turn_id)
            except Exception:
                pass
            generation_kwargs.setdefault("eos_token_id", eos_ids)
        
        return generation_kwargs

    
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
    
    @staticmethod
    def _ensure_nltk_resources() -> None:
        """Pre-download NLTK resources to avoid repeated downloads during metrics computation."""
        try:
            import nltk
            import os
            import sys
            # Suppress download messages
            with open(os.devnull, 'w') as devnull:
                old_stdout, old_stderr = sys.stdout, sys.stderr
                try:
                    sys.stdout, sys.stderr = devnull, devnull
                    for pkg in ["wordnet", "punkt", "punkt_tab", "omw-1.4"]:
                        try:
                            nltk.download(pkg, quiet=True)
                        except Exception:
                            pass
                finally:
                    sys.stdout, sys.stderr = old_stdout, old_stderr
        except ImportError:
            pass  # NLTK not installed, skip

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
        
        # Cross-attention (part of bridge for SequenceTokenBridge)
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
        
        # Removed cross-phase log persistence to avoid replay artifacts

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
            if getattr(self, "_debug_stop_triggered", False):
                print("Debug prompt flag triggered; exiting after first batch.")
                return
            
            # Log phase-specific metrics
            if self.config.is_ref_device:
                epoch_train_metrics[f'{RunMode.TRAIN}/current_phase'] = 1 if self.current_phase == 'phase1' else 2
                        
            if self.wandb_wrapper is not None and self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
                # Log only current training epoch metrics (no cross-phase replay), anchored to current global step
                self.wandb_wrapper.log(dict(epoch_train_metrics), step=int(self.global_step))
            
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
                epoch,
                max_batches=getattr(self.config, "validation_max_batches", None)
            )
            
            # Save best model (only on reference device)
            if self.config.is_ref_device:
                current_val_loss = epoch_metrics[f'{RunMode.VALIDATE}/loss']
                if current_val_loss < self.best_val_loss:
                    self.best_val_loss = current_val_loss
                    self._save_model(
                        epoch=epoch,
                        loss=current_val_loss,
                        is_best=True
                    )
                
                # Also save regular checkpoint
                self._save_model(
                    epoch=epoch,
                    loss=current_val_loss,
                    is_best=False
                )
            
            # Sync after validation epoch, before next epoch            
            if self.wandb_wrapper is not None and self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
                # Log validation epoch metrics without replaying training scalars, anchor to current global step
                lr_metrics = {k: v for k, v in epoch_metrics.items() if "lr_" in k}
                val_epoch_payload = {**epoch_metrics, **lr_metrics, f"{RunMode.VALIDATE}/best_loss": self.best_val_loss}
                # Also emit concise 'val/' aliases for key loss scalars so dashboards stay consistent
                val_aliases: dict[str, float] = {}
                for key, value in val_epoch_payload.items():
                    if not key.startswith(f"{RunMode.VALIDATE}/"):
                        continue
                    suffix = key.split("/", 1)[1]
                    if suffix.endswith("loss") or suffix.startswith("pattern_loss") or suffix.startswith("cf_loss"):
                        val_aliases[f"val/{suffix}"] = value
                val_epoch_payload.update(val_aliases)
                self.wandb_wrapper.log(val_epoch_payload, step=int(self.global_step))

            # Sync the process group
            DistributedUtils.sync_process_group(
                world_size=self.config.world_size,
                device_ids=self.config.device
            )
                
    def _run_epoch(
        self,
        mode: RunMode,
        epoch: int,
        max_batches: int | None = None,
        snapshot_step: int | None = None,
        snapshot: bool = False
    )->dict[str, float]:
        """
        Run an epoch of training, validation, or testing.
        
        Args:
            mode: The execution mode (TRAIN, VALIDATE, TEST)
            epoch: The current epoch
            max_batches: Optional cap on number of batches to process (for snapshots)
            snapshot_step: Optional global step identifier for snapshot logging
            snapshot: Whether this run is a mid-epoch snapshot (affects logging side effects)
            
        Returns:
            dict[str, float]: Dictionary containing the metrics for the epoch
        """
        assert mode in [RunMode.TRAIN, RunMode.VALIDATE, RunMode.TEST]
        
        # Set the model to training or evaluation mode
        self.model.train(mode == RunMode.TRAIN)
        if not hasattr(self, "_debug_stop_triggered"):
            self._debug_stop_triggered = False

        # Reset accumulation counter at the start of each training epoch
        if mode == RunMode.TRAIN:
            try:
                self._grad_accum_counter = 0
            except Exception:
                pass
        
        if self.train_dataloader is None and self.validation_dataloader is None and self.test_dataloader is None:
            raise ValueError("Train, validation, or test dataloader is not set")
        
        # Get the dataloader and step function
        dataloader: DataLoader = self.train_dataloader if mode == RunMode.TRAIN else self.validation_dataloader
        step_fn: Callable | None = self._train_step if mode == RunMode.TRAIN else self._val_step

        # Prepare full-epoch aggregation of predictions/references for validation
        if mode == RunMode.VALIDATE:
            self._val_agg_preds: list[str] = []
            self._val_agg_refs: list[str] = []
        
        # Ensure distributed samplers reshuffle appropriately
        self._set_sampler_epoch(
            dataloader=dataloader,
            mode=mode,
            epoch=epoch,
            snapshot_step=snapshot_step,
            snapshot=snapshot
        )

        # Create a progress bar for the epoch
        phase_str = f"[Phase {1 if self.current_phase == 'phase1' else 2}]"
        try:
            dataloader_len = len(dataloader)  # type: ignore[arg-type]
        except TypeError:
            dataloader_len = None
        total_for_tqdm = max_batches if max_batches is not None else dataloader_len
        snapshot_suffix = ""
        if snapshot:
            if snapshot_step is not None:
                snapshot_suffix = f" (snapshot step {snapshot_step})"
            else:
                snapshot_suffix = " (snapshot)"
        data_iter: tqdm = tqdm(
            dataloader, 
            desc=f"{phase_str} {mode} epoch {epoch}/{self.config.num_epochs}{snapshot_suffix}",
            leave=True,
            disable=not self.config.is_ref_device,
            total=total_for_tqdm
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
        batches_processed: int = 0
        first_batch_logged = False

        pattern_logits_batches: list[torch.Tensor] = []
        pattern_targets_batches: list[torch.Tensor] = []
        debug_config = getattr(self.config, 'debug_config', None) or {}

        if mode == RunMode.VALIDATE:
            self._bertscore_skip_logged = False
            worst_batch_metrics, best_batch_metrics, random_batch_metrics, random_batch_idx = self._init_validation_metrics(
                dataloader,
                max_batches=max_batches
            )
            # Write validation generations JSON using inference-style generation
            write_val_generations = bool(getattr(self.config, "write_val_generations", True))
            json_path = None
            if write_val_generations:
                # Initialize JSON file for incremental writing
                json_path = self._get_val_generation_json_path(
                    epoch,
                    step=snapshot_step if snapshot else None,
                    prefix=getattr(self.config, 'validation_snapshot_prefix', 'step') if snapshot else None
                )
                os.makedirs(os.path.dirname(json_path), exist_ok=True)
                # Initialize with empty dict
                if self.config.is_ref_device:
                    with open(json_path, 'w', encoding='utf-8') as f:
                        json.dump({}, f)

            # Reset category metrics calculator for this validation epoch
            if self.category_metrics_calculator is not None:
                self.category_metrics_calculator.reset()
            # Reset validation telemetry
            self._val_eot_counter = []
            self._val_len_counter = []
        
        for batch_idx, batch in enumerate(data_iter):            
            # Preprocess the batch
            ecg_signal: torch.Tensor = batch['signal'].to(self.config.device)
            input_ids: torch.Tensor = batch['input_ids'].to(self.config.device)
            attention_mask: torch.Tensor = batch['attention_mask'].to(self.config.device)
            labels: torch.Tensor = batch['labels'].to(self.config.device) if 'labels' in batch else input_ids.clone()
            pattern_targets: torch.Tensor | None = None
            if 'pattern_targets' in batch:
                pattern_targets = batch['pattern_targets'].to(self.config.device, dtype=torch.float32)
            
            prompt_input_ids_tensor: torch.Tensor | None = None
            prompt_attention_mask_tensor: torch.Tensor | None = None
            if 'prompt_input_ids' in batch:
                prompt_input_ids_tensor = batch['prompt_input_ids'].to(self.config.device)
                prompt_attention_mask_tensor = (
                    batch['prompt_attention_mask'].to(self.config.device)
                    if 'prompt_attention_mask' in batch else None
                )

                # Debug ablations: shuffle prompts within batch
                if getattr(self.config, 'debug_shuffle_prompts', False):
                    perm = torch.randperm(prompt_input_ids_tensor.size(0), device=prompt_input_ids_tensor.device)
                    prompt_input_ids_tensor = prompt_input_ids_tensor[perm]
                    if prompt_attention_mask_tensor is not None:
                        prompt_attention_mask_tensor = prompt_attention_mask_tensor[perm]
                    if self.config.is_ref_device and batch_idx == 0:
                        print("[DEBUG] Shuffled prompt_input_ids across batch.")

            # Debug ablations: zero or shuffle ECG signals
            ablate_mode = getattr(self.config, 'debug_ablate_ecg', None)
            if ablate_mode in ("zero", "shuffle"):
                if ablate_mode == "zero":
                    ecg_signal = torch.zeros_like(ecg_signal)
                    if self.config.is_ref_device and batch_idx == 0:
                        print("[DEBUG] Zeroed ECG signals for ablation.")
                elif ablate_mode == "shuffle":
                    perm = torch.randperm(ecg_signal.size(0), device=ecg_signal.device)
                    ecg_signal = ecg_signal[perm]
                    if self.config.is_ref_device and batch_idx == 0:
                        print("[DEBUG] Shuffled ECG signals across batch.")

            # Run the step function
            if prompt_input_ids_tensor is not None:
                # Pass prompt_input_ids for both train and validate to prevent answer leakage
                outputs = step_fn(
                    ecg_signal=ecg_signal,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    prompt_input_ids=prompt_input_ids_tensor,
                    prompt_attention_mask=prompt_attention_mask_tensor,
                    pattern_targets=pattern_targets,
                )
            else:
                outputs = step_fn(
                    ecg_signal=ecg_signal, 
                    input_ids=input_ids, 
                    attention_mask=attention_mask, 
                    labels=labels,
                    pattern_targets=pattern_targets,
                )

            if getattr(self.config, 'debug_prompt_stop_after_first_batch', False) and batch_idx == 0:
                print("Debug flag set; stopping after first batch.")
                self._debug_stop_triggered = True
                # Still allow first-batch logging before breaking
                if not first_batch_logged and self.config.is_ref_device and mode == RunMode.VALIDATE:
                    first_batch_logged = self._print_first_batch_details(
                        outputs,
                        batch,
                        labels,
                        pattern_targets
                    )
                if getattr(self.config, 'debug_prompt_dump', False) and self.config.is_ref_device:
                    self._debug_dump_prompt_batch(batch, labels, prefix=f"{mode} batch {batch_idx}")
                break

            if (
                not first_batch_logged
                and batch_idx == 0
                and self.config.is_ref_device
                and mode == RunMode.VALIDATE
            ):
                first_batch_logged = self._print_first_batch_details(
                    outputs,
                    batch,
                    labels,
                    pattern_targets
                )
                if getattr(self.config, 'debug_prompt_dump', False):
                    self._debug_dump_prompt_batch(batch, labels, prefix="validate batch 0")
            
            # initialize metrics
            metrics: dict[str, float] = {}
            metrics['loss'] = outputs['loss'].item()  # type: ignore[index]
            pattern_loss_value = outputs.get('pattern_loss')
            if pattern_loss_value is not None:
                if torch.is_tensor(pattern_loss_value):
                    metrics['pattern_loss'] = float(pattern_loss_value.item())
                else:
                    metrics['pattern_loss'] = float(pattern_loss_value)
            cf_loss_value = outputs.get('cf_loss')
            if cf_loss_value is not None:
                if torch.is_tensor(cf_loss_value):
                    metrics['cf_loss'] = float(cf_loss_value.item())
                else:
                    metrics['cf_loss'] = float(cf_loss_value)
            
            # Extract learning rate and gradient metrics
            for key, value in outputs.items():  # type: ignore[attr-defined]
                if key.startswith('lr_'):
                    metrics[key] = float(value) if isinstance(value, torch.Tensor) else float(value)
                elif key.startswith('grad_norm/'):
                    metrics[key] = float(value.detach().item()) if isinstance(value, torch.Tensor) else float(value)
            
            # Compute rouge score, bleu score, and meteor score
            if mode == RunMode.VALIDATE:
                # --- Validation telemetry (EOT hit-rate and generated length) ---
                try:
                    model_for_generation = self.model.module if hasattr(self.model, 'module') else self.model
                    decoder = getattr(model_for_generation, 'decoder', None)
                    tokenizer = getattr(decoder, 'tokenizer', None) if decoder is not None else None
                    gen_ids = outputs.get('generated_ids')
                    if tokenizer is not None and isinstance(gen_ids, torch.Tensor):
                        eot_ids_list = []
                        # Primary end-of-turn token
                        try:
                            eot_tok = "<|eot_id|>"
                            eot_id = tokenizer.convert_tokens_to_ids(eot_tok)
                            if eot_id is not None and eot_id != -1:
                                eot_ids_list.append(int(eot_id))
                        except Exception:
                            pass
                        # Fallback to eos if available
                        eos_id = getattr(tokenizer, 'eos_token_id', None)
                        if eos_id is not None:
                            if isinstance(eos_id, (list, tuple)) and eos_id:
                                eot_ids_list.append(int(eos_id[0]))
                            elif isinstance(eos_id, int):
                                eot_ids_list.append(int(eos_id))
                        # Compute hit-rate per batch
                        if eot_ids_list:
                            has_eot = torch.zeros(gen_ids.size(0), dtype=torch.bool, device=gen_ids.device)
                            for eid in eot_ids_list:
                                has_eot |= (gen_ids == eid).any(dim=1)
                            self._val_eot_counter.append(float(has_eot.float().mean().item()))
                        else:
                            self._val_eot_counter.append(0.0)
                        self._val_len_counter.append(float(gen_ids.size(-1)))
                except Exception:
                    pass
                # Process and append to JSON immediately across all devices (write occurs on ref device)
                if write_val_generations and json_path is not None and outputs.get('generated_ids') is not None:
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

                    # Add learning rate and auxiliary loss metrics to log_dict
                    for key, value in gathered_metrics.items():
                        if f"{mode}/lr_" in key:
                            log_dict[key] = value
                        if key == f"{mode}/pattern_loss":
                            log_dict[key] = value
                        if key == f"{mode}/cf_loss":
                            log_dict[key] = value
                        if key.startswith(f"{mode}/grad_norm"):
                            log_dict[key] = value
                    # Also publish unprefixed LR aliases for convenience in W&B dashboards
                    # llm
                    llm_key = f"{mode}/lr_llm"
                    if llm_key in gathered_metrics:
                        log_dict["lr_llm"] = gathered_metrics[llm_key]
                    # adapter (lowercase alias only)
                    adapter_key = f"{mode}/lr_adapter"
                    if adapter_key in gathered_metrics:
                        log_dict["lr_adapter"] = gathered_metrics[adapter_key]
                    # cross_attention if present
                    cross_key = f"{mode}/lr_cross_attention"
                    if cross_key in gathered_metrics:
                        log_dict["lr_cross_attention"] = gathered_metrics[cross_key]
            
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

            if mode == RunMode.VALIDATE:
                logits_batch = outputs.get('pattern_logits')
                targets_batch = outputs.get('pattern_targets')
                if logits_batch is not None and targets_batch is not None:
                    pattern_logits_batches.append(logits_batch)
                    pattern_targets_batches.append(targets_batch)
            
            # Add gradient metrics to tqdm postfix for quick inspection
            if debug_config.get('log_gradient_norms', False) and mode == RunMode.TRAIN:
                grad_metrics = {
                    key: value
                    for key, value in gathered_metrics.items()
                    if key.startswith(f"{mode}/grad_norm/")
                }
                if grad_metrics:
                    for key, value in grad_metrics.items():
                        suffix = key[len(f"{mode}/grad_norm/"):]
                        postfix_key = f"grad_{suffix.replace('/', '_')}"
                        value_float = float(value)
                        postfix_dict[postfix_key] = f"{value_float:.2e}"
                    if log_dict is not None:
                        for key, value in grad_metrics.items():
                            log_dict[key] = float(value)

            if log_dict is not None and self.wandb_wrapper is not None and self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
                # Log batch metrics anchored to current training step to preserve monotonicity
                self.wandb_wrapper.log(log_dict, step=int(self.global_step))

            if mode == RunMode.TRAIN:
                # Increase step and run CF eval only on optimizer steps (end of accumulation cycle)
                if bool(outputs.get('did_step', False)):
                    self.global_step += 1
                    # Optional CF evaluation as early signal
                    self._maybe_run_cf_evaluation()
                    self._handle_train_metric_snapshot(
                        epoch=epoch,
                        batch_idx=batch_idx,
                        batch=batch,
                        ecg_signal=ecg_signal,
                        labels=labels,
                        prompt_input_ids=prompt_input_ids_tensor,
                        prompt_attention_mask=prompt_attention_mask_tensor
                    )
                    # Run validation snapshot check only on optimizer steps
                    self._handle_validation_snapshot(
                        epoch=epoch,
                        snapshot_running=snapshot
                    )
            
            data_iter.set_postfix(postfix_dict)
            
            # Show sample outputs periodically if debugging
            if debug_config.get('show_sample_outputs', False) and mode == RunMode.VALIDATE:
                log_frequency = debug_config.get('log_frequency', 50)
                if batch_idx % log_frequency == 0 and self.config.is_ref_device:
                    self._log_sample_generation(
                        outputs=outputs,
                        labels=labels,
                        epoch=epoch,
                        batch_idx=batch_idx,
                        batch=batch
                    )

            batches_processed += 1
            if max_batches is not None and batches_processed >= max_batches:
                break
        
        # Show epoch summary with debugging info
        if debug_config.get('verbose_loss_logging', False) and self.config.is_ref_device:
            self._log_epoch_summary(mode, epoch, epoch_metrics, total_loss, max(1, batches_processed))

        # Compute unified validation metrics across all samples using HF evaluate + SacreBLEU
        if mode == RunMode.VALIDATE:
            try:
                # Gather predictions/references from all ranks
                world_size = int(getattr(self.config, 'world_size', 1))
                local_pred_refs = (getattr(self, '_val_agg_preds', []), getattr(self, '_val_agg_refs', []))
                gather_list: list[Optional[tuple[list[str], list[str]]]] = [None for _ in range(world_size)]
                if world_size > 1 or local_pred_refs is not None:
                    DistributedUtils.all_gather_object(gather_list, local_pred_refs)
                # Only reference device computes aggregated metrics
                if self.config.is_ref_device:
                    preds_all: list[str] = []
                    refs_all: list[str] = []
                    for item in gather_list:
                        if item is None:
                            continue
                        preds_all.extend(item[0])
                        refs_all.extend(item[1])
                    if preds_all and refs_all:
                        try:
                            from utils.metrics.aggregate_text_metrics import aggregate_text_metrics
                            agg = aggregate_text_metrics(preds_all, refs_all)
                            for k, v in agg.items():
                                epoch_metrics[f"{mode}/{k}"] = float(v)
                        except Exception as _agg_exc:
                            print(f"Warning: failed to compute aggregated validation metrics: {_agg_exc}")
            except Exception as _agg_exc:
                if self.config.is_ref_device:
                    print(f"Warning: failed to compute aggregated validation metrics: {_agg_exc}")

        # === New Block: Log best and worst metrics as HTML to wandb ===
        if (
            mode == RunMode.VALIDATE
            and self.wandb_wrapper is not None
            and self.wandb_wrapper.is_initialized()
            and self.config.is_ref_device
        ):
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
                    # Use the already-computed aggregated epoch metrics for overall values
                    overall_results = {}
                    for mkey in ("rouge1", "rouge2", "rougeL", "bleu1", "bleu4", "meteor"):
                        ek = f"{RunMode.VALIDATE}/{mkey}"
                        if ek in epoch_metrics and isinstance(epoch_metrics[ek], (int, float, np.floating)):
                            overall_results[mkey] = float(epoch_metrics[ek])
                    
                    # Format for logging
                    log_prefix = "val_snapshot" if snapshot else "val"
                    category_log_dict = self.category_metrics_calculator.format_results_for_logging(
                        category_results, overall_results, log_prefix
                    )
                    
                    # Print category statistics
                    stats = self.category_metrics_calculator.get_category_statistics()
                    print(f"\n=== Category Metrics Summary (Epoch {epoch}) ===")
                    for category, category_stats in stats.items():
                        print(f"{category}: {category_stats['n_samples']} samples")
                    
                    try:
                        total_samples = int(sum(s.get('n_samples', 0) for s in stats.values()))
                    except Exception:
                        total_samples = 0
                    print(f"\nOverall Metrics (aggregated across {total_samples} samples):")
                    for metric, score in sorted(overall_results.items()):
                        try:
                            print(f"  {metric}: {float(score):.4f}")
                        except Exception:
                            print(f"  {metric}: {score}")
                    
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
            import wandb
            log_prefix = "val_snapshot" if snapshot else "val"
            self.wandb_wrapper.log({
                f"{log_prefix}/best_metrics_html": wandb.Html(best_html),
                f"{log_prefix}/worst_metrics_html": wandb.Html(worst_html),
                f"{log_prefix}/random_metrics_html": wandb.Html(random_html),
                **category_log_dict
            }, step=int(getattr(self, 'global_step', 0)))
            
            # JSON export is done incrementally during validation
            # Add validation telemetry aggregates
            if hasattr(self, '_val_eot_counter') and hasattr(self, '_val_len_counter'):
                try:
                    import numpy as _np
                    eot_values = _np.array(self._val_eot_counter, dtype=float) if self._val_eot_counter else _np.array([0.0])
                    len_values = _np.array(self._val_len_counter, dtype=float) if self._val_len_counter else _np.array([0.0])
                    log_prefix = "val_snapshot" if snapshot else "val"
                    telemetry_payload = {
                        f"{log_prefix}/generated_len_mean": float(len_values.mean()) if len(len_values) else 0.0,
                        f"{log_prefix}/generated_len_p95": float(_np.percentile(len_values, 95)) if len(len_values) else 0.0,
                        f"{log_prefix}/eot_reached_rate": float(eot_values.mean()) if len(eot_values) else 0.0,
                    }
                    self.wandb_wrapper.log(telemetry_payload, step=int(getattr(self, 'global_step', 0)))
                except Exception:
                    pass
            
            # Plot ECG waveforms with Q&A annotations if configured
            if not snapshot:
                json_path = self._get_val_generation_json_path(epoch)
                if os.path.exists(json_path):
                    self._plot_validation_ecgs(epoch, json_path)
        # === End new block ===
                
        # Normalize only metrics that were accumulated as sums over batches.
        # Aggregated text metrics (computed once over all samples) must NOT be divided again.
        denominator = max(1, batches_processed)
        skip_norm_prefixes = (
            f"{RunMode.VALIDATE}/rouge",  # rouge1, rouge2, rougeL
            f"{RunMode.VALIDATE}/bleu",   # bleu1, bleu4
            f"{RunMode.VALIDATE}/meteor",
            f"{RunMode.VALIDATE}/bertscore",  # bertscore_* if present
        )
        keys = list(epoch_metrics.keys())
        for k in keys:
            try:
                if any(k.startswith(pref) for pref in skip_norm_prefixes):
                    continue
                if k.endswith("/batches_processed"):
                    continue
                epoch_metrics[k] /= denominator
            except Exception:
                # Leave non-numeric entries untouched
                pass

        epoch_metrics[f"{mode}/batches_processed"] = float(batches_processed)

        if mode == RunMode.VALIDATE:
            world_size = getattr(self.config, "world_size", 1)
            local_payload = None
            if pattern_logits_batches and pattern_targets_batches:
                local_payload = (
                    torch.cat(pattern_logits_batches, dim=0),
                    torch.cat(pattern_targets_batches, dim=0),
                )
            gather_list: List[Optional[Tuple[torch.Tensor, torch.Tensor]]] = [None for _ in range(world_size)]
            if world_size > 1 or local_payload is not None:
                DistributedUtils.all_gather_object(gather_list, local_payload)
            if self.config.is_ref_device:
                combined_logits: list[torch.Tensor] = []
                combined_targets: list[torch.Tensor] = []
                for item in gather_list:
                    if item is None:
                        continue
                    combined_logits.append(item[0])
                    combined_targets.append(item[1])
                if combined_logits and combined_targets:
                    all_logits = torch.cat(combined_logits, dim=0)
                    all_targets = torch.cat(combined_targets, dim=0)
                    probs = torch.sigmoid(all_logits)
                    # Ensure CPU for numpy conversion to avoid device errors
                    probs_np = probs.detach().cpu().numpy()
                    targets_np = all_targets.detach().cpu().numpy()
                    try:
                        macro_auc = roc_auc_score(targets_np, probs_np, average='macro')
                    except ValueError:
                        macro_auc = float("nan")
                    try:
                        micro_auc = roc_auc_score(targets_np, probs_np, average='micro')
                    except ValueError:
                        micro_auc = float("nan")
                    epoch_metrics[f"{mode}/pattern_auc_macro"] = float(macro_auc)
                    epoch_metrics[f"{mode}/pattern_auc_micro"] = float(micro_auc)
                    positive_counts = targets_np.sum(axis=0)
                    for idx, label in enumerate(ECG_PATTERNS):
                        if idx >= targets_np.shape[1]:
                            break
                        if np.unique(targets_np[:, idx]).size < 2:
                            continue
                        try:
                            label_auc = roc_auc_score(targets_np[:, idx], probs_np[:, idx])
                        except ValueError:
                            continue
                        label_key = (
                            label.lower()
                            .replace(" ", "_")
                            .replace("/", "_")
                            .replace("-", "_")
                            .replace("(", "")
                            .replace(")", "")
                            .replace(",", "")
                            .replace("'", "")
                        )
                        epoch_metrics[f"{mode}/pattern_auc/{label_key}"] = float(label_auc)
                        epoch_metrics[f"{mode}/pattern_support/{label_key}"] = float(positive_counts[idx])

                    # Optional: log logits/targets histograms to W&B for inspection
                    if self.wandb_wrapper is not None and self.wandb_wrapper.is_initialized():
                        try:
                            import wandb
                            log_prefix = "val_snapshot" if snapshot else "val"
                            # Flatten to 1D for histogram
                            logits_np = all_logits.detach().cpu().flatten().numpy()
                            targets_hist_np = all_targets.detach().cpu().flatten().numpy()
                            step_for_log = int(getattr(self, 'global_step', 0))
                            self.wandb_wrapper.log({
                                f"{log_prefix}/pattern_logits_hist": wandb.Histogram(logits_np),
                                f"{log_prefix}/pattern_targets_hist": wandb.Histogram(targets_hist_np)
                            }, step=step_for_log)
                        except Exception as _wandb_exc:
                            # Non-fatal; skip histogram logging if unavailable
                            pass

        # Create validation metric plots on the reference device for both full epochs and snapshots
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
                current_step = int(getattr(self, "global_step", 0))
                step_suffix = ""
                if snapshot and current_step > 0:
                    step_suffix = f"_step_{current_step:06d}"
                elif current_step > 0:
                    step_suffix = f"_step_{current_step:06d}"
                if snapshot:
                    snapshot_tag = "snapshot"
                    if not step_suffix:
                        step_suffix = "_step_unknown"
                    plot_filename = f"{snapshot_tag}_epoch_{epoch:03d}{step_suffix}_metrics.png"
                else:
                    plot_filename = f"epoch_{epoch:03d}{step_suffix}_metrics.png"
                plot_path = os.path.join(plot_dir, plot_filename)

                labels = list(metric_values.keys())
                scores = [metric_values[label] for label in labels]

                fig, ax = plt.subplots(figsize=(6, 4))
                bar_container = ax.bar(labels, scores, color="#4C72B0")
                upper_ylim = min(1.1, max(scores) + 0.05)
                upper_ylim = max(upper_ylim, 0.2)
                ax.set_ylim(0.0, upper_ylim)
                ax.set_ylabel("Score")
                title_prefix = "Validation Snapshot" if snapshot else "Validation Metrics"
                title_suffix = f" (step {current_step:,})" if current_step > 0 else ""
                ax.set_title(f"{title_prefix} - Epoch {epoch}{title_suffix}")
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
                        log_prefix = "val_snapshot" if snapshot else "val"
                        if snapshot and current_step > 0:
                            plot_key = f"{log_prefix}/metrics_plot_step_{current_step}"
                        else:
                            plot_key = f"{log_prefix}/metrics_plot_epoch_{epoch}"
                        log_payload = {
                            plot_key: wandb.Image(plot_path)
                        }
                        if current_step > 0:
                            log_payload[f"{plot_key}_step"] = float(current_step)
                        self.wandb_wrapper.log(log_payload, step=int(current_step) if current_step > 0 else int(getattr(self, 'global_step', 0)))
                    except Exception as exc:
                        print(f"Warning: Failed to log validation metrics plot to Weights & Biases: {exc}")

                scope = "snapshot" if snapshot else "validation"
                if current_step > 0:
                    print(f"Saved {scope} metric plot (step {current_step}): {plot_path}")
                else:
                    print(f"Saved {scope} metric plot: {plot_path}")
            elif metric_values:
                print("matplotlib is unavailable; skipping validation metric plot generation.")

        # Return the epoch metrics
        return epoch_metrics

    def _set_sampler_epoch(
        self,
        dataloader: DataLoader,
        mode: RunMode,
        epoch: int,
        snapshot_step: int | None,
        snapshot: bool
    ) -> None:
        """Reseed distributed samplers so validation snapshots can shuffle."""
        sampler = getattr(dataloader, "sampler", None)
        if sampler is None or not hasattr(sampler, "set_epoch"):
            return

        if mode == RunMode.TRAIN:
            sampler.set_epoch(int(epoch + (getattr(self.config, "seed", 0) or 0)))
            return

        if not getattr(self.config, "validation_shuffle", False):
            return

        seed_offset = int(getattr(self.config, "seed", 0) or 0)
        if snapshot and snapshot_step is not None:
            sampler.set_epoch(seed_offset + int(snapshot_step))
        else:
            sampler.set_epoch(seed_offset + int(epoch * 9973))

    def _handle_train_metric_snapshot(
        self,
        epoch: int,
        batch_idx: int,
        batch: dict[str, Any],
        ecg_signal: torch.Tensor,
        labels: torch.Tensor,
        prompt_input_ids: torch.Tensor | None,
        prompt_attention_mask: torch.Tensor | None
    ) -> None:
        """Compute and log mid-epoch training metrics when requested."""
        interval = getattr(self.config, "train_metric_interval", None)
        if interval is None or interval <= 0:
            return

        batches_to_collect = max(1, getattr(self.config, "train_metric_batches", 1))
        # Trigger snapshots on the very first step and then every `interval` steps
        eff_step = int(self.global_step)
        if eff_step % interval == 0 or eff_step == 1:
            self._train_metric_remaining = batches_to_collect
            self._train_metric_accumulator = {}
            self._train_metric_batches_collected = 0

        if self._train_metric_remaining <= 0:
            return

        if self.config.is_ref_device:
            try:
                metrics = self._compute_train_metrics_for_batch(
                    batch=batch,
                    ecg_signal=ecg_signal,
                    labels=labels,
                    prompt_input_ids=prompt_input_ids,
                    prompt_attention_mask=prompt_attention_mask,
                    batch_idx=batch_idx
                )
            except Exception as exc:
                print(f"⚠️ Train snapshot metrics failed at step {self.global_step}: {exc}")
                traceback.print_exc()
                metrics = {}

            for key, value in metrics.items():
                if isinstance(value, (int, float, np.floating)):
                    self._train_metric_accumulator[key] = self._train_metric_accumulator.get(key, 0.0) + float(value)
            self._train_metric_batches_collected += 1

        self._train_metric_remaining -= 1

        if self._train_metric_remaining == 0:
            if self.config.is_ref_device and self._train_metric_batches_collected > 0:
                averaged_metrics = {
                    f"train_snapshot/{metric}": total / self._train_metric_batches_collected
                    for metric, total in self._train_metric_accumulator.items()
                }
                averaged_metrics["train_snapshot/batches"] = float(self._train_metric_batches_collected)
                averaged_metrics["train_snapshot/global_step"] = float(self.global_step)
                averaged_metrics["train_snapshot/epoch"] = float(epoch)
                if self.wandb_wrapper is not None and self.wandb_wrapper.is_initialized():
                    # Log only current train snapshot metrics (commit=False to avoid step collisions)
                    payload = dict(averaged_metrics)
                    self.wandb_wrapper.log(payload, step=int(self.global_step))
            DistributedUtils.sync_process_group(
                world_size=self.config.world_size,
                device_ids=self.config.device
            )

    def _compute_train_metrics_for_batch(
        self,
        batch: dict[str, Any],
        ecg_signal: torch.Tensor,
        labels: torch.Tensor,
        prompt_input_ids: torch.Tensor | None,
        prompt_attention_mask: torch.Tensor | None,
        batch_idx: int
    ) -> dict[str, float]:
        """Generate outputs for a training batch and compute requested metrics."""
        if self.train_dataloader is None:
            return {}

        model_for_generation = self.model.module if hasattr(self.model, "module") else self.model
        was_training = model_for_generation.training

        original_category_calculator = self.category_metrics_calculator
        self.category_metrics_calculator = None

        try:
            model_for_generation.eval()
            with torch.no_grad():
                # Build generation kwargs matching inference script
                generation_kwargs = self._build_generation_kwargs()
                
                generated_ids = model_for_generation.generate_report_with_question(
                    ecg_signal,
                    prompt_input_ids=prompt_input_ids,
                    prompt_attention_mask=prompt_attention_mask,
                    max_token_length=self.config.max_token_length,
                    **generation_kwargs,
                )

            outputs = {
                "generated_ids": generated_ids,
                "loss": torch.tensor(0.0, device=self.config.device)
            }

            dummy_best: dict[str, list[dict[str, Union[float, list[str]]]]] = {}
            dummy_worst: dict[str, list[dict[str, Union[float, list[str]]]]] = {}
            dummy_random: dict[str, list[dict[str, Union[float, list[str]]]]] = {}

            metrics = self._compute_metrics(
                outputs,
                labels,
                self.train_dataloader,
                dummy_best,
                dummy_worst,
                dummy_random,
                random_batch=False,
                batch=batch,
                batch_idx=batch_idx
            )

            filtered_metrics = {
                key: float(value)
                for key, value in metrics.items()
                if isinstance(value, (int, float, np.floating))
            }
            return filtered_metrics
        finally:
            self.category_metrics_calculator = original_category_calculator
            model_for_generation.train(was_training)

    def _handle_validation_snapshot(
        self,
        epoch: int,
        snapshot_running: bool
    ) -> None:
        """Run validation snapshots mid-epoch when configured.
        
        Only called on optimizer steps (when did_step=True). Uses multiple guards
        to ensure validation only runs at the configured interval.
        """
        if snapshot_running:
            return

        interval = getattr(self.config, "validation_step_interval", None)
        if interval is None or interval <= 0:
            return
        
        # Cast to int to avoid any floating point issues
        interval = int(interval)
        current_step = int(self.global_step)
        
        # Only run at exact multiples of interval (not step 0)
        if current_step == 0 or current_step % interval != 0:
            return
        
        # CRITICAL: Prevent running if we haven't advanced at least `interval` steps
        # since the last validation. This guards against any edge cases where
        # validation might be triggered on consecutive steps.
        last_step = int(self._last_validation_snapshot_step)
        if last_step >= 0 and (current_step - last_step) < interval:
            if self.config.is_ref_device:
                print(f"[Validation] Skipping step {current_step}: only {current_step - last_step} steps since last validation at {last_step}")
            return
        
        # Mark this step as processed BEFORE running to prevent re-entry
        self._last_validation_snapshot_step = current_step

        snapshot_batches_cfg = getattr(self.config, "validation_snapshot_batches", 0)
        if snapshot_batches_cfg and snapshot_batches_cfg > 0:
            snapshot_batches = snapshot_batches_cfg
        else:
            snapshot_batches = None

        metrics = self._run_epoch(
            RunMode.VALIDATE,
            epoch,
            max_batches=snapshot_batches,
            snapshot_step=self.global_step,
            snapshot=True
        )

        checkpoint_path = None
        if self.config.is_ref_device:
            loss_key = f"{RunMode.VALIDATE}/loss"
            val_loss_raw = metrics.get(loss_key)
            if isinstance(val_loss_raw, (int, float, np.floating)):
                snapshot_loss = float(val_loss_raw)
                is_best = False
                if snapshot_loss < self.best_val_loss:
                    self.best_val_loss = snapshot_loss
                    is_best = True
                self._save_model(
                    epoch=epoch,
                    loss=snapshot_loss,
                    is_best=is_best,
                    step=self.global_step
                )
                # Get checkpoint path for inference-style validation
                checkpoint_path = os.path.join(
                    self.config.output_dir,
                    f'checkpoint_step_{self.global_step}.pt'
                )


        if self.config.is_ref_device:
            log_payload: dict[str, float] = {}
            for key, value in metrics.items():
                if not isinstance(value, (int, float, np.floating)):
                    continue
                if key.startswith(f"{RunMode.VALIDATE}/"):
                    metric_name = key.split("/", 1)[1]
                    log_payload[f"val_snapshot/{metric_name}"] = float(value)
            log_payload["val_snapshot/global_step"] = float(self.global_step)
            log_payload["val_snapshot/epoch"] = float(epoch)
            log_payload["val_snapshot/best_loss"] = float(self.best_val_loss)
            if snapshot_batches is not None:
                log_payload["val_snapshot/batches_requested"] = float(snapshot_batches)
            if self.wandb_wrapper is not None and self.wandb_wrapper.is_initialized():
                # Anchor validation snapshot logs to current global step (commit=False to avoid step collisions)
                self.wandb_wrapper.log(log_payload, step=int(self.global_step))

        DistributedUtils.sync_process_group(
            world_size=self.config.world_size,
            device_ids=self.config.device
        )
    
    def _decode_generation(
        self,
        tokenizer,
        generated_ids: torch.Tensor,
        label_ids: torch.Tensor,
        batch: Optional[dict] = None,
        sample_idx: int = 0,
    ) -> Tuple[str, str]:
        """
        Unified decoding function for all paths (validation, inference, metrics, JSON).

        Returns (prediction_text, reference_text) with proper ECG token offset handling.

        IMPORTANT: Uses raw 'answer_text' from batch when available for reference,
        matching the inference script behavior. This ensures metrics are computed
        on the original ground truth text, not tokenized/truncated labels.

        Args:
            tokenizer: The tokenizer instance
            generated_ids: Generated token ids for a single sample (1D tensor)
            label_ids: Label token ids for a single sample (1D tensor)
            batch: Optional batch dictionary containing prompt_input_ids and answer_text
            sample_idx: Index of the sample in the batch (for extracting prompt_input_ids)

        Returns:
            Tuple of (sanitized_prediction, sanitized_reference)
        """
        if generated_ids is None or generated_ids.numel() == 0:
            return "", ""

        gen_tensor = generated_ids.detach().cpu() if generated_ids.is_cuda else generated_ids.detach()

        # Decode prediction: MedGemma uses inputs_embeds, so generated contains only new tokens
        gen_ids = gen_tensor.tolist()
        raw_prediction = " ".join(tokenizer.decode(gen_ids, skip_special_tokens=True).strip().split())

        # Get reference: PREFER raw answer_text from batch (matches inference script)
        # This avoids tokenization/truncation artifacts in ground truth
        raw_reference = ""
        if batch is not None:
            try:
                answer_texts = batch.get('answer_text', None)
                if answer_texts is not None:
                    if isinstance(answer_texts, (list, tuple)) and sample_idx < len(answer_texts):
                        raw_reference = str(answer_texts[sample_idx]).strip()
                    elif isinstance(answer_texts, str):
                        raw_reference = answer_texts.strip()
            except Exception:
                pass

        # Fallback to decoding labels if answer_text not available
        if not raw_reference:
            label_tensor = label_ids.detach().cpu() if label_ids.is_cuda else label_ids.detach()
            ref_ids = label_tensor[label_tensor != -100].tolist()
            raw_reference = " ".join(tokenizer.decode(ref_ids, skip_special_tokens=True).strip().split())

        # Sanitize both outputs
        prediction = self._sanitize_chat_text(raw_prediction)
        reference = self._sanitize_chat_text(raw_reference)

        return prediction, reference
    
    def _extract_assistant_text(self, tokenizer, generated_ids: torch.Tensor, label_ids: torch.Tensor) -> str:
        """
        DEPRECATED: Use _decode_generation instead for proper ECG token handling.
        Kept for backwards compatibility but now delegates to _decode_generation.
        """
        prediction, _ = self._decode_generation(tokenizer, generated_ids, label_ids)
        return prediction

    @staticmethod
    def _sanitize_chat_text(text: str, max_length: int = 512) -> str:
        """Trim special chat tokens, EOT artifacts, tidy semicolons, and optionally truncate.

        Handles broken variants of the end-of-turn token (e.g. "<|eot_", "|eot_id|>")
        that sometimes appear when the tokenizer doesn't mark them as special.
        Also handles MedGemma's <end_of_turn> and <start_of_turn> tokens.
        Also normalizes whitespace and excessive/leading semicolons.
        """
        if not text:
            return ""

        cleaned = str(text)

        # 1) Robustly trim at end-of-turn artifacts (handle broken variants and MedGemma)
        try:
            # Match MedGemma's <end_of_turn> or LLaMA's <|eot_id|> and variants
            eot_match = re.search(
                r"<end_of_turn>|<\|\s*eot[^>]*>?|\|\s*eot[^\s]*|<start_of_turn>model|<start_of_turn>user",
                cleaned,
                flags=re.IGNORECASE
            )
            if eot_match is not None:
                cleaned = cleaned[: eot_match.start()]
        except Exception:
            # Fallback: common delimiters
            for delimiter in ("<end_of_turn>", "<|eot_id|>", "|eot_id|>", "<start_of_turn>"):
                if delimiter in cleaned:
                    cleaned = cleaned.split(delimiter, 1)[0]

        # 2) Strip remaining special header markers (LLaMA and MedGemma)
        special_tokens = (
            "<|start_header_id|>",
            "<|end_header_id|>",
            "<|assistant|>",
            "<|user|>",
            "<|system|>",
            "<|start_of_turn|>",
            "<|end_of_turn|>",
            "<start_of_turn>",
            "<end_of_turn>",
            "<start_of_image>",
            "<end_of_image>",
            "model",  # MedGemma turn marker
        )
        for token in special_tokens:
            cleaned = cleaned.replace(token, "")
        # Also strip residual textual header crumbs that may appear without brackets
        cleaned = re.sub(r"\b(start_header_id|end_header_id)\b\|?", "", cleaned, flags=re.IGNORECASE)

        # 3) Strip common HTML fragments the model sometimes emits
        cleaned = re.sub(r"<\s*br\s*/?>", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"</?p[^>]*>", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"</?div[^>]*>", " ", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"</?span[^>]*>", " ", cleaned, flags=re.IGNORECASE)
        # Remove any remaining generic tags
        cleaned = re.sub(r"<[^>]+>", " ", cleaned)

        cleaned = cleaned.replace("\u2581", " ")

        # 4) Normalize whitespace early so punctuation cleanup behaves well
        cleaned = cleaned.strip()
        if cleaned:
            cleaned = " ".join(cleaned.split())

        # 5) Tidy semicolons:
        #    - collapse consecutive semicolons
        #    - ensure a single space after semicolons
        #    - remove leading/trailing semicolons
        if cleaned:
            cleaned = re.sub(r"\s*;\s*", "; ", cleaned)
            cleaned = re.sub(r"(?:;\s*){2,}", "; ", cleaned)
            cleaned = re.sub(r"^;\s*", "", cleaned)
            cleaned = re.sub(r";\s*\.", ".", cleaned)
            cleaned = re.sub(r";\s*$", "", cleaned)

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
        """
        Extract the human-readable question/prompt text from a batch.
        
        Priority order:
        1. prompt_text - The actual question text (preferred)
        2. prompt / question - Alternative column names for question
        3. rendered_prompt - Full chat template (fallback, less clean)
        4. prompt_input_ids - Decode from token IDs (last resort)
        """
        # Check prompt_text first - this is the actual question
        for key in ('prompt_text', 'prompt', 'question'):
            value = batch.get(key)
            if isinstance(value, (list, tuple)) and index < len(value):
                candidate = value[index]
                if candidate is not None:
                    cleaned = str(candidate).strip()
                    if cleaned:
                        return cleaned
            elif value is not None and not isinstance(value, (list, tuple)) and index == 0:
                cleaned = str(value).strip()
                if cleaned:
                    return cleaned

        # Fallback to rendered_prompt (contains full chat template)
        rendered = batch.get('rendered_prompt')
        if isinstance(rendered, (list, tuple)) and index < len(rendered):
            candidate = rendered[index]
        elif rendered is not None and not isinstance(rendered, (list, tuple)) and index == 0:
            candidate = rendered
        else:
            candidate = None
        
        if candidate is not None:
            cleaned = self._sanitize_chat_text(str(candidate))
            if cleaned:
                return cleaned

        # Last resort: decode from prompt_input_ids
        if 'prompt_input_ids' in batch:
            prompt_ids = batch['prompt_input_ids'][index]
            prompt_mask = None
            if 'prompt_attention_mask' in batch:
                prompt_mask = batch['prompt_attention_mask'][index]
            decoded = self._decode_prompt_from_ids(tokenizer, prompt_ids, prompt_mask)
            if decoded:
                return decoded

        return ""

    def _print_first_batch_details(
        self,
        outputs: dict[str, Any],
        batch: dict[str, Any],
        labels: torch.Tensor,
        pattern_targets: torch.Tensor | None,
    ) -> bool:
        """Pretty-print prompt, prediction, and multilabel targets for the first batch."""
        if 'generated_ids' not in outputs:
            return False

        generated_ids = outputs['generated_ids']
        if not isinstance(generated_ids, torch.Tensor) or generated_ids.size(0) == 0:
            return False

        model = self.model.module if hasattr(self.model, 'module') else self.model
        decoder = getattr(model, 'decoder', None)
        tokenizer = getattr(decoder, 'tokenizer', None) if decoder is not None else None
        if tokenizer is None:
            print("First-batch debug skipped: decoder tokenizer unavailable.")
            return True

        prompt_text = self._extract_prompt_text(tokenizer, batch, 0)

        first_generated = generated_ids[0].detach().cpu()
        first_label = labels[0].detach().cpu() if labels is not None and labels.size(0) > 0 else None
        predicted_text = ""
        try:
            if first_label is not None:
                # Use unified decoding function for consistent results
                predicted_text, _ = self._decode_generation(
                    tokenizer,
                    first_generated,
                    first_label,
                    batch=batch,
                    sample_idx=0
                )
            else:
                predicted_text = self._sanitize_chat_text(
                    tokenizer.decode(first_generated.tolist(), skip_special_tokens=True)
                )
        except Exception as exc:
            predicted_text = f"<decode error: {exc}>"

        positive_targets: list[str] = []
        if pattern_targets is not None and pattern_targets.size(0) > 0:
            first_targets = pattern_targets[0].detach().float().cpu()
            for idx, score in enumerate(first_targets.tolist()):
                if idx >= len(ECG_PATTERNS):
                    break
                if float(score) >= 0.5:
                    positive_targets.append(f"{ECG_PATTERNS[idx]} ({score:.2f})")

        print("\n" + "=" * 80)
        print("First validation batch snapshot")
        print("-" * 80)
        print(f"Prompt: {prompt_text if prompt_text else '<empty>'}")
        print("-" * 80)
        print(f"Predicted report: {predicted_text if predicted_text else '<empty>'}")
        print("-" * 80)
        if positive_targets:
            print("Positive ECG patterns: " + ", ".join(positive_targets))
        else:
            print("Positive ECG patterns: none")
        print("=" * 80 + "\n")

        if getattr(self.config, 'debug_prompt_dump', False):
            try:
                full_input = batch.get('input_ids')
                attn_mask = batch.get('attention_mask')
                prompt_ids = batch.get('prompt_input_ids')
                prompt_mask = batch.get('prompt_attention_mask')
                label_mask = labels[0] if labels is not None and labels.size(0) > 0 else None

                def _decode(ids: torch.Tensor, mask: torch.Tensor | None = None, skip_special: bool = False) -> str:
                    ids_cpu = ids.detach().cpu()
                    if mask is not None:
                        ids_cpu = ids_cpu[mask.detach().cpu().bool()]
                    return tokenizer.decode(ids_cpu.tolist(), skip_special_tokens=skip_special)

                if prompt_ids is not None:
                    print("Prompt tokens (decoded):")
                    print(_decode(prompt_ids[0], prompt_mask[0] if prompt_mask is not None else None, skip_special=False))
                if full_input is not None:
                    print("Full input (decoded):")
                    print(_decode(full_input[0], attn_mask[0] if attn_mask is not None else None, skip_special=False))
                if label_mask is not None and full_input is not None:
                    kept = label_mask != -100
                    print("Non-masked label tokens (decoded):")
                    print(_decode(full_input[0][kept], None, skip_special=True))
            except Exception as exc:
                print(f"debug_prompt_dump failed: {exc}")

        return True

    def _debug_dump_prompt_batch(self, batch: dict[str, Any], labels: torch.Tensor, prefix: str = "") -> None:
        """Decode prompt/input/label tokens for a single batch example."""
        try:
            model = self.model.module if hasattr(self.model, 'module') else self.model
            decoder = getattr(model, 'decoder', None)
            tokenizer = getattr(decoder, 'tokenizer', None) if decoder is not None else None
            if tokenizer is None:
                print("debug_prompt_dump: tokenizer unavailable.")
                return

            prompt_ids = batch.get('prompt_input_ids')
            prompt_mask = batch.get('prompt_attention_mask')
            input_ids = batch.get('input_ids')
            attn_mask = batch.get('attention_mask')
            label_mask = labels[0].detach().cpu() if labels is not None and labels.size(0) > 0 else None

            def _decode(ids: torch.Tensor, mask: torch.Tensor | None = None, skip_special: bool = False) -> str:
                ids_cpu = ids.detach().cpu()
                if mask is not None:
                    ids_cpu = ids_cpu[mask.detach().cpu().bool()]
                return tokenizer.decode(ids_cpu.tolist(), skip_special_tokens=skip_special)

            print("\n[DEBUG PROMPT DUMP]", prefix)
            if prompt_ids is not None:
                print("Prompt tokens (decoded):")
                print(_decode(prompt_ids[0], prompt_mask[0] if prompt_mask is not None else None, skip_special=False))
            if input_ids is not None:
                print("Full input (decoded):")
                print(_decode(input_ids[0], attn_mask[0] if attn_mask is not None else None, skip_special=False))
            if label_mask is not None and input_ids is not None:
                kept = label_mask != -100
                ids_cpu = input_ids[0].detach().cpu()
                print("Non-masked label tokens (decoded):")
                print(_decode(ids_cpu[kept], None, skip_special=True))
            print("[END DEBUG PROMPT DUMP]\n")
        except Exception as exc:
            print(f"debug_prompt_dump failed: {exc}")

    def _log_sample_generation(
        self,
        outputs: dict,
        labels: torch.Tensor,
        epoch: int,
        batch_idx: int,
        batch: Optional[dict] = None
    ):
        """Log sample generations for debugging using unified decoding."""
        try:
            model = self.model.module if hasattr(self.model, 'module') else self.model
            tokenizer = model.decoder.tokenizer

            generated_ids = outputs['generated_ids'][0] if 'generated_ids' in outputs else None
            label_ids = labels[0]

            if generated_ids is not None:
                # Use unified decoding function for consistent results
                generated_text, label_text = self._decode_generation(
                    tokenizer,
                    generated_ids.cpu(),
                    label_ids.cpu(),
                    batch=batch,
                    sample_idx=0
                )

                print("\n" + "="*60)
                print(f"Sample Generation (Epoch {epoch}, Batch {batch_idx})")
                print("-"*60)
                print(f"Generated: {generated_text[:200]}..." if len(generated_text) > 200 else f"Generated: {generated_text}")
                print("-"*60)
                print(f"Expected:  {label_text[:200]}..." if len(label_text) > 200 else f"Expected:  {label_text}")
                print("="*60 + "\n")
        except Exception as e:
            print(f"Error logging sample: {e}")
    
    # ------------------------------------------------------------------
    # CF evaluation (Choice-Free) – early signal during training
    # ------------------------------------------------------------------
    def _maybe_run_cf_evaluation(self) -> None:
        """
        Run CF evaluation periodically during training.

        DDP-safe behavior:
          - Single-GPU (world_size == 1): run CF eval as usual.
          - Multi-GPU (world_size > 1): run CF eval on every rank by default
            to keep collective ordering aligned; only skip if explicitly
            disabled via config.cf_eval_ddp_mode = "off".
        """
        if getattr(self, "cf_evaluator", None) is None:
            return

        try:
            interval = int(getattr(self.config, "cf_eval_interval", 0) or 0)
        except Exception:
            interval = 0
        if interval <= 0:
            return

        step = int(getattr(self, "global_step", 0))
        if step == 0 or (step % interval) != 0:
            return

        world_size = int(getattr(self.config, "world_size", 1) or 1)
        ddp_mode = str(getattr(self.config, "cf_eval_ddp_mode", "all_ranks")).lower()

        # Multi-GPU: run on all ranks by default to avoid desync; allow explicit opt-out.
        if world_size > 1:
            if ddp_mode in ("off", "disable", "disabled"):
                return
            self._run_cf_evaluation(step)
            return

        # Single-GPU: plain behavior.
        self._run_cf_evaluation(step)

    @torch.no_grad()
    def _run_cf_evaluation(self, step: int) -> None:
        """Run CF evaluation and log metrics to wandb/console."""
        if getattr(self, "cf_evaluator", None) is None:
            return

        was_training = self.model.training
        self.model.eval()

        # Resolve tokenizer from decoder
        base_model = self.model.module if hasattr(self.model, "module") else self.model
        decoder = getattr(base_model, "decoder", None)
        tokenizer = getattr(decoder, "tokenizer", None) if decoder is not None else None
        if tokenizer is None:
            if self.config.is_ref_device:
                print("[CF Eval] Skipped: decoder tokenizer unavailable")
            if was_training:
                self.model.train(True)
            return

        try:
            max_samples = int(getattr(self.config, "cf_eval_samples", 1000) or 1000)
            log_details = bool(getattr(self.config, "cf_eval_log_details", False))

            if log_details and hasattr(self.cf_evaluator, 'evaluate_with_details'):
                metrics, details = self.cf_evaluator.evaluate_with_details(
                    model=base_model,
                    tokenizer=tokenizer,
                    # For detailed logging, limit to cf_eval_log_k items to keep payload small
                    max_samples=max_samples,
                    max_details=int(getattr(self.config, "cf_eval_log_k", 10) or 10),
                )
            else:
                metrics = self.cf_evaluator.evaluate(
                    model=base_model,
                    tokenizer=tokenizer,
                    max_samples=max_samples,
                )
                details = []
        except Exception as exc:
            if self.config.is_ref_device:
                print(f"[CF Eval] Error during evaluation at step {step}: {exc}")
            if was_training:
                self.model.train(True)
            return

        # Log to wandb (reference device only)
        if (
            self.wandb_wrapper is not None
            and self.wandb_wrapper.is_initialized()
            and self.config.is_ref_device
        ):
            payload = {**metrics, "cf_step": float(step)}
            # Attach compact JSON for a few examples if requested
            if details:
                try:
                    payload["cf_examples_json"] = json.dumps(details, ensure_ascii=False)
                except Exception:
                    pass
            try:
                self.wandb_wrapper.log(payload, step=int(self.global_step))
            except Exception:
                self.wandb_wrapper.log(payload)

        # Print to console (reference device only)
        if self.config.is_ref_device:
            print(f"\n--- CF Evaluation (Step {step}) ---")
            for k, v in sorted(metrics.items()):
                try:
                    print(f"  {k}: {float(v):.3f}")
                except Exception:
                    print(f"  {k}: {v}")
            print("---\n")
            if details:
                print("CF Examples (first few):")
                preview = details[: min(3, len(details))]
                for ex in preview:
                    print(f"  [{ex.get('category','')}] Q: {ex.get('question','')}")
                    print(f"    pred: idx={ex.get('pred_index')}, ans={ex.get('pred_answer')}")
                    print(f"    gt:   idx={ex.get('ground_truth_indices')}")
                print()

        # Optionally dump full per-question argmax predictions to a JSON file
        try:
            if (
                self.config.is_ref_device
                and bool(getattr(self.config, "cf_eval_write_predictions", False))
                and hasattr(self.cf_evaluator, "evaluate_with_details")
            ):
                max_records = int(getattr(self.config, "cf_eval_predictions_max", 0) or 0)
                # If max_records <= 0, fall back to the lighter cf_eval_samples
                pred_eval_samples = max_records if max_records > 0 else max_samples
                pred_detail_cap = (
                    max_records if max_records > 0
                    else int(getattr(self.config, "cf_eval_log_k", 10) or 10)
                )
                metrics_full, details_full = self.cf_evaluator.evaluate_with_details(
                    model=base_model,
                    tokenizer=tokenizer,
                    max_samples=pred_eval_samples,
                    max_details=pred_detail_cap,
                )
                out_dir = self._get_val_generations_dir()
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, f"cf_predictions_step_{int(step)}.json")
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(details_full, f, ensure_ascii=False, indent=2)
                print(f"[CF Eval] Wrote per-question argmax predictions: {out_path} ({len(details_full)} records)")
        except Exception as dump_exc:
            if self.config.is_ref_device:
                print(f"[CF Eval] Failed to write prediction JSON: {dump_exc}")

        if was_training:
            self.model.train(True)

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

    def _collect_grad_norms(self) -> Dict[str, float]:
        """Compute gradient norms for key sub-modules after backward pass."""
        base_model = self.model.module if hasattr(self.model, 'module') else self.model
        decoder = getattr(base_model, 'decoder', None)
        grad_norms: Dict[str, float] = {}

        def _norm(params: List[torch.nn.Parameter]) -> float:
            total = 0.0
            for param in params:
                if param is None or param.grad is None:
                    continue
                grad = param.grad.detach()
                total += grad.float().pow(2).sum().item()
            return math.sqrt(total) if total > 0.0 else 0.0

        # LLM / decoder head
        llm_module = None
        if decoder is not None:
            llm_module = getattr(decoder, 'llm_model', None)
            if llm_module is None:
                llm_module = getattr(decoder, 'llm', None)
        if llm_module is not None:
            llm_params = [p for p in llm_module.parameters() if p.requires_grad]
            grad_norms['llm'] = _norm(llm_params)
            lora_params = [param for name, param in llm_module.named_parameters()
                           if 'lora_' in name and param.requires_grad]
            if lora_params:
                grad_norms['lora'] = _norm(lora_params)

        # Q-Former / bridge
        bridge = getattr(decoder, 'bridge', None) if decoder is not None else None
        if bridge is not None:
            bridge_params = [p for p in bridge.parameters() if p.requires_grad]
            if bridge_params:
                grad_norms['qformer'] = _norm(bridge_params)

        # Adapter (if separate from bridge)
        adapter = getattr(decoder, 'adapter', None) if decoder is not None else None
        if adapter is not None:
            adapter_params = [p for p in adapter.parameters() if p.requires_grad]
            if adapter_params:
                grad_norms['adapter'] = _norm(adapter_params)

        # Cross-attention blocks (if present)
        cross_layers = getattr(decoder, 'cross_attention_layers', None) if decoder is not None else None
        if cross_layers:
            cross_params: List[torch.nn.Parameter] = []
            for layer in cross_layers:
                cross_params.extend([p for p in layer.parameters() if p.requires_grad])
            if cross_params:
                grad_norms['cross_attention'] = _norm(cross_params)

        # ECG tokenizer (encoder + quantizer)
        tokenizer_params: List[torch.nn.Parameter] = []
        encoder = getattr(base_model, 'encoder', None)
        if encoder is not None:
            tokenizer_params.extend([p for p in encoder.parameters() if p.requires_grad])
        quantizer = getattr(base_model, 'quantizer', None)
        if quantizer is not None:
            tokenizer_params.extend([p for p in quantizer.parameters() if p.requires_grad])
        if tokenizer_params:
            grad_norms['ecg_tokenizer'] = _norm(tokenizer_params)

        # Optimizer parameter groups for granular visibility
        if self.optimizer is not None:
            for group_idx, group in enumerate(self.optimizer.param_groups):
                params = [
                    p for p in group.get('params', [])
                    if p is not None and p.requires_grad and p.grad is not None
                ]
                if not params:
                    continue
                group_name = str(group.get('name') or f"group_{group_idx}")
                grad_norms[f"optimizer/{group_name}"] = _norm(params)

        return grad_norms

    def _train_step(
        self, 
        ecg_signal: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        prompt_input_ids: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        pattern_targets: torch.Tensor | None = None,
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
        # Clear gradients (only at the start of an accumulation cycle)
        assert self.optimizer is not None
        if (self._grad_accum_counter % self.grad_accum) == 0:
            self.optimizer.zero_grad(set_to_none=True)
        
        # Forward pass with autocast for mixed precision using bfloat16
        with autocast('cuda', dtype=torch.bfloat16):
            outputs: dict[str, torch.Tensor] = self.model(
                ecg_signal=ecg_signal,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                prompt_input_ids=prompt_input_ids,  # Pass for cross-attention
                prompt_attention_mask=prompt_attention_mask,
                pattern_targets=pattern_targets,
            )
            loss: torch.Tensor = outputs['loss']
            # Keep an unscaled copy for logging; use scaled loss for backward
            raw_loss: torch.Tensor = loss.detach()
            if self.grad_accum > 1:
                loss = loss / self.grad_accum

        # Backward pass - bfloat16 doesn't need gradient scaling
        loss.backward()
        self._grad_accum_counter += 1

        did_step = False
        if (self._grad_accum_counter % self.grad_accum) == 0:
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
            did_step = True
                
        # Get learning rate metrics (only change on optimizer step)
        lr_metrics = {}
        if did_step:
            for pg in self.optimizer.param_groups if self.optimizer else []:
                if "name" in pg:
                    lr_metrics[f"lr_{pg['name']}"] = pg["lr"]
        
        # Step the scheduler if it should be updated per-iteration and we just stepped optimizer
        if did_step and self.scheduler and self.scheduler_per_iteration:
            self.scheduler.step()
        
        result: dict[str, Any] = {
            "loss": raw_loss if 'raw_loss' in locals() else loss,
            "did_step": did_step,
            **lr_metrics
        }
        cf_loss_tensor = outputs.get("cf_loss")
        if cf_loss_tensor is not None:
            cf_loss_value = float(cf_loss_tensor.detach().item()) if torch.is_tensor(cf_loss_tensor) else float(cf_loss_tensor)
            result["cf_loss"] = cf_loss_value
        pattern_loss_tensor = outputs.get("pattern_loss")
        if pattern_loss_tensor is not None:
            pattern_loss_value = float(pattern_loss_tensor.detach().item()) if torch.is_tensor(pattern_loss_tensor) else float(pattern_loss_tensor)
            result["pattern_loss"] = pattern_loss_value
        grad_norms = self._collect_grad_norms()
        for key, value in grad_norms.items():
            result[f"grad_norm/{key}"] = value
        return result

    def _val_step(
        self, 
        ecg_signal: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        prompt_input_ids: torch.Tensor | None = None,
        prompt_attention_mask: torch.Tensor | None = None,
        pattern_targets: torch.Tensor | None = None,
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
                prompt_attention_mask=prompt_attention_mask,
                pattern_targets=pattern_targets,
            )
            loss: torch.Tensor = outputs['loss']
            
            skip_generation = bool(getattr(self.config, "skip_val_generation", False))
            generated_ids = None
            if not skip_generation:
                model_for_generation = self.model.module if hasattr(self.model, 'module') else self.model
                model_for_generation.eval()
                # Build generation kwargs matching inference script
                generation_kwargs = self._build_generation_kwargs()
                
                generated_ids = model_for_generation.generate_report_with_question(
                    ecg_signal,
                    prompt_input_ids=prompt_input_ids,
                    prompt_attention_mask=prompt_attention_mask,
                    max_token_length=self.config.max_token_length,
                    **generation_kwargs,
                )

            # Get learning rate metrics
            lr_metrics = {}
            for pg in self.optimizer.param_groups if self.optimizer else []:
                if "name" in pg:
                    lr_metrics[f"lr_{pg['name']}"] = pg["lr"]

            result: dict[str, Any] = {
                "loss": loss,
                "generated_ids": generated_ids,
                **lr_metrics
            }
            cf_loss_tensor = outputs.get("cf_loss")
            if cf_loss_tensor is not None:
                result["cf_loss"] = float(cf_loss_tensor.detach().item()) if torch.is_tensor(cf_loss_tensor) else float(cf_loss_tensor)
            pattern_loss_tensor = outputs.get("pattern_loss")
            if pattern_loss_tensor is not None:
                result["pattern_loss"] = float(pattern_loss_tensor.detach().item()) if torch.is_tensor(pattern_loss_tensor) else float(pattern_loss_tensor)
            pattern_logits = outputs.get("pattern_logits")
            pattern_targets_out = outputs.get("pattern_targets")
            if pattern_logits is not None:
                result["pattern_logits"] = pattern_logits.detach().float().cpu()
            if pattern_targets_out is not None:
                result["pattern_targets"] = pattern_targets_out.detach().float().cpu()
            return result

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

            model_for_generation = self.model.module if hasattr(self.model, 'module') else self.model
            model_for_generation.eval()
            # Build generation kwargs matching inference script
            generation_kwargs = self._build_generation_kwargs()
            
            generated_ids = model_for_generation.generate_report_with_question(
                ecg_signal,
                prompt_input_ids=prompt_input_ids,
                prompt_attention_mask=prompt_attention_mask,
                max_token_length=self.config.max_token_length,
                **generation_kwargs,
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
            
            # Derive category hint directly from batch categories
            outputs: dict[str, torch.Tensor] = self._inference_step(
                ecg_signal=ecg_signal,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                prompt_input_ids=(batch['prompt_input_ids'].to(self.config.device) if getattr(self.config, 'instruct_mode', False) and 'prompt_input_ids' in batch else None),
                prompt_attention_mask=(batch['prompt_attention_mask'].to(self.config.device) if getattr(self.config, 'instruct_mode', False) and 'prompt_input_ids' in batch else None),
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

                # Use unified decoding function for consistent results
                decoded_prediction, decoded_reference = self._decode_generation(
                    tokenizer,
                    gen,
                    lab,
                    batch=batch,
                    sample_idx=idx
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

    def test(self):
        """
        Run standalone test on the model.
        
        Returns:
            dict[str, float]: Dictionary containing test metrics
        """
        if self.test_dataloader is None:
            raise ValueError("Test dataloader is not set")
        
        # Set model to evaluation mode
        self.model.eval()
        
        if self.config.is_ref_device:
            print("Starting standalone test...")
        
        self.config.num_epochs = 1
        
        # Run test epoch (using epoch=0 as placeholder since this is standalone)
        test_metrics: dict[str, float] = self._run_epoch(
            RunMode.TEST,
            epoch=0
        )
        
        # Log test results to console
        if self.config.is_ref_device:
            print("\n" + "="*60)
            print("TEST RESULTS")
            print("="*60)
            for metric_name, metric_value in test_metrics.items():
                print(f"{metric_name}: {metric_value:.4f}")
            print("="*60 + "\n")
        
        # Log to wandb if available
        if self.wandb_wrapper is not None and self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
            # Add a prefix to distinguish standalone test from training test
            standalone_metrics = {f"standalone_{k}": v for k, v in test_metrics.items()}
            self.wandb_wrapper.log(standalone_metrics)
                    
    def validate(self):
        """
        Run standalone validation on the model.
        
        Returns:
            dict[str, float]: Dictionary containing validation metrics
        """
        if self.validation_dataloader is None:
            raise ValueError("Validation dataloader is not set")
        
        # Set model to evaluation mode
        self.model.eval()
        
        if self.config.is_ref_device:
            print("Starting standalone validation...")
        
        self.config.num_epochs = 1
        
        # Run validation epoch (using epoch=0 as placeholder since this is standalone)
        validation_metrics: dict[str, float] = self._run_epoch(
            RunMode.VALIDATE,
            epoch=0
        )
        
        # Log validation results to console
        if self.config.is_ref_device:
            print("\n" + "="*60)
            print("VALIDATION RESULTS")
            print("="*60)
            for metric_name, metric_value in validation_metrics.items():
                print(f"{metric_name}: {metric_value:.4f}")
            print("="*60 + "\n")
        
        # Log to wandb if available
        if self.wandb_wrapper is not None and self.wandb_wrapper.is_initialized() and self.config.is_ref_device:
            # Add a prefix to distinguish standalone validation from training validation
            standalone_metrics = {f"standalone_{k}": v for k, v in validation_metrics.items()}
            self.wandb_wrapper.log(standalone_metrics)
        
    def _save_model(
        self,
        epoch: int,
        loss: float,
        is_best: bool = False,
        step: Optional[int] = None
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
            'step': step,
            'model_state_dict': model_state_dict,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'scaler_state_dict': self.scaler.state_dict() if self.scaler else None,
            'loss': loss,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'use_lora': self.config.use_lora  # Store LoRA flag for loading
        }
        
        # Save checkpoint with epoch or step granularity
        if step is not None:
            checkpoint_path: str = os.path.join(save_dir, f'checkpoint_step_{step}.pt')
        else:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        
        if step is not None:
            # Keep only the most recent snapshot checkpoint
            if (
                self._last_snapshot_checkpoint is not None
                and self._last_snapshot_checkpoint != checkpoint_path
                and os.path.exists(self._last_snapshot_checkpoint)
            ):
                os.remove(self._last_snapshot_checkpoint)
                if self.config.is_ref_device:
                    print(f"Deleted old snapshot checkpoint: {self._last_snapshot_checkpoint}")
            self._last_snapshot_checkpoint = checkpoint_path
        else:
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
            
            log_payload = {
                "checkpoint/epoch": epoch,
                "checkpoint/loss": loss,
                **lr_metrics
            }
            if step is not None:
                log_payload["checkpoint/step"] = float(step)
            # Anchor checkpoint logs to the provided step or current global step
            step_for_log = int(step) if step is not None else int(getattr(self, 'global_step', 0))
            self.wandb_wrapper.log(log_payload, step=step_for_log)

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
        dataset_ref = dataloader.dataset
        tokenizer = getattr(dataset_ref, "tokenizer", None)
        if tokenizer is None and hasattr(dataset_ref, "dataset"):
            base = getattr(dataset_ref, "dataset")
            while base is not None and tokenizer is None:
                tokenizer = getattr(base, "tokenizer", None)
                base = getattr(base, "dataset", None) if hasattr(base, "dataset") else None
        # Handle ConcatDataset (.datasets is a list of sub-datasets)
        if tokenizer is None and hasattr(dataset_ref, "datasets"):
            for sub_ds in dataset_ref.datasets:
                tokenizer = getattr(sub_ds, "tokenizer", None)
                if tokenizer is not None:
                    break
        if tokenizer is None:
            raise AttributeError("Unable to locate tokenizer on the provided dataloader dataset.")
        labels_for_metrics = labels

        # Decode predictions, references and prompts for logging/metrics enrichment
        batch_predictions: list[str] = []
        batch_references: list[str] = []
        batch_categories: list[str] = []
        batch_prompts: list[str] = []
        gen_ids = outputs.get('generated_ids')
        has_generation = isinstance(gen_ids, torch.Tensor)

        if batch is not None and has_generation:
            generated_ids = gen_ids

            for i in range(generated_ids.size(0)):
                gen_tensor = generated_ids[i].cpu()
                label_tensor = labels[i].cpu()
                
                # Use unified decoding function for consistent results
                prediction, reference = self._decode_generation(
                    tokenizer,
                    gen_tensor,
                    label_tensor,
                    batch=batch,
                    sample_idx=i
                )

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
            # Accumulate full-epoch predictions/references for unified epoch metrics
            if hasattr(self, '_val_agg_preds') and hasattr(self, '_val_agg_refs'):
                    self._val_agg_preds.extend(batch_predictions)
                    self._val_agg_refs.extend(batch_references)

        # If no generations are available (e.g., generation skipped for fast eval), skip text metrics
        if not has_generation:
            return computed_metrics

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
            
            # Prefer prompt-only ids to help metrics trim prompts when available
            input_ids = None
            prompt_input_ids = None
            if batch is not None and 'prompt_input_ids' in batch:
                prompt_input_ids = batch['prompt_input_ids'].to(self.config.device)
            if batch is not None and 'input_ids' in batch:
                input_ids = batch['input_ids'].to(self.config.device)
            
            # Get num_ecg_tokens for proper decoding offset
            # Q-Former uses num_query_tokens, projection bridge uses num_ecg_tokens
            num_ecg_tokens = getattr(self.config, 'num_query_tokens', 0)
            if num_ecg_tokens is None:
                num_ecg_tokens = 0
            num_ecg_tokens = int(num_ecg_tokens)
            if num_ecg_tokens == 0:
                num_ecg_tokens = getattr(self.config, 'num_ecg_tokens', 0)
                if num_ecg_tokens is None:
                    num_ecg_tokens = 0
                num_ecg_tokens = int(num_ecg_tokens)
            
            LLM_metrics: dict[str, Union[float, list[str]]] = registered_metrics.compute_score(
                gen_ids,
                labels_for_metrics,
                tokenizer,  # type: ignore
                input_ids=input_ids,
                prompt_input_ids=prompt_input_ids,
                num_ecg_tokens=num_ecg_tokens
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
            dataloader: DataLoader,
            max_batches: int | None = None
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
            # Select a random batch index using either max_batches or dataloader length
            try:
                dataloader_len = len(dataloader)  # type: ignore[arg-type]
            except TypeError:
                dataloader_len = None
            if max_batches is not None and max_batches > 0:
                if dataloader_len is None:
                    total_batches = max_batches
                else:
                    total_batches = min(max_batches, dataloader_len)
            else:
                total_batches = dataloader_len if dataloader_len is not None else 1
            total_batches = max(1, total_batches)
            random_batch_idx = random.randint(0, total_batches - 1)
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

    def _get_val_generation_json_path(
        self,
        epoch: int,
        step: int | None = None,
        prefix: str | None = None
    ) -> str:
        """Return the validation generations path for an epoch (and optional step snapshot)."""
        if step is None:
            filename = f"val_generations_epoch_{epoch}.json"
        else:
            step_prefix = prefix if prefix else "step"
            filename = f"val_generations_epoch_{epoch:03d}_{step_prefix}_{step}.json"
        return os.path.join(
            self._get_val_generations_dir(),
            filename
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

                # Use unified decoding function for consistent results across all paths
                generation, ground_truth = self._decode_generation(
                    tokenizer,
                    gen_tensor,
                    label_tensor,
                    batch=batch,
                    sample_idx=i
                )

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

            # Add a small preview of 5 randomly selected samples at the end of the JSON.
            try:
                preview_candidates = [
                    k for k, v in existing_data.items()
                    if isinstance(v, dict) and 'Question' in v and 'Generation' in v and k != '__sample_preview__'
                ]
                if preview_candidates:
                    take = min(5, len(preview_candidates))
                    chosen = random.sample(preview_candidates, take)
                    preview = []
                    for key in chosen:
                        entry = existing_data.get(key, {})
                        preview.append({
                            'waveform': key,
                            'question': entry.get('Question', ''),
                            'generation': entry.get('Generation', ''),
                            'ground_truth': entry.get('Ground truth', ''),
                        })
                    existing_data['__sample_preview__'] = preview
            except Exception:
                pass

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
                # Skip non-dict entries (e.g., '__sample_preview__' which is a list)
                if not isinstance(ecg_info, dict):
                    continue
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
                }, step=int(getattr(self, 'global_step', 0)))
                print(f"✓ Logged {len(plot_images)} ECG plots to WandB")

        except Exception as e:
            print(f"Error in ECG plotting: {e}")
            traceback.print_exc()
