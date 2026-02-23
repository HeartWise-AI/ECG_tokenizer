from typing import Tuple, Optional, Dict, Any, List
from dataclasses import dataclass, field

from utils.enums import ConfigName
from utils.registry import ConfigRegistry
from utils.config.heartwise_config import HeartWiseConfig

@dataclass
@ConfigRegistry.register(ConfigName.ECG_TOKENIZER_LLM_FINETUNING)
class LLMFinetuningConfig(HeartWiseConfig):   
    # Pipeline parameters (required - no defaults)
    pretrained_tokenizer_path: str
    model_name: str
    runner_name: str
    num_epochs: int
    base_checkpoint_path: str
    
    # Training hyperparameters (required - no defaults)
    llm_lr: float
    adapter_lr: float
    llm_weight_decay: float
    adapter_weight_decay: float
    scheduler_type: str
    lr_step_period: int
    factor: float
    optimizer: str
    step_size: int
    gamma: float
    num_warmup_percent: float
    num_hard_restarts_cycles: float
    warm_restart_tmult: int

    # VQVAE parameters (required - no defaults)
    decoder_mode: str
    decoder_name: str
        
    # LLM tokenizer parameters (required - no defaults)
    max_token_length: int
    tokenizer_name: str
    num_ecg_tokens: int

    # Model parameters (required - no defaults)
    huggingface_model_name: str
    bridge_name: str
    llm_input_embedding_size: int
    
    # Metrics (required - no defaults)
    metrics: list[str]
    
    # Dataset parameters (required - no defaults)
    train_dataset_path: str
    validation_dataset_path: str
    num_workers: int
    batch_size: int
    signal_path_column: str
    
    # Waveform parameters (required - no defaults)
    ecg_waveform_length: int
    ecg_num_leads: int
    
    # High-level data mode toggle: 'qa' (instruction tuning) or 'cf' (choice-free)
    data_mode: str = "qa"
    
    # Sequence token adapter parameters (now optional with defaults)
    use_cross_attention: bool = False
    num_attention_heads: int = 8
    enable_attention_visualization: bool = False
    attention_log_frequency: int = 100
    
    # Fields with defaults (must come after all required fields)
    stage1_checkpoint_path: Optional[str] = None
    ecg_token_start_id: Optional[int] = None
    intermediate_dim: Optional[int] = None
    ecg_codebook_size: int = 192
    bridge_mid_dim: int = 512
    bridge_num_visual_tokens: Optional[int] = None
    # Preferred Q-Former query count (replaces bridge_num_visual_tokens for Q-Former bridges)
    num_query_tokens: Optional[int] = None
    bridge_num_heads: int = 8
    bridge_dropout: float = 0.1
    bridge_num_special_tokens: int = 4
    bridge_qformer_layers: Optional[int] = None
    bridge_text_hidden_size: Optional[int] = None
    bridge_bias_last_codebook: Optional[float] = None
    bridge_codebook_dropout: Optional[float] = None
    bridge_cross_every: Optional[int] = None
    instruction_dropout: float = 0.0
    default_generation_kwargs: Optional[Dict[str, Any]] = None

    # Codebook selection parameters (for multi-codebook models)
    num_quantizers: Optional[int] = None
    num_codebooks_kept: Optional[int] = None  # None = keep all codebooks
    codebook_offset: int = 0                   # Skip the first N codebooks

    ecg_projection_config: Optional[Dict[str, Any]] = None
    # Projection bridge tuning (optional)
    bridge_use_sinusoidal_pos_emb: bool = False
    bridge_pos_embedding_max_len: Optional[int] = None
    bridge_softmax_temp: Optional[float] = None
    bridge_mix_residual: Optional[float] = None
    bridge_add_modality_embed: Optional[bool] = None
    bridge_add_cls_token: Optional[bool] = None
    # Default to computing BERTScore on 10 batches unless overridden in config
    bertscore_max_batches: Optional[int] = 10
    processor_name: Optional[str] = None
    use_auto_processor: bool = False
    
    # Per-category metrics configuration (with defaults)
    compute_category_metrics: bool = False
    category_metrics: list[str] = field(default_factory=lambda: ["rouge", "bleu", "meteor"])
    
    # Dataset column configuration (with defaults)
    prompt_column: str = "prompt"              # Input question/prompt column
    answer_column: str = "generated_answer"    # Expected output/answer column
    category_column: str = "prompt_category"   # For per-category metrics
    pattern_label_columns: Tuple[str, ...] = field(default_factory=tuple)  # Multilabel ECG targets
    pattern_loss_weight: float = 0.3

    # Weighted sampling for minority class upsampling
    use_weighted_sampling: bool = False        # Enable WeightedRandomSampler for training
    sample_weight_column: str = "sample_weight"  # Column containing per-sample weights
    # Optional BCE pos_weight for auxiliary pattern head (float or list of floats)
    pattern_bce_pos_weight: Optional[Any] = None
    
    # Instruction tuning (with default)
    instruct_mode: bool = False
    # MedGemma-style chat prompts with explicit image placeholders
    medgemma_prompt_style: bool = False
    # Optional one-time debug print of a formatted sample at startup
    debug_print_example: bool = False
    # Debug: assert/log ECG injection after <start_of_image>
    debug_ecg_injection: bool = False

    # Prefix tuning toggle (with default)
    prefix_tuning: bool = False
    
    # LoRA parameters (with defaults)
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[list[str]] = None  # Will default to common targets
    lora_bias: str = "none"  # "none", "all", or "lora_only"
    lora_top_k_layers: Optional[int] = None
    # Deprecated nested LoRA config (kept for backward compatibility; flattened in __post_init__)
    lora_config: Optional[Dict[str, Any]] = None
    
    # Training optimization parameters (with defaults)
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    dataloader_pin_memory: bool = False
    max_sequence_length: Optional[int] = None
    
    # Two-stage training configuration (with defaults)
    training_phases: Optional[Dict[str, Any]] = None
    # Optional sweep override: freeze/unfreeze phase2 LLM backbone
    phase2_freeze_llm: Optional[bool] = None
    
    # Debug and monitoring configuration (with defaults)
    debug_config: Optional[Dict[str, Any]] = None

    # Checkpoint management
    resume_checkpoint_path: Optional[str] = None
    resume_global_step: Optional[int] = None  # Global step to resume from (for mid-epoch resume)
    resume_best_val_loss: Optional[float] = None  # Best validation loss to resume from
    
    # ECG plotting configuration for validation
    plot_validation_ecgs: bool = False
    num_validation_plots: int = 6
    plot_selection_strategy: str = "worst_random_best"  # "worst_random_best", "random", "all"
    validation_shuffle: bool = False
    validation_step_interval: Optional[int] = None
    validation_snapshot_batches: int = 0
    validation_snapshot_prefix: str = "step"
    validation_subset_size: Optional[int] = None
    validation_balance_prompt_categories: bool = False
    validation_sampling_seed: Optional[int] = None
    train_metric_interval: Optional[int] = None
    train_metric_batches: int = 1
    # Debug options
    debug_prompt_dump: bool = False                # Decode prompt/input/label for first batch
    debug_prompt_stop_after_first_batch: bool = False  # Exit loops after first batch when debugging
    debug_ablate_ecg: Optional[str] = None         # None, "zero", or "shuffle"
    debug_shuffle_prompts: bool = False            # Shuffle prompt_input_ids within batch

    # Multi-dataset support
    train_dataset_paths: Optional[List[str]] = None       # List of train parquet paths (overrides train_dataset_path)
    dataset_weights: Optional[List[float]] = None          # Per-dataset sampling weights
    validation_dataset_paths: Optional[List[str]] = None   # List of validation parquet paths

    # ECG augmentation
    ecg_augmentation_enabled: bool = False                 # Enable random ECG perturbations
    ecg_augmentation_prob: float = 0.5                     # Per-sample augmentation probability
    ecg_augmentation_amplitude_min: float = 0.8            # Global amplitude scale lower bound
    ecg_augmentation_amplitude_max: float = 1.2            # Global amplitude scale upper bound
    ecg_augmentation_noise_snr_min: float = 20.0           # Gaussian noise SNR lower bound (dB, lower=louder)
    ecg_augmentation_noise_snr_max: float = 40.0           # Gaussian noise SNR upper bound (dB)
    ecg_augmentation_wander_min: float = 0.01              # Baseline wander amplitude lower bound
    ecg_augmentation_wander_max: float = 0.05              # Baseline wander amplitude upper bound

    # CF evaluation configuration (early-signal, evaluation-only)
    use_cf_eval: bool = False
    cf_eval_dataset_path: str = "ecg_cf_eval/test_cf.json"
    cf_eval_interval: int = 500     # Evaluate every N training steps
    cf_eval_samples: int = 1000     # Subsample size for speed during training
    # Optional: log example-level CF details to wandb/console
    cf_eval_log_details: bool = False
    cf_eval_log_k: int = 10
    # Optional: write per-question CF argmax predictions to JSON
    # When max_records <= 0, evaluates and writes for all available records
    cf_eval_write_predictions: bool = False
    cf_eval_predictions_max: int = 0
    # Validation controls
    validation_max_batches: Optional[int] = None   # cap validation batches (None = full)
    write_val_generations: bool = True            # write val_generations JSON during validation
    # Optional perf knob: skip expensive generation during validation (loss/aux metrics only)
    skip_val_generation: bool = False

    def __post_init__(self):
        """Flatten legacy nested LoRA configs into flat fields."""
        if self.lora_config:
            cfg = self.lora_config
            self.lora_r = cfg.get("r", self.lora_r)
            self.lora_alpha = cfg.get("lora_alpha", self.lora_alpha)
            self.lora_dropout = cfg.get("lora_dropout", self.lora_dropout)
            self.lora_target_modules = cfg.get("target_modules", self.lora_target_modules)
            self.lora_bias = cfg.get("bias", self.lora_bias)
            self.lora_top_k_layers = cfg.get("top_k_layers", self.lora_top_k_layers)
            # Clear nested copy so downstream code uses flat fields only
            self.lora_config = None
