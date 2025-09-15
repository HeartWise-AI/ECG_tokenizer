from typing import Tuple, Optional, Dict, Any
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
    adapter_name: str
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
    
    # Sequence token adapter parameters (required - no defaults)
    use_cross_attention: bool
    num_attention_heads: int
    enable_attention_visualization: bool
    attention_log_frequency: int
    
    # Fields with defaults (must come after all required fields)
    ecg_token_start_id: Optional[int] = None
    intermediate_dim: Optional[int] = None
    
    # Per-category metrics configuration (with defaults)
    compute_category_metrics: bool = False
    category_metrics: list[str] = field(default_factory=lambda: ["rouge", "bleu", "meteor"])
    
    # Dataset column configuration (with defaults)
    prompt_column: str = "prompt"              # Input question/prompt column
    answer_column: str = "generated_answer"    # Expected output/answer column  
    category_column: str = "prompt_category"   # For per-category metrics
    
    # Instruction tuning (with default)
    instruct_mode: bool = False
    
    # LoRA parameters (with defaults)
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[list[str]] = None  # Will default to common targets
    lora_bias: str = "none"  # "none", "all", or "lora_only"
    lora_config: Optional[Dict[str, Any]] = None  # Nested LoRA config
    
    # Training optimization parameters (with defaults)
    gradient_accumulation_steps: int = 1
    dataloader_pin_memory: bool = False
    max_sequence_length: Optional[int] = None
    
    # Two-stage training configuration (with defaults)
    training_phases: Optional[Dict[str, Any]] = None
    
    # Debug and monitoring configuration (with defaults)
    debug_config: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Process nested lora_config if provided"""
        if self.lora_config is not None:
            # Extract individual LoRA parameters from nested config
            self.lora_r = self.lora_config.get('r', self.lora_r)
            self.lora_alpha = self.lora_config.get('lora_alpha', self.lora_alpha)
            self.lora_dropout = self.lora_config.get('lora_dropout', self.lora_dropout)
            self.lora_target_modules = self.lora_config.get('target_modules', self.lora_target_modules)
            self.lora_bias = self.lora_config.get('bias', self.lora_bias)
