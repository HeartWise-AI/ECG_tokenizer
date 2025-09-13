from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass, field

from utils.enums import ConfigName
from utils.registry import ConfigRegistry
from utils.config.heartwise_config import HeartWiseConfig

@dataclass
@ConfigRegistry.register(ConfigName.ECG_TOKENIZER_LLM_FINETUNING)
class LLMFinetuningConfig(HeartWiseConfig):   
    # Pipeline parameters
    pretrained_tokenizer_path: str
    model_name: str
    runner_name: str
    num_epochs: int
    base_checkpoint_path: str
    
    # Training hyperparameters
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

    # VQVAE parameters
    decoder_mode: str
    decoder_name: str
        
    # LLM tokenizer parameters
    max_token_length: int
    tokenizer_name: str
    num_ecg_tokens: int
    ecg_token_start_id: Optional[int]

    # Model parameters
    huggingface_model_name: str
    adapter_name: str
    llm_input_embedding_size: int
    
    # Metrics
    metrics: list[str]
    
    # Dataset parameters
    train_dataset_path: str
    validation_dataset_path: str
    num_workers: int
    batch_size: int
    signal_path_column: str
    
    # Waveform parameters
    ecg_waveform_length: int
    ecg_num_leads: int
    
    # Sequence token adapter parameters (optional - with defaults)
    use_cross_attention: bool
    num_attention_heads: int
    intermediate_dim: Optional[int]
    enable_attention_visualization: bool
    attention_log_frequency: int
    
    # Instruction tuning
    instruct_mode: bool = False
    
    # LoRA parameters
    use_lora: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: Optional[list[str]] = None  # Will default to common targets
    lora_bias: str = "none"  # "none", "all", or "lora_only"
    lora_config: Optional[Dict[str, Any]] = field(default=None)  # Nested LoRA config
    
    # Training optimization parameters
    gradient_accumulation_steps: int = 1
    dataloader_pin_memory: bool = False
    max_sequence_length: Optional[int] = None
    
    def __post_init__(self):
        """Process nested lora_config if provided"""
        if self.lora_config is not None:
            # Extract individual LoRA parameters from nested config
            self.lora_r = self.lora_config.get('r', self.lora_r)
            self.lora_alpha = self.lora_config.get('lora_alpha', self.lora_alpha)
            self.lora_dropout = self.lora_config.get('lora_dropout', self.lora_dropout)
            self.lora_target_modules = self.lora_config.get('target_modules', self.lora_target_modules)
            self.lora_bias = self.lora_config.get('bias', self.lora_bias)
