from typing import Optional, List
from dataclasses import dataclass

from utils.enums import ConfigName
from utils.registry import ConfigRegistry
from utils.config.heartwise_config import HeartWiseConfig

@dataclass
@ConfigRegistry.register(ConfigName.ECG_TOKENIZER_INSTRUCT_TUNING)
class InstructTuningConfig(HeartWiseConfig):   
    # Required pipeline parameters
    pretrained_tokenizer_path: str
    model_name: str
    runner_name: str
    num_epochs: int
    base_checkpoint_path: str
    train_dataset_path: str
    val_dataset_path: str
    
    # Required model parameters
    huggingface_model_name: str
    llm_input_embedding_size: int
    decoder_name: str
    adapter_name: str
    
    # ECG tokenizer configuration (with defaults)
    encoder_name: str = "Conv_Encoder"
    quantizer_name: str = "ECG_Tokenizer_Quantizer"
    num_quantizers: int = 4
    codebook_size: int = 1024
    decoder_mode: str = "llm"
    
    # Instruction tuning specific (with defaults)
    template_style: str = "alpaca"
    max_seq_length: int = 512
    include_metadata: bool = True
    enhanced_prompt: bool = True
    
    # Training hyperparameters (with defaults)
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    
    # Advanced training settings (with defaults)
    fp16: bool = True
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    remove_unused_columns: bool = False
    
    # LoRA settings (with defaults)
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: Optional[List[str]] = None
    
    # Early stopping (with defaults)
    use_early_stopping: bool = True
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # LLM tokenizer parameters (with defaults)
    max_token_length: int = 256
    tokenizer_name: str = "gpt2"
    
    # Metrics (with defaults)
    metrics: List[str] = None
    
    def __post_init__(self):
        super().__post_init__()
        if self.metrics is None:
            self.metrics = ["rouge", "bleu"]
