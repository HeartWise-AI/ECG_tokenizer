    
from typing import Tuple
from dataclasses import dataclass

from utils.enums import ConfigName
from utils.registry import ConfigRegistry
from utils.config.heartwise_config import HeartWiseConfig

@dataclass
@ConfigRegistry.register(ConfigName.ECG_TOKENIZER_LLM_FINETUNING)
class LLMFinetuningConfig(HeartWiseConfig):   
    # Pipeline parameters
    pretrained_tokenizer_path: str
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

    # Model parameters
    huggingface_model_name: str
    trainable_model_name: str
    adapter_name: str
    gpt2_embedding_size: int
    
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