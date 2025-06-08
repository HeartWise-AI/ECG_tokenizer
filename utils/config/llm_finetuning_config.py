    
from typing import Tuple
from dataclasses import dataclass

from utils.enums import ConfigName
from utils.registry import ConfigRegistry
from utils.config.heartwise_config import HeartWiseConfig

@dataclass
@ConfigRegistry.register(ConfigName.ECG_TOKENIZER_LLM_FINETUNING)
class LLMFinetuningConfig(HeartWiseConfig):   
    # Training hyperparameters
    num_epochs: int
    num_workers: int
    runner_name: str
    llm_lr: float
    llm_weight_decay: float
    embedding_adapter_lr: float
    embedding_adapter_weight_decay: float
    embedding_adapter_dropout: float
    batch_size: int
    criterion: str
    optimizer: str
    scheduler_type: str
    step_size: int
    gamma: float
    num_warmup_percent: float
    num_hard_restarts_cycles: float
    warm_restart_tmult: int
    
    # Tokenizer parameters
    tokenizer_name: str
    max_token_length: int

    # Model parameters
    huggingface_model_name: str
    trainable_model_name: str
    embedding_adapter_name: str
    gpt2_embedding_size: int

    # ECG tokenizer parameters
    ecg_tokenizer_path: str
    ecg_tokenizer_num_quantizers: int
    ecg_tokenizer_codebook_size: int
    ecg_encoder_name: str
    ecg_quantizer_name: str
    ecg_decoder_name: str
    ecg_embedding_size: Tuple[int, int, int]
    
    # ECG parameters
    ecg_waveform_length: int
    ecg_num_leads: int

    # Metrics
    metrics: list[str]

    # Data and checkpoint paths
    base_checkpoint_path: str
    train_dataset_path: str
    validation_dataset_path: str
    output_dir: str
    
    # Inference parameters
    checkpoint_dir: str
    inference_dataset_path: str
