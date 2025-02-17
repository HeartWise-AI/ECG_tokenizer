    
from typing import Optional
from dataclasses import dataclass

from utils.registry import ConfigRegistry
from utils.config.heartwise_config import HeartWiseConfig

@dataclass
@ConfigRegistry.register("ECG_tokenizer_LLM_finetuning")
class LLMFinetuningConfig(HeartWiseConfig):   
    # Training hyperparameters
    seed: int
    lr: float
    num_workers: int
    batch_size: int
    weight_decay: float
    num_epochs: int
    criterion: str
    optimizer: str
    runner_name: str

    # Tokenizer parameters
    tokenizer_name: str
    max_token_length: int

    # Model parameters
    huggingface_model_name: str
    trainable_model_name: str
    embedding_reducer_name: str
    embedding_size: int

    # Metrics
    metrics: list[str]

    # Data and checkpoint paths
    base_checkpoint_path: str
    train_dataset_path: str
    validation_dataset_path: str
    embeddings_path: str
    output_dir: str