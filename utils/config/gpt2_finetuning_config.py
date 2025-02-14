    
from typing import Optional
from dataclasses import dataclass

from utils.registry import ConfigRegistry
from utils.config.heartwise_config import HeartWiseConfig

@dataclass
@ConfigRegistry.register("ECG_tokenizer_gpt2_finetuning")
class GPT2FinetuningConfig(HeartWiseConfig):   
    # Training hyperparameters
    seed: int
    lr: float
    num_workers: int
    batch_size: int
    weight_decay: float
    num_epochs: int
    criterion: str
    optimizer: str
    
    # Data and checkpoint paths
    base_checkpoint_path: str
    train_dataset_path: str
    test_dataset_path: str
    embedding_dir: str