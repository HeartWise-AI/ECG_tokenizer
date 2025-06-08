    
from typing import Optional
from dataclasses import dataclass

from utils.enums import ConfigName
from utils.registry import ConfigRegistry
from utils.config.heartwise_config import HeartWiseConfig

@dataclass
@ConfigRegistry.register(ConfigName.ECG_TOKENIZER_LINEAR_PROBING)
class LinearProbingConfig(HeartWiseConfig):
    # Model architecture parameters
    num_layers: int
    hidden_dim: int
    embedding_dim: int
    prev_embedding_dim: int
    num_classes: int
    num_quantizers: int
    
    # Training hyperparameters
    lr: float
    batch_size: int
    weight_decay: float
    num_epochs: int
    criterion: str
    
    # Data and checkpoint paths
    base_checkpoint_path: str
    csv_file: str
    parquet_file: str
    model_path: str
    embedding_dir: str
    
    # Experiment tracking
    name: str
    classifier_experiment_name: Optional[str]