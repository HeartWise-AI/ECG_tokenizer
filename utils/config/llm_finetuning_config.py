    
from dataclasses import dataclass

from utils.registry import ConfigRegistry
from utils.config.heartwise_config import HeartWiseConfig

@dataclass
@ConfigRegistry.register("ECG_tokenizer_LLM_finetuning")
class LLMFinetuningConfig(HeartWiseConfig):   
    # Training hyperparameters
    seed: int
    llm_lr: float
    embedding_reducer_lr: float
    num_workers: int
    batch_size: int
    llm_weight_decay: float
    embedding_reducer_weight_decay: float
    num_epochs: int
    criterion: str
    optimizer: str
    runner_name: str
    scheduler_type: str
    step_size: int
    gamma: float
    run_mode: str
    reducer_dropout: float
    
    # Tokenizer parameters
    tokenizer_name: str
    max_token_length: int

    # Model parameters
    huggingface_model_name: str
    trainable_model_name: str
    embedding_reducer_name: str
    gpt2_embedding_size: int

    # Metrics
    metrics: list[str]

    # Data and checkpoint paths
    base_checkpoint_path: str
    train_dataset_path: str
    validation_dataset_path: str
    train_embeddings_path: str
    validation_embeddings_path: str
    output_dir: str
    checkpoint_dir: str
    
    # Inference parameters
    inference_dataset_path: str
