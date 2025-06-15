    
from dataclasses import dataclass

from utils.registry import ConfigRegistry
from utils.config.heartwise_config import HeartWiseConfig

@dataclass
@ConfigRegistry.register("ECG_tokenizer_LLM_finetuning")
class LLMFinetuningConfig(HeartWiseConfig):   
    # Training hyperparameters (non-default fields)
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
    num_warmup_percent: float
    num_hard_restarts_cycles: float
    warm_restart_tmult: int
    
    # Tokenizer parameters
    tokenizer_name: str
    max_token_length: int

    # Model parameters
    huggingface_model_name: str
    trainable_model_name: str
    embedding_reducer_name: str
    
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
    
    # Model type and embedding sizes (default fields)
    model_type: str = "gpt2"  # Model type: gpt2, bloom, opt, mistral, gptneo, gptj
    gpt2_embedding_size: int = 768
    bloom_embedding_size: int = 1024  # BLOOM-560m
    opt_embedding_size: int = 768     # OPT-125m (350m model is broken on HuggingFace)
    mistral_embedding_size: int = 4096  # Mistral-7B
    gptneo_embedding_size: int = 768    # GPT-Neo-125M
    gptj_embedding_size: int = 4096     # GPT-J-6B
    
    # Validation speed optimization
    validation_text_generation_frequency: float = 0.1  # Generate text for 10% of validation batches
    full_validation_frequency: int = 1  # Run full validation every N epochs (1 = every epoch)
    
    def get_embedding_size(self) -> int:
        """Get the embedding size for the current model type."""
        model_type_mapping = {
            "gpt2": self.gpt2_embedding_size,
            "bloom": self.bloom_embedding_size,
            "opt": self.opt_embedding_size,
            "mistral": self.mistral_embedding_size,
            "gptneo": self.gptneo_embedding_size,
            "gptj": self.gptj_embedding_size
        }
        return model_type_mapping.get(self.model_type, self.gpt2_embedding_size)
    
    def get_model_class_name(self) -> str:
        """Get the model class name for the current model type."""
        model_type_mapping = {
            "gpt2": "GPT2_WithEmbedding",
            "bloom": "BLOOM_WithEmbedding",
            "opt": "OPT_WithEmbedding",
            "mistral": "Mistral_WithEmbedding",
            "gptneo": "GPTNeo_WithEmbedding",
            "gptj": "GPTJ_WithEmbedding"
        }
        return model_type_mapping.get(self.model_type, "GPT2_WithEmbedding")
    
    def get_default_reducer_name(self) -> str:
        """Get the default reducer name for the current model type."""
        model_type_mapping = {
            "gpt2": "GPT2_SimpleEmbeddingReducer",
            "bloom": "BLOOM_SimpleEmbeddingReducer", 
            "opt": "OPT_SimpleEmbeddingReducer",
            "mistral": "Mistral_SimpleEmbeddingReducer",
            "gptneo": "GPTNeo_SimpleEmbeddingReducer",
            "gptj": "GPTJ_SimpleEmbeddingReducer"
        }
        return model_type_mapping.get(self.model_type, "GPT2_SimpleEmbeddingReducer")
