
import torch

from transformers import GPT2Tokenizer

from utils.registry import (
    ProjectRegistry, 
    ModelRegistry
)
from utils.config import LLMFinetuningConfig
from utils.wandb_wrapper import WandbWrapper
from models.embedding_reducer import EmbeddingReducer
from models.gpt2_with_embeddings import GPT2WithEmbedding
from data.ecg_clinical_report_dataset import ECGClinicalReportDataset


@ProjectRegistry.register("ECG_tokenizer_LLM_finetuning")
class LLMFinetuningProject:
    def __init__(
        self,
        config: LLMFinetuningConfig,
        wandb_wrapper: WandbWrapper
    ):
        self.config: LLMFinetuningConfig = config
        self.wandb_wrapper: WandbWrapper = wandb_wrapper

    def run(self):
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        tokenizer.pad_token = tokenizer.eos_token

        training_set = ECGClinicalReportDataset(
            embeddings_path=self.config.embeddings_path,
            reports_path=self.config.train_dataset_path,
            tokenizer=tokenizer,
            max_length=self.config.max_token_length
        )

        validation_set = ECGClinicalReportDataset(
            embeddings_path=self.config.embeddings_path,
            reports_path=self.config.validation_dataset_path,
            tokenizer=tokenizer,
            max_length=self.config.max_token_length
        )
        
        device = torch.device(self.config.device)
        
        reducer: EmbeddingReducer = ModelRegistry.get(self.config.embedding_reducer_name)(
            output_size=self.config.embedding_size
        )
        model: GPT2WithEmbedding = ModelRegistry.get(self.config.trainable_model_name)(
            gpt2_model_name=self.config.huggingface_model_name, 
            embedding_size=self.config.embedding_size, 
            reducer=reducer
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=self.config.lr)
