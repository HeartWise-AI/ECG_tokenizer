
import torch

from transformers import GPT2Tokenizer

from utils.registry import ProjectRegistry
from utils.config import GPT2FinetuningConfig
from utils.wandb_wrapper import WandbWrapper
from models.embedding_reducer import EmbeddingReducer
from models.gpt2_with_embeddings import GPT2WithEmbedding
from data.ecg_clinical_report_dataset import ECGClinicalReportDataset

@ProjectRegistry.register("ECG_tokenizer_gpt2_finetuning")
class GPT2FinetuningProject:
    def __init__(
        self,
        config: GPT2FinetuningConfig,
        wandb_wrapper: WandbWrapper
    ):
        self.config: GPT2FinetuningConfig = config
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
        model = GPT2WithEmbedding(
            gpt2_model_name=self.config.model_name, 
            embedding_size=self.config.embedding_size, 
            reducer=EmbeddingReducer(
                output_size=self.config.embedding_size
            )
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
