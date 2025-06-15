from typing import Any, Dict, Union
from transformers import BertTokenizer
import torch
from torch.utils.data import DataLoader

from utils.registry import (
    ProjectRegistry, 
    ModelRegistry,
    RunnerRegistry
)
from projects.base_project import BaseProject
from utils.files_handler import load_api_keys
from utils.wandb_wrapper import WandbWrapper
from utils.config import BertReportClassifierConfig
from utils.huggingface_wrapper import HuggingFaceWrapper
from models.bert_classifier import BertClassifier
from runners.bert_report_classifier_runner import BertReportClassifierRunner
from data.bert_clinical_report_dataset import get_distributed_clinical_report_dataloader


@ProjectRegistry.register("BERT_Report_Classifier")
class BertReportClassifierProject(BaseProject):
    def __init__(
        self, 
        config: BertReportClassifierConfig,
        wandb_wrapper: WandbWrapper
    ):
        super().__init__(config, wandb_wrapper)
    
    def run(self):
        super().run()
        
    def _setup_inference_objects(self)->dict[str, Any]:
        huggingface_token: str = load_api_keys(self.config.api_keys_path)["HUGGING_FACE_TOKEN"]
        
        model_path: str = HuggingFaceWrapper.get_model(
            repo_id=self.config.huggingface_model_name,
            local_dir=self.config.store_model_path,
            hugging_face_api_key=huggingface_token
        )
        
        tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_path)
        
        model: BertClassifier = ModelRegistry.get(self.config.pipeline_project)(
            model_path=model_path,
            num_classes=self.config.num_classes,
        )
        model.to(self.config.device)
        
        validation_dataloader = get_distributed_clinical_report_dataloader(
            predicted_reports_path=self.config.predictions_reports_path,
            tokenizer=tokenizer,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            num_replicas=self.config.world_size,
            rank=self.config.device,
            shuffle=False,
            pin_memory=True
        )
        
        return {
            "val_dataloader": validation_dataloader,
            "model": model
        }
    
    def _setup_extraction_objects(self)->dict[str, Any]:
        raise NotImplementedError("Extraction is not implemented for this project")
    
    def _setup_training_objects(self)->dict[str, Any]:
        raise NotImplementedError("Training is not implemented for this project")

