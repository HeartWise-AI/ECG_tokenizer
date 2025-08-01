from typing import Any
from transformers import BertTokenizer

from utils.registry import (
    ProjectRegistry, 
    ModelRegistry
)
from utils.enums import ProjectName
from utils.wandb_wrapper import WandbWrapper
from utils.files_handler import load_api_keys
from utils.config import BertReportClassifierConfig
from utils.huggingface_wrapper import HuggingFaceWrapper
from projects.base_project import BaseProject
from models.bert_classifier import BertClassifier
from data.bert_clinical_report_dataset import get_distributed_clinical_report_dataloader


@ProjectRegistry.register(ProjectName.BERT_REPORT_CLASSIFIER)
class BertReportClassifierProject(BaseProject):
    """BERT-based clinical report classification project.
    
    Implements inference pipeline for classifying clinical reports using
    pretrained BERT models from HuggingFace. Supports distributed inference
    across multiple devices.
    """    
    def __init__(
        self, 
        config: BertReportClassifierConfig,
        wandb_wrapper: WandbWrapper
    ):
        """Initialize BERT report classifier project.
        
        Args:
            config: BERT classification configuration
            wandb_wrapper: Weights & Biases logging wrapper
        """        
        super().__init__(config, wandb_wrapper)
        self.config: BertReportClassifierConfig = config # cast to BertReportClassifierConfig to avoid type errors
    
    def run(self):
        """Execute the BERT classification workflow."""
        super().run()
        
    def _setup_inference_objects(self)->dict[str, Any]:
        """Setup objects required for clinical report inference.
        
        Downloads pretrained BERT model from HuggingFace, initializes tokenizer
        and classifier, and prepares data loader for inference on clinical reports.
        
        Returns:
            Dictionary containing validation data loader and model for inference
        """
        huggingface_token: str = load_api_keys(self.config.api_keys_path)["HUGGING_FACE_TOKEN"]
        
        model_path: str = HuggingFaceWrapper.get_model(
            repo_id=self.config.huggingface_model_name,
            local_dir=self.config.store_model_path,
            hugging_face_api_key=huggingface_token
        )
        
        tokenizer: BertTokenizer = BertTokenizer.from_pretrained(model_path)
        
        model: BertClassifier = ModelRegistry.get(self.config.model_name)(
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
        """Setup objects for extraction mode.
        
        Raises:
            NotImplementedError: Extraction not implemented for BERT classifier
        """        
        raise NotImplementedError("Extraction is not implemented for this project")
    
    def _setup_training_objects(self)->dict[str, Any]:
        """Setup objects for training mode.
        
        Raises:
            NotImplementedError: Training not implemented for BERT classifier
        """        
        raise NotImplementedError("Training is not implemented for this project")
    
    def _setup_validation_objects(self)->dict[str, Any]:
        """Setup objects for validation mode.
        
        Raises:
            NotImplementedError: Validation not implemented for BERT classifier
        """        
        raise NotImplementedError("Validation is not implemented for this project")