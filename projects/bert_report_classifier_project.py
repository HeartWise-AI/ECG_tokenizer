from typing import Any
from transformers import BertTokenizer

from utils.registry import (
    ProjectRegistry, 
    ModelRegistry
)
from utils.files_handler import load_api_keys
from utils.wandb_wrapper import WandbWrapper
from utils.config import BertReportClassifierConfig
from utils.huggingface_wrapper import HuggingFaceWrapper
from models.bert_classifier import BertClassifier
from data.bert_clinical_report_dataset import get_distributed_clinical_report_dataloader


@ProjectRegistry.register("BERT_Report_Classifier")
class BertReportClassifierProject:
    def __init__(
        self, 
        config: BertReportClassifierConfig,
        wandb_wrapper: WandbWrapper
    ):
        self.config = config
        self.wandb_wrapper = wandb_wrapper
        
    def _setup_training_objects(self)->dict[str, Any]:
        raise NotImplementedError("Subclasses must implement this method")
    
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
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            num_replicas=self.config.world_size,
            rank=self.config.device,
            shuffle=False,
            pin_memory=True
        )
        
        return {
            "validation_dataloader": validation_dataloader,
            "tokenizer": tokenizer
        }
        
    def run(self):
        runner_args = {
            "config": self.config,
            "wandb_wrapper": self.wandb_wrapper
        }
        if self.config.run_mode == "train":
            raise NotImplementedError("Training is not implemented")
        elif self.config.run_mode == "inference":
            inference_objects: dict[str, Any] = self._setup_inference_objects()
            runner_args.update(inference_objects)
        
        print('ready to execute')
          
        # runner: BertReportClassifierRunner = RunnerRegistry.get(self.config.runner_name)(**runner_args)
        # runner.execute(mode=self.config.run_mode)