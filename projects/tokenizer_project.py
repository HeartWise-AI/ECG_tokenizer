from typing import Any
from transformers import BertTokenizer

from utils.registry import (
    ProjectRegistry, 
    ModelRegistry,
    RunnerRegistry
)
from utils.files_handler import load_api_keys
from utils.wandb_wrapper import WandbWrapper
from utils.config import ECGTokenizerTrainingConfig
from utils.huggingface_wrapper import HuggingFaceWrapper
from models.bert_classifier import BertClassifier
from runners.bert_report_classifier_runner import BertReportClassifierRunner
from data.bert_clinical_report_dataset import get_distributed_clinical_report_dataloader


@ProjectRegistry.register("ECG_Tokenizer_Training")
class ECGTokenizerTrainingProject:
    def __init__(
        self, 
        config: ECGTokenizerTrainingConfig,
        wandb_wrapper: WandbWrapper
    ):
        self.config = config
        self.wandb_wrapper = wandb_wrapper
        
    def _setup_training_objects(self)->dict[str, Any]:
        raise NotImplementedError("Subclasses must implement this method")
    
    def _setup_inference_objects(self)->dict[str, Any]:
        raise NotImplementedError("Subclasses must implement this method")
    def run(self):
        runner_args = {
            "config": self.config,
            "wandb_wrapper": self.wandb_wrapper
        }
        if self.config.run_mode == "train":
            raise NotImplementedError("Training is not implemented")
        elif self.config.run_mode == "inference":
            raise NotImplementedError("Inference is not implemented")
                  
        # runner: BertReportClassifierRunner = RunnerRegistry.get(self.config.pipeline_project)(**runner_args)
        # runner.execute(mode=self.config.run_mode)