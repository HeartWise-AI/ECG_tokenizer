import os
import torch

from typing import Any, Union
from abc import ABC, abstractmethod

from utils.enums import RunMode
from utils.registry import RunnerRegistry
from utils.wandb_wrapper import WandbWrapper
from utils.config.heartwise_config import HeartWiseConfig
from utils.files_handler import generate_output_dir_name, backup_config
from runners.tokenizer_runner import ECGTokenizerRunner
from runners.llm_finetuning_runner import LLMFinetuningRunner
from runners.bert_report_classifier_runner import BertReportClassifierRunner

class BaseProject(ABC):
    def __init__(
        self, 
        config: HeartWiseConfig,
        wandb_wrapper: WandbWrapper
    ):
        self.config: HeartWiseConfig = config
        self.wandb_wrapper: WandbWrapper = wandb_wrapper
        
    @abstractmethod
    def _setup_inference_objects(self)->dict[str, Any]:
        pass
    
    @abstractmethod
    def _setup_training_objects(self)->dict[str, Any]:
        pass
    
    @abstractmethod
    def _setup_extraction_objects(self)->dict[str, Any]:
        pass
    
    def _setup_project(self):
        # Generate the output directory name
        self.config.output_dir = generate_output_dir_name(
            config=self.config, 
            run_id=self.wandb_wrapper.get_run_id() if self.wandb_wrapper.is_initialized() else None
        )
        
        # Create the output directory
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        # Backup the configuration file
        backup_config(
            config=self.config,
            output_dir=self.config.output_dir
        )      
        
    def _load_checkpoint(
        self, 
        checkpoint_path: str
    )->dict[str, Any]:
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint file does not exist: {checkpoint_path}")
        
        print(
            f"[{self.__class__.__name__}] Loading checkpoint: {checkpoint_path}"
        )
        
        return torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    def run(self):    
        if self.config.is_ref_device:
            self._setup_project()    
        
        runner_args = {
            "config": self.config,
            "wandb_wrapper": self.wandb_wrapper
        }
        if self.config.run_mode == RunMode.TRAIN:
            runner_args.update(self._setup_training_objects())
            
        elif self.config.run_mode == RunMode.EXTRACT_EMBEDDINGS:
            runner_args.update(self._setup_extraction_objects())
        
        elif self.config.run_mode == RunMode.INFERENCE:
            runner_args.update(self._setup_inference_objects())
        
        runner: Union[
            ECGTokenizerRunner, 
            LLMFinetuningRunner,
            BertReportClassifierRunner
        ] = RunnerRegistry.get(self.config.pipeline_project)(**runner_args)
        runner.execute(mode=self.config.run_mode)    