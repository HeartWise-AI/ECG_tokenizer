import os
import torch

from typing import Any, Union
from abc import ABC, abstractmethod

from utils.enums import RunMode
from runners.types import RunnerT, RunnerClassT
from utils.registry import RunnerRegistry
from utils.wandb_wrapper import WandbWrapper
from utils.config.heartwise_config import HeartWiseConfig
from utils.files_handler import generate_output_dir_name, backup_config

class BaseProject(ABC):
    """Abstract base class for ML project execution with different run modes.
    
    Handles project setup, checkpoint loading, and runner orchestration for
    training, inference, and embedding extraction workflows.
    """    
    def __init__(
        self, 
        config: Any,
        wandb_wrapper: WandbWrapper
    ):
        """Initialize project with configuration and logging.
        
        Args:
            config: Project configuration object
            wandb_wrapper: Weights & Biases logging wrapper
        """        
        self.config: Any = config
        self.wandb_wrapper: WandbWrapper = wandb_wrapper
        
    @abstractmethod
    def _setup_inference_objects(self)->dict[str, Any]:
        """Setup objects required for inference mode.
        
        Returns:
            dict[str, Any]: A dictionary containing the objects required for inference.
        """
        pass
    
    @abstractmethod
    def _setup_training_objects(self)->dict[str, Any]:
        """Setup objects required for training mode.
        
        Returns:
            dict[str, Any]: A dictionary containing the objects required for training.
        """
        pass
    
    @abstractmethod
    def _setup_validation_objects(self)->dict[str, Any]:
        """Setup objects required for validation mode.
        
        Returns:
            dict[str, Any]: A dictionary containing the objects required for validation.
        """
        pass
    
    def _setup_extraction_objects(self)->dict[str, Any]:
        """Setup objects required for extraction mode.
        
        Returns:
            dict[str, Any]: A dictionary containing the objects required for extraction.
        """
        pass
    
    def _setup_project(self):
        """Initialize project directories and backup the base_config file.
        
        This method is called by the run method to initialize the project directories and backup the base_config file.
        """
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
        """Load model checkpoint from file.
        
        Args:
            checkpoint_path: Path to checkpoint file
            
        Returns:
            Loaded checkpoint dictionary
            
        Raises:
            ValueError: If checkpoint file doesn't exist
        """
        if not os.path.exists(checkpoint_path):
            raise ValueError(f"Checkpoint file does not exist: {checkpoint_path}")
        
        print(
            f"[{self.__class__.__name__}] Loading checkpoint: {checkpoint_path}"
        )
        
        return torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    
    def run(self): 
        """Execute project workflow based on configured run mode.
        
        Sets up appropriate objects for the run mode (train/inference/extraction)
        and executes the corresponding runner.
        """          
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
        
        elif self.config.run_mode == RunMode.VALIDATE:
            runner_args.update(self._setup_validation_objects())
        
        runner_class: RunnerClassT = RunnerRegistry.get(self.config.runner_name)
        runner: RunnerT = runner_class(**runner_args)
        runner.execute(mode=self.config.run_mode)    