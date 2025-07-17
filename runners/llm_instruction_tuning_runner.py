#!/usr/bin/env python3
"""
LLM Instruction Tuning Runner for ECG Tokenizer.

This runner implements instruction tuning (also known as instruct-tuning) for
any Hugging Face LLM decoder in the ECG tokenizer system.
"""

import os
import sys
import torch
import logging
from typing import Dict, Any, Optional
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.registry import RunnerRegistry
from utils.enums import RunnerName
from runners.base_runner import BaseRunner
from utils.instruct_tuning.instructor import ECGInstructTuner
from utils.instruct_tuning.data_formatter import ECGInstructDataFormatter
from utils.config import InstructTuningConfig
from datasets import Dataset
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@RunnerRegistry.register(RunnerName.INSTRUCT_TUNING_RUNNER)
class InstructTuningRunner(BaseRunner):
    """
    Runner for instruction tuning ECG LLM models.
    
    This runner:
    1. Loads and formats ECG QA data for instruction tuning
    2. Initializes the ECG tokenizer model with any Hugging Face LLM decoder
    3. Performs instruction tuning using SFTTrainer
    4. Saves the instruction-tuned model
    """
    
    def __init__(self, config: InstructTuningConfig, experiment_dir: str):
        """
        Initialize the instruction tuning runner.
        
        Args:
            config: Instruction tuning configuration
            experiment_dir: Directory for experiment outputs
        """
        super().__init__(config, experiment_dir)
        self.config: InstructTuningConfig = config
        
        # Initialize components
        self.data_formatter = ECGInstructDataFormatter(
            template_style=config.template_style
        )
        self.instruct_tuner = None
        
        logger.info(f"Initialized InstructTuningRunner for {config.huggingface_model_name}")
        logger.info(f"Template style: {config.template_style}")
        logger.info(f"Output directory: {config.output_dir}")
    
    def setup(self) -> None:
        """Setup the instruction tuning components."""
        logger.info("Setting up instruction tuning components...")
        
        # Create output directories
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.logging_dir, exist_ok=True)
        
        # Initialize the instruct tuner
        self.instruct_tuner = ECGInstructTuner(self.config)
        
        logger.info("Setup completed successfully")
    
    def load_and_format_data(self) -> Dict[str, Dataset]:
        """
        Load and format ECG QA data for instruction tuning.
        
        Returns:
            Dictionary containing train/eval/test datasets
        """
        logger.info("Loading and formatting ECG QA data...")
        
        datasets = {}
        
        # Load training data
        if hasattr(self.config, 'train_data_path') and self.config.train_data_path:
            if os.path.exists(self.config.train_data_path):
                with open(self.config.train_data_path, 'r') as f:
                    train_data = json.load(f)
                
                datasets['train'] = self.data_formatter.format_dataset(
                    train_data,
                    include_metadata=self.config.include_metadata,
                    enhanced_prompt=self.config.enhanced_prompt
                )
                logger.info(f"Loaded {len(datasets['train'])} training examples")
            else:
                logger.warning(f"Training data path not found: {self.config.train_data_path}")
        
        # Load evaluation data
        if hasattr(self.config, 'eval_data_path') and self.config.eval_data_path:
            if os.path.exists(self.config.eval_data_path):
                with open(self.config.eval_data_path, 'r') as f:
                    eval_data = json.load(f)
                
                datasets['eval'] = self.data_formatter.format_dataset(
                    eval_data,
                    include_metadata=self.config.include_metadata,
                    enhanced_prompt=self.config.enhanced_prompt
                )
                logger.info(f"Loaded {len(datasets['eval'])} evaluation examples")
            else:
                logger.warning(f"Evaluation data path not found: {self.config.eval_data_path}")
        
        # Load test data
        if hasattr(self.config, 'test_data_path') and self.config.test_data_path:
            if os.path.exists(self.config.test_data_path):
                with open(self.config.test_data_path, 'r') as f:
                    test_data = json.load(f)
                
                datasets['test'] = self.data_formatter.format_dataset(
                    test_data,
                    include_metadata=self.config.include_metadata,
                    enhanced_prompt=self.config.enhanced_prompt
                )
                logger.info(f"Loaded {len(datasets['test'])} test examples")
            else:
                logger.warning(f"Test data path not found: {self.config.test_data_path}")
        
        if not datasets:
            raise ValueError("No datasets loaded. Please check your data paths.")
        
        return datasets
    
    def run_instruction_tuning(self, datasets: Dict[str, Dataset]) -> None:
        """
        Run instruction tuning on the loaded datasets.
        
        Args:
            datasets: Dictionary containing train/eval/test datasets
        """
        logger.info("Starting instruction tuning...")
        
        # Prepare training arguments
        train_dataset = datasets.get('train')
        eval_dataset = datasets.get('eval')
        
        if train_dataset is None:
            raise ValueError("No training dataset provided")
        
        # Run instruction tuning
        results = self.instruct_tuner.train(
            train_dataset=train_dataset,
            eval_dataset=eval_dataset
        )
        
        logger.info("Instruction tuning completed successfully")
        logger.info(f"Training results: {results}")
        
        # Save the instruction-tuned model
        self.instruct_tuner.save_model(self.config.output_dir)
        logger.info(f"Model saved to {self.config.output_dir}")
    
    def evaluate_model(self, datasets: Dict[str, Dataset]) -> Dict[str, Any]:
        """
        Evaluate the instruction-tuned model.
        
        Args:
            datasets: Dictionary containing datasets
            
        Returns:
            Evaluation results
        """
        logger.info("Evaluating instruction-tuned model...")
        
        eval_dataset = datasets.get('eval') or datasets.get('test')
        if eval_dataset is None:
            logger.warning("No evaluation dataset available")
            return {}
        
        # Run evaluation
        results = self.instruct_tuner.evaluate(eval_dataset)
        
        logger.info(f"Evaluation results: {results}")
        return results
    
    def run(self) -> None:
        """Run the complete instruction tuning pipeline."""
        logger.info("Starting instruction tuning pipeline...")
        
        try:
            # Setup components
            self.setup()
            
            # Load and format data
            datasets = self.load_and_format_data()
            
            # Run instruction tuning
            self.run_instruction_tuning(datasets)
            
            # Evaluate model
            eval_results = self.evaluate_model(datasets)
            
            logger.info("Instruction tuning pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Instruction tuning pipeline failed: {str(e)}")
            raise


def main():
    """Main function for standalone execution."""
    import argparse
    from utils.config import load_config
    
    parser = argparse.ArgumentParser(description="ECG LLM Instruction Tuning")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True,
        help="Path to instruction tuning config file"
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        default="./experiments/instruct_tuning",
        help="Directory for experiment outputs"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config, InstructTuningConfig)
    
    # Create experiment directory
    os.makedirs(args.experiment_dir, exist_ok=True)
    
    # Run instruction tuning
    runner = InstructTuningRunner(config, args.experiment_dir)
    runner.run()


if __name__ == "__main__":
    main()