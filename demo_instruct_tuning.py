#!/usr/bin/env python3
"""
Demo script for ECG LLM Instruction Tuning.

This script demonstrates how to run instruction tuning on ECG QA data
with any Hugging Face LLM decoder.
"""

import os
import sys
import json
import logging
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.config import load_config
from utils.config.instruct_tuning_config import InstructTuningConfig
from runners.llm_instruction_tuning_runner import InstructTuningRunner
from utils.instruct_tuning.data_formatter import ECGInstructDataFormatter

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def create_sample_eval_data():
    """Create a sample evaluation dataset by splitting the training data."""
    train_path = "/volume/ECG_tokenizer/test_ecg_qa_sample/train.json"
    eval_path = "/volume/ECG_tokenizer/test_ecg_qa_sample/eval.json"
    
    if os.path.exists(train_path) and not os.path.exists(eval_path):
        logger.info("Creating sample evaluation dataset...")
        
        with open(train_path, 'r') as f:
            data = json.load(f)
        
        # Split data: 80% train, 20% eval
        split_idx = int(0.8 * len(data))
        train_data = data[:split_idx]
        eval_data = data[split_idx:]
        
        # Save eval data
        with open(eval_path, 'w') as f:
            json.dump(eval_data, f, indent=2)
        
        # Update train data
        with open(train_path, 'w') as f:
            json.dump(train_data, f, indent=2)
        
        logger.info(f"Created eval dataset with {len(eval_data)} samples")
        logger.info(f"Updated train dataset with {len(train_data)} samples")


def demonstrate_data_formatting():
    """Demonstrate different instruction templates."""
    logger.info("=== Demonstrating Data Formatting ===")
    
    # Load sample data
    sample_path = "/volume/ECG_tokenizer/test_ecg_qa_sample/train.json"
    if not os.path.exists(sample_path):
        logger.error(f"Sample data not found at {sample_path}")
        return
    
    with open(sample_path, 'r') as f:
        data = json.load(f)
    
    # Take first sample
    sample = data[0] if data else None
    if not sample:
        logger.error("No sample data available")
        return
    
    # Demonstrate different templates
    templates = ["alpaca", "vicuna", "chat", "medical"]
    
    for template in templates:
        logger.info(f"\n--- {template.upper()} Template ---")
        
        formatter = ECGInstructDataFormatter(template_style=template)
        formatted = formatter.format_single_example(
            sample, 
            include_metadata=True, 
            enhanced_prompt=True
        )
        
        print(f"Template: {template}")
        print(f"Formatted text preview (first 300 chars):")
        print(formatted['text'][:300] + "...")
        print()


def run_instruct_tuning_demo(model_name: str = "gpt2"):
    """
    Run a demonstration of instruction tuning.
    
    Args:
        model_name: Name of the model to use (gpt2 or llama32_1b)
    """
    logger.info(f"=== Running Instruction Tuning Demo with {model_name} ===")
    
    # Create sample eval data if needed
    create_sample_eval_data()
    
    # Select config file
    if model_name == "gpt2":
        config_path = "/volume/ECG_tokenizer/config/instruct_tuning/gpt2_instruct_base.yaml"
    elif model_name == "llama32_1b":
        config_path = "/volume/ECG_tokenizer/config/instruct_tuning/llama32_1b_instruct_base.yaml"
    else:
        raise ValueError(f"Unsupported model: {model_name}")
    
    if not os.path.exists(config_path):
        logger.error(f"Config file not found: {config_path}")
        return
    
    # Load configuration
    logger.info(f"Loading config from {config_path}")
    config = load_config(config_path, InstructTuningConfig)
    
    # Create experiment directory
    experiment_dir = f"./experiments/instruct_tuning_demo_{model_name}"
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Run instruction tuning
    try:
        runner = InstructTuningRunner(config, experiment_dir)
        runner.run()
        
        logger.info("✅ Instruction tuning demo completed successfully!")
        
    except Exception as e:
        logger.error(f"❌ Instruction tuning demo failed: {str(e)}")
        raise


def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ECG LLM Instruction Tuning Demo")
    parser.add_argument(
        "--action",
        choices=["format_demo", "train", "both"],
        default="both",
        help="Action to perform"
    )
    parser.add_argument(
        "--model",
        choices=["gpt2", "llama32_1b"],
        default="gpt2",
        help="Model to use for training"
    )
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting ECG LLM Instruction Tuning Demo")
    
    try:
        if args.action in ["format_demo", "both"]:
            demonstrate_data_formatting()
        
        if args.action in ["train", "both"]:
            run_instruct_tuning_demo(args.model)
            
        logger.info("🎉 Demo completed successfully!")
        
    except Exception as e:
        logger.error(f"💥 Demo failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
