import os
import pytest
import tempfile
import yaml
import json
from typing import Dict, Any
from dataclasses import dataclass

from utils.config import HeartWiseConfig, ECGTokenizerTrainingConfig


@pytest.fixture
def sample_yaml_config():
    """Returns a sample YAML config for testing"""
    return {
        "pipeline_project": "ECG_Tokenizer_Training",
        "wandb_project": "ecg_tokenizer",
        "wandb_entity": "test_entity",
        "use_wandb": True,
        "output_dir": "/path/to/output",
        "run_mode": "train",
        "num_epochs": 10,
        "seed": 42,
        "lr": 0.001,
        "scheduler_name": "step",
        "lr_step_period": 5,
        "factor": 0.1,
        "optimizer": "adam",
        "weight_decay": 0.0001,
        "step_size": 10,
        "gamma": 0.5,
        "use_amp": True,
        "encoder_name": "cnn",
        "quantizer_name": "vq",
        "decoder_name": "cnn",
        "num_quantizers": 4,
        "codebook_size": 512,
        "train_dataset_path": "/path/to/train",
        "validation_dataset_path": "/path/to/val",
        "num_workers": 4,
        "batch_size": 32,
        "waveform_length": 5000,
        "num_leads": 12,
        "normalize_waveforms": True,
        "lead_stats": {"lead_I": {"mean": 0.0, "std": 1.0}}
    }


@pytest.fixture
def sample_config_file():
    """Creates a temporary YAML file with a sample config"""
    with tempfile.TemporaryDirectory() as temp_dir:
        config_path = os.path.join(temp_dir, "config.yaml")
        config = {
            "pipeline_project": "ECG_Tokenizer_Training",
            "wandb_project": "ecg_tokenizer",
            "wandb_entity": "test_entity",
            "use_wandb": True,
            "output_dir": "/path/to/output",
            "run_mode": "train",
            "num_epochs": 10,
            "seed": 42
        }
        
        with open(config_path, 'w') as f:
            yaml.dump(config, f)
            
        yield config_path


@pytest.fixture
def sample_api_keys_file():
    """Creates a temporary JSON file with sample API keys"""
    with tempfile.TemporaryDirectory() as temp_dir:
        keys_path = os.path.join(temp_dir, "api_keys.json")
        keys = {
            "HUGGING_FACE_TOKEN": "hf_abcdefg",
            "OPENAI_API_KEY": "sk-abcdefg"
        }
        
        with open(keys_path, 'w') as f:
            json.dump(keys, f)
            
        yield keys_path


@pytest.fixture
def basic_heartwise_config():
    """Returns a basic HeartWiseConfig instance"""
    return HeartWiseConfig(
        pipeline_project="test_project",
        wandb_project="test_wandb",
        wandb_entity="test_entity",
        use_wandb=False
    )


@pytest.fixture
def full_ecg_tokenizer_config():
    """Returns a complete ECGTokenizerTrainingConfig instance"""
    return ECGTokenizerTrainingConfig(
        pipeline_project="ECG_Tokenizer_Training",
        wandb_project="ecg_tokenizer",
        wandb_entity="test_entity",
        use_wandb=False,
        output_dir="/path/to/output",
        run_mode="train",
        num_epochs=10,
        seed=42,
        lr=0.001,
        scheduler_name="step",
        lr_step_period=5,
        factor=0.1,
        optimizer="adam",
        weight_decay=0.0001,
        step_size=10,
        gamma=0.5,
        use_amp=True,
        encoder_name="cnn",
        quantizer_name="vq",
        decoder_name="cnn",
        num_quantizers=4,
        codebook_size=512,
        train_dataset_path="/path/to/train",
        validation_dataset_path="/path/to/val",
        num_workers=4,
        batch_size=32,
        waveform_length=5000,
        num_leads=12,
        normalize_waveforms=True,
        lead_stats={"lead_I": {"mean": 0.0, "std": 1.0}}
    )


@dataclass
class MockArgs:
    """Mock command line arguments for testing"""
    seed: int = 123
    batch_size: int = 64
    other_param: str = "ignored"


@pytest.fixture
def mock_args():
    """Returns mock command line arguments"""
    return MockArgs()