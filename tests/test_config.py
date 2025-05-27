import os
import pytest
from unittest.mock import patch, mock_open
import tempfile
import yaml
import json
from dataclasses import dataclass

from utils.config import (
    HeartWiseConfig,
    LinearProbingConfig,
    LLMFinetuningConfig,
    BertReportClassifierConfig,
    ECGTokenizerTrainingConfig
)
from utils.registry import ConfigRegistry


class TestHeartWiseConfig:
    """Tests for the HeartWiseConfig base class"""

    def test_to_dict(self):
        """Test that to_dict correctly converts the config to a dictionary"""
        config = HeartWiseConfig(
            pipeline_project="test_project",
            wandb_project="test_wandb",
            wandb_entity="test_entity",
            use_wandb=True,
            base_config_path="config/gpt2/base_config.yaml"
        )
        config_dict = config.to_dict()
        
        assert isinstance(config_dict, dict)
        assert config_dict["pipeline_project"] == "test_project"
        assert config_dict["wandb_project"] == "test_wandb"
        assert config_dict["wandb_entity"] == "test_entity"
        assert config_dict["use_wandb"] is True
        assert config_dict["base_config_path"] == "config/gpt2/base_config.yaml"
    @patch.dict(os.environ, {"LOCAL_RANK": "1", "WORLD_SIZE": "4"})
    def test_set_gpu_info_in_place(self):
        """Test that set_gpu_info_in_place correctly sets GPU info from environment variables"""
        config = HeartWiseConfig(
            pipeline_project="test_project",
            wandb_project="test_wandb",
            wandb_entity="test_entity",
            use_wandb=True,
            base_config_path="config/gpt2/base_config.yaml"
        )
        
        HeartWiseConfig.set_gpu_info_in_place(config)
        
        assert config.device == 1
        assert config.world_size == 4
        assert config.is_ref_device is False

    @patch("utils.config.heartwise_config.load_yaml")
    def test_from_yaml(self, mock_load_yaml):
        """Test that from_yaml correctly loads a config from a YAML file"""
        # Set up the mock to return a predefined config
        mock_config = {
            "pipeline_project": "ECG_Tokenizer_Training",
            "wandb_project": "ecg_tokenizer",
            "wandb_entity": "test_entity",
            "use_wandb": True,
            "run_mode": "train",
            "num_epochs": 10,
            "seed": 42,
            "base_checkpoint_path": "/path/to/checkpoint",
            "lr": 0.001,
            "scheduler_type": "step",
            "lr_step_period": 5,
            "factor": 0.1,
            "optimizer": "adam",
            "weight_decay": 0.0001,
            "step_size": 10,
            "gamma": 0.5,
            "num_warmup_percent": 0.1,
            "num_hard_restarts_cycles": 1,
            "warm_restart_tmult": 2,
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
            "lead_stats": {"lead_I": {"mean": 0.0, "std": 1.0}},
            "base_config_path": "config/gpt2/base_config.yaml"
        }
        mock_load_yaml.return_value = mock_config
        
        config = HeartWiseConfig.from_yaml("dummy_path.yaml")
        
        assert isinstance(config, ECGTokenizerTrainingConfig)
        assert config.pipeline_project == "ECG_Tokenizer_Training"
        assert config.num_epochs == 10
        assert config.waveform_length == 5000

    @patch("utils.config.heartwise_config.load_yaml")
    def test_from_yaml_missing_pipeline_project(self, mock_load_yaml):
        """Test that from_yaml raises an error when pipeline_project is missing"""
        mock_config = {
            "wandb_project": "ecg_tokenizer",
            "wandb_entity": "test_entity",
            "use_wandb": True
        }
        mock_load_yaml.return_value = mock_config
        
        with pytest.raises(ValueError, match="pipeline_project is not set in the yaml file"):
            HeartWiseConfig.from_yaml("dummy_path.yaml")

    def test_update_config_with_args(self):
        """Test that update_config_with_args correctly updates a config with command line arguments"""
        # Create a mock args object
        @dataclass
        class Args:
            seed: int = 123
            batch_size: int = 64
            other_param: str = "ignored"
        
        args = Args()
        
        # Create a base config
        base_config = ECGTokenizerTrainingConfig(
            pipeline_project="ECG_Tokenizer_Training",
            wandb_project="ecg_tokenizer",
            wandb_entity="test_entity",
            use_wandb=True,
            run_mode="train",
            num_epochs=10,
            seed=42,
            base_checkpoint_path="/path/to/checkpoint",
            lr=0.001,
            scheduler_type="step",
            lr_step_period=5,
            factor=0.1,
            optimizer="adam",
            weight_decay=0.0001,
            step_size=10,
            gamma=0.5,
            num_warmup_percent=0.1,
            num_hard_restarts_cycles=1,
            warm_restart_tmult=2,
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
            lead_stats={"lead_I": {"mean": 0.0, "std": 1.0}},
            base_config_path="config/gpt2/base_config.yaml"
        )
        
        updated_config = HeartWiseConfig.update_config_with_args(base_config, args)
        
        assert updated_config.seed == 123  # Should be updated
        assert updated_config.batch_size == 64  # Should be updated
        assert updated_config.num_epochs == 10  # Should remain unchanged


class TestECGTokenizerTrainingConfig:
    """Tests for the ECGTokenizerTrainingConfig class"""
    
    def test_config_registration(self):
        """Test that ECGTokenizerTrainingConfig is properly registered"""
        config_class = ConfigRegistry.get("ECG_Tokenizer_Training")
        assert config_class == ECGTokenizerTrainingConfig

    def test_config_creation(self):
        """Test that ECGTokenizerTrainingConfig can be created with the required parameters"""
        config = ECGTokenizerTrainingConfig(
            pipeline_project="ECG_Tokenizer_Training",
            wandb_project="ecg_tokenizer",
            wandb_entity="test_entity",
            use_wandb=True,
            run_mode="train",
            num_epochs=10,
            seed=42,
            base_checkpoint_path="/path/to/checkpoint",
            lr=0.001,
            scheduler_type="step",
            lr_step_period=5,
            factor=0.1,
            optimizer="adam",
            weight_decay=0.0001,
            step_size=10,
            gamma=0.5,
            num_warmup_percent=0.1,
            num_hard_restarts_cycles=1,
            warm_restart_tmult=2,
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
            lead_stats={"lead_I": {"mean": 0.0, "std": 1.0}},
            base_config_path="config/gpt2/base_config.yaml"
        )
        
        assert config.pipeline_project == "ECG_Tokenizer_Training"
        assert config.waveform_length == 5000
        assert config.num_leads == 12


class TestConfigRegistry:
    """Tests for the ConfigRegistry class"""
    
    def test_registry_get(self):
        """Test that ConfigRegistry.get returns the correct class"""
        assert ConfigRegistry.get("ECG_Tokenizer_Training") == ECGTokenizerTrainingConfig
        
    def test_registry_get_invalid(self):
        """Test that ConfigRegistry.get raises an error for invalid names"""
        with pytest.raises(ValueError, match="config InvalidConfig not found in registry"):
            ConfigRegistry.get("InvalidConfig")
            
    def test_registry_list_registered(self):
        """Test that ConfigRegistry.list_registered returns all registered classes"""
        registered = ConfigRegistry.list_registered()
        assert "ECG_Tokenizer_Training" in registered
        assert registered["ECG_Tokenizer_Training"] == ECGTokenizerTrainingConfig 