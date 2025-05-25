import os
import json
import pytest
from unittest.mock import patch, mock_open
import tempfile
import unittest
from unittest.mock import MagicMock

from utils.files_handler import (
    load_yaml,
    load_api_keys,
    save_to_csv,
    save_json,
    generate_output_dir_name
)


class TestLoadYaml:
    """Tests for the load_yaml function"""
    
    def test_load_yaml_valid(self):
        """Test that load_yaml correctly loads a valid YAML file"""
        yaml_content = """
        pipeline_project: test_project
        wandb_project: test_wandb
        wandb_entity: test_entity
        use_wandb: true
        """
        
        with patch("builtins.open", mock_open(read_data=yaml_content)):
            config = load_yaml("dummy_path.yaml")
            
        assert config["pipeline_project"] == "test_project"
        assert config["wandb_project"] == "test_wandb"
        assert config["wandb_entity"] == "test_entity"
        assert config["use_wandb"] is True
    
    def test_load_yaml_file_not_found(self):
        """Test that load_yaml raises an error when the file is not found"""
        with patch("builtins.open", side_effect=FileNotFoundError()):
            with pytest.raises(FileNotFoundError):
                load_yaml("nonexistent_file.yaml")


class TestLoadApiKeys:
    """Tests for the load_api_keys function"""
    
    def test_load_api_keys_valid(self):
        """Test that load_api_keys correctly loads a valid JSON file"""
        api_keys_content = """
        {
            "HUGGING_FACE_TOKEN": "hf_abcdefg",
            "OPENAI_API_KEY": "sk-abcdefg"
        }
        """
        
        with patch("builtins.open", mock_open(read_data=api_keys_content)):
            keys = load_api_keys("dummy_path.json")
            
        assert keys["HUGGING_FACE_TOKEN"] == "hf_abcdefg"
        assert keys["OPENAI_API_KEY"] == "sk-abcdefg"
    
    def test_load_api_keys_file_not_found(self):
        """Test that load_api_keys raises a FileNotFoundError when the file is not found"""
        with patch("builtins.open", side_effect=FileNotFoundError()):
            with pytest.raises(FileNotFoundError, match="The API keys file at nonexistent_file.json does not exist"):
                load_api_keys("nonexistent_file.json")
    
    def test_load_api_keys_invalid_json(self):
        """Test that load_api_keys raises a JSONDecodeError when the file is not valid JSON"""
        invalid_json = """
        {
            "HUGGING_FACE_TOKEN": "hf_abcdefg",
            "OPENAI_API_KEY": "sk-abcdefg"
        """  # Missing closing bracket
        
        with patch("builtins.open", mock_open(read_data=invalid_json)):
            with patch("json.load", side_effect=json.JSONDecodeError("Expecting ',' delimiter", invalid_json, 110)):
                with pytest.raises(json.JSONDecodeError):
                    load_api_keys("invalid_json.json")


class TestSaveJson:
    """Tests for the save_json function"""
    
    def test_save_json(self):
        """Test that save_json correctly saves a dictionary to a JSON file"""
        test_data = {
            "key1": "value1",
            "key2": 42,
            "key3": [1, 2, 3],
            "key4": {"nested": "value"}
        }
        
        # Use a real temporary file for this test
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "output", "test.json")
            
            save_json(test_data, file_path)
            
            # Check that the file was created
            assert os.path.exists(file_path)
            
            # Read the file and check the content
            with open(file_path, "r") as f:
                saved_data = json.load(f)
                
            assert saved_data == test_data


class TestSaveToCsv:
    """Tests for the save_to_csv function"""
    
    def test_save_to_csv(self):
        """Test that save_to_csv correctly saves metrics to a CSV file"""
        metrics = {
            "model1": {"accuracy": 0.95, "f1": 0.92, "precision": 0.94},
            "model2": {"accuracy": 0.87, "f1": 0.84, "recall": 0.85}
        }
        
        # Use a real temporary file for this test
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "output", "metrics.csv")
            
            save_to_csv(metrics, file_path)
            
            # Check that the file was created
            assert os.path.exists(file_path)
            
            # Read the content as text
            with open(file_path, "r") as f:
                content = f.read()
                
            # Check that the headers are present
            assert "Key" in content
            assert "accuracy" in content
            assert "f1" in content
            assert "precision" in content
            assert "recall" in content
            
            # Check that the model names are present
            assert "model1" in content
            assert "model2" in content


class TestFilesHandler(unittest.TestCase):
    
    @patch('time.strftime')
    def test_generate_output_dir_name_with_run_id(self, mock_strftime):
        # Set up mock time
        mock_time = "20230101-120000"
        mock_strftime.return_value = mock_time
        
        # Create a mock config object
        mock_config = MagicMock()
        mock_config.base_checkpoint_path = "/path/to/checkpoints"
        mock_config.pipeline_project = "ECG_tokenizer_project"
        mock_config.wandb_project = "test_project"
        
        # Test with a run_id
        run_id = "test_run_123"
        expected_dir = os.path.join(
            "/path/to/checkpoints",
            "ECG_tokenizer_project",
            "test_project",
            f"{run_id}_{mock_time}"
        )
        
        result = generate_output_dir_name(mock_config, run_id)
        self.assertEqual(result, expected_dir)
        mock_strftime.assert_called_once_with("%Y%m%d-%H%M%S")
    
    @patch('time.strftime')
    def test_generate_output_dir_name_without_run_id(self, mock_strftime):
        # Set up mock time
        mock_time = "20230101-120000"
        mock_strftime.return_value = mock_time
        
        # Create a mock config object
        mock_config = MagicMock()
        mock_config.base_checkpoint_path = "/path/to/checkpoints"
        mock_config.pipeline_project = "ECG_tokenizer_project"
        mock_config.wandb_project = "test_project"
        
        # Test without a run_id
        expected_dir = os.path.join(
            "/path/to/checkpoints",
            "ECG_tokenizer_project",
            "test_project",
            f"{mock_time}_no_wandb"
        )
        
        result = generate_output_dir_name(mock_config)
        self.assertEqual(result, expected_dir)
        mock_strftime.assert_called_once_with("%Y%m%d-%H%M%S") 