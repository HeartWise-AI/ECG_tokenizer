import json
import yaml
from typing import Dict, Any

'''
Adapted from: https://github.com/HeartWise-AI/DeepCORO_CLIP/blob/jd/support_multigpu-issue_7/utils/files_handler.py
'''

def load_yaml(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def generate_output_dir_name(args, run_id):
    """
    Generates a directory name for output based on the provided configuration.
    """
    import time

    model_dir = (
        f"run_{run_id}"
    )

    return model_dir

def load_api_keys(path: str) -> dict[str, str]:
    """
    Load API keys from a JSON file.

    Args:
        path (str): Path to the JSON file containing the API keys.

    Returns:
        dict[str, str]: A dictionary containing the API keys.
        
    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If the file is not a valid JSON file.
    """
    try:
        with open(path) as f:
            keys = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"The API keys file at {path} does not exist.")
    except json.JSONDecodeError:
        raise json.JSONDecodeError(f"The API keys file at {path} is not a valid JSON file.")
    return keys

