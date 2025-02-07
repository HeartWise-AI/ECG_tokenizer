import os
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