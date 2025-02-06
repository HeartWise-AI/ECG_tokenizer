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

    current_time = time.strftime("%Y%m%d-%H%M%S")
    batch_size = args.batch_size
    lr = args.lr
    tag = args.tag if args.tag else "default"
    project = args.project if args.project else "default_project"

    model_dir = (
        f"{tag}_b{batch_size}_lr{lr}_{current_time}_{run_id}"
    )

    dir_name = os.path.join(project, model_dir)
    return dir_name