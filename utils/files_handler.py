import os
import csv
import json
import yaml
import shutil

from typing import Dict, Any


def load_yaml(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config

def generate_output_dir_name(
    config: dict[str, Any], 
    run_id: str | None = None
)->str:
    """
    Generates a directory name for output based on the provided configuration.
    """
    import time
    current_time = time.strftime("%Y%m%d-%H%M%S")

    run_folder = f"{run_id}_{current_time}" if run_id is not None else f"{current_time}_no_wandb"

    model_dir = os.path.join(
        config.base_checkpoint_path, 
        config.pipeline_project,
        config.wandb_project,
        run_folder
    )
    return model_dir

def backup_config(
    config: dict[str, Any],
    output_dir: str
) -> None:
    """
    Backup the configuration file to the output directory by copying the original config file.
    """
    config_path: str = os.path.join(output_dir, "config.yaml")
    shutil.copyfile(config.base_config_path, config_path)

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
        # Simply re-raise the exception without trying to create a new one
        raise
    return keys

def save_to_csv(
    metrics: dict, 
    path: str
) -> None:
    """
    Save metrics dictionary to a CSV file.

    This function collects all unique subkeys from the nested dictionaries in the
    metrics and writes them into a CSV file with a consistent column order.
    If the output directory does not exist, it is created.

    Args:
        metrics (dict): A dictionary where each key maps to a dictionary of metric values.
        path (str): The file path where the CSV file will be saved.

    Returns:
        None
    """
    if not os.path.exists('/'.join(path.split('/')[:-1])):
        os.makedirs('/'.join(path.split('/')[:-1]))

    # Collect all unique subkeys
    all_subkeys = set()
    for key, value in metrics.items():
        if isinstance(value, dict):
            all_subkeys.update(value.keys())

    # Sort the subkeys for consistent column order
    sorted_subkeys = sorted(all_subkeys)

    # Prepare the rows
    rows = []
    for key, value in metrics.items():
        if isinstance(value, dict):
            row = {'Key': key}
            for subkey in sorted_subkeys:
                row[subkey] = value.get(subkey, '')
            rows.append(row)

    # Write to CSV
    with open(path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Key'] + sorted_subkeys)
        writer.writeheader()
        writer.writerows(rows)

def save_json(
    data: dict, 
    path: str
) -> None:
    """
    Save a dictionary as a JSON file.

    This function writes the provided dictionary to a JSON file with an indentation
    of 4 spaces. If the directory for the specified path does not exist, it is created.

    Args:
        data (dict): The data to be saved as JSON.
        path (str): The destination file path for the JSON file.

    Returns:
        None
    """
    if not os.path.exists('/'.join(path.split('/')[:-1])):
        os.makedirs('/'.join(path.split('/')[:-1]))
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)