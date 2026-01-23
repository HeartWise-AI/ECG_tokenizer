"""
File handling utilities for ECG Tokenizer inference pipeline.

Following DeepECG_Docker pattern for:
- Unified CSV/Parquet loading and saving
- Base64-encoded ECG signal storage for preprocessed data
"""

import os
import csv
import json
import base64
import numpy as np
import pandas as pd
from typing import Optional


def load_df(path: str) -> pd.DataFrame:
    """
    Load a DataFrame from CSV or Parquet file.
    
    Args:
        path: Path to the file (.csv or .parquet)
        
    Returns:
        Loaded DataFrame
        
    Raises:
        ValueError: If file extension is not supported
    """
    if path.endswith('.csv'):
        df = pd.read_csv(path)
    elif path.endswith('.parquet'):
        df = pd.read_parquet(path)
    else:
        raise ValueError("Unsupported file extension. Only .csv and .parquet are supported.")
    return df


def save_df(df: pd.DataFrame, path: str) -> None:
    """
    Save a DataFrame to CSV or Parquet file.
    
    Args:
        df: DataFrame to save
        path: Path to save the file (.csv or .parquet)
        
    Raises:
        ValueError: If file extension is not supported
    """
    os.makedirs(os.path.dirname(path), exist_ok=True) if os.path.dirname(path) else None
    
    if path.endswith('.csv'):
        df.to_csv(path, index=False)
    elif path.endswith('.parquet'):
        df.to_parquet(path, index=False)
    else:
        raise ValueError("Unsupported file extension. Only .csv and .parquet are supported.")


def save_to_csv(metrics: dict, path: str) -> None:
    """
    Save metrics dictionary to CSV file.
    
    Args:
        metrics: Dictionary of metrics
        path: Path to save the CSV file
    """
    parent_dir = os.path.dirname(path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)

    all_subkeys = set()
    for key, value in metrics.items():
        if isinstance(value, dict):
            all_subkeys.update(value.keys())

    sorted_subkeys = sorted(all_subkeys)

    rows = []
    for key, value in metrics.items():
        if isinstance(value, dict):
            row = {'Key': key}
            for subkey in sorted_subkeys:
                row[subkey] = value.get(subkey, '')
            rows.append(row)

    with open(path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=['Key'] + sorted_subkeys)
        writer.writeheader()
        writer.writerows(rows)


def save_json(data: dict, path: str) -> None:
    """
    Save dictionary to JSON file.
    
    Args:
        data: Dictionary to save
        path: Path to save the JSON file
    """
    parent_dir = os.path.dirname(path)
    if parent_dir and not os.path.exists(parent_dir):
        os.makedirs(parent_dir)
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, default=str)


class ECGFileHandler:
    """
    Handler for ECG signal file operations.
    
    Supports:
    - .base64: Base64-encoded float32 arrays (compact, text-safe)
    - .npy: Standard numpy array files
    """
    
    @staticmethod
    def save_ecg_signal(ecg_signal: np.ndarray, filename: str) -> None:
        """
        Save ECG signal to file.
        
        Args:
            ecg_signal: ECG signal array, shape (2500, 12) or (12, 2500)
            filename: Path to save the file (.base64 or .npy)
        """
        parent_dir = os.path.dirname(filename)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
            
        if filename.endswith('.base64'):
            ecg_signal = ecg_signal.astype(np.float32)
            base64_str = base64.b64encode(ecg_signal.tobytes()).decode('utf-8')
            with open(filename, 'w') as f:
                f.write(base64_str)
        else:
            np.save(filename, ecg_signal)
    
    @staticmethod
    def load_ecg_signal(filename: str) -> np.ndarray:
        """
        Load ECG signal from file.
        
        Args:
            filename: Path to the file (.base64 or .npy)
            
        Returns:
            ECG signal array, shape (2500, 12)
        """
        if filename.endswith('.base64'):
            with open(filename, 'r') as f:
                base64_str = f.read()
            np_array = np.frombuffer(base64.b64decode(base64_str), dtype=np.float32)
        else:
            np_array = np.load(filename)
        writable_array = np.copy(np_array)
        return writable_array.reshape(-1, 12)
    
    @staticmethod
    def list_files(directory_path: str, extension: Optional[str] = None) -> list:
        """
        List files in a directory.
        
        Args:
            directory_path: Path to the directory
            extension: Optional file extension filter (e.g., '.base64')
            
        Returns:
            List of file paths
        """
        files = [
            os.path.join(directory_path, f) 
            for f in os.listdir(directory_path) 
            if os.path.isfile(os.path.join(directory_path, f))
        ]
        if extension:
            files = [f for f in files if f.endswith(extension)]
        return files
