import os
import csv
import json
import struct
import base64
import yaml
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd

try:
    import wfdb
except ImportError:
    wfdb = None


def load_yaml(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


# ---------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------

def load_df(path: str) -> pd.DataFrame:
    """
    Load a DataFrame from CSV or Parquet file.
    """
    if path.endswith('.csv'):
        return pd.read_csv(path)
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    raise ValueError("Unsupported file extension. Only .csv and .parquet are supported.")


def save_df(df: pd.DataFrame, path: str) -> None:
    """
    Save a DataFrame to CSV or Parquet file.
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if path.endswith('.csv'):
        df.to_csv(path, index=False)
    elif path.endswith('.parquet'):
        df.to_parquet(path, index=False)
    else:
        raise ValueError("Unsupported file extension. Only .csv and .parquet are supported.")

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


# ---------------------------------------------------------------------
# ECG signal IO
# ---------------------------------------------------------------------

LEAD_ORDER = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]


class ECGFileHandler:
    """
    Handler for ECG signal file operations.
    
    Supports:
    - .npy: preferred storage
    - .hea: WFDB header files (read-only)
    - .xml: CLSA and GE MUSE XML formats (read-only)
    """
    
    @staticmethod
    def save_ecg_signal(ecg_signal: np.ndarray, filename: str) -> None:
        """
        Save ECG signal to .npy
        """
        parent_dir = os.path.dirname(filename)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        np.save(filename, ecg_signal.astype(np.float32))
    
    @staticmethod
    def load_ecg_signal(filename: str) -> np.ndarray:
        """
        Load ECG signal from .npy, .hea (WFDB), or .xml (CLSA/GE MUSE).
        Returns shape (N, 12) float32.
        """
        filename = str(filename)
        if filename.endswith('.hea'):
            if wfdb is None:
                raise ImportError("wfdb is required to read .hea files")
            record_path = filename[:-4]
            record = wfdb.rdrecord(record_path)
            np_array = record.p_signal
        elif filename.endswith('.xml'):
            return ECGFileHandler._load_xml_signal(filename)
        else:
            np_array = np.load(filename, allow_pickle=False)
        writable_array = np.copy(np_array)
        return writable_array.reshape(-1, 12)

    @staticmethod
    def load_ecg_signal_raw(filename: str) -> np.ndarray:
        """
        Load ECG signal preserving its original shape (no reshape).

        Use this when the caller handles shape normalization (e.g.
        AnalysisPipeline._canonicalize_signal which correctly transposes
        (12, N) arrays instead of reshaping them).
        """
        filename = str(filename)
        if filename.endswith('.hea'):
            if wfdb is None:
                raise ImportError("wfdb is required to read .hea files")
            record_path = filename[:-4]
            record = wfdb.rdrecord(record_path)
            return np.copy(record.p_signal)
        elif filename.endswith('.xml'):
            return ECGFileHandler._load_xml_signal(filename)
        else:
            return np.copy(np.load(filename, allow_pickle=False))
    
    @staticmethod
    def list_files(directory_path: str, extension: Optional[str] = None) -> list:
        files = [
            os.path.join(directory_path, f)
            for f in os.listdir(directory_path)
            if os.path.isfile(os.path.join(directory_path, f))
        ]
        if extension:
            files = [f for f in files if f.endswith(extension)]
        return files

    @staticmethod
    def _parse_xml_to_dict(element):
        if len(element) == 0:
            return element.text
        result = {}
        for child in element:
            child_result = ECGFileHandler._parse_xml_to_dict(child)
            if child.tag in result:
                if not isinstance(result[child.tag], list):
                    result[child.tag] = [result[child.tag]]
                result[child.tag].append(child_result)
            else:
                result[child.tag] = child_result
        return result

    @staticmethod
    def _flatten_dict(d, parent_key=''):
        items = []
        if isinstance(d, dict):
            for k, v in d.items():
                new_key = f'{parent_key}.{k}' if parent_key else k
                items.extend(ECGFileHandler._flatten_dict(v, new_key).items())
        elif isinstance(d, list):
            for i, item in enumerate(d):
                items.extend(ECGFileHandler._flatten_dict(item, f'{parent_key}.{i}').items())
        else:
            items.append((parent_key, d))
        return dict(items)

    @staticmethod
    def _decode_base64_waveform(raw_wave: str) -> np.ndarray:
        arr = base64.b64decode(bytes(raw_wave, "utf-8"))
        byte_array = struct.unpack(f"{len(arr) // 2}h", arr)
        return np.array(byte_array, dtype=np.float32)

    @staticmethod
    def _derive_missing_leads(leads: dict[str, Any]) -> None:
        if leads["III"] is None:
            leads["III"] = np.subtract(leads["II"], leads["I"])
        if leads["aVR"] is None:
            leads["aVR"] = np.add(leads["I"], leads["II"]) * (-0.5)
        if leads["aVL"] is None:
            leads["aVL"] = np.subtract(leads["I"], 0.5 * leads["II"])
        if leads["aVF"] is None:
            leads["aVF"] = np.subtract(leads["II"], 0.5 * leads["I"])

    @staticmethod
    def _extract_leads_clsa(data_dict: dict) -> np.ndarray:
        leads: dict[str, Any] = {lead: None for lead in LEAD_ORDER}
        lead_order_str = data_dict['RestingECGMeasurements.MeasurementTable.LeadOrder']
        for i, lead in enumerate(lead_order_str.replace(' ', '').split(',')):
            raw = data_dict[f'StripData.WaveformData.{i}'].lstrip('\t').split(',')
            leads[lead] = np.array(raw, dtype=float)

        ECGFileHandler._derive_missing_leads(leads)

        non_empty_dim = next(l.shape[0] for l in leads.values() if l is not None)
        for lead in leads:
            if leads[lead] is None:
                leads[lead] = np.full(non_empty_dim, np.nan)
        return np.vstack([leads[lead] for lead in LEAD_ORDER])

    @staticmethod
    def _extract_leads_muse(data_dict: dict) -> np.ndarray:
        leads: dict[str, Any] = {lead: None for lead in LEAD_ORDER}
        for i in range(12):
            key = f'Waveform.1.LeadData.{i}.LeadID'
            if key in data_dict:
                wave = data_dict[f'Waveform.1.LeadData.{i}.WaveFormData']
                leads[data_dict[key]] = ECGFileHandler._decode_base64_waveform(wave)

        ECGFileHandler._derive_missing_leads(leads)

        non_empty_dim = next(l.shape[0] for l in leads.values() if l is not None)
        for lead in leads:
            if leads[lead] is None:
                leads[lead] = np.full(non_empty_dim, np.nan)
        return np.vstack([leads[lead] for lead in LEAD_ORDER])

    @staticmethod
    def _load_xml_signal(filename: str) -> np.ndarray:
        """
        Parse a CLSA or GE MUSE XML file and return shape (samples, 12).
        Raises ValueError for unsupported XML formats.
        """
        tree = ET.parse(filename)
        data_dict = ECGFileHandler._flatten_dict(
            ECGFileHandler._parse_xml_to_dict(tree.getroot())
        )

        if 'RestingECGMeasurements.MeasurementTable.LeadOrder' in data_dict:
            leads = ECGFileHandler._extract_leads_clsa(data_dict)
        elif any(f'Waveform.1.LeadData.{i}.LeadID' in data_dict for i in range(12)):
            leads = ECGFileHandler._extract_leads_muse(data_dict)
        else:
            raise ValueError(
                f"Unsupported ECG XML format in {filename}. "
                "Pre-convert to .npy or add a handler in ECGFileHandler."
            )

        return leads.T
