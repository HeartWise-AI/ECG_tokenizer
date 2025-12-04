"""
Configuration for Classical ECG Baseline

This module contains configuration settings for the classical baseline pipeline
that integrates with the ECG_tokenizer infrastructure.
"""

import os
from pathlib import Path
from typing import Dict, Any
import yaml

# Default paths within ECG_tokenizer structure
BASE_DIR = Path('/volume/ECG_tokenizer')
CLASSICAL_BASELINE_DIR = BASE_DIR / 'classical_baseline'
OUTPUT_DIR = CLASSICAL_BASELINE_DIR / 'output'
CONFIG_DIR = CLASSICAL_BASELINE_DIR / 'config'

# Create directories if they don't exist
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CONFIG_DIR.mkdir(parents=True, exist_ok=True)

# Default configuration
DEFAULT_CONFIG = {
    'data': {
        'expected_waveform_length': 2500,
        'num_leads': 12,
        'normalize_waveforms': True,
        'signal_path_column': 'waveform_path_psa',  # Changed from 'waveform_path_psa' to match MIMIC dataset
        'test_size': 0.2,
        'validation_size': 0.1,
        'random_state': 42
    },
    'signal_processing': {
        'wavelet': {
            'name': 'db4',
            'levels': 6
        },
        'emd': {
            'max_imf': 8
        },
        'vmd': {
            'alpha': 2000,
            'tau': 0,
            'K': 8,
            'DC': 0,
            'init': 1,
            'tol': 1e-7
        },
        'sax': {
            'n_segments': 20,
            'alphabet_size': 26,  # Changed to 26 for full alphabet
            'bow_window_size': 12,
            'bow_word_size': 4,
            'bow_n_bins': 10
        }
    },
    'machine_learning': {
        'cv_folds': 5,
        'n_jobs': -1,
        'classifiers': {
            'random_forest': {
                'n_estimators': 100,
                'max_depth': None,
                'min_samples_split': 2,
                'min_samples_leaf': 1
            },
            'gradient_boosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 3
            },
            'svm': {
                'kernel': 'rbf',
                'C': 1.0,
                'gamma': 'scale'
            },
            'logistic_regression': {
                'max_iter': 1000,
                'C': 1.0,
                'solver': 'lbfgs'
            }
        }
    },
    'output': {
        'save_features': True,
        'save_models': True,
        'save_predictions': True,
        'output_dir': str(OUTPUT_DIR)
    }
}

# MIMIC dataset configurations
MIMIC_CONFIGS = {
    'mimic_iv_ecg': {
        'description': 'MIMIC-IV ECG dataset configuration',
        'expected_waveform_length': 5000,
        'sampling_rate': 500,
        'num_leads': 12,
        'patterns': [
            "Sinusal", "Regular", "Bradycardia", "Afib", "Left bundle branch block",
            "Right bundle branch block", "Left ventricular hypertrophy", 
            "Atrial tachycardia (>= 100 BPM)", "Premature ventricular complex"
        ]
    }
}

# Lead statistics (to be updated with actual dataset statistics)
DEFAULT_LEAD_STATS = {
    'I': {'mean': 0.0, 'std': 1.0},
    'II': {'mean': 0.0, 'std': 1.0},
    'III': {'mean': 0.0, 'std': 1.0},
    'aVR': {'mean': 0.0, 'std': 1.0},
    'aVL': {'mean': 0.0, 'std': 1.0},
    'aVF': {'mean': 0.0, 'std': 1.0},
    'V1': {'mean': 0.0, 'std': 1.0},
    'V2': {'mean': 0.0, 'std': 1.0},
    'V3': {'mean': 0.0, 'std': 1.0},
    'V4': {'mean': 0.0, 'std': 1.0},
    'V5': {'mean': 0.0, 'std': 1.0},
    'V6': {'mean': 0.0, 'std': 1.0}
}


def load_config(config_path: str = None) -> Dict[str, Any]:
    """Load configuration from file or return default"""
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    else:
        return DEFAULT_CONFIG


def save_config(config: Dict[str, Any], config_path: str):
    """Save configuration to file"""
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def get_dataset_paths():
    """Get available dataset paths"""
    # Look for common MIMIC dataset locations
    possible_paths = [
        '/volume/ECG_tokenizer/output/MHI/mimic_mhi_psa_train_updated_stratified_20%.parquet',
        '/volume/ECG_tokenizer/output/MIMIC/MIMIC_train_cleaned.parquet',
        '/volume/ECG_tokenizer/output/MIMIC/MIMIC_test_cleaned.parquet',
        '/volume/ECG_tokenizer/data/mimic_preprocessed.parquet',
        '/volume/ECG_tokenizer/output/MIMIC/train.parquet',
        '/volume/ECG_tokenizer/output/MIMIC/test.parquet',
        '/volume/data/mimic_iv_ecg/processed/train.parquet',
        '/volume/data/mimic_iv_ecg/processed/test.parquet'
    ]
    
    existing_paths = []
    for path in possible_paths:
        if os.path.exists(path):
            existing_paths.append(path)
    
    return existing_paths


def detect_lead_stats(parquet_file: str) -> Dict[str, Dict[str, float]]:
    """Detect lead statistics from a sample of the dataset"""
    try:
        import pandas as pd
        import numpy as np
        from classical_baseline import ECGTokenizerClassifierDataset
        
        # Load a sample of the dataset
        dataset = ECGTokenizerClassifierDataset(
            parquet_file=parquet_file,
            expected_waveform_length=5000,
            num_leads=12,
            normalize_waveforms=False,  # Don't normalize to get raw stats
            lead_stats=None
        )
        
        # Sample signals to compute statistics
        sample_size = min(100, len(dataset))
        signals = []
        
        for i in range(sample_size):
            try:
                sample = dataset[i]
                signal = sample['signal']  # Shape: (leads, time)
                signals.append(signal)
            except:
                continue
        
        if not signals:
            print("Warning: Could not load any signals, using default stats")
            return DEFAULT_LEAD_STATS
        
        # Compute lead statistics
        all_signals = np.stack(signals, axis=0)  # Shape: (samples, leads, time)
        lead_stats = {}
        
        from utils.constants import standard_lead_order
        
        for lead_idx, lead_name in enumerate(standard_lead_order):
            if lead_idx < all_signals.shape[1]:
                lead_data = all_signals[:, lead_idx, :].flatten()
                lead_stats[lead_name] = {
                    'mean': float(np.mean(lead_data)),
                    'std': float(np.std(lead_data))
                }
        
        print(f"Computed lead statistics from {sample_size} samples")
        return lead_stats
        
    except Exception as e:
        print(f"Warning: Could not compute lead statistics: {e}")
        return DEFAULT_LEAD_STATS


def create_default_config_file():
    """Create default configuration file"""
    config_file = CONFIG_DIR / 'classical_baseline_config.yaml'
    save_config(DEFAULT_CONFIG, str(config_file))
    print(f"Default configuration saved to: {config_file}")
    return str(config_file)
