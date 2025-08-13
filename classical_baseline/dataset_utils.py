"""
Enhanced ECG Dataset Loader for Classical Baseline

This module extends the ECG_tokenizer dataset infrastructure to support
filtering for specific datasets (e.g., MIMIC only) from mixed datasets.
"""

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import sys
sys.path.append('/volume/ECG_tokenizer')

from data.ecg_tokenizer_classifier_dataset import ECGTokenizerClassifierDataset
from utils.constants import ECG_PATTERNS, lead_to_idx


class FilteredECGDataset(ECGTokenizerClassifierDataset):
    """
    Extended dataset class that can filter data by dataset type
    """
    
    def __init__(
        self, 
        parquet_file: str, 
        expected_waveform_length: int,
        num_leads: int,
        normalize_waveforms: bool,
        lead_stats: dict[str, dict[str, float]],
        signal_path_column: str = 'waveform_path_psa',
        dataset_filter: str = None,
        dataset_column: str = 'dataset'
    ):
        # Initialize parent class
        super().__init__(
            parquet_file=parquet_file,
            expected_waveform_length=expected_waveform_length,
            num_leads=num_leads,
            normalize_waveforms=normalize_waveforms,
            lead_stats=lead_stats,
            signal_path_column=signal_path_column
        )
        
        self.dataset_filter = dataset_filter
        self.dataset_column = dataset_column
        
        # Apply dataset filtering if specified
        if self.dataset_filter and self.dataset_column in self.data.columns:
            original_size = len(self.data)
            
            # Filter for specific dataset (case-insensitive)
            mask = self.data[self.dataset_column].str.contains(
                self.dataset_filter, case=False, na=False
            )
            self.data = self.data[mask].reset_index(drop=True)
            
            filtered_size = len(self.data)
            print(f"Dataset filtered for '{self.dataset_filter}': {filtered_size}/{original_size} samples")
            
        elif self.dataset_filter:
            print(f"Warning: Dataset column '{self.dataset_column}' not found, using all data")
        
        print(f"Final dataset size: {len(self.data)} samples")


def create_mimic_dataset(
    parquet_file: str,
    expected_waveform_length: int = 5000,
    num_leads: int = 12,
    normalize_waveforms: bool = True,
    lead_stats: dict = None,
    signal_path_column: str = 'waveform_path_psa'
) -> FilteredECGDataset:
    """
    Create a dataset filtered for MIMIC data only
    """
    return FilteredECGDataset(
        parquet_file=parquet_file,
        expected_waveform_length=expected_waveform_length,
        num_leads=num_leads,
        normalize_waveforms=normalize_waveforms,
        lead_stats=lead_stats,
        signal_path_column=signal_path_column,
        dataset_filter='MIMIC',
        dataset_column='dataset'
    )
