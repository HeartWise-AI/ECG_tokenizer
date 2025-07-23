import os
import torch
import numpy as np
import pandas as pd

from utils.ddp import DistributedUtils
from transformers import GPT2Tokenizer, BatchEncoding
from torch.utils.data import Dataset, DataLoader, default_collate
from utils.config.llm_finetuning_config import LLMFinetuningConfig
from models.types import AutoTokenizerT


class ECGClinicalReportDataset(Dataset):
    def __init__(
        self, 
        dataset_path: str, 
        signal_path_column: str,
        ecg_waveform_length: int,
        ecg_num_leads: int,
        tokenizer: AutoTokenizerT, 
        max_length: int = 512
    ):
        """
        Args:
            dataset_path (str): Path to the dataset.
            tokenizer (PreTrainedTokenizer): Tokenizer for the clinical reports.
            max_length (int): Maximum token length for the reports.
        """
        try:
            self.df: pd.DataFrame = pd.read_parquet(dataset_path)
        except Exception as e:
            print(f"Error reading parquet file: {e}")
            raise Exception(f"Error reading parquet file: {e}")
        
        self.ecg_waveform_length: int = ecg_waveform_length
        self.ecg_num_leads: int = ecg_num_leads
        self.tokenizer: AutoTokenizerT = tokenizer
        self.max_length: int = max_length
        self.signal_path_column: str = signal_path_column
        
    def __len__(self):
        return len(self.df)

    def load_ecg_signal(self, waveform_path: str) -> np.ndarray:
        try:
            waveform: np.ndarray = np.load(waveform_path)
        except Exception as e:
            print(f"Error loading ECG signal: {e}")
            raise Exception(f"Error loading ECG signal: {e}")
        
        # Hack for MHI dataset stored as 3D array with shape (2500, 12, 1)
        if len(waveform.shape) == 3:
            waveform = waveform.squeeze(-1)
            
        assert len(waveform.shape) == 2, f"Unnormalized signal has shape {waveform.shape}"
        
        return waveform

    def __getitem__(self, idx: int) -> dict | None:
        try:
            # Get the row
            row = self.df.iloc[idx]
            
            # Check if the waveform path or report is missing
            if pd.isnull(row[self.signal_path_column]) or pd.isnull(row['report']):
                print(f"Missing waveform_path or report for index {idx}, skipping sample. "
                      f"waveform_path: {row.get(self.signal_path_column)}, report: {row.get('report')}")
                return self.__getitem__((idx + 1) % len(self))

            # Load the waveform
            waveform: np.ndarray = self.load_ecg_signal(
                waveform_path=row[self.signal_path_column]
            )
            
            if np.isnan(waveform).any():
                return self.__getitem__((idx + 1) % len(self))
            
            current_length: int = waveform.shape[0]
            if current_length > self.ecg_waveform_length:
                step: int = waveform.shape[0] // self.ecg_waveform_length
                waveform = waveform[::step, :]
            
            if waveform.shape[0] != self.ecg_waveform_length:
                return self.__getitem__((idx + 1) % len(self))
            
            if waveform.shape[1] != self.ecg_num_leads:
                return self.__getitem__((idx + 1) % len(self))
            
            # Tokenize the report
            encoding: BatchEncoding = self.tokenizer.encode_plus(
                row['report'],
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )          
            input_ids: torch.Tensor = encoding['input_ids'].squeeze()  # Shape: (max_length)
            attention_mask: torch.Tensor = encoding['attention_mask'].squeeze()  # Shape: (max_length)
            
            return {
                'signal': np.transpose(waveform, (1, 0)),  # (2500, 12)
                'input_ids': input_ids,  # (max_length)
                'attention_mask': attention_mask,  # (max_length)
                'waveform_name': row['waveform_name']
            }
            
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            return None
            
            
def get_clinical_report_dataloader(
    config: LLMFinetuningConfig,
    shuffle: bool = True,
    pin_memory: bool = True
):
    dataset: ECGClinicalReportDataset = ECGClinicalReportDataset(
        dataset_path=config.train_dataset_path, 
        signal_path_column=config.signal_path_column,
        ecg_waveform_length=config.ecg_waveform_length,
        ecg_num_leads=config.ecg_num_leads,
        tokenizer=config.tokenizer, 
        max_length=config.max_length
    )
    return DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=shuffle, 
        num_workers=config.num_workers, 
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn
    )
    
def get_distributed_clinical_report_dataloader(
    dataset_path: str,
    signal_path_column: str,
    ecg_waveform_length: int,
    ecg_num_leads: int,
    tokenizer: AutoTokenizerT,
    max_token_length: int = 512,
    batch_size: int = 32,
    num_workers: int = 16,
    num_replicas: int = 1,
    rank: int = 0,
    shuffle: bool = True,
    pin_memory: bool = True
):
    dataset: ECGClinicalReportDataset = ECGClinicalReportDataset(
        dataset_path=dataset_path, 
        signal_path_column=signal_path_column,
        ecg_waveform_length=ecg_waveform_length,
        ecg_num_leads=ecg_num_leads,
        tokenizer=tokenizer, 
        max_length=max_token_length
    )
    
    return DistributedUtils.get_distributed_dataloader(
        dataset=dataset,
        batch_size=batch_size, 
        num_workers=num_workers, 
        pin_memory=pin_memory, 
        num_replicas=num_replicas, 
        rank=rank,
        shuffle=shuffle,
        collate_fn=custom_collate_fn
    )

def custom_collate_fn(batch):
    """
    Custom collate function which filters out any None items in the batch.
    """
    filtered_batch = [item for item in batch if item is not None]
    if len(filtered_batch) == 0:
        raise ValueError("All items in the batch were invalid. Check dataset integrity or file paths.")
    return default_collate(filtered_batch)
