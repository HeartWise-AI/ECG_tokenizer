import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from utils.ddp import DistributedUtils
from utils.constants import lead_to_idx

class ECGDataset(Dataset):
    def __init__(
        self, 
        parquet_file: str, 
        expected_waveform_length: int = 2500,
        num_leads: int = 12,
        normalize_waveforms: bool = True,
        lead_stats: dict[str, dict[str, float]] | None = None,
        signal_path_column: str = 'waveform_path_psa'
    ):
        try:
            self.data: pd.DataFrame = pd.read_parquet(parquet_file)
        except Exception as e:
            print(f"Error reading parquet file: {e}")
            raise Exception(f"Error reading parquet file: {e}")
        
        self.expected_waveform_length: int = expected_waveform_length
        self.num_leads: int = num_leads
        self.normalize_waveforms: bool = normalize_waveforms
        self.lead_stats: dict[str, dict[str, float]] | None = lead_stats
        self.signal_path_column: str = signal_path_column
        
        if self.normalize_waveforms and self.lead_stats is None:
            raise ValueError("lead_stats must be provided if normalize_waveforms is True") 
        
        if self.normalize_waveforms and self.lead_stats is not None:
            print(f"Normalizing waveforms with lead statistics: {self.lead_stats}")
        elif not self.normalize_waveforms and self.lead_stats is not None:
            print(f"Not normalizing waveforms")
        else:
            print(f"Not normalizing waveforms")
            
    def __len__(self):
        return len(self.data)
        
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

    def __getitem__(self, idx):
        try:
            if not os.path.exists(self.data.iloc[idx][self.signal_path_column]):
                return self.__getitem__((idx + 1) % len(self))

            waveform: np.ndarray = self.load_ecg_signal(
                waveform_path=self.data.iloc[idx][self.signal_path_column]
            )
            
            if np.isnan(waveform).any():
                return self.__getitem__((idx + 1) % len(self))
            
            current_length: int = waveform.shape[0]
            if current_length > self.expected_waveform_length:
                step: int = waveform.shape[0] // self.expected_waveform_length
                waveform = waveform[::step, :]
            
            if waveform.shape[0] != self.expected_waveform_length:
                return self.__getitem__((idx + 1) % len(self))
            
            if waveform.shape[1] != self.num_leads:
                return self.__getitem__((idx + 1) % len(self))
            
            # Only normalize if flag is set and we have lead statistics
            try:
                if self.normalize_waveforms and self.lead_stats is not None:
                    signal = np.zeros_like(waveform)
                    # Normalize each lead separately using its statistics
                    for lead_name, lead_idx in lead_to_idx.items():
                        mean = self.lead_stats[lead_name]["mean"]
                        std = self.lead_stats[lead_name]["std"]
                        signal[:, lead_idx] = (waveform[:, lead_idx] - mean) / std
                        
                else:
                    signal = waveform
            except Exception as e:
                print(f"Error normalizing signal: {e}")
                raise Exception(f"Error normalizing signal: {e}")

            return {
                'signal': np.transpose(signal, (1, 0)), 
                'waveform_path': self.data.iloc[idx][self.signal_path_column]
            }
        
        except Exception as e:
            print(f"Error processing index {self.data.iloc[idx][self.signal_path_column]}: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))


def get_distributed_ecg_dataloader(
    parquet_file: str,
    expected_waveform_length: int = 2500,
    num_leads: int = 12,
    normalize_waveforms: bool = True,
    lead_stats: dict[str, dict[str, float]] | None = None,
    signal_path_column: str = 'waveform_path_psa',
    batch_size: int = 32,
    num_workers: int = 16,
    num_replicas: int = 1,
    rank: int = 0,
    shuffle: bool = True,
    pin_memory: bool = True
):
    dataset: ECGDataset = ECGDataset(
        parquet_file=parquet_file,
        expected_waveform_length=expected_waveform_length,
        num_leads=num_leads,
        normalize_waveforms=normalize_waveforms,
        lead_stats=lead_stats,
        signal_path_column=signal_path_column
    )
    
    return DistributedUtils.get_distributed_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        pin_memory=pin_memory
    )