import os
import numpy as np
import pandas as pd

from torch.utils.data import Dataset
from utils.ddp import DistributedUtils
from utils.constants import lead_to_idx
class ECGDataset(Dataset):
    def __init__(
        self, 
        parquet_file: str = None, 
        expected_waveform_length: int = 2500,
        num_leads: int = 12,
        normalize_waveforms: bool = True,
        lead_stats: dict[str, dict[str, float]] = None
    ):
        self.data = pd.read_parquet(parquet_file)
        self.expected_waveform_length = expected_waveform_length
        self.num_leads = num_leads
        self.normalize_waveforms = normalize_waveforms
        self.lead_stats = lead_stats
        
    def __len__(self):
        return len(self.data)
        
    def load_signal(self, waveform_path: str):
        return np.load(waveform_path)
    
    def load_signal(self, waveform_path: str):
        return np.load(waveform_path)
    
    def __getitem__(self, idx):
        
        try:
            if not os.path.exists(self.data.iloc[idx]['waveform_path']):
                return self.__getitem__((idx + 1) % len(self))
            
            unnormalized_signal: np.ndarray = self.load_signal(waveform_path=self.data.iloc[idx]['waveform_path'])
            
            # Hack for MHI dataset stored as 3D array with shape (2500, 12, 1)
            if len(unnormalized_signal.shape) == 3:
                unnormalized_signal = unnormalized_signal.squeeze(-1)
            
            assert len(unnormalized_signal.shape) == 2, f"Unnormalized signal has shape {unnormalized_signal.shape}"
            
            if np.isnan(unnormalized_signal).any():
                return self.__getitem__((idx + 1) % len(self))
            
            current_length: int = unnormalized_signal.shape[0]
            if current_length > self.expected_waveform_length:
                step: int = unnormalized_signal.shape[0] // self.expected_waveform_length
                unnormalized_signal = unnormalized_signal[::step, :]
            
            if unnormalized_signal.shape[0] != self.expected_waveform_length:
                return self.__getitem__((idx + 1) % len(self))
            
            if unnormalized_signal.shape[1] != self.num_leads:
                return self.__getitem__((idx + 1) % len(self))
            
            # signal_min: float = unnormalized_signal.min()
            # signal_max: float = unnormalized_signal.max()
            # signal_range: float = signal_max - signal_min
            
            # if signal_range == 0:
            #     return self.__getitem__((idx + 1) % len(self))
            
            # Only normalize if flag is set and we have lead statistics
            try:
                if self.normalize_waveforms and self.lead_stats is not None:
                    signal = np.zeros_like(unnormalized_signal)
                    # Normalize each lead separately using its statistics
                    for lead_name, lead_idx in lead_to_idx.items():
                        mean = self.lead_stats[lead_name]["mean"]
                        std = self.lead_stats[lead_name]["std"]
                        signal[:, lead_idx] = (unnormalized_signal[:, lead_idx] - mean) / std
                        
                    # # Min-max normalization to scale between -1 and 1
                    # signal_min = signal.min()
                    # signal_max = signal.max()
                    # signal = 2 * (signal - signal_min) / (signal_max - signal_min) - 1
                else:
                    signal = unnormalized_signal
            except Exception as e:
                print(f"Error normalizing signal: {e}")
                raise Exception(f"Error normalizing signal: {e}")

            return {
                'signal': np.transpose(signal, (1, 0)), 
                'waveform_path': self.data.iloc[idx]['waveform_path']
            }
        
        except Exception as e:
            print(f"Error processing index {self.data.iloc[idx]['waveform_path']}: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))


def get_distributed_ecg_dataloader(
    parquet_file: str,
    expected_waveform_length: int = 2500,
    num_leads: int = 12,
    normalize_waveforms: bool = True,
    lead_stats: dict[str, dict[str, float]] = None,
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
        lead_stats=lead_stats
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