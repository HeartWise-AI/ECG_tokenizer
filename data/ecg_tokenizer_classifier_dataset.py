import numpy as np
import pandas as pd

from torch.utils.data import Dataset

from utils.ddp import DistributedUtils
from utils.constants import ECG_PATTERNS, lead_to_idx


class ECGTokenizerClassifierDataset(Dataset):
    def __init__(
        self, 
        parquet_file: str, 
        expected_waveform_length: int,
        num_leads: int,
        normalize_waveforms: bool,
        lead_stats: dict[str, dict[str, float]],
        signal_path_column: str = 'waveform_path_psa'
    ):
        self.data: pd.DataFrame = pd.read_parquet(parquet_file)
        self.expected_waveform_length: int = expected_waveform_length
        self.num_leads: int = num_leads
        self.normalize_waveforms: bool = normalize_waveforms
        self.lead_stats: dict[str, dict[str, float]] = lead_stats
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
    
    def load_signal(self, waveform_path: str):
        return np.load(waveform_path)
    
    def __getitem__(self, index: int):
        row: pd.Series = self.data.iloc[index]
        
        unnormalized_signal: np.ndarray = self.load_signal(
            waveform_path=row[self.signal_path_column]
        ).astype(np.float32)

        # Hack for MHI dataset stored as 3D array with shape (2500, 12, 1)
        if len(unnormalized_signal.shape) == 3:
            unnormalized_signal = unnormalized_signal.squeeze(-1)
        
        assert len(unnormalized_signal.shape) == 2, f"Unnormalized signal has shape {unnormalized_signal.shape}"
        
        current_length: int = unnormalized_signal.shape[0]
        if current_length > self.expected_waveform_length:
            step: int = unnormalized_signal.shape[0] // self.expected_waveform_length
            unnormalized_signal = unnormalized_signal[::step, :]
                
        if self.normalize_waveforms and self.lead_stats is not None:
            signal: np.ndarray = np.zeros_like(unnormalized_signal)
            # Normalize each lead separately using its statistics
            for lead_name, lead_idx in lead_to_idx.items():
                mean: float = self.lead_stats[lead_name]["mean"]
                std: float = self.lead_stats[lead_name]["std"]
                signal[:, lead_idx] = (unnormalized_signal[:, lead_idx] - mean) / std       
        else:
            signal: np.ndarray = unnormalized_signal
        
        labels: np.ndarray = row[ECG_PATTERNS].to_numpy().astype(np.float32)
        return {
            'signal': np.transpose(signal, (1, 0)), 
            'labels': labels
        }
    
def get_distributed_ecg_tokenizer_classifier_dataloader(
    parquet_file: str,
    expected_waveform_length: int,
    num_leads: int,
    normalize_waveforms: bool,
    lead_stats: dict[str, dict[str, float]],
    batch_size: int,
    num_workers: int,
    num_replicas: int,
    rank: int,
    shuffle: bool,
    pin_memory: bool,
    signal_path_column: str = 'waveform_path_psa'
):
    dataset: ECGTokenizerClassifierDataset = ECGTokenizerClassifierDataset(
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
        