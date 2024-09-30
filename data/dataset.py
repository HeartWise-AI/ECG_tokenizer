"""
Dataset class to load the MHI data (for now)
Args:
            parquet_file (string): Path to the .parquet file.
            root_dir (string): Directory with all the numpy files.
            transform (callable, optional): Optional transform to be applied
                                            on an ECG signal sample.


Author: Rohan Banerjee
"""


import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os

class ECGDataset(Dataset):
    def __init__(self, parquet_file, root_dir, transform=None):
        self.data_frame = pd.read_parquet(parquet_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Fetch the path to the numpy file
        npy_path = os.path.join(self.root_dir, self.data_frame.iloc[idx]['npy_path'])
        
        # Load the numpy ECG signal
        ecg_signal = np.load(npy_path)

        sample = {'ecg_signal': ecg_signal}

        if self.transform:
            sample['ecg_signal'] = self.transform(sample['ecg_signal'])

        return sample
