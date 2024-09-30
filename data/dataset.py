import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from sklearn.model_selection import train_test_split


"""
Dataset class to load the MHI or MIMIC-IV data
Args:
            parquet_file (string): Path to the .parquet file (for the first dataset).
            csv_file (string): Path to the .csv file (for the MIMIC-IV dataset).
            root_dir (string): Directory with all the waveform files or numpy files.
            transform (callable, optional): Optional transform to be applied on a sample.
            split (string): 'train' or 'test' split.
            test_size (float): Proportion of the data to use as test data (for MIMIC-IV dataset).
            random_state (int): Random state for train-test split reproducibility.

Author: Rohan Banerjee
"""

class ECGDataset(Dataset):
    def __init__(self, parquet_file=None, csv_file=None, transform=None, split='train', test_size=0.2, random_state=42):
        self.transform = transform
        self.split = split

        if parquet_file:
            # Load data from parquet file for the first dataset
            self.data_frame = pd.read_parquet(parquet_file)
        elif csv_file:
            # Load data from csv file for the MIMIC-IV dataset
            self.data_frame = pd.read_csv(csv_file)
            self.data_frame = self._random_split(test_size, random_state)

    def _random_split(self, test_size, random_state):
        """
        Perform a random train-test split (MIMIC-IV dataset).
        """
        # Random train-test split
        train_df, test_df = train_test_split(self.data_frame, test_size=test_size, random_state=random_state)

        if self.split == 'train':
            return train_df
        else:
            return test_df


    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if 'npy_path' in self.data_frame.columns:
            # Load data for the first dataset (with .npy files)
            npy_path = os.path.join(self.root_dir, self.data_frame.iloc[idx]['npy_path'])
            signal = np.load(npy_path)
        elif 'waveform_path' in self.data_frame.columns:
            # Load data for the MIMIC-IV dataset (with full waveform paths)
            waveform_path = self.data_frame.iloc[idx]['waveform_path']
            signal = np.load(waveform_path)  # Loading the .npy file from the full path

        sample = {'signal': signal}

        if self.transform:
            sample['signal'] = self.transform(sample['signal'])

        return sample
