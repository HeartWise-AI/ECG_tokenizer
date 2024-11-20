import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from sklearn.model_selection import train_test_split
import yaml

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)


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
    def __init__(self, parquet_file=None, csv_file=None, transform=None, split='train', test_size=0.01, random_state=config["training"]["seed"]):
        self.transform = transform
        self.split = split

        if parquet_file:
            # Load data from parquet file for the first dataset
            self.data_frame = pd.read_parquet(parquet_file)
        elif csv_file:
            # Load data from csv file for the MIMIC-IV dataset
            self.data_frame = pd.read_csv(csv_file)
            self.data_frame = self._random_split(test_size, random_state)

        self.waveform_length, self.leads = self.get_signal(1).shape

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

    def get_signal(self, idx):
        if 'npy_path' in self.data_frame.columns:
            # Load data for the first dataset (with .npy files)
            npy_path = os.path.join(self.root_dir, self.data_frame.iloc[idx]['npy_path'])
            unnormalized_signal = np.load(npy_path)
        elif 'waveform_path' in self.data_frame.columns:
            # Load data for the MIMIC-IV dataset (with full waveform paths)
            waveform_path = self.data_frame.iloc[idx]['waveform_path']
            unnormalized_signal = np.load(waveform_path)  # Loading the .npy file from the full path
        return unnormalized_signal

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        unnormalized_signal = self.get_signal(idx)
        
        # Check for NaN values in the signal
        if np.isnan(unnormalized_signal).any():
            return self.__getitem__((idx + 1) % len(self))  # Skip this sample if NaN values are found


        epsilon = 1e-8  # Small value to avoid division by zero in cases where the unnormalized_signal.max() and the unnormalized_signal.min() values are the same
        signal = (unnormalized_signal - unnormalized_signal.min()) / (unnormalized_signal.max() - unnormalized_signal.min() + epsilon) * 2 - 1
    
        sample = {'signal': signal}

        # if self.transform:
        #     sample['signal'] = self.transform(sample['signal'])

        # sample['signal'] = sample['signal'].permute(1, 0)
        return sample

class ECGDatasetLLM(Dataset):
    def __init__(self, parquet_file=None, csv_file=None, model=None, transform=None, split='train', test_size=0.2, random_state=config["training"]["seed"]):
        self.transform = transform
        self.split = split
        self.model = model

        if parquet_file:
            # Load data from parquet file for the first dataset
            self.data_frame = pd.read_parquet(parquet_file)
        elif csv_file:
            # Load data from csv file for the MIMIC-IV dataset
            self.data_frame = pd.read_csv(csv_file)
            self.data_frame = self._random_split(test_size, random_state)

        self.waveform_length, self.leads = self.get_signal(1).shape

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


    def get_signal(self, idx):
        if 'npy_path' in self.data_frame.columns:
            # Load data for the first dataset (with .npy files)
            npy_path = os.path.join(self.root_dir, self.data_frame.iloc[idx]['npy_path'])
            unnormalized_signal = np.load(npy_path)
        elif 'waveform_path' in self.data_frame.columns:
            # Load data for the MIMIC-IV dataset (with full waveform paths)
            waveform_path = self.data_frame.iloc[idx]['waveform_path']
            unnormalized_signal = np.load(waveform_path)  # Loading the .npy file from the full path

        return unnormalized_signal
    
    def get_report(self, idx):
        return self.data_frame.iloc[idx]['report']
    
    def set_model(self, model):
        """Set the model only if it hasn't been set before."""
        if self.model is not None:
            raise ValueError("Model has already been set.")
        self.model = model
    
    # get VQ embeddings for the ECG signal
    def encode_ecg_to_tokens(self, ecg_signal):
        if self.model is None:
            raise ValueError("Model has not been set in the dataset.")
        
        with torch.no_grad():
            _, encoded_tokens, _ = self.model(ecg_signal) 
        return encoded_tokens

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        unnormalized_signal = self.get_signal(idx)
        report = self.get_report(idx)
        
        # Check for NaN values in the signal
        if np.isnan(unnormalized_signal).any():
            return self.__getitem__((idx + 1) % len(self))  # Skip this sample if NaN values are found


        epsilon = 1e-8  # Small value to avoid division by zero in cases where the unnormalized_signal.max() and the unnormalized_signal.min() values are the same
        signal = (unnormalized_signal - unnormalized_signal.min()) / (unnormalized_signal.max() - unnormalized_signal.min() + epsilon) * 2 - 1
        signal = torch.from_numpy(signal).float()
        signal = signal.permute(1, 0).unsqueeze(0)

        device = next(self.model.parameters()).device
        signal = signal.to(device)
        
        encoded_tokens = self.encode_ecg_to_tokens(signal)
        # sample = {'signal': signal, 'report': report}

        return encoded_tokens, report