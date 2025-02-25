import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import random
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import yaml
from typing import Optional
import argparse
"""
Dataset classed to load the MHI or MIMIC-IV data (signals and labels)
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
    def __init__(
        self, 
        parquet_file: str = None, 
        csv_file: str = None, 
        transform: callable = None, 
        split: str = 'train', 
        test_size: float = 0.0001, 
        random_state: int = 1234, 
        expected_waveform_length: int = 5000, 
        num_leads: int = 12,
        dataset: str = None
    ):
        self.transform: callable = transform
        self.split: str = split
        self.expected_waveform_length: int = expected_waveform_length
        self.num_leads: int = num_leads
        self.dataset: str = dataset
        if parquet_file:
            # Load data from parquet file for the first dataset
            self.data_frame: pd.DataFrame = pd.read_parquet(parquet_file)
        elif csv_file:
            # Load data from csv file for the MIMIC-IV dataset
            self.data_frame: pd.DataFrame = pd.read_csv(csv_file)
            self.data_frame: pd.DataFrame = self._random_split(test_size, random_state)

    def _random_split(
        self, 
        test_size: float, 
        random_state: int
    ) -> pd.DataFrame:
        """
        Perform a random train-test split (MIMIC-IV dataset).
        """
        # Random train-test split
        train_df: pd.DataFrame
        test_df: pd.DataFrame
        train_df, test_df = train_test_split(
            self.data_frame, 
            test_size=test_size, 
            random_state=random_state, 
        )

        if self.split == 'train':
            return train_df
        else:
            return test_df

    def __len__(self) -> int:
        return len(self.data_frame)

    def load_signal(
        self, 
        waveform_path: str
    ) -> np.ndarray:
        return np.load(waveform_path) 

    def __getitem__(
        self, 
        idx: int
    ) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        try:
            unnormalized_signal: np.ndarray = self.load_signal(waveform_path=self.data_frame.iloc[idx]['waveform_path'])
            
            # Hack for MHI dataset stored as 3D array with shape (2500, 12, 1)
            if len(unnormalized_signal.shape) == 3:
                unnormalized_signal = unnormalized_signal.squeeze(-1)
            
            if np.isnan(unnormalized_signal).any():
                return self.__getitem__((idx + 1) % len(self))

            current_length: int = unnormalized_signal.shape[0]
            if current_length < self.expected_waveform_length:
                pad_size: int = self.expected_waveform_length - current_length
                # Pad timesteps dimension at the end
                unnormalized_signal = np.pad(
                    unnormalized_signal, 
                    ((0, pad_size), (0, 0)), 
                    mode='constant', 
                    constant_values=0
                )
            elif current_length > self.expected_waveform_length:
                step: int = unnormalized_signal.shape[0] // self.expected_waveform_length
                unnormalized_signal = unnormalized_signal[::step, :]
            
            if unnormalized_signal.shape[0] != self.expected_waveform_length:
                return self.__getitem__((idx + 1) % len(self))

            if unnormalized_signal.shape[1] != self.num_leads:
                return self.__getitem__((idx + 1) % len(self))
            
            epsilon: float = 1e-8
            signal_min: float = unnormalized_signal.min()
            signal_max: float = unnormalized_signal.max()
            signal_range: float = signal_max - signal_min
            
            # Skip samples with zero or near-zero range
            if signal_range == 0:
                print(f"Skipping {self.data_frame.iloc[idx]['waveform_path']}: signal has no variation (min={signal_min}, max={signal_max})")
                return self.__getitem__((idx + 1) % len(self))
            
            signal: np.ndarray = (unnormalized_signal - signal_min) / signal_range * 2 - 1

            if self.dataset == 'mimic':
                signal[:, [4, 5]] = signal[:, [5, 4]]

            return {'signal': np.transpose(signal, (1, 0))}
        except Exception as e:
            print(f"Error processing index {self.data_frame.iloc[idx]['waveform_path']}: {str(e)}")
            return self.__getitem__((idx + 1) % len(self))

class ECGDatasetLLM(Dataset):
    def __init__(
        self, 
        embeddings_folder: str = None, 
        model: callable = None, 
        transform: callable = None, 
        split: str = 'train', 
        test_size: float = 0.2, 
        random_state: int = 42
    ):
        self.transform: callable = transform
        self.split: str = split
        self.model: callable = model
    
        if parquet_file:
            self.data_frame: pd.DataFrame = pd.read_parquet(parquet_file)
        elif csv_file:
            self.data_frame: pd.DataFrame = pd.read_csv(csv_file)
            self.data_frame: pd.DataFrame = self._random_split(test_size, random_state)

        signal_shape: tuple[int, int] = self.get_signal(1).shape
        signal_shape: tuple[int, int] = signal_shape[:-1] if len(signal_shape) == 3 else signal_shape
        self.waveform_length: int = signal_shape[0]
        self.leads: int = signal_shape[1]

    def _random_split(
        self, 
        test_size: float, 
        random_state: int
    ) -> pd.DataFrame:
        """
        Perform a random train-test split (MIMIC-IV dataset).
        """
        train_df: pd.DataFrame
        test_df: pd.DataFrame
        train_df, test_df = train_test_split(self.data_frame, test_size=test_size, random_state=random_state)

        if self.split == 'train':
            return train_df
        else:
            return test_df

    def __len__(self) -> int:
        return len(self.data_frame)


    def get_signal(
        self, 
        idx: int
    ) -> np.ndarray:
        if 'npy_path' in self.data_frame.columns:
            npy_path: str = os.path.join(self.root_dir, self.data_frame.iloc[idx]['npy_path'])
            unnormalized_signal: np.ndarray = np.load(npy_path)
        elif 'waveform_path' in self.data_frame.columns:
            waveform_path: str = self.data_frame.iloc[idx]['waveform_path']
            unnormalized_signal: np.ndarray = np.load(waveform_path)

        return unnormalized_signal
    
    def get_report(
        self, 
        idx: int
    ) -> str:
        return self.data_frame.iloc[idx]['report']
    
    def set_model(
        self, 
        model: callable
    ):
        """Set the model only if it hasn't been set before."""
        if self.model is not None:
            raise ValueError("Model has already been set.")
        self.model = model
    
    def encode_ecg_to_tokens(
        self, 
        ecg_signal: np.ndarray
    ) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model has not been set in the dataset.")
        
        with torch.no_grad():
            _, encoded_tokens, _ = self.model(ecg_signal) 
        return encoded_tokens

    def __getitem__(
        self, 
        idx: int
    ) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        unnormalized_signal: np.ndarray = self.get_signal(idx)
        report: str = self.get_report(idx)
        
        if np.isnan(unnormalized_signal).any():
            return self.__getitem__((idx + 1) % len(self))  


        epsilon: float = 1e-8  
        signal: np.ndarray = (unnormalized_signal - unnormalized_signal.min()) / (unnormalized_signal.max() - unnormalized_signal.min() + epsilon) * 2 - 1
        signal: torch.Tensor = torch.from_numpy(signal).float()
        signal: torch.Tensor = signal.permute(1, 0).unsqueeze(0)

        device: torch.device = next(self.model.parameters()).device
        signal: torch.Tensor = signal.to(device)
        
        encoded_tokens: torch.Tensor = self.encode_ecg_to_tokens(signal)

        return encoded_tokens, report
    
class ECGDatasetClassifier(Dataset):
    def __init__(self, 
        parquet_file: str = None, 
        transform: callable = None, 
        split: str = 'train', 
        val_size: float = 0.01, 
        test_size: float = 0.01, 
        random_state: int = 42
    ):
        self.transform: callable = transform
        self.split: str = split

        if parquet_file:
            self.data_frame: pd.DataFrame = pd.read_parquet(parquet_file)
            self.train_df: pd.DataFrame
            self.val_df: pd.DataFrame
            self.test_df: pd.DataFrame
            self.train_df, self.val_df, self.test_df = self._stratified_split(val_size, test_size, random_state)
            if self.split == 'train':
                self.data_frame = self.train_df
            elif self.split == 'val':
                self.data_frame = self.val_df
            elif self.split == 'test':
                self.data_frame = self.test_df

        unnormalized_signal: np.ndarray
        unnormalized_signal, _ = self.get_signal(0)
        self.waveform_length: int = unnormalized_signal.shape[0]
        self.leads: int = unnormalized_signal.shape[1]

    def _random_split(
        self, 
        test_size: float, 
        random_state: int
    ) -> pd.DataFrame:
        """
        Perform a random train-test split (MIMIC-IV dataset).
        """ 
        train_df: pd.DataFrame
        test_df: pd.DataFrame
        train_df, test_df = train_test_split(self.data_frame, test_size=test_size, random_state=random_state)

        if self.split == 'train':
            return train_df
        else:
            return test_df
        
    def _stratified_split(
        self, 
        val_size: float, 
        test_size: float, 
        random_state: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Stratified train-validation-test split based on the label distribution.
        """
        labels: np.ndarray = self.data_frame[[
            'Sinusal', 'Regular', 'Monomorph', 
                'QS complex in V1-V2-V3', 'R complex in V5-V6', 
                'T wave inversion (inferior - II, III, aVF)', 
                'Left bundle branch block', 'RaVL > 11 mm', 'SV1 + RV5 or RV6 > 35 mm', 
                'T wave inversion (lateral -I, aVL, V5-V6)', 'T wave inversion (anterior - V3-V4)', 
                'Left axis deviation', 'Left ventricular hypertrophy', 'Bradycardia', 
                'Q wave (inferior - II, III, aVF)', 'Afib', 'Irregularly irregular', 
                'Atrial tachycardia (>= 100 BPM)', 'Nonspecific intraventricular conduction delay', 
                'Premature ventricular complex', 'Polymorph', 'T wave inversion (septal- V1-V2)', 
                'Right bundle branch block', 'Ventricular paced', 'ST elevation (anterior - V3-V4)', 
                'ST elevation (septal - V1-V2)', '1st degree AV block', 'Premature atrial complex', 
                'Atrial flutter', "rSR' in V1-V2", 'qRS in V5-V6-I, aVL', 
                'Left anterior fascicular block', 'Right axis deviation', '2nd degree AV block - mobitz 1', 
                'ST depression (inferior - II, III, aVF)', 'Acute pericarditis', 
                'ST elevation (inferior - II, III, aVF)', 'Low voltage', 'Regularly irregular', 
                'Junctional rhythm', 'Left atrial enlargement', 'ST elevation (lateral - I, aVL, V5-V6)', 
                'Atrial paced', 'Right ventricular hypertrophy', 'Delta wave', 'Wolff-Parkinson-White (Pre-excitation syndrome)', 
                'Prolonged QT', 'ST depression (anterior - V3-V4)', 'QRS complex negative in III', 
                'Q wave (lateral- I, aVL, V5-V6)', 'Supraventricular tachycardia', 'ST downslopping', 
                'ST depression (lateral - I, avL, V5-V6)', '2nd degree AV block - mobitz 2', 'U wave', 
                'R/S ratio in V1-V2 >1', 'RV1 + SV6 > 11 mm', 'Left posterior fascicular block', 
                'Right atrial enlargement', 'ST depression (septal- V1-V2)', 'Q wave (septal- V1-V2)', 
                'Q wave (anterior - V3-V4)', 'ST upslopping', 'Right superior axis', 'Ventricular tachycardia', 
                'ST elevation (posterior - V7-V8-V9)', 'Ectopic atrial rhythm (< 100 BPM)', 
                'Lead misplacement', 'Third Degree AV Block', 'Acute MI', 'Early repolarization', 
                'Q wave (posterior - V7-V9)', 'Bi-atrial enlargement', 'LV pacing', 'Brugada', 
                'Ventricular Rhythm', 'no_qrs'
        ]].values
        
        stratifier: MultilabelStratifiedShuffleSplit = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(stratifier.split(self.data_frame, labels))
        
        train_val_df: pd.DataFrame = self.data_frame.iloc[train_idx].reset_index(drop=True)
        test_df: pd.DataFrame = self.data_frame.iloc[test_idx].reset_index(drop=True)
        
        stratifier: MultilabelStratifiedShuffleSplit = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
        train_idx, val_idx = next(stratifier.split(train_val_df, labels[train_idx]))
        train_df: pd.DataFrame = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df: pd.DataFrame = train_val_df.iloc[val_idx].reset_index(drop=True)

        return train_df, val_df, test_df


    def __len__(self) -> int:
        return len(self.data_frame)

    def get_signal(
        self, 
        idx: int
    ) -> tuple[np.ndarray, np.ndarray]:
        if 'npy_path' in self.data_frame.columns:
            npy_path: str = self.data_frame.iloc[idx]['npy_path']
            unnormalized_signal: np.ndarray = np.load(npy_path)
            labels: np.ndarray = self.data_frame.iloc[idx][[
                'Sinusal', 'Regular', 'Monomorph', 
                'QS complex in V1-V2-V3', 'R complex in V5-V6', 
                'T wave inversion (inferior - II, III, aVF)', 
                'Left bundle branch block', 'RaVL > 11 mm', 'SV1 + RV5 or RV6 > 35 mm', 
                'T wave inversion (lateral -I, aVL, V5-V6)', 'T wave inversion (anterior - V3-V4)', 
                'Left axis deviation', 'Left ventricular hypertrophy', 'Bradycardia', 
                'Q wave (inferior - II, III, aVF)', 'Afib', 'Irregularly irregular', 
                'Atrial tachycardia (>= 100 BPM)', 'Nonspecific intraventricular conduction delay', 
                'Premature ventricular complex', 'Polymorph', 'T wave inversion (septal- V1-V2)', 
                'Right bundle branch block', 'Ventricular paced', 'ST elevation (anterior - V3-V4)', 
                'ST elevation (septal - V1-V2)', '1st degree AV block', 'Premature atrial complex', 
                'Atrial flutter', "rSR' in V1-V2", 'qRS in V5-V6-I, aVL', 
                'Left anterior fascicular block', 'Right axis deviation', '2nd degree AV block - mobitz 1', 
                'ST depression (inferior - II, III, aVF)', 'Acute pericarditis', 
                'ST elevation (inferior - II, III, aVF)', 'Low voltage', 'Regularly irregular', 
                'Junctional rhythm', 'Left atrial enlargement', 'ST elevation (lateral - I, aVL, V5-V6)', 
                'Atrial paced', 'Right ventricular hypertrophy', 'Delta wave', 'Wolff-Parkinson-White (Pre-excitation syndrome)', 
                'Prolonged QT', 'ST depression (anterior - V3-V4)', 'QRS complex negative in III', 
                'Q wave (lateral- I, aVL, V5-V6)', 'Supraventricular tachycardia', 'ST downslopping', 
                'ST depression (lateral - I, avL, V5-V6)', '2nd degree AV block - mobitz 2', 'U wave', 
                'R/S ratio in V1-V2 >1', 'RV1 + SV6 > 11 mm', 'Left posterior fascicular block', 
                'Right atrial enlargement', 'ST depression (septal- V1-V2)', 'Q wave (septal- V1-V2)', 
                'Q wave (anterior - V3-V4)', 'ST upslopping', 'Right superior axis', 'Ventricular tachycardia', 
                'ST elevation (posterior - V7-V8-V9)', 'Ectopic atrial rhythm (< 100 BPM)', 
                'Lead misplacement', 'Third Degree AV Block', 'Acute MI', 'Early repolarization', 
                'Q wave (posterior - V7-V9)', 'Bi-atrial enlargement', 'LV pacing', 'Brugada', 
                'Ventricular Rhythm', 'no_qrs']]
    

        labels: torch.Tensor = torch.tensor(labels.values.astype(np.int32), dtype=torch.int32)
        labels: torch.Tensor = labels.to(torch.float)
        return unnormalized_signal, labels

    def __getitem__(
        self, 
        idx: int
    ) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        signal_info: tuple[np.ndarray, np.ndarray] = self.get_signal(idx)
        unnormalized_signal: np.ndarray = signal_info[0]
        labels: np.ndarray = signal_info[1]
    
        if np.isnan(unnormalized_signal).any():
            return self.__getitem__((idx + 1) % len(self))

        epsilon: float = 1e-8 
        signal: np.ndarray = (unnormalized_signal - unnormalized_signal.min()) / (unnormalized_signal.max() - unnormalized_signal.min() + epsilon) * 2 - 1
        signal: torch.Tensor = torch.tensor(signal, dtype=torch.float32)
        sample: dict = {'signal': signal, 'labels': labels}
        sample['signal'] = sample['signal'].permute(1, 0)
        return sample
    
class ECGDatasetEmbeddings(Dataset):
    def __init__(
        self, 
        parquet_file: Optional[str] = None, 
        csv_file: Optional[str] = None, 
        transform: Optional[callable] = None, 
        split: Optional[str] = 'train', 
        test_size: Optional[float] = 0.1, 
        random_state: Optional[int] = 42
    ):
        self.transform: Optional[callable] = transform
        self.split: str = split

        if parquet_file:
            self.data_frame: pd.DataFrame = pd.read_parquet(parquet_file)
        elif csv_file:
            self.data_frame: pd.DataFrame = pd.read_csv(csv_file)
            self.data_frame: pd.DataFrame = self._random_split(test_size, random_state)

        unnormalized_signal: np.ndarray
        unnormalized_signal, _ = self.get_signal(1)
        self.waveform_length: int = unnormalized_signal.shape[0]
        self.leads: int = unnormalized_signal.shape[1]

    def _random_split(
        self, 
        test_size: float, 
        random_state: int
    ) -> pd.DataFrame:
        """
        Perform a random train-test split (MIMIC-IV dataset).
        """
        train_df: pd.DataFrame
        test_df: pd.DataFrame
        train_df, test_df = train_test_split(self.data_frame, test_size=test_size, random_state=random_state)

        if self.split == 'train':
            return train_df
        else:
            return test_df

    def __len__(self) -> int:
        return len(self.data_frame)

    def get_signal(
        self, 
        idx: int
    ) -> tuple[np.ndarray, str]:
        if 'npy_path' in self.data_frame.columns:
            npy_path: str = os.path.join(self.root_dir, self.data_frame.iloc[idx]['npy_path'])
            unnormalized_signal: np.ndarray = np.load(npy_path)
            return unnormalized_signal, npy_path
        elif 'waveform_path' in self.data_frame.columns:
            waveform_path = self.data_frame.iloc[idx]['waveform_path']
            unnormalized_signal = np.load(waveform_path)
            return unnormalized_signal, waveform_path

    def __getitem__(
        self, 
        idx: int
    ) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        signal_info: tuple[np.ndarray, str] = self.get_signal(idx)
        unnormalized_signal: np.ndarray = signal_info[0]
        waveform_path: str = signal_info[1]
        
        if np.isnan(unnormalized_signal).any():
            return self.__getitem__((idx + 1) % len(self))

        epsilon: float = 1e-8 
        signal: np.ndarray = (unnormalized_signal - unnormalized_signal.min()) / (unnormalized_signal.max() - unnormalized_signal.min() + epsilon) * 2 - 1
    
        sample = {'signal': signal, 'waveform_path': waveform_path}
        return sample
    

class ECGDatasetLinearProbe(Dataset):
    def __init__(
        self, 
        parquet_file: Optional[str] = None, 
        embedding_folder: Optional[str] = None, 
        transform: Optional[callable] = None, 
        split: Optional[str] = 'train', 
        val_size: Optional[float] = 0.01, 
        test_size: Optional[float] = 0.01, 
        random_state: Optional[int] = 1234
    ):
        self.transform: Optional[callable] = transform
        self.split: str = split
        self.embedding_folder: Optional[str] = embedding_folder

        if parquet_file:
            self.data_frame: pd.DataFrame = pd.read_parquet(parquet_file)

            embedding_files: set[str] = set(f.replace('_embedding.npy', '') for f in os.listdir(self.embedding_folder) if f.endswith('_embedding.npy'))
            self.data_frame: pd.DataFrame = self.data_frame[self.data_frame['npy_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0] in embedding_files)]

            self.train_df, self.val_df, self.test_df = self._stratified_split(val_size, test_size, random_state)
            if self.split == 'train':
                self.data_frame = self.train_df
            elif self.split == 'val':
                self.data_frame = self.val_df
            elif self.split == 'test':
                self.data_frame = self.test_df

    def _stratified_split(
        self, 
        val_size: float, 
        test_size: float, 
        random_state: int
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Stratified train-validation-test split based on the label distribution.
        """
        labels: np.ndarray = self.data_frame[[
            'Sinusal', 'Regular', 'Monomorph', 
                'QS complex in V1-V2-V3', 'R complex in V5-V6', 
                'T wave inversion (inferior - II, III, aVF)', 
                'Left bundle branch block', 'RaVL > 11 mm', 'SV1 + RV5 or RV6 > 35 mm', 
                'T wave inversion (lateral -I, aVL, V5-V6)', 'T wave inversion (anterior - V3-V4)', 
                'Left axis deviation', 'Left ventricular hypertrophy', 'Bradycardia', 
                'Q wave (inferior - II, III, aVF)', 'Afib', 'Irregularly irregular', 
                'Atrial tachycardia (>= 100 BPM)', 'Nonspecific intraventricular conduction delay', 
                'Premature ventricular complex', 'Polymorph', 'T wave inversion (septal- V1-V2)', 
                'Right bundle branch block', 'Ventricular paced', 'ST elevation (anterior - V3-V4)', 
                'ST elevation (septal - V1-V2)', '1st degree AV block', 'Premature atrial complex', 
                'Atrial flutter', "rSR' in V1-V2", 'qRS in V5-V6-I, aVL', 
                'Left anterior fascicular block', 'Right axis deviation', '2nd degree AV block - mobitz 1', 
                'ST depression (inferior - II, III, aVF)', 'Acute pericarditis', 
                'ST elevation (inferior - II, III, aVF)', 'Low voltage', 'Regularly irregular', 
                'Junctional rhythm', 'Left atrial enlargement', 'ST elevation (lateral - I, aVL, V5-V6)', 
                'Atrial paced', 'Right ventricular hypertrophy', 'Delta wave', 'Wolff-Parkinson-White (Pre-excitation syndrome)', 
                'Prolonged QT', 'ST depression (anterior - V3-V4)', 'QRS complex negative in III', 
                'Q wave (lateral- I, aVL, V5-V6)', 'Supraventricular tachycardia', 'ST downslopping', 
                'ST depression (lateral - I, avL, V5-V6)', '2nd degree AV block - mobitz 2', 'U wave', 
                'R/S ratio in V1-V2 >1', 'RV1 + SV6 > 11 mm', 'Left posterior fascicular block', 
                'Right atrial enlargement', 'ST depression (septal- V1-V2)', 'Q wave (septal- V1-V2)', 
                'Q wave (anterior - V3-V4)', 'ST upslopping', 'Right superior axis', 'Ventricular tachycardia', 
                'ST elevation (posterior - V7-V8-V9)', 'Ectopic atrial rhythm (< 100 BPM)', 
                'Lead misplacement', 'Third Degree AV Block', 'Acute MI', 'Early repolarization', 
                'Q wave (posterior - V7-V9)', 'Bi-atrial enlargement', 'LV pacing', 'Brugada', 
                'Ventricular Rhythm', 'no_qrs'
        ]].values

        stratifier: MultilabelStratifiedShuffleSplit = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(stratifier.split(self.data_frame, labels))

        train_val_df = self.data_frame.iloc[train_idx].reset_index(drop=True)
        test_df = self.data_frame.iloc[test_idx].reset_index(drop=True)
        stratifier = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
        train_idx, val_idx = next(stratifier.split(train_val_df, labels[train_idx]))
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

        return train_df, val_df, test_df

    def __len__(self) -> int:
        return len(self.data_frame)

    def get_embedding_and_labels(
        self, 
        idx: int
    ) -> tuple[np.ndarray, torch.Tensor, list[str]]:
        label_row: pd.Series = self.data_frame.iloc[idx]
        base_id: str = os.path.splitext(os.path.basename(label_row['npy_path']))[0]
        embedding_path: str = os.path.join(self.embedding_folder, f"{base_id}_embedding.npy")

        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")  

        embedding: np.ndarray = np.load(embedding_path)
        # Extract label values using the desired class columns
        labels_series: pd.Series = label_row[[ 
            'Sinusal', 'Regular', 'Monomorph', 
                'QS complex in V1-V2-V3', 'R complex in V5-V6', 
                'T wave inversion (inferior - II, III, aVF)', 
                'Left bundle branch block', 'RaVL > 11 mm', 'SV1 + RV5 or RV6 > 35 mm', 
                'T wave inversion (lateral -I, aVL, V5-V6)', 'T wave inversion (anterior - V3-V4)', 
                'Left axis deviation', 'Left ventricular hypertrophy', 'Bradycardia', 
                'Q wave (inferior - II, III, aVF)', 'Afib', 'Irregularly irregular', 
                'Atrial tachycardia (>= 100 BPM)', 'Nonspecific intraventricular conduction delay', 
                'Premature ventricular complex', 'Polymorph', 'T wave inversion (septal- V1-V2)', 
                'Right bundle branch block', 'Ventricular paced', 'ST elevation (anterior - V3-V4)', 
                'ST elevation (septal - V1-V2)', '1st degree AV block', 'Premature atrial complex', 
                'Atrial flutter', "rSR' in V1-V2", 'qRS in V5-V6-I, aVL', 
                'Left anterior fascicular block', 'Right axis deviation', '2nd degree AV block - mobitz 1', 
                'ST depression (inferior - II, III, aVF)', 'Acute pericarditis', 
                'ST elevation (inferior - II, III, aVF)', 'Low voltage', 'Regularly irregular', 
                'Junctional rhythm', 'Left atrial enlargement', 'ST elevation (lateral - I, aVL, V5-V6)', 
                'Atrial paced', 'Right ventricular hypertrophy', 'Delta wave', 'Wolff-Parkinson-White (Pre-excitation syndrome)', 
                'Prolonged QT', 'ST depression (anterior - V3-V4)', 'QRS complex negative in III', 
                'Q wave (lateral- I, aVL, V5-V6)', 'Supraventricular tachycardia', 'ST downslopping', 
                'ST depression (lateral - I, avL, V5-V6)', '2nd degree AV block - mobitz 2', 'U wave', 
                'R/S ratio in V1-V2 >1', 'RV1 + SV6 > 11 mm', 'Left posterior fascicular block', 
                'Right atrial enlargement', 'ST depression (septal- V1-V2)', 'Q wave (septal- V1-V2)', 
                'Q wave (anterior - V3-V4)', 'ST upslopping', 'Right superior axis', 'Ventricular tachycardia', 
                'ST elevation (posterior - V7-V8-V9)', 'Ectopic atrial rhythm (< 100 BPM)', 
                'Lead misplacement', 'Third Degree AV Block', 'Acute MI', 'Early repolarization', 
                'Q wave (posterior - V7-V9)', 'Bi-atrial enlargement', 'LV pacing', 'Brugada', 
                'Ventricular Rhythm', 'no_qrs'
        ]]

        labels_values: torch.Tensor = torch.tensor(labels_series.values.astype(np.int32), dtype=torch.float32)
        labels: list[str] = labels_series.index.tolist()
        return embedding, labels_values, labels

    def __getitem__(
        self, 
        idx: int
    ) -> dict:
        if torch.is_tensor(idx):
            idx = idx.tolist()

        embedding, labels_values, labels = self.get_embedding_and_labels(idx)
        embedding: torch.Tensor = torch.tensor(embedding, dtype=torch.float32)

        sample: dict = {'embedding': embedding, 'labels_values': labels_values, 'labels': labels}

        if self.transform:
            sample['embedding'] = self.transform(sample['embedding'])

        return sample