import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import os
import random
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit
import yaml

with open('/home/rbanerjee/ECG_tokenizer/config.yaml', 'r') as file:
    config = yaml.safe_load(file)


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
    def __init__(self, parquet_file=None, csv_file=None, transform=None, split='train', test_size=0.0001, random_state=config["training"]["seed"]):
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
            npy_path = os.path.join(self.root_dir, self.data_frame.iloc[idx]['npy_path'])
            unnormalized_signal = np.load(npy_path)
        elif 'waveform_path' in self.data_frame.columns:
            # Load data for the MIMIC-IV dataset (with full waveform paths)
            waveform_path = self.data_frame.iloc[idx]['waveform_path']
            unnormalized_signal = np.load(waveform_path) 
        return unnormalized_signal

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        unnormalized_signal = self.get_signal(idx)
        
        if np.isnan(unnormalized_signal).any():
            return self.__getitem__((idx + 1) % len(self)) 

        epsilon = 1e-8 
        signal = (unnormalized_signal - unnormalized_signal.min()) / (unnormalized_signal.max() - unnormalized_signal.min() + epsilon) * 2 - 1
    
        sample = {'signal': signal}

        return sample

class ECGDatasetLLM(Dataset):
    def __init__(self, embeddings_folder=None, model=None, transform=None, split='train', test_size=0.2, random_state=config["training"]["seed"]):
        self.transform = transform
        self.split = split
        self.model = model

        if parquet_file:
            self.data_frame = pd.read_parquet(parquet_file)
        elif csv_file:
            self.data_frame = pd.read_csv(csv_file)
            self.data_frame = self._random_split(test_size, random_state)

        self.waveform_length, self.leads = self.get_signal(1).shape

    def _random_split(self, test_size, random_state):
        """
        Perform a random train-test split (MIMIC-IV dataset).
        """
        train_df, test_df = train_test_split(self.data_frame, test_size=test_size, random_state=random_state)

        if self.split == 'train':
            return train_df
        else:
            return test_df

    def __len__(self):
        return len(self.data_frame)


    def get_signal(self, idx):
        if 'npy_path' in self.data_frame.columns:
            npy_path = os.path.join(self.root_dir, self.data_frame.iloc[idx]['npy_path'])
            unnormalized_signal = np.load(npy_path)
        elif 'waveform_path' in self.data_frame.columns:
            waveform_path = self.data_frame.iloc[idx]['waveform_path']
            unnormalized_signal = np.load(waveform_path)

        return unnormalized_signal
    
    def get_report(self, idx):
        return self.data_frame.iloc[idx]['report']
    
    def set_model(self, model):
        """Set the model only if it hasn't been set before."""
        if self.model is not None:
            raise ValueError("Model has already been set.")
        self.model = model
    
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
        
        if np.isnan(unnormalized_signal).any():
            return self.__getitem__((idx + 1) % len(self))  


        epsilon = 1e-8  
        signal = (unnormalized_signal - unnormalized_signal.min()) / (unnormalized_signal.max() - unnormalized_signal.min() + epsilon) * 2 - 1
        signal = torch.from_numpy(signal).float()
        signal = signal.permute(1, 0).unsqueeze(0)

        device = next(self.model.parameters()).device
        signal = signal.to(device)
        
        encoded_tokens = self.encode_ecg_to_tokens(signal)

        return encoded_tokens, report
    
class ECGDatasetClassifier(Dataset):
    def __init__(self, parquet_file=None, transform=None, split='train', val_size=0.01, test_size=0.01, random_state=config["training"]["seed"]):
        self.transform = transform
        self.split = split

        if parquet_file:
            self.data_frame = pd.read_parquet(parquet_file)
            self.train_df, self.val_df, self.test_df = self._stratified_split(val_size, test_size, random_state)
            if self.split == 'train':
                self.data_frame = self.train_df
            elif self.split == 'val':
                self.data_frame = self.val_df
            elif self.split == 'test':
                self.data_frame = self.test_df

        unnormalized_signal, _ = self.get_signal(0)
        self.waveform_length, self.leads = unnormalized_signal.shape

    def _random_split(self, test_size, random_state):
        """
        Perform a random train-test split (MIMIC-IV dataset).
        """
        train_df, test_df = train_test_split(self.data_frame, test_size=test_size, random_state=random_state)

        if self.split == 'train':
            return train_df
        else:
            return test_df
        
    def _stratified_split(self, val_size, test_size, random_state):
        """
        Stratified train-validation-test split based on the label distribution.
        """
        labels = self.data_frame[[
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
        
        stratifier = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(stratifier.split(self.data_frame, labels))
        
        train_val_df = self.data_frame.iloc[train_idx].reset_index(drop=True)
        test_df = self.data_frame.iloc[test_idx].reset_index(drop=True)
        
        stratifier = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
        train_idx, val_idx = next(stratifier.split(train_val_df, labels[train_idx]))
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

        return train_df, val_df, test_df


    def __len__(self):
        return len(self.data_frame)

    def get_signal(self, idx):
        if 'npy_path' in self.data_frame.columns:
            npy_path = self.data_frame.iloc[idx]['npy_path']
            unnormalized_signal = np.load(npy_path)
            labels = self.data_frame.iloc[idx][[
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
    

        labels = torch.tensor(labels.values.astype(np.int32), dtype=torch.int32)
        labels = labels.to(torch.float)
        return unnormalized_signal, labels

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        unnormalized_signal, labels = self.get_signal(idx)
    
        if np.isnan(unnormalized_signal).any():
            return self.__getitem__((idx + 1) % len(self))

        epsilon = 1e-8 
        signal = (unnormalized_signal - unnormalized_signal.min()) / (unnormalized_signal.max() - unnormalized_signal.min() + epsilon) * 2 - 1
        signal = torch.tensor(signal, dtype=torch.float32)
        sample = {'signal': signal, 'labels': labels}
        sample['signal'] = sample['signal'].permute(1, 0)
        return sample
    
class ECGDatasetEmbeddings(Dataset):
    def __init__(self, parquet_file=None, csv_file=None, transform=None, split='train', test_size=0.1, random_state=config["training"]["seed"]):
        self.transform = transform
        self.split = split

        if parquet_file:
            self.data_frame = pd.read_parquet(parquet_file)
        elif csv_file:
            self.data_frame = pd.read_csv(csv_file)
            self.data_frame = self._random_split(test_size, random_state)

        unnormalized_signal, _ = self.get_signal(1)
        self.waveform_length, self.leads = unnormalized_signal.shape

    def _random_split(self, test_size, random_state):
        """
        Perform a random train-test split (MIMIC-IV dataset).
        """
        train_df, test_df = train_test_split(self.data_frame, test_size=test_size, random_state=random_state)

        if self.split == 'train':
            return train_df
        else:
            return test_df

    def __len__(self):
        return len(self.data_frame)

    def get_signal(self, idx):
        if 'npy_path' in self.data_frame.columns:
            npy_path = os.path.join(self.root_dir, self.data_frame.iloc[idx]['npy_path'])
            unnormalized_signal = np.load(npy_path)
            return unnormalized_signal, npy_path
        elif 'waveform_path' in self.data_frame.columns:
            waveform_path = self.data_frame.iloc[idx]['waveform_path']
            unnormalized_signal = np.load(waveform_path)
            return unnormalized_signal, waveform_path

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        unnormalized_signal, waveform_path = self.get_signal(idx)
        
        if np.isnan(unnormalized_signal).any():
            return self.__getitem__((idx + 1) % len(self))

        epsilon = 1e-8 
        signal = (unnormalized_signal - unnormalized_signal.min()) / (unnormalized_signal.max() - unnormalized_signal.min() + epsilon) * 2 - 1
    
        sample = {'signal': signal, 'waveform_path': waveform_path}
        return sample
    

class ECGDatasetLinearProbe(Dataset):
    def __init__(self, parquet_file=None, embedding_folder=None, transform=None, split='train', val_size=0.1, test_size=0.1, random_state=42):
        self.transform = transform
        self.split = split
        self.embedding_folder = embedding_folder

        if parquet_file:
            self.data_frame = pd.read_parquet(parquet_file)

            embedding_files = set(f.replace('_embedding.npy', '') for f in os.listdir(self.embedding_folder) if f.endswith('_embedding.npy'))
            self.data_frame = self.data_frame[self.data_frame['npy_path'].apply(lambda x: os.path.splitext(os.path.basename(x))[0] in embedding_files)]

            self.train_df, self.val_df, self.test_df = self._stratified_split(val_size, test_size, random_state)
            if self.split == 'train':
                self.data_frame = self.train_df
            elif self.split == 'val':
                self.data_frame = self.val_df
            elif self.split == 'test':
                self.data_frame = self.test_df

    def _stratified_split(self, val_size, test_size, random_state):
        """
        Stratified train-validation-test split based on the label distribution.
        """
        labels = self.data_frame[[
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

        stratifier = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
        train_idx, test_idx = next(stratifier.split(self.data_frame, labels))

        train_val_df = self.data_frame.iloc[train_idx].reset_index(drop=True)
        test_df = self.data_frame.iloc[test_idx].reset_index(drop=True)
        stratifier = MultilabelStratifiedShuffleSplit(n_splits=1, test_size=val_size, random_state=random_state)
        train_idx, val_idx = next(stratifier.split(train_val_df, labels[train_idx]))
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)

        return train_df, val_df, test_df

    def __len__(self):
        return len(self.data_frame)

    def get_embedding_and_labels(self, idx):
        label_row = self.data_frame.iloc[idx]
        base_id = os.path.splitext(os.path.basename(label_row['npy_path']))[0]
        embedding_path = os.path.join(self.embedding_folder, f"{base_id}_embedding.npy")

        if not os.path.exists(embedding_path):
            raise FileNotFoundError(f"Embedding file not found: {embedding_path}")  

        embedding = np.load(embedding_path)
        # Extract label values using the desired class columns
        labels_series = label_row[[ 
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

        labels_values = torch.tensor(labels_series.values.astype(np.int32), dtype=torch.float32)
        labels = labels_series.index.tolist()
        return embedding, labels_values, labels

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        embedding, labels_values, labels = self.get_embedding_and_labels(idx)
        embedding = torch.tensor(embedding, dtype=torch.float32)

        sample = {'embedding': embedding, 'labels_values': labels_values, 'labels': labels}

        if self.transform:
            sample['embedding'] = self.transform(sample['embedding'])

        return sample