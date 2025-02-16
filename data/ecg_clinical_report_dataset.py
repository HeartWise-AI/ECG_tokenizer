import torch
import numpy as np
import pandas as pd

from utils.ddp import DistributedUtils
from transformers import GPT2Tokenizer
from torch.utils.data import Dataset, DataLoader
from utils.config.heartwise_config import HeartWiseConfig


class ECGClinicalReportDataset(Dataset):
    def __init__(
        self, 
        embeddings_path: str, 
        reports_path: str, 
        tokenizer: GPT2Tokenizer, 
        max_length: int = 512
    ):
        """
        Args:
            embeddings_path (str): Path to the ECG embeddings.
            reports_path (str): Path to the clinical reports.
            tokenizer (PreTrainedTokenizer): Tokenizer for the clinical reports.
            max_length (int): Maximum token length for the reports.
        """
        self.embeddings_path: str = embeddings_path
        self.df: pd.DataFrame = pd.read_parquet(reports_path)
        self.tokenizer: GPT2Tokenizer = tokenizer
        self.max_length: int = max_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            waveform_path = self.df.iloc[idx]['waveform_path']
            waveform_path = waveform_path.split('/')[-1]
            waveform_path = waveform_path.split('.')[0]
            embedding_path = self.embeddings_path + waveform_path + '_embedding.npy'
            
            # Try to load the embedding
            embedding = torch.tensor(
                np.load(embedding_path), 
                dtype=torch.float
            )  # Shape: (8, 128, 160)
            report = self.df.iloc[idx]['report']
            
            encoding = self.tokenizer.encode_plus(
                report,
                add_special_tokens=True,
                max_length=self.max_length,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            
            input_ids = encoding['input_ids'].squeeze()  # Shape: (max_length)
            attention_mask = encoding['attention_mask'].squeeze()  # Shape: (max_length)
            
            return {
                'embedding': embedding,  # (8, 128, 160)
                'input_ids': input_ids,  # (max_length)
                'attention_mask': attention_mask  # (max_length)
            }
            
        except (FileNotFoundError, OSError) as e:
            print(f"Could not load embedding for index {idx}, trying next item...")
            if idx + 1 < len(self):
                return self.__getitem__(idx + 1)
            else:
                raise Exception("No valid items found in the remaining dataset")
            
            
def get_clinical_report_dataloader(
    config: HeartWiseConfig,
    shuffle: bool = True,
    pin_memory: bool = True
):
    dataset: ECGClinicalReportDataset = ECGClinicalReportDataset(
        embeddings_path=config.embeddings_path, 
        reports_path=config.reports_path, 
        tokenizer=config.tokenizer, 
        max_length=config.max_length
    )
    return DataLoader(
        dataset, 
        batch_size=config.batch_size, 
        shuffle=shuffle, 
        num_workers=config.num_workers, 
        pin_memory=pin_memory
    )
    
def get_distributed_clinical_report_dataloader(
    reports_path: str,
    embeddings_path: str,
    tokenizer: GPT2Tokenizer,
    max_token_length: int = 512,
    batch_size: int = 32,
    num_workers: int = 16,
    num_replicas: int = 1,
    rank: int = 0,
    shuffle: bool = True,
    pin_memory: bool = True
):
    dataset: ECGClinicalReportDataset = ECGClinicalReportDataset(
        embeddings_path=embeddings_path, 
        reports_path=reports_path, 
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
        shuffle=shuffle
    )
