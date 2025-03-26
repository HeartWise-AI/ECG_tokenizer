import os
import torch
import numpy as np
import pandas as pd

from utils.ddp import DistributedUtils
from transformers import GPT2Tokenizer
from torch.utils.data import (
    Dataset, 
    DataLoader, 
    ConcatDataset
)
from utils.config.heartwise_config import HeartWiseConfig


class ECGClinicalReportDataset(Dataset):
    """
    Dataset for loading ECG clinical reports and corresponding embeddings.
    
    This dataset reads clinical reports from a parquet file and loads the 
    corresponding ECG embedding from a file. It uses a tokenizer to encode 
    the textual report into token ids.

    Attributes:
        embeddings_path (str): Directory path containing ECG embeddings files.
        df (pd.DataFrame): DataFrame containing clinical reports and waveform information.
        tokenizer (GPT2Tokenizer): Tokenizer to encode textual reports.
        max_length (int): Maximum length for tokenized reports.
    """
    def __init__(
        self, 
        embeddings_path: str, 
        reports_path: str, 
        tokenizer: GPT2Tokenizer, 
        max_length: int = 512
    ):
        """
        Initialize the ECGClinicalReportDataset.

        Args:
            embeddings_path (str): Path to the directory containing ECG embeddings.
            reports_path (str): Path to the clinical reports parquet file.
            tokenizer (GPT2Tokenizer): Tokenizer for processing textual reports.
            max_length (int, optional): Maximum token length for the reports. Defaults to 512.
        """
        self.embeddings_path: str = embeddings_path
        self.df: pd.DataFrame = pd.read_parquet(reports_path)
        self.tokenizer: GPT2Tokenizer = tokenizer
        self.max_length: int = max_length

    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict or None: A dictionary containing the processed 'embedding', 'input_ids', 
            'attention_mask', and 'waveform_name' if the sample is valid; otherwise, None.
        """
        try:
            row = self.df.iloc[idx]
            if pd.isnull(row['waveform_path']) or pd.isnull(row['report']):
                print(f"Missing waveform_path or report for index {idx}, skipping sample. "
                      f"waveform_path: {row.get('waveform_path')}, report: {row.get('report')}")
                return None

            # Get the embedding path
            waveform_path = row['waveform_path']
            waveform_path = waveform_path.split('/')[-1]
            waveform_name = waveform_path.split('.')[0]
            embedding_path = os.path.join(self.embeddings_path, waveform_name + '_embedding.npy')
            
            # Try to load the embedding
            try:
                embedding = torch.tensor(
                    np.load(embedding_path),
                    dtype=torch.float
                )  # Shape: (8, 128, 160)
            except (FileNotFoundError, OSError) as e:
                print(f"Could not load embedding for index {idx}, skipping this item. "
                      f"embedding_path: {embedding_path}")
                return None
            
            report = row['report']
            
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
                'attention_mask': attention_mask,  # (max_length)
                'waveform_name': waveform_name
            }
            
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            return None
            
            
def get_clinical_report_dataloader(
    config: HeartWiseConfig,
    shuffle: bool = True,
    pin_memory: bool = True
):
    """
    Create a DataLoader for the ECG clinical report dataset.
    
    Uses the provided HeartWiseConfig to set up the dataset and returns a DataLoader
    to iterate over the dataset.

    Args:
        config (HeartWiseConfig): Configuration object containing dataset and loader parameters.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
        pin_memory (bool, optional): Whether to use pinned memory. Defaults to True.

    Returns:
        DataLoader: A DataLoader object for the ECGClinicalReportDataset.
    """
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
        pin_memory=pin_memory,
        collate_fn=custom_collate_fn
    )
    
    
def get_distributed_clinical_report_dataloader(
    reports_path: dict[str, str],
    embeddings_path: dict[str, str],
    tokenizer: GPT2Tokenizer,
    max_token_length: int = 512,
    batch_size: int = 32,
    num_workers: int = 16,
    num_replicas: int = 1,
    rank: int = 0,
    shuffle: bool = True,
    pin_memory: bool = True
):
    """
    Create a distributed DataLoader for concatenated ECG clinical report datasets.
    
    This function creates individual datasets for each key present in the provided dictionaries
    for reports and embeddings, concatenates them into a single dataset, and returns a distributed
    DataLoader that partitions the merged dataset across multiple processes for distributed training.

    Args:
        reports_path (dict[str, str]): Dictionary mapping keys to clinical report file paths.
        embeddings_path (dict[str, str]): Dictionary mapping keys to ECG embeddings directory paths.
        tokenizer (GPT2Tokenizer): Tokenizer instance to process textual reports.
        max_token_length (int, optional): Maximum token length for each report. Defaults to 512.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 32.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 16.
        num_replicas (int, optional): Number of processes participating in distributed training. Defaults to 1.
        rank (int, optional): Rank of the current process in distributed training. Defaults to 0.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
        pin_memory (bool, optional): Whether to use pinned memory for data loading. Defaults to True.

    Returns:
        DataLoader: A distributed DataLoader for the concatenated ECG clinical report dataset.
    """
    # Init the datasets list
    datasets: list[ECGClinicalReportDataset] = []
    
    # Create the datasets for each key in embeddings_path
    for key in embeddings_path:
        dataset: ECGClinicalReportDataset = ECGClinicalReportDataset(
            embeddings_path=embeddings_path[key], 
            reports_path=reports_path[key], 
            tokenizer=tokenizer, 
            max_length=max_token_length
        )
        datasets.append(dataset)
    
    # Concatenate the datasets into a single dataset
    dataset: ConcatDataset = ConcatDataset(datasets)
    
    # Return the distributed DataLoader for the concatenated dataset
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
    Custom collate function which filters out any None items in the batch and collates valid samples.
    
    Args:
        batch (list): List of samples from the dataset, potentially containing None values.

    Raises:
        ValueError: If all items in the batch are invalid (None).

    Returns:
        dict or Tensor: Collated valid samples using torch.utils.data.default_collate.
    """
    filtered_batch = [item for item in batch if item is not None]
    if len(filtered_batch) == 0:
        raise ValueError("All items in the batch were invalid. Check dataset integrity or file paths.")
    return torch.utils.data.default_collate(filtered_batch)
