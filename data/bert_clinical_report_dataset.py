import pandas as pd
from torch.utils.data import (
    DataLoader, 
    Dataset
)
from transformers import BertTokenizer

from utils.ddp import DistributedUtils
from utils.config.heartwise_config import HeartWiseConfig    


class BertClinicalReportDataset(Dataset):
    """
    PyTorch Dataset for processing clinical reports using BERT.

    This dataset reads a CSV file containing clinical reports with columns for
    predicted and reference reports. Each report is tokenized using a provided
    BertTokenizer. The dataset returns a dictionary for each sample containing 
    the tokenized predicted and reference reports.

    Attributes:
        predicted_report (pd.Series): Series containing the predicted reports.
        reference_report (pd.Series): Series containing the reference reports.
        tokenizer (BertTokenizer): Tokenizer for encoding the reports.
    """
    def __init__(
        self, 
        df_path: str,
        tokenizer: BertTokenizer
    ):
        """
        Initialize the BertClinicalReportDataset.

        Reads the CSV file from the given path and extracts the predicted and 
        reference reports.

        Args:
            df_path (str): Path to the CSV file containing clinical reports.
            tokenizer (BertTokenizer): Tokenizer to encode the textual reports.
        """
        df = pd.read_csv(df_path)
        self.predicted_report = df['predicted_report']
        self.reference_report = df['reference_report']
        self.tokenizer = tokenizer
        
    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: Total number of samples.
        """
        return len(self.predicted_report)
    
    def __getitem__(self, idx):
        """
        Retrieve a sample from the dataset and tokenize its reports.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            dict: A dictionary containing:
                - 'encoded_predicted_report': Tokenized version of the predicted report.
                - 'encoded_reference_report': Tokenized version of the reference report.
        """
        predicted_report = self.predicted_report.loc[idx]
        reference_report = self.reference_report.loc[idx]
        
        encoded_predicted_report = self.tokenizer(
            predicted_report,
            padding='max_length',
            max_length=512,
            truncation=True,
            return_tensors='pt'
        )

        encoded_reference_report = self.tokenizer(
            reference_report,
            padding='max_length',
            max_length=512,
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'encoded_predicted_report': encoded_predicted_report, 
            'encoded_reference_report': encoded_reference_report, 
        }

def get_clinical_report_dataloader(
    config: HeartWiseConfig,
    tokenizer: BertTokenizer,
    shuffle: bool = True,
    pin_memory: bool = True
):
    """
    Create a DataLoader for the BertClinicalReportDataset.

    This function initializes a BertClinicalReportDataset using the provided 
    configuration and tokenizer, and then returns a DataLoader for it.

    Args:
        config (HeartWiseConfig): Configuration object with dataset parameters,
            including the path to the predicted reports CSV file and batch size.
        tokenizer (BertTokenizer): Tokenizer for processing the reports.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
        pin_memory (bool, optional): Whether to pin memory. Defaults to True.

    Returns:
        DataLoader: DataLoader for the BertClinicalReportDataset.
    """
    return DataLoader(
        BertClinicalReportDataset(
            df_path=config.predicted_reports_path,
            tokenizer=tokenizer
        ), 
        batch_size=config.batch_size, 
        shuffle=shuffle, 
        pin_memory=pin_memory
    )
    
def get_distributed_clinical_report_dataloader(
    predicted_reports_path: str,
    tokenizer: BertTokenizer,
    batch_size: int = 32,
    num_workers: int = 16,
    num_replicas: int = 1,
    rank: int = 0,
    shuffle: bool = True,
    pin_memory: bool = True
):
    """
    Create a distributed DataLoader for the BertClinicalReportDataset.

    This function initializes a BertClinicalReportDataset with the specified CSV path
    and tokenizer, and then returns a DataLoader suitable for distributed training,
    using a DistributedSampler internally.

    Args:
        predicted_reports_path (str): Path to the CSV file containing predicted reports.
        tokenizer (BertTokenizer): Tokenizer for encoding the reports.
        batch_size (int, optional): Batch size for the DataLoader. Defaults to 32.
        num_workers (int, optional): Number of worker processes for data loading. Defaults to 16.
        num_replicas (int, optional): Total number of processes in distributed training. Defaults to 1.
        rank (int, optional): Rank of the current process. Defaults to 0.
        shuffle (bool, optional): Whether to shuffle the dataset. Defaults to True.
        pin_memory (bool, optional): Whether to pin memory. Defaults to True.

    Returns:
        DataLoader: Distributed DataLoader for the BertClinicalReportDataset.
    """
    return DistributedUtils.get_distributed_dataloader(
        dataset=BertClinicalReportDataset(
            df_path=predicted_reports_path,
            tokenizer=tokenizer
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
    )
