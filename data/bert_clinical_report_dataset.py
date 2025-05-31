
import pandas as pd
from torch.utils.data import (
    DataLoader, 
    Dataset
)
from transformers import BertTokenizer

from utils.ddp import DistributedUtils
from utils.config.bert_classifier_config import BertReportClassifierConfig    

    
class BertClinicalReportDataset(Dataset):
    def __init__(
        self, 
        df_path: str,
        tokenizer: BertTokenizer
    ):
        df = pd.read_csv(df_path)
        self.predicted_report = df['predicted_report']
        self.reference_report = df['reference_report']
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.predicted_report)
    
    def __getitem__(self, idx):
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
    config: BertReportClassifierConfig,
    tokenizer: BertTokenizer,
    shuffle: bool = True,
    pin_memory: bool = True
):
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
