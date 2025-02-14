import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer
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