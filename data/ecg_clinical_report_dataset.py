import os
import torch
import numpy as np
import pandas as pd
import torch.nn.functional as F
from typing import cast, List
from transformers import PreTrainedTokenizerBase

from utils.ddp import DistributedUtils
from transformers import GPT2Tokenizer, BatchEncoding
from torch.utils.data import Dataset, DataLoader, default_collate
from utils.config.llm_finetuning_config import LLMFinetuningConfig
from models.types import AutoTokenizerT


class ECGClinicalReportDataset(Dataset):
    def __init__(
        self, 
        dataset_path: str, 
        signal_path_column: str,
        ecg_waveform_length: int,
        ecg_num_leads: int,
        tokenizer: AutoTokenizerT, 
        max_length: int = 512,
        instruct_mode: bool = False
    ):
        """
        Args:
            dataset_path (str): Path to the dataset.
            tokenizer (PreTrainedTokenizer): Tokenizer for the clinical reports.
            max_length (int): Maximum token length for the reports.
        """
        try:
            self.df: pd.DataFrame = pd.read_parquet(dataset_path)
        except Exception as e:
            print(f"Error reading parquet file: {e}")
            raise Exception(f"Error reading parquet file: {e}")
        
        self.ecg_waveform_length: int = ecg_waveform_length
        self.ecg_num_leads: int = ecg_num_leads
        self.tokenizer: AutoTokenizerT = tokenizer
        self._pt_tokenizer: PreTrainedTokenizerBase = cast(PreTrainedTokenizerBase, tokenizer)
        self.max_length: int = max_length
        self.signal_path_column: str = signal_path_column
        self.instruct_mode: bool = instruct_mode
        
    def __len__(self):
        return len(self.df)

    def load_ecg_signal(self, waveform_path: str) -> np.ndarray:
        try:
            waveform: np.ndarray = np.load(waveform_path)
        except Exception as e:
            print(f"Error loading ECG signal: {e}")
            raise Exception(f"Error loading ECG signal: {e}")
        
        # Hack for MHI dataset stored as 3D array with shape (2500, 12, 1)
        if len(waveform.shape) == 3:
            waveform = waveform.squeeze(-1)
            
        assert len(waveform.shape) == 2, f"Unnormalized signal has shape {waveform.shape}"
        
        return waveform

    def __getitem__(self, idx: int) -> dict | None:
        try:
            # Get the row
            row = self.df.iloc[idx]
            
            # Check if the waveform path or report is missing
            if pd.isnull(row[self.signal_path_column]) or pd.isnull(row['report']):
                print(f"Missing waveform_path or report for index {idx}, skipping sample. "
                      f"waveform_path: {row.get(self.signal_path_column)}, report: {row.get('report')}")
                return self.__getitem__((idx + 1) % len(self))

            # Load the waveform
            waveform: np.ndarray = self.load_ecg_signal(
                waveform_path=row[self.signal_path_column]
            )
            
            if np.isnan(waveform).any():
                return self.__getitem__((idx + 1) % len(self))
            
            current_length: int = waveform.shape[0]
            if current_length > self.ecg_waveform_length:
                step: int = waveform.shape[0] // self.ecg_waveform_length
                waveform = waveform[::step, :]
            
            if waveform.shape[0] != self.ecg_waveform_length:
                return self.__getitem__((idx + 1) % len(self))
            
            if waveform.shape[1] != self.ecg_num_leads:
                return self.__getitem__((idx + 1) % len(self))
            
            # Tokenization logic
            if self.instruct_mode:
                question_text: str = ""
                if 'question' in self.df.columns and not pd.isnull(row['question']):
                    question_text = str(row['question'])
                report_text: str = str(row['report'])

                # Construct LLaMA 3.2 chat template with ECG integration
                # System message for ECG analysis task
                system_message = "You are a medical expert assistant specialized in ECG analysis. Analyze the provided ECG signal and respond with a clinical report."
                
                # User message with ECG placeholder and question
                user_content = f"<|start_ecg|>\n[ECG_SIGNAL]\n<|end_ecg|>\n\n{question_text}" if question_text else "<|start_ecg|>\n[ECG_SIGNAL]\n<|end_ecg|>\n\nPlease analyze the above ECG signal and provide a clinical report."
                
                # Create messages for chat template
                messages_prompt = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_content}
                ]
                
                messages_full = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": report_text}
                ]
                
                # Apply chat template
                prompt_text = self._pt_tokenizer.apply_chat_template(
                    messages_prompt, 
                    tokenize=False, 
                    add_generation_prompt=True
                )
                full_text = self._pt_tokenizer.apply_chat_template(
                    messages_full, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                
                # Replace ECG placeholder with special tokens for tokenization
                # The actual ECG embedding will replace the ECG token during training/inference
                ecg_token_placeholder = "<|start_ecg|>\n[ECG_SIGNAL]\n<|end_ecg|>"
                ecg_token_replacement = "<|start_ecg|><|end_ecg|>"  # Simplified to just the boundary tokens
                
                prompt_text = prompt_text.replace(ecg_token_placeholder, ecg_token_replacement)
                full_text = full_text.replace(ecg_token_placeholder, ecg_token_replacement)
                
                # Tokenize both
                prompt_encoding = self._pt_tokenizer.encode_plus(
                    prompt_text,
                    add_special_tokens=False,  # Chat template already adds special tokens
                    return_tensors=None
                )
                full_encoding = self._pt_tokenizer.encode_plus(
                    full_text,
                    add_special_tokens=False,  # Chat template already adds special tokens
                    return_tensors=None
                )
                
                prompt_ids = prompt_encoding.input_ids
                full_ids = full_encoding.input_ids

                # Convert to tensors and pad/truncate to max_length  
                input_ids = torch.tensor(full_ids[:self.max_length], dtype=torch.long)
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
                if input_ids.numel() < self.max_length:
                    pad_len = self.max_length - input_ids.numel()
                    pad_id_attr = getattr(self._pt_tokenizer, 'pad_token_id', None)
                    eos_attr = getattr(self._pt_tokenizer, 'eos_token_id', None) 
                    if isinstance(eos_attr, list):
                        eos_id = int(eos_attr[0]) if len(eos_attr) > 0 else 0
                    elif eos_attr is None:
                        eos_id = 0
                    else:
                        eos_id = int(eos_attr)
                    pad_id = int(pad_id_attr) if pad_id_attr is not None else eos_id
                    input_ids = F.pad(input_ids, (0, pad_len), value=pad_id)
                    attention_mask = F.pad(attention_mask, (0, pad_len), value=0)

                # Prepare prompt-only ids and mask for generation
                prompt_input_ids = torch.tensor(prompt_ids[: self.max_length], dtype=torch.long)
                prompt_attention_mask = torch.ones_like(prompt_input_ids, dtype=torch.long)
                if prompt_input_ids.numel() < self.max_length:
                    pad_len = self.max_length - prompt_input_ids.numel()
                    prompt_input_ids = F.pad(prompt_input_ids, (0, pad_len), value=pad_id)
                    prompt_attention_mask = F.pad(prompt_attention_mask, (0, pad_len), value=0)

                # Create labels: ignore prompt (question + assistant header) and padding
                prompt_len = min(len(prompt_ids), self.max_length)
                labels = input_ids.clone()
                labels[:prompt_len] = -100
                labels = labels.masked_fill(attention_mask == 0, -100)

                return {
                    'signal': np.transpose(waveform, (1, 0)),
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'prompt_input_ids': prompt_input_ids,
                    'prompt_attention_mask': prompt_attention_mask,
                    'labels': labels,
                    'waveform_name': row['waveform_name']
                }
            else:
                # Tokenize the report only (legacy behavior)
                encoding: BatchEncoding = self._pt_tokenizer.encode_plus(
                    row['report'],
                    add_special_tokens=True,
                    max_length=self.max_length,
                    padding='max_length',
                    truncation=True,
                    return_tensors='pt'
                )
                input_ids = cast(torch.Tensor, encoding['input_ids']).squeeze()
                attention_mask = cast(torch.Tensor, encoding['attention_mask']).squeeze()

                return {
                    'signal': np.transpose(waveform, (1, 0)),
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'waveform_name': row['waveform_name']
                }
            
        except Exception as e:
            print(f"Error processing index {idx}: {e}")
            return None
            
            
def get_clinical_report_dataloader(
    config: LLMFinetuningConfig,
    shuffle: bool = True,
    pin_memory: bool = True
):
    dataset: ECGClinicalReportDataset = ECGClinicalReportDataset(
        dataset_path=config.train_dataset_path, 
        signal_path_column=config.signal_path_column,
        ecg_waveform_length=config.ecg_waveform_length,
        ecg_num_leads=config.ecg_num_leads,
        tokenizer=config.tokenizer, 
        max_length=config.max_length,
        instruct_mode=getattr(config, 'instruct_mode', False)
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
    dataset_path: str,
    signal_path_column: str,
    ecg_waveform_length: int,
    ecg_num_leads: int,
    tokenizer: AutoTokenizerT,
    max_token_length: int = 512,
    batch_size: int = 32,
    num_workers: int = 16,
    num_replicas: int = 1,
    rank: int = 0,
    shuffle: bool = True,
    pin_memory: bool = True,
    instruct_mode: bool = False
):
    dataset: ECGClinicalReportDataset = ECGClinicalReportDataset(
        dataset_path=dataset_path, 
        signal_path_column=signal_path_column,
        ecg_waveform_length=ecg_waveform_length,
        ecg_num_leads=ecg_num_leads,
        tokenizer=tokenizer, 
        max_length=max_token_length,
        instruct_mode=instruct_mode
    )
    
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
    Custom collate function which filters out any None items in the batch.
    """
    filtered_batch = [item for item in batch if item is not None]
    if len(filtered_batch) == 0:
        raise ValueError("All items in the batch were invalid. Check dataset integrity or file paths.")
    return default_collate(filtered_batch)
