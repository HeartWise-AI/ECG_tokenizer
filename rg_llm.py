import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from models.vqvae import SimpleVQAutoEncoder
import torch.nn as nn
import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
import yaml
from data.dataset import ECGDataset, ECGDatasetLLM

with open('/mnt/rbanerjee/code/ECG_tokenizer/config.yaml', 'r') as file:
    config = yaml.safe_load(file)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def encode_ecg_to_tokens(ecg_signal):
    with torch.no_grad():
        _, encoded_tokens, _ = model(ecg_signal)
    return encoded_tokens


csv_file = config["dataset"]["csv_file"]
dataset_mimic_train = ECGDatasetLLM(csv_file=csv_file, split='train')
print(dataset_mimic_train[0])
train_loader = DataLoader(dataset_mimic_train, batch_size=512, shuffle=True, num_workers=4)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
language_decoder = GPT2LMHeadModel.from_pretrained('gpt2')

for param in language_decoder.parameters():
    param.requires_grad = False

model = SimpleVQAutoEncoder(
    timesteps=dataset_mimic_train.waveform_length,
    codebook_size=config["training"]["num_codes"],
    rotation_trick=False 
).to(device)
checkpoint = torch.load(config["evaluation"]["model_path"], weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()




