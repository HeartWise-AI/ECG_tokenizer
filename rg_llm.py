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
csv_file = config["dataset"]["csv_file"]

def evaluate(model, data_loader):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in data_loader:
            signals = batch['signal'].float().to(device)
            signals = signals.permute(0, 2, 1)
            out, indices, cmt_loss = model(signals)
            rec_loss = (out - signals).abs().mean()
            total_loss += rec_loss.item()
    model.train()
    return total_loss / len(data_loader)

def create_dataset_and_model(csv_file, config, device):
    temp_data = pd.read_csv(csv_file)
    temp_waveform = np.load(temp_data.iloc[0]['waveform_path'])
    waveform_length = temp_waveform.shape[0]

    model = SimpleVQAutoEncoder(
        timesteps=waveform_length,
        codebook_size=config["training"]["num_codes"],
        rotation_trick=False 
    ).to(device)
    checkpoint = torch.load(config["evaluation"]["model_path"], weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    # print(model.layers[4].codebook.shape)

    dataset = ECGDatasetLLM(csv_file=csv_file, split='train', model=model)

    return dataset, model

dataset_mimic_train, model = create_dataset_and_model(csv_file, config, device)
dataloader = DataLoader(dataset_mimic_train, batch_size=32, shuffle=True)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
language_decoder = GPT2LMHeadModel.from_pretrained('gpt2')

for param in language_decoder.parameters():
    param.requires_grad = False

class ECGReportGenerator(nn.Module):
    def __init__(self, vq_embedding_dim, language_model):
        super(ECGReportGenerator, self).__init__()
        self.language_model = language_model
        self.embedding_layer = nn.Embedding(vq_embedding_dim, language_model.config.n_embd)

    def forward(self, ecg_tokens, input_ids):
        ecg_embeddings = self.embedding_layer(ecg_tokens)
        outputs = self.language_model(inputs_embeds=ecg_embeddings, labels=input_ids)
        return outputs