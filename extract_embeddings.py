import torch
import numpy as np
from torch.utils.data import DataLoader
from vector_quantize_pytorch import VectorQuantize, ResidualVQ
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.cuda.amp import autocast
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from sklearn.metrics import accuracy_score
import argparse
import wandb

from data.dataset import ECGDatasetEmbeddings
from models.models import (
    VQVAE, 
    SimpleVQAutoEncoder, 
    ResVQAutoEncoder, 
    CodebookClassifier
)
import os
from tqdm.auto import trange
import wandb
import yaml
import logging

from tqdm import tqdm

"""
Passes signal through the trained VQVAE model and saves the embeddings i.e. the codebook embeddings

Author: Rohan Banerjee
"""

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_embeddings(model, data_loader, device, save_dir, dataset_name):
    dataset_dir = os.path.join(save_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)
    for batch in tqdm(data_loader):
        signals = batch['signal'].float().to(device)
        waveform_path = batch['waveform_path']
        residual_vq_layer = None
        signals = signals.permute(0, 2, 1)

        with torch.no_grad():
            out, indices, cmt_loss = model(signals)
            for i, layer in enumerate(model.module.layers):
                if isinstance(layer, ResidualVQ):
                    residual_vq_layer = layer
                    break
        
        batch_embeddings = residual_vq_layer.get_codes_from_indices(indices)

        for idx in range(len(waveform_path)):
            single_embedding = batch_embeddings[:, idx:idx+1, :, :]
            single_embedding = single_embedding.squeeze(1)
            embedding_np = single_embedding.detach().cpu().numpy()
            
            original_filename = os.path.basename(waveform_path[idx])
            filename_without_ext = os.path.splitext(original_filename)[0]
            save_path = os.path.join(save_dir, dataset_name, f"{filename_without_ext}_embedding.npy")
            np.save(save_path, embedding_np)
        
    return


def main():
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description='Extract embeddings')
    parser.add_argument('--base_config', type=str, help='Path to base_config file', required=True)
    args: argparse.Namespace = parser.parse_args()
    with open(args.base_config, 'r') as file:
        config = yaml.safe_load(file)

    dataset_mimic_train: ECGDatasetEmbeddings = ECGDatasetEmbeddings(
        parquet_file=config["train_parquet_MIMIC_file"],
        expected_waveform_length=config["waveform_length"],
        num_leads=config["num_leads"]
    )
    print(f"len(dataset_mimic_train): {len(dataset_mimic_train)}")

    dataset_mhi_train: ECGDatasetEmbeddings = ECGDatasetEmbeddings(
        parquet_file=config["train_parquet_MHI_file"],
        expected_waveform_length=config["waveform_length"],
        num_leads=config["num_leads"]
    )
    print(f"len(dataset_mhi_train): {len(dataset_mhi_train)}")

    dataset_code_15_train: ECGDatasetEmbeddings = ECGDatasetEmbeddings(
        parquet_file=config["code_15_dataset_path"],
        expected_waveform_length=config["waveform_length"],
        num_leads=config["num_leads"]
    )
    print(f"len(dataset_code_15_train): {len(dataset_code_15_train)}")

    combined_train_dataset: torch.utils.data.ConcatDataset = torch.utils.data.ConcatDataset(
        [
            dataset_mimic_train, 
            dataset_mhi_train,
            dataset_code_15_train
        ]
    )


    dataset_mimic_test: ECGDatasetEmbeddings = ECGDatasetEmbeddings(
        parquet_file=config["test_parquet_MIMIC_file"],
        expected_waveform_length=config["waveform_length"],
        num_leads=config["num_leads"]
    )
    print(f"len(dataset_mimic_test): {len(dataset_mimic_test)}")
    dataset_mhi_test: ECGDatasetEmbeddings = ECGDatasetEmbeddings(
        parquet_file=config["test_parquet_MHI_file"],
        expected_waveform_length=config["waveform_length"],
        num_leads=config["num_leads"]
    )
    print(f"len(dataset_mhi_test): {len(dataset_mhi_test)}")

    combined_test_dataset: torch.utils.data.ConcatDataset = torch.utils.data.ConcatDataset(
        [
            dataset_mimic_test, 
            dataset_mhi_test
        ]
    )

    data_train_loader: DataLoader = DataLoader(
        combined_train_dataset, 
        batch_size=config["batch_size"], 
        shuffle=True, 
        num_workers=16
    )

    data_test_loader: DataLoader = DataLoader(
        combined_test_dataset, 
        batch_size=config["batch_size"], 
        shuffle=False, 
        num_workers=16
    )

    embedding_dir = config["embedding_dir"]
    model: ResVQAutoEncoder = ResVQAutoEncoder(
        timesteps=config["waveform_length"],
        codebook_size=config["num_codes"],
        implicit_neural_codebook=True
    ).to(device)

    model = nn.DataParallel(model)

    checkpoint = torch.load(config["model_path"], weights_only=True, map_location=device)
    state_dict = checkpoint['model_state_dict']
    model_state_dict = model.module.state_dict()
    new_state_dict = {}

    for key in state_dict:
        if key.startswith("module.") and not any(k.startswith("module.") for k in model_state_dict.keys()):
            new_key = key[len("module."):]
        elif not key.startswith("module.") and any(k.startswith("module.") for k in model_state_dict.keys()):
            new_key = "module." + key
        else:
            new_key = key
        new_state_dict[new_key] = state_dict[key]

    model.module.load_state_dict(new_state_dict, strict=False)

    model.eval()

    get_embeddings(model, data_train_loader, device, save_dir=embedding_dir, dataset_name="train")
    get_embeddings(model, data_test_loader, device, save_dir=embedding_dir, dataset_name="test")

if __name__ == '__main__':
    main()