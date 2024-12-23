import torch
import numpy as np
from torch.utils.data import DataLoader
from vector_quantize_pytorch import VectorQuantize, ResidualVQ
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.cuda.amp import autocast
import torch.distributed as dist
from sklearn.metrics import accuracy_score
import argparse
import wandb

from data.dataset import ECGDatasetClassifier
from models.vqvae import VQVAE, SimpleVQAutoEncoder, ResVQAutoEncoder, CodebookClassifier
import os
from tqdm.auto import trange
import wandb
import yaml
import logging

from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def evaluate(model, data_loader, classifier, criterion):
    print("Starting evaluation...")
    model.eval()
    classifier.eval()
    total_loss = 0
    all_predictions = []
    all_labels = []
    print(f"len test: {len(data_loader)}")

    residual_vq_layer = None
    with torch.no_grad():
        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            signals = batch['signal'].float().to(device)
            labels = batch['labels'].to(device)

            with autocast():
                out, indices, cmt_loss = model(signals)
                for i, layer in enumerate(model.layers):
                    if isinstance(layer, ResidualVQ):
                        residual_vq_layer = layer
                        break
                embeddings = residual_vq_layer.get_codes_from_indices(indices)
                combined_embeddings = embeddings.permute(1, 2, 3, 0).reshape(signals.shape[0], -1)
                combined_embeddings = combined_embeddings.squeeze().to(device)
                predictions = classifier(combined_embeddings)
                print(predictions.shape)
                loss = criterion(predictions, labels)
                total_loss += loss.item()

                all_predictions.extend((predictions > 0.5).cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                wandb.log({
                    "test_loss": loss.item()
                })

        if i % 10 == 0:  # Reduce logging frequency
            print(f"Evaluated {i + 1}/{len(data_loader)} batches")
            print(f"Loss {total_loss / (i + 1)}")
        
    test_accuracy = accuracy_score(np.array(all_labels), np.array(all_predictions))
    avg_loss = total_loss / len(data_loader)
    # model.train()
    print("Evaluation completed.")
    return avg_loss, test_accuracy

def train(model, classifier, train_loader, val_loader, test_loader, optimizer, criterion, num_codes, num_epochs):
    
    def iterate_dataset(data_loader):
        data_iter = iter(data_loader)
        while True:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                batch = next(data_iter)
            signals = batch['signal'].float().to(device)
            labels = batch['labels'].to(device)
            signals = signals.permute(0, 2, 1)
            yield signals, labels

    print(f"train set len {len(train_loader)}")
    residual_vq_layer = None
    prev_batch = None

    for _ in (pbar := trange(num_epochs)):
        optimizer.zero_grad()
        x, labels = next(iterate_dataset(train_loader))
        x = x.permute(0, 2, 1)
        if prev_batch is not None:
            print(f"Epoch {_ + 1} - {(x == prev_batch).all().unique()}")
        prev_batch = x 
        with torch.no_grad():
            out, indices, cmt_loss = model(x)
            for i, layer in enumerate(model.layers):
                if isinstance(layer, ResidualVQ):
                    residual_vq_layer = layer
                    break

        embeddings = residual_vq_layer.get_codes_from_indices(indices)
        combined_embeddings = embeddings.permute(1, 2, 3, 0).reshape(x.shape[0], -1)
        combined_embeddings = combined_embeddings.squeeze().to(device)
        predictions = classifier(combined_embeddings)
        loss = criterion(predictions, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_prob = torch.sigmoid(predictions)
        
        train_accuracy = accuracy_score(labels.cpu().numpy(), pred_prob.cpu().detach().numpy() > 0.5)
        print(f"Epoch {_ + 1}, Training Loss: {loss.item():.4f}, Training Accuracy: {train_accuracy:.10f}")

        if (_ + 1) % 10 == 0:
            test_loss = evaluate(model, test_loader, classifier, criterion)
            print({"test_loss": test_loss})
        pbar.set_description(f"classification loss: {loss.item():.3f}")
        wandb.log({
            "classification_loss": loss.item()
        })

        if (_ + 1) % 10 == 0:
            val_loss, val_accuracy = evaluate(model, val_loader, classifier, criterion)
            print(f"Epoch {_ + 1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.4f}")
    return


def main():
    seed = config["training"]["seed"]
    torch.random.manual_seed(seed)
    wandb.init(project="ECG_tokenizer_linear_probing", entity="rohanbanerjee", name="test", config=config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    parquet_file = config["dataset"]["parquet_file"]
    num_classes = 77
    embedding_dim = 160
    prev_embedding_dim = 128
    num_quantizers = 8
    num_epochs = 1000

    dataset_mimic_train = ECGDatasetClassifier(parquet_file=parquet_file, split='train')
    train_loader = DataLoader(dataset_mimic_train, batch_size=8, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)

    dataset_mimic_val = ECGDatasetClassifier(parquet_file=parquet_file, split='val')
    val_loader = DataLoader(dataset_mimic_val, batch_size=128, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)

    dataset_mimic_test = ECGDatasetClassifier(parquet_file=parquet_file, split='test')
    test_loader = DataLoader(dataset_mimic_test, batch_size=128, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)
    num_codes = config["training"]["num_codes"]

    model = ResVQAutoEncoder(
        timesteps=dataset_mimic_train.waveform_length,
        codebook_size=num_codes,
        implicit_neural_codebook=True
    ).to(device)
    checkpoint = torch.load(config["evaluation"]["model_path"], weights_only=True)
    state_dict = checkpoint['model_state_dict']
    model_state_dict = model.state_dict()
    new_state_dict = {}

    for key in state_dict:
        if key.startswith("module.") and not any(k.startswith("module.") for k in model_state_dict.keys()):
            new_key = key[len("module."):]
        elif not key.startswith("module.") and any(k.startswith("module.") for k in model_state_dict.keys()):
            new_key = "module." + key
        else:
            new_key = key
        new_state_dict[new_key] = state_dict[key]

    model.load_state_dict(new_state_dict, strict=True)

    model.eval()

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    classifier = CodebookClassifier(num_classes=num_classes, num_quantizers=num_quantizers, prev_embedding_dim=prev_embedding_dim, embedding_dim=embedding_dim).to(device)

    optimizer = optim.Adam(classifier.parameters(), lr=0.00001)
    criterion = nn.BCEWithLogitsLoss()
    # train(model, classifier, train_loader, test_loader, optimizer, criterion, num_codes, num_epochs)
    train(model.module if isinstance(model, nn.DataParallel) else model, classifier, train_loader, val_loader, test_loader, optimizer, criterion, num_codes, num_epochs)

if __name__ == '__main__':
    main()