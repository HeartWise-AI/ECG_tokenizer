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

from data.dataset import ECGDatasetLinearProbe
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


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def evaluate(data_loader, classifier, criterion):

    classifier.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            embeddings = batch['embedding'].float()
            embeddings = embeddings[-1].unsqueeze(0)
            embeddings = embeddings.permute(1, 0, 2, 3)
            embeddings = embeddings.to(device)
            labels = batch['labels'].to(device)
            predictions = classifier(embeddings)
            loss = criterion(predictions, labels)
            total_loss += loss.item()
            pred_prob = torch.sigmoid(predictions)
            pred_vals = (pred_prob > 0.5).int().cpu()
            all_preds.extend(pred_vals)
            all_labels.extend(labels.int().cpu().detach().numpy())
    
    print(len(data_loader))
    avg_loss = total_loss / len(data_loader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    accuracy = accuracy_score(all_labels, all_preds)
    classifier.train()
    
    return avg_loss, accuracy


def train(classifier, train_loader, val_loader, test_loader, optimizer, criterion, num_epochs):
    classifier.train()

    pbar = trange(num_epochs)
    for epoch in pbar:
        epoch_losses = []
        all_preds = []
        all_labels = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            embeddings = batch['embedding'].float().to(device)
            labels = batch['labels'].to(device)

            embeddings = embeddings[-1].unsqueeze(0)
            embeddings = embeddings.permute(1, 0, 2, 3)
            embeddings = embeddings.to(device)
            predictions = classifier(embeddings)
            loss = criterion(predictions, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_prob = torch.sigmoid(predictions)
            pred_vals = (pred_prob > 0.5).int().cpu()
            all_preds.extend(pred_vals)
            all_labels.extend(labels.int().cpu().detach().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        train_accuracy = accuracy_score(all_labels, all_preds)
        print(f"Epoch {epoch + 1}, Training Loss: {loss.item():.4f}, Training Accuracy: {train_accuracy * 100:.10f}")

        if (epoch + 1) % 5 == 0:
            val_loss, val_accuracy = evaluate(val_loader, classifier, criterion)
            print(f"Epoch {epoch + 1}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy * 100:.4f}")
            wandb.log({
                "test_loss": val_loss,
                "test_accuracy": val_accuracy * 100
            })
        
        wandb.log({
            "classification_loss": loss.item(),
            "training_accuracy": train_accuracy * 100
        })
    return


def main():
    seed = config["training"]["seed"]
    torch.random.manual_seed(seed)
    wandb.init(project="ECG_tokenizer_linear_probing", entity="rohanbanerjee", name="test_run_one_codebook", config=config)
    parquet_file = config["dataset"]["parquet_file"]
    embedding_folder = config["evaluation"]["embedding_dir"]
    num_classes = 1
    embedding_dim = 160
    prev_embedding_dim = 128
    num_quantizers = 1
    num_epochs = 1000

    dataset_mimic_train = ECGDatasetLinearProbe(parquet_file=parquet_file, embedding_folder=embedding_folder, split='train')
    train_loader = DataLoader(dataset_mimic_train, batch_size=8, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)

    dataset_mimic_val = ECGDatasetLinearProbe(parquet_file=parquet_file, embedding_folder=embedding_folder, split='val')
    val_loader = DataLoader(dataset_mimic_val, batch_size=8, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)

    dataset_mimic_test = ECGDatasetLinearProbe(parquet_file=parquet_file, embedding_folder=embedding_folder, split='test')
    test_loader = DataLoader(dataset_mimic_test, batch_size=128, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)

    classifier = CodebookClassifier(num_classes=num_classes, num_quantizers=num_quantizers, prev_embedding_dim=prev_embedding_dim, embedding_dim=embedding_dim).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-6, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
    # criterion = FocalLoss(logits=True)
    criterion = nn.BCEWithLogitsLoss()
    train(classifier, train_loader, val_loader, test_loader, optimizer, criterion, num_epochs)
    
if __name__ == '__main__':
    main()