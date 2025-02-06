import torch
import numpy as np
from torch.utils.data import DataLoader
from vector_quantize_pytorch import VectorQuantize, ResidualVQ
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms
from torch.cuda.amp import autocast
import torch.distributed as dist
import argparse
import wandb
from metrics import compute_metrics
import pandas as pd

from data.dataset import ECGDatasetLinearProbe
from models.models import VQVAE, SimpleVQAutoEncoder, ResVQAutoEncoder, CodebookClassifier
import os
from tqdm.auto import trange
import wandb
import yaml
import logging

from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('config.yaml', 'r') as file:
    config = yaml.safe_load(file)

def print_class_distribution(loader, split_name):
    total_samples = 0
    positive_samples = 0
    for batch in tqdm(loader):
        labels = batch['labels_values']
        total_samples += len(labels)
        positive_samples += labels.sum().item()
    
    print(f"{split_name} set distribution:")
    print(f"Positive samples: {positive_samples} ({(positive_samples/total_samples)*100:.2f}%)")
    print(f"Negative samples: {total_samples-positive_samples} ({((total_samples-positive_samples)/total_samples)*100:.2f}%)")
    print(f"Total samples: {total_samples}")


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

def get_criterion(criterion_name):
    if criterion_name == "focal_loss":
        return FocalLoss(logits=True)
    elif criterion_name == "bce_loss":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError("criterion must be either 'focal_loss' or 'bce_loss'")

def evaluate(data_loader, classifier, criterion, labels):

    classifier.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    predictions_list = []
    labels_list = []
    
    with torch.no_grad():
        for batch in data_loader:
            embeddings = batch['embedding'].float()
            embeddings = embeddings.to(device)
            labels_values = batch['labels_values'].to(device)
            predictions = classifier(embeddings)
            loss = criterion(predictions, labels_values)
            pred_prob = torch.sigmoid(predictions)
            labels_np = labels_values.int().cpu().detach().numpy()
            predictions_list.append(pred_prob.cpu())
            labels_list.append(labels_np)

    all_preds = np.vstack(predictions_list)
    all_labels = np.vstack(labels_list)
    df_preds = pd.DataFrame(all_preds, columns=labels)
    df_gt = pd.DataFrame(all_labels, columns=labels)
    metrics = compute_metrics(df_gt, df_preds)
    print(f"Validation Loss: {loss.item():.4f}")
    # wandb.log({'val/validation_loss': loss.item()})
    
    return metrics


def train(classifier, train_loader, val_loader, optimizer, criterion, num_epochs, labels, scheduler, checkpoint_path):
    classifier.train()

    pbar = trange(num_epochs)
    for epoch in pbar:
        all_preds = []
        all_labels = []
        predictions_list = []
        labels_list = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            embeddings = batch['embedding'].float().to(device)
            labels_values = batch['labels_values'].to(device)

            embeddings = embeddings.to(device)
            predictions = classifier(embeddings)
            loss = criterion(predictions, labels_values)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pred_prob = torch.sigmoid(predictions)
            labels_np = labels_values.int().cpu().detach().numpy()
            predictions_list.append(pred_prob.cpu().detach().numpy())
            labels_list.append(labels_np)

        all_preds = np.vstack(predictions_list)
        all_labels = np.vstack(labels_list)
        df_preds = pd.DataFrame(all_preds, columns=labels)
        df_gt = pd.DataFrame(all_labels, columns=labels)
        metrics = compute_metrics(df_gt, df_preds)
        print(f"Epoch {epoch}, Training Loss: {loss.item():.4f}")
        print(f"Metrics: {metrics['Rhythm Disorders']}")
        wandb.log({'train/train_loss': loss.item()})
        wandb.log({'train/Rhythm Disorders': metrics['Rhythm Disorders']})
        wandb.log({'train/Conduction Disorder': metrics['Conduction Disorder']})
        wandb.log({'train/Enlargement of the heart chambers': metrics['Enlargement of the heart chambers']})
        wandb.log({'train/Pericarditis': metrics['Pericarditis']})
        wandb.log({'train/Infarction or ischemia': metrics['Infarction or ischemia']})
        wandb.log({'train/Other diagnoses': metrics['Other diagnoses']})

        os.makedirs(checkpoint_path, exist_ok=True)
        checkpoint_file = os.path.join(checkpoint_path, f"model_epoch_{epoch}.pth")
        torch.save(classifier.state_dict(), checkpoint_file)
        print(f"Model weights saved to {checkpoint_file}")

        val_metrics = evaluate(val_loader, classifier, criterion, labels)
        print(f"Validation Metrics: {val_metrics['Rhythm Disorders']}")
        wandb.log({'val/Rhythm Disorders': val_metrics['Rhythm Disorders']})
        wandb.log({'val/Conduction Disorder': metrics['Conduction Disorder']})
        wandb.log({'val/Enlargement of the heart chambersmetrics': val_metrics['Enlargement of the heart chambers']})
        wandb.log({'val/Pericarditis': val_metrics['Pericarditis']})
        wandb.log({'val/Infarction or ischemia': val_metrics['Infarction or ischemia']})
        wandb.log({'val/Other diagnoses': val_metrics['Other diagnoses']})
    
        scheduler.step()
    return

def main():
    seed = config["training"]["seed"]
    torch.random.manual_seed(seed)
    experiment_name = config["classifier"]["experiment_name"]
    wandb.init(project="ECG_tokenizer_linear_probing", entity="mhi_ai", name=experiment_name, config=config)
    parquet_file = config["dataset"]["parquet_file"]
    embedding_folder = config["evaluation"]["embedding_dir"]
    checkpoint_path = os.path.join(config["classifier"]["base_checkpoint_path"], experiment_name)
    num_classes = config["classifier"]["num_classes"]
    embedding_dim = 160
    prev_embedding_dim = 128
    num_quantizers = config["classifier"]["num_quantizers"]
    num_epochs = config["classifier"]["num_epochs"]

    dataset_mimic_train = ECGDatasetLinearProbe(parquet_file=parquet_file, embedding_folder=embedding_folder, split='train')
    train_loader = DataLoader(dataset_mimic_train, batch_size=config["classifier"]["batch_size"], shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    labels = dataset_mimic_train[0]['labels']

    dataset_mimic_val = ECGDatasetLinearProbe(parquet_file=parquet_file, embedding_folder=embedding_folder, split='val')
    val_loader = DataLoader(dataset_mimic_val, batch_size=config["classifier"]["batch_size"], shuffle=False, num_workers=16, pin_memory=True, drop_last=True)

    dataset_mimic_test = ECGDatasetLinearProbe(parquet_file=parquet_file, embedding_folder=embedding_folder, split='test')
    test_loader = DataLoader(dataset_mimic_test, batch_size=config["classifier"]["batch_size"], shuffle=False, num_workers=16, pin_memory=True, drop_last=True)

    classifier = CodebookClassifier(num_classes=num_classes, num_quantizers=num_quantizers, prev_embedding_dim=prev_embedding_dim, embedding_dim=embedding_dim, num_layers=config["classifier"]["num_layers"], hidden_dim=config["classifier"]["hidden_dim"]).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=float(config["classifier"]["lr"]), weight_decay=float(config["classifier"]["weight_decay"]))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)
    criterion = criterion = get_criterion(config["classifier"]["criterion"])
    train(classifier, train_loader, val_loader, optimizer, criterion, num_epochs, labels, scheduler, checkpoint_path)
    test_metrics = evaluate(test_loader, classifier, criterion, labels)
    print(f"Test Metrics: {test_metrics}")
    wandb.log({'test/all_metrics': test_metrics})
    
if __name__ == '__main__':
    main()