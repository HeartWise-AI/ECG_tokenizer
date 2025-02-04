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
            embeddings = embeddings.permute(1, 0, 2, 3)
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
    
    return metrics


def train(classifier, train_loader, val_loader, test_loader, optimizer, criterion, num_epochs, labels, scheduler, checkpoint_path):
    classifier.train()

    pbar = trange(num_epochs)
    for epoch in pbar:
        epoch_losses = []
        all_preds = []
        all_labels = []
        predictions_list = []
        labels_list = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            embeddings = batch['embedding'].float().to(device)
            labels_values = batch['labels_values'].to(device)

            embeddings = embeddings.permute(1, 0, 2, 3)
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
        print(f"Epoch {epoch + 1}, Training Loss: {loss.item():.4f}")
        print(f"Metrics: {metrics['Rhythm Disorders']}")
        wandb.log({'Rhythm Disorders': metrics['Rhythm Disorders']})
        wandb.log({'Enlargement of the heart chambersmetrics': metrics['Enlargement of the heart chambers']})
        wandb.log({'Pericarditis': metrics['Pericarditis']})
        wandb.log({'Infarction or ischemia': metrics['Infarction or ischemia']})
        wandb.log({'Other diagnoses': metrics['Other diagnoses']})

        if (epoch + 1) % 5 == 0:
            val_metrics = evaluate(val_loader, classifier, criterion, labels)
            print(f"Validation Metrics: {val_metrics['Rhythm Disorders']}")
            wandb.log({'Validation Rhythm Disorders': val_metrics['Rhythm Disorders']})
            wandb.log({'Validation Enlargement of the heart chambersmetrics': val_metrics['Enlargement of the heart chambers']})
            wandb.log({'Validation Pericarditis': val_metrics['Pericarditis']})
            wandb.log({'Validation Infarction or ischemia': val_metrics['Infarction or ischemia']})
            wandb.log({'Validation Other diagnoses': val_metrics['Other diagnoses']})

            checkpoint_dir = checkpoint_path
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pth")
            torch.save(classifier.state_dict(), checkpoint_path)
            print(f"Model weights saved to {checkpoint_path}")
        
        scheduler.step()
    return

def main():
    seed = config["training"]["seed"]
    torch.random.manual_seed(seed)
    wandb.init(project="ECG_tokenizer_linear_probing", entity="mhi_ai", name="lp_shallow", config=config)
    parquet_file = config["dataset"]["parquet_file"]
    embedding_folder = config["evaluation"]["embedding_dir"]
    checkpoint_path = config["evaluation"]["classifier_checkpoint"]
    num_classes = 77
    embedding_dim = 160
    prev_embedding_dim = 128
    num_quantizers = 8
    num_epochs = 1000

    dataset_mimic_train = ECGDatasetLinearProbe(parquet_file=parquet_file, embedding_folder=embedding_folder, split='train')
    train_loader = DataLoader(dataset_mimic_train, batch_size=8, shuffle=True, num_workers=16, pin_memory=True, drop_last=True)
    labels = dataset_mimic_train[0]['labels']

    dataset_mimic_val = ECGDatasetLinearProbe(parquet_file=parquet_file, embedding_folder=embedding_folder, split='val')
    val_loader = DataLoader(dataset_mimic_val, batch_size=8, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)

    dataset_mimic_test = ECGDatasetLinearProbe(parquet_file=parquet_file, embedding_folder=embedding_folder, split='test')
    test_loader = DataLoader(dataset_mimic_test, batch_size=128, shuffle=False, num_workers=16, pin_memory=True, drop_last=True)

    classifier = CodebookClassifier(num_classes=num_classes, num_quantizers=num_quantizers, prev_embedding_dim=prev_embedding_dim, embedding_dim=embedding_dim).to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=1e-6, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-7)
    criterion = FocalLoss(logits=True)
    # criterion = nn.BCEWithLogitsLoss()
    train(classifier, train_loader, val_loader, test_loader, optimizer, criterion, num_epochs, labels, scheduler, checkpoint_path)
    
if __name__ == '__main__':
    main()