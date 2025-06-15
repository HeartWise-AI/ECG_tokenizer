from transformers import GPT2Tokenizer
from data.ecg_clinical_report_dataset import ECGClinicalReportDataset

import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from models.gpt2_with_embeddings import GPT2WithEmbedding
from models.embedding_reducer import EmbeddingReducer
import wandb
import os
from pathlib import Path

# Initialize wandb
wandb.init(
    project="ecg-gpt2",
    config={
        "architecture": "GPT2WithEmbedding",
        "dataset": "MIMIC-ECG",
        "learning_rate": 5e-5,
        "epochs": 3,
        "batch_size": 32,
        "max_length": 512,
        "embedding_size": 768,
    },
    entity="jacques-delfrate",     
)

# Create checkpoint directory
save_dir = Path("checkpoints")
save_dir.mkdir(exist_ok=True)

# Initialize components
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.pad_token = tokenizer.eos_token


all_embeddings_path = '/media/data1/datasets/ECG_Tokenizer/embeddings/experiment_deeper_1024_rvq_QINCo/'
all_reports_path = "/media/data1/datasets/DeepECG/SSL_pretraining/split/MIMIC/mimic_v4_clean_train.parquet"

dataset = ECGClinicalReportDataset(
    embeddings_path=all_embeddings_path,
    reports_path=all_reports_path,
    tokenizer=tokenizer,
    max_length=512,
    subset_fraction=1.0  # Use only 10% of the data
)


# Split dataset
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoaders
num_workers = 16
batch_size = 32
train_loader = DataLoader(
    train_dataset, 
    batch_size=batch_size, 
    shuffle=True,
    drop_last=True,
    num_workers=num_workers,
    pin_memory=True,
    
)
val_loader = DataLoader(
    val_dataset, 
    batch_size=batch_size,
    shuffle=False,
    drop_last=True,
    num_workers=num_workers,
    pin_memory=True,
)

# Device
device = torch.device('cuda:1')

# Initialize model
gpt2_embedding_size = 768
model = GPT2WithEmbedding(
    gpt2_model_name='gpt2', 
    embedding_size=gpt2_embedding_size, 
    reducer=EmbeddingReducer(output_size=gpt2_embedding_size)
).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_epochs = 3

# Training Loop
best_val_loss = float('inf')
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    loop = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=True)
    for batch_idx, batch in enumerate(loop):
        optimizer.zero_grad()
        embeddings = batch['embedding'].to(device)
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = input_ids.clone()

        outputs = model(
            ecg_embeddings=embeddings,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        # Update total loss and mean loss
        total_loss += loss.item()
        mean_loss = total_loss / (batch_idx + 1)
        
        # Log only mean training loss
        wandb.log({
            "train/loss": mean_loss,
            "step": batch_idx + (epoch * len(train_loader))
        })
        
        # Update progress bar
        loop.set_postfix({
            'loss': f'{loss.item():.4f}',
            'mean_loss': f'{mean_loss:.4f}'
        })
    
    # Validation
    model.eval()
    val_loss = 0
    val_loop = tqdm(val_loader, desc='Validation', leave=False)
    with torch.no_grad():
        for batch_idx, batch in enumerate(val_loop):
            embeddings = batch['embedding'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = input_ids.clone()

            outputs = model(
                ecg_embeddings=embeddings,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            val_loss += outputs.loss.item()
            mean_val_loss = val_loss / (batch_idx + 1)
            
            # Log only mean validation loss
            wandb.log({
                "val/loss": mean_val_loss,
                "step": batch_idx + (epoch * len(train_loader))
            })
            
            val_loop.set_postfix({'val_loss': f'{mean_val_loss:.4f}'})
    
    avg_val_loss = val_loss / len(val_loader)
    print(f'Epoch {epoch+1}/{num_epochs} - Validation Loss: {avg_val_loss:.4f}')
    
    # Save model
    checkpoint_path = save_dir / f"model_epoch_{epoch+1}.pt"
    torch.save(model.state_dict(), checkpoint_path)
    
    # Log epoch metrics
    wandb.log({
        "train/epoch_loss": total_loss / len(train_loader),
        "val/epoch_loss": avg_val_loss,
        "epoch": epoch
    })

# Close wandb run
wandb.finish()