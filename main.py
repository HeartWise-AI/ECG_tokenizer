import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn
from torchvision import datasets, transforms

from data.dataset import ECGDataset
from models.vqvae import VQVAE, SimpleVQAutoEncoder
import os
from tqdm.auto import trange
import wandb

"""
Author: Rohan Banerjee

Relevant issues from lucid-rains repos: #28, #44
"""

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def save_checkpoint(model, optimizer, epoch, checkpoint_dir='checkpoints/'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, f'vqvae_epoch_{epoch}.pth')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at epoch {epoch}")

def load_checkpoint(model, optimizer, checkpoint_dir='checkpoints/'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
        return 0  

    checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pth')]
    if not checkpoints:
        return 0

    latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('_')[-1].split('.')[0]))
    checkpoint_path = os.path.join(checkpoint_dir, latest_checkpoint)
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return start_epoch

# Define the training loop
def train_vqvae(model, dataloader, num_epochs=1, learning_rate=1e-5, checkpoint_dir='checkpoints/'):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    start_epoch = load_checkpoint(model, optimizer, checkpoint_dir=checkpoint_dir)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0

        progress_bar = tqdm(dataloader, desc=f"Epoch [{epoch + 1}/{num_epochs}]", leave=False)

        for batch in dataloader:
            signals = batch['signal'].float().to(device)
            signals = signals.permute(0, 2, 1)
            # import pdb; pdb.set_trace()

            optimizer.zero_grad()

            # Forward pass
            x_reconstructed, vq_loss = model(signals)

            # Check for NaN values in model outputs
            if torch.isnan(x_reconstructed).any() or torch.isnan(vq_loss).any():
                raise ValueError("NaN values found in model outputs")


            # Calculate reconstruction loss
            recon_loss = criterion(x_reconstructed, signals)
            loss = recon_loss + vq_loss

            # Check for NaN values in loss
            if torch.isnan(loss).any():
                raise ValueError("NaN values found in loss")

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            print(total_loss)

            progress_bar.set_postfix({"Loss": total_loss / (len(dataloader))})

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")

        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch + 1, checkpoint_dir)

def train(model, train_loader, optimizer, num_codes, train_iterations=1000, alpha=10):
    
    def iterate_dataset(data_loader):
        data_iter = iter(data_loader)
        while True:
            try:
                batch = next(data_iter)
            except StopIteration:
                data_iter = iter(data_loader)
                batch = next(data_iter)
            signals = batch['signal'].float().to(device)
            signals = signals.permute(0, 2, 1)
            yield signals

    for _ in (pbar := trange(train_iterations)):
        # import pdb; pdb.set_trace()
        optimizer.zero_grad()
        x = next(iterate_dataset(train_loader))
        print(x.min(), x.max())
        # x = x.unsqueeze(1)
        # import pdb; pdb.set_trace()
        out, indices, cmt_loss = model(x)
        rec_loss = (out - x).abs().mean()
        (rec_loss + alpha * cmt_loss).backward()

        optimizer.step()
        pbar.set_description(
            f"rec loss: {rec_loss.item():.3f} | "
            + f"cmt loss: {cmt_loss.item():.3f} | "
            + f"active %: {indices.unique().numel() / num_codes * 100:.3f}"
        )
        # Initialize Weights and Biases
        wandb.init(project="ECG_tokenizer", entity="rohanbanerjee")

        # Log the losses and active indices
        wandb.log({
            "rec_loss": rec_loss.item(),
            "cmt_loss": cmt_loss.item(),
            "active_percentage": indices.unique().numel() / num_codes * 100
        })

        if (epoch + 1) % 5 == 0:
            save_checkpoint(model, optimizer, epoch + 1, checkpoint_dir)
    return

def main():

    # Load your dataset
    csv_file = '/mnt/rbanerjee/data/MIMIC-IV/mimic_index.corrected.csv'  # Update with the path to your CSV file
    dataset_mimic = ECGDataset(csv_file=csv_file, split='train')
    train_loader = DataLoader(dataset_mimic, batch_size=512, shuffle=True, num_workers=4)

    #=======================================================================================================
    # Define the VQ-VAE model
    in_channels = 12  # ECG signals
    hidden_channels = 64  # Hidden dimension
    embedding_dim = 64  # Latent space embedding dimension
    num_embeddings = 512  # Number of discrete embeddings
    commitment_cost = 0.25  # Weight for the commitment loss

    # model = VQVAE(in_channels, hidden_channels, embedding_dim, num_embeddings, commitment_cost)
    # model = model.to(device)

    # # Train the VQ-VAE model
    # train_vqvae(model, train_loader, num_epochs=10, checkpoint_dir='checkpoints/')
    #========================================================================================================

    lr = 3e-4
    train_iter = 1000
    num_codes = 2048
    seed = 1234

    print("baseline")
    torch.random.manual_seed(seed)
    # import pdb; pdb.set_trace()
    model = SimpleVQAutoEncoder(
        timesteps=dataset_mimic.waveform_length,
        codebook_size=num_codes
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    train(model, train_loader, train_iterations=train_iter, optimizer=opt, num_codes=num_codes)

if __name__ == '__main__':
    main()
