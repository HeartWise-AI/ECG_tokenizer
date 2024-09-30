import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import torch.nn as nn

from data.dataset import ECGDataset
from models.vqvae import VQVAE

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the training loop
def train_vqvae(model, dataloader, num_epochs=1, learning_rate=1e-3):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            signals = batch['signal'].unsqueeze(1).float()  # Add a channel dimension
            signals = signals.to(device)

            optimizer.zero_grad()

            # Forward pass
            x_reconstructed, vq_loss = model(signals)

            # Calculate reconstruction loss
            recon_loss = criterion(x_reconstructed, signals)
            loss = recon_loss + vq_loss

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(dataloader):.4f}")

def main():

    # Load your dataset
    csv_file = '/media/data1/ravram/MIMIC-IV/mimic_index.corrected.csv'  # Update with the path to your CSV file
    dataset_mimic = ECGDataset(csv_file=csv_file, split='train')
    train_loader = DataLoader(dataset_mimic, batch_size=1, shuffle=True, num_workers=4)

    # Define the VQ-VAE model
    in_channels = 1  # ECG signals, so single channel
    hidden_channels = 64  # Hidden dimension
    embedding_dim = 64  # Latent space embedding dimension
    num_embeddings = 512  # Number of discrete embeddings
    commitment_cost = 0.25  # Weight for the commitment loss

    model = VQVAE(in_channels, hidden_channels, embedding_dim, num_embeddings, commitment_cost)
    model = model.to(device)

    # Train the VQ-VAE model
    train_vqvae(model, train_loader, num_epochs=20)

if __name__ == '__main__':
    main()
