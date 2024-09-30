import torch 
import torch.nn as nn
import torch.nn.functional as F

"""
VQ-VAE vanilla implementation
Author: Rohan Banerjee
"""

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, embedding_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.conv1d(hidden_channels,  embedding_dim, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        return x

class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_channels, out_channels):
        super(Decoder, self).__init__()
        self.conv1 = nn.ConvTranspose1d(embedding_dim, hidden_channels, kernel_size=4, stride=2 ,padding=1)
        self.conv2 = nn.ConvTranspose1d(hidden_channels, out_channels, kernel_size=4, stride=2 ,padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x
    
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, z):
        # Flatten the input latent space
        z_flattened = z.permute(0, 2, 1).contiguous().view(-1, self.embedding_dim)

        # Find the nearest embedding index for each point in the latent space
        distances = torch.sum(z_flattened ** 2, dim=1, keepdim=True) - 2 * torch.matmul(z_flattened, self.embeddings.weight.T) + torch.sum(self.embeddings.weight ** 2, dim=1)
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)

        # Get the corresponding embeddings
        quantized = torch.index_select(self.embeddings.weight, 0, encoding_indices.squeeze()).view(z.shape)

        # Calculate losses
        e_latent_loss = F.mse_loss(quantized.detach(), z)
        q_latent_loss = F.mse_loss(quantized, z.detach())
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through gradient pass
        quantized = z + (quantized - z).detach()
        
        return quantized, loss

class VQVAE(nn.Module):
    def __init__(self, in_channels, hidden_channels, embedding_dim, num_embeddings, commitment_cost):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_channels, hidden_channels, embedding_dim)
        self.decoder = Decoder(embedding_dim, hidden_channels, in_channels)
        self.vq = VectorQuantizer(num_embeddings, embedding_dim, commitment_cost)

    def forward(self, x):
        z = self.encoder(x)  
        quantized, vq_loss = self.vq(z) 
        x_reconstructed = self.decoder(quantized)
        return x_reconstructed, vq_loss