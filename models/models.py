import torch 
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import VectorQuantize, ResidualVQ

"""
VQ-VAE vanilla implementation
Author: Rohan Banerjee
"""

class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, embedding_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv1d(hidden_channels,  embedding_dim, kernel_size=4, stride=2, padding=1)

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

class SimpleVQAutoEncoder(nn.Module):
    def __init__(self, timesteps, **vq_kwargs):
            super().__init__()
            self.layers = nn.ModuleList(
            [
                    nn.Conv1d(12, 32, kernel_size=4, stride=2, padding=1), 
                    nn.MaxPool1d(kernel_size=2, stride=2),
                    nn.GELU(),
                    nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
                    VectorQuantize(dim=timesteps // 8,
                                    decay = 0.8,             # the exponential moving average decay, lower means the dictionary will change faster
                                    commitment_weight = 0.25,
                                    **vq_kwargs),
                    nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=1),
                    nn.GELU(),
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.ConvTranspose1d(32, 12, kernel_size=4, stride=2, padding=1),
                ]
            )
            return

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, VectorQuantize):
                x, indices, commit_loss = layer(x)
            else:
                x = layer(x)
           
        return x.clamp(-1, 1), indices, commit_loss
    
class ResVQAutoEncoder(nn.Module):
    def __init__(
        self, 
        timesteps, 
        **vq_kwargs
    ):
        super().__init__()
        if timesteps == 5000:
            ResidualVQ_input = 160
            final_padding = 16
        else:
            ResidualVQ_input = 82
            final_padding = 18
        self.layers = nn.ModuleList(
            [
                nn.Conv1d(12, 32, kernel_size=4, stride=2, padding=16), 
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
                nn.GELU(),
                nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=8),
                nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
                nn.GELU(),
                nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=2),
                ResidualVQ(dim=ResidualVQ_input,
                            num_quantizers = 8,
                            commitment_weight = 0.25,
                            **vq_kwargs),
                nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=2),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=8),
                nn.GELU(),
                nn.Upsample(scale_factor=2, mode="nearest"),
                nn.ConvTranspose1d(32, 12, kernel_size=2, stride=2, padding=final_padding)
            ]
        )

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if isinstance(layer, ResidualVQ):
                x, indices, commit_loss = layer(x)
            else:
                x = layer(x)
           
        return x, indices, commit_loss
    
class CodebookClassifier(nn.Module):
    def __init__(self, num_classes, num_quantizers, prev_embedding_dim, embedding_dim, num_layers=5, hidden_dim=4096):
        super(CodebookClassifier, self).__init__()
        layers = [nn.Flatten()]
        input_dim = num_quantizers * prev_embedding_dim * embedding_dim

        if num_layers == 0:
            # Directly connect input to output.
            layers.append(nn.Linear(input_dim, num_classes))
        else:
            # First hidden layer.
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            
            current_dim = hidden_dim
            # Create additional hidden layers with a reduction strategy.
            for _ in range(num_layers - 1):
                # For instance, reduce dimension by half each time (with a lower bound of 256).
                next_dim = current_dim // 2
                layers.append(nn.Linear(current_dim, next_dim))
                layers.append(nn.ReLU())
                current_dim = next_dim

            # Final classification layer.
            layers.append(nn.Linear(current_dim, num_classes))

        self.classifier = nn.Sequential(*layers)
    
    def forward(self, codebook_embeddings):
        return self.classifier(codebook_embeddings)