import torch.nn as nn
from vector_quantize_pytorch import ResidualVQ

from utils.registry import ModelRegistry

@ModelRegistry.register("ECG_Tokenizer_Conv_Encoder")
class ECG_Tokenizer_Conv_Encoder(nn.Module):
    """
    Encoder module that processes the input with convolutional layers to extract features.

    Expected input shape: (batch_size, 12, length)
    """
    def __init__(self):
        super(ECG_Tokenizer_Conv_Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList([
            # First convolutional block
            nn.Conv1d(12, 32, kernel_size=4, stride=2, padding=16),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.GELU(),
            # Second convolutional block
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=8),
            nn.MaxPool1d(kernel_size=2, stride=2, padding=1),
            nn.GELU(),
            # Third convolutional block (stops before quantization)
            nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=2)
        ])

    def forward(self, x):
        for layer in self.encoder_layers:
            x = layer(x)
        return x

@ModelRegistry.register("ECG_Tokenizer_Quantizer")
class ECG_Tokenizer_Quantizer(nn.Module):
    """
    Quantizer module that wraps the Residual Vector Quantization layer.

    It takes the features provided by the Encoder and quantizes them,
    returning quantized features along with indices and commitment loss.
    """
    def __init__(
        self, 
        num_quantizers: int, 
        codebook_size: int
    ):
        super(ECG_Tokenizer_Quantizer, self).__init__()
        # Adjust the latent dimension based on input timesteps.
        latent_dim = 82

        self.quantizer = ResidualVQ(
            dim=latent_dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            commitment_weight=0.25,
        )

    def forward(self, x):
        # The ResidualVQ layer returns (quantized, indices, commit_loss)
        quantized, indices, commit_loss = self.quantizer(x)
        return quantized, indices, commit_loss

@ModelRegistry.register("ECG_Tokenizer_Conv_Decoder")
class ECG_Tokenizer_Conv_Decoder(nn.Module):
    """
    Decoder module that reconstructs the input from the quantized latent representation.

    Expected input shape: (batch_size, 128, length_after_encoder)
    """
    def __init__(self):
        super(ECG_Tokenizer_Conv_Decoder, self).__init__()
        # Determine final padding based on timesteps.
        final_padding = 18

        self.decoder_layers = nn.ModuleList([
            nn.ConvTranspose1d(128, 64, kernel_size=4, stride=2, padding=2),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ConvTranspose1d(64, 32, kernel_size=4, stride=2, padding=8),
            nn.GELU(),
            nn.Upsample(scale_factor=2, mode="nearest"),
            nn.ConvTranspose1d(32, 12, kernel_size=2, stride=2, padding=final_padding)
        ])

    def forward(self, x):
        for layer in self.decoder_layers:
            x = layer(x)
        return x.clamp(-1, 1)

@ModelRegistry.register("ECG_Tokenizer_Training")
class ECG_Tokenizer_Wrapper(nn.Module):
    """
    Combined tokenization wrapper that encapsulates the encoder, quantizer, and decoder.
    
    This wrapper serves as a shell that orchestrates:
      - Encoder: extracts the ECG features.
      - Quantizer: applies residual vector quantization.
      - Decoder: reconstructs the signal from the quantized features.
    
    The forward pass returns the reconstructed output, along with the quantization indices
    and the associated commitment loss.
    """
    def __init__(
        self, 
        encoder_name="ECG_Tokenizer_Conv_Encoder", 
        quantizer_name="ECG_Tokenizer_Quantizer", 
        decoder_name="ECG_Tokenizer_Conv_Decoder",
        num_quantizers=8,
        codebook_size=512
    ):
        super(ECG_Tokenizer_Wrapper, self).__init__()

        # Save the names for potential reference
        self.encoder_name = encoder_name
        self.quantizer_name = quantizer_name
        self.decoder_name = decoder_name

        # Retrieve the components from the registry using the provided names
        self.encoder = ModelRegistry.get(encoder_name)()
        self.quantizer = ModelRegistry.get(quantizer_name)(
            num_quantizers=num_quantizers,
            codebook_size=codebook_size
        )
        self.decoder = ModelRegistry.get(decoder_name)()

    def forward(self, x):
        features = self.encoder(x)
        quantized, indices, commit_loss = self.quantizer(features)
        reconstruction = self.decoder(quantized)
        return reconstruction, indices, commit_loss

@ModelRegistry.register("ECG_CodebookClassifier")
class ECG_CodebookClassifier(nn.Module):
    def __init__(self, num_classes, num_quantizers, prev_embedding_dim, embedding_dim, num_layers=5, hidden_dim=4096):
        super(ECG_CodebookClassifier, self).__init__()
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