import torch.nn as nn
from vector_quantize_pytorch import ResidualVQ

from utils.registry import ModelRegistry
from utils.enums import DecoderMode

@ModelRegistry.register("Conv_Encoder")
class Conv_Encoder(nn.Module):
    """
    Encoder module that processes the input with convolutional layers to extract features.

    Expected input shape: (batch_size, 12, length)
    """
    def __init__(self, input_channels=12):
        super(Conv_Encoder, self).__init__()
        self.encoder_layers = nn.ModuleList([
            # First convolutional block
            nn.Conv1d(input_channels, 32, kernel_size=4, stride=2, padding=16),
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

@ModelRegistry.register('Linear_Classifier_Decoder')
class Linear_Classifier_Decoder(nn.Module):
    """
    Classifier decoder module that generates class probabilities from the quantized latent representation.

    Expected input shape: (batch_size, 128, length_after_encoder)
    Output shape: (batch_size, num_classes)
    
    This is a linear probing implementation for classifying 77 independent classes.
    """
    def __init__(
        self, 
        num_classes=77,
        dropout_rate=0.3
    ):
        super(Linear_Classifier_Decoder, self).__init__()
        self.num_classes = num_classes
        
        # Global pooling + classifier approach
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),  # Global average pooling across the time dimension
            nn.Flatten(),             # Flatten to [batch_size, 128]
            nn.Linear(128, 256),      # First dense layer
            nn.BatchNorm1d(256),      # Batch normalization for better training stability
            nn.ReLU(),                # Activation function
            nn.Dropout(dropout_rate),          # Dropout for regularization
            nn.Linear(256, 128),      # Second dense layer
            nn.BatchNorm1d(128),      # Another batch normalization
            nn.ReLU(),                # Activation function
            nn.Dropout(dropout_rate),          # More dropout
            nn.Linear(128, num_classes) # Output layer for the 77 classes (no activation - will be applied in loss)
        )

    def forward(self, x):
        return self.classifier(x)

@ModelRegistry.register("ResNet_Classifier_Decoder")
class ResNet_Classifier_Decoder(nn.Module):
    """
    ResNet classifier decoder module that generates class probabilities from the quantized latent representation.
    
    Expected input shape: (batch_size, 128, length_after_encoder)
    Output shape: (batch_size, num_classes)
    """
    def __init__(self, num_classes=77, dropout_rate=0.3):
        super(ResNet_Classifier_Decoder, self).__init__()
        self.block1 = self._make_layer(128, 128, blocks=1, stride=1)
        self.block2 = self._make_layer(128, 256, blocks=1, stride=2)
        self.block3 = self._make_layer(256, 256, blocks=1, stride=1)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

class BasicBlock(nn.Module):
    """
    Basic residual block for 1D signals.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )
        
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out

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
            implicit_neural_codebook=True
        )

    def forward(self, x):
        # The ResidualVQ layer returns (quantized, indices, commit_loss)
        quantized, indices, commit_loss = self.quantizer(x)
        return quantized, indices, commit_loss

@ModelRegistry.register("Conv_Decoder")
class Conv_Decoder(nn.Module):
    """
    Decoder module that reconstructs the input from the quantized latent representation.

    Expected input shape: (batch_size, 128, length_after_encoder)
    """
    def __init__(self):
        super(Conv_Decoder, self).__init__()
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
        return x   

@ModelRegistry.register("ECG_Tokenizer_Training")
class ECG_Tokenizer_Wrapper(nn.Module):
    """
    Combined tokenization wrapper that encapsulates the encoder, quantizer, and decoder.
    
    This wrapper serves as a shell that orchestrates:
      - Encoder: extracts the ECG features.
      - Quantizer: applies residual vector quantization.
      - Decoder: reconstructs the signal from the quantized features or performs classification.
    
    The forward pass returns either:
      - For reconstruction: (reconstruction, indices, commit_loss)
      - For classification: (class_logits, indices, commit_loss)
    """
    def __init__(
        self, 
        encoder_name="Conv_Encoder", 
        quantizer_name="ECG_Tokenizer_Quantizer", 
        decoder_name="Linear_Decoder",
        num_quantizers=8,
        codebook_size=512,
        decoder_mode=DecoderMode.RECONSTRUCTION,
        num_classes=77
    ):
        super(ECG_Tokenizer_Wrapper, self).__init__()

        # Save the names for potential reference
        self.encoder_name = encoder_name
        self.quantizer_name = quantizer_name
        self.decoder_name = decoder_name
        
        # Use the DecoderMode enum instead of a string
        self.decoder_mode = decoder_mode if isinstance(decoder_mode, DecoderMode) else DecoderMode(decoder_mode)

        # Retrieve the components from the registry using the provided names
        self.encoder = ModelRegistry.get(encoder_name)()
        self.quantizer = ModelRegistry.get(quantizer_name)(
            num_quantizers=num_quantizers,
            codebook_size=codebook_size
        )
        
        # Initialize appropriate decoder based on mode
        if self.decoder_mode == DecoderMode.CLASSIFICATION and decoder_name == "Linear_Classifier_Decoder":
            self.decoder = ModelRegistry.get(decoder_name)(num_classes=num_classes)
        else:
            self.decoder = ModelRegistry.get(decoder_name)()

    def forward(self, x):
        features = self.encoder(x)
        quantized, indices, commit_loss = self.quantizer(features)
        reconstructed_output = self.decoder(quantized)
        return reconstructed_output, indices, commit_loss

@ModelRegistry.register("ECG_CodebookClassifier")
class ECG_CodebookClassifier(nn.Module):
    def __init__(
        self, 
        num_classes, 
        num_quantizers, 
        prev_embedding_dim, 
        embedding_dim, 
        num_layers=5, 
        hidden_dim=4096
    ):
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