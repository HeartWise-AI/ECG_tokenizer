import torch
import torch.nn as nn
from typing import List
from models.local_residual_vq import ResidualVQ

from utils.registry import ModelRegistry
from utils.enums import DecoderMode, ModelName

@ModelRegistry.register(ModelName.CONV_ENCODER)
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

@ModelRegistry.register(ModelName.RESIDUAL_CONV_ENCODER)
class Residual_Conv_Encoder(nn.Module):
    """
    Residual encoder with skip connections for better ECG feature preservation.
    
    Expected input shape: (batch_size, 12, length)
    """
    def __init__(self, input_channels=12):
        super(Residual_Conv_Encoder, self).__init__()
        
        # Match the original encoder's pattern more closely
        # First block: 12 -> 32 channels (with larger stride to mimic conv+pool)
        self.conv1 = nn.Conv1d(input_channels, 32, kernel_size=4, stride=2, padding=16)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.block1 = ResidualBlock1D(32, 32, stride=1, downsample=False)
        
        # Second block: 32 -> 64 channels
        self.conv2 = nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=8)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=1)
        self.block2 = ResidualBlock1D(64, 64, stride=1, downsample=False)
        
        # Third block: 64 -> 128 channels  
        self.conv3 = nn.Conv1d(64, 128, kernel_size=4, stride=2, padding=2)
        self.block3 = ResidualBlock1D(128, 128, stride=1, downsample=False)

    def forward(self, x):
        # First conv + pool + residual refinement
        x = self.conv1(x)
        x = self.pool1(x)
        x = nn.GELU()(x)
        x = self.block1(x)
        
        # Second conv + pool + residual refinement
        x = self.conv2(x)
        x = self.pool2(x)
        x = nn.GELU()(x)
        x = self.block2(x)
        
        # Third conv + residual refinement
        x = self.conv3(x)
        x = self.block3(x)
        
        return x

class ResidualBlock1D(nn.Module):
    """
    1D Residual block for ECG signal processing.
    """
    def __init__(self, in_channels, out_channels, stride=1, downsample=False):
        super(ResidualBlock1D, self).__init__()
        
        # Main path - use consistent padding
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, 
                              stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.gelu1 = nn.GELU()
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, 
                              stride=1, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Skip connection
        self.downsample_layer = None
        if downsample or in_channels != out_channels:
            # Match dimensions for skip connection
            self.downsample_layer = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, 
                         stride=stride, padding=0),
                nn.BatchNorm1d(out_channels)
            )
        
        self.gelu2 = nn.GELU()

    def forward(self, x):
        identity = x
        
        # Main path
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.gelu1(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Skip connection
        if self.downsample_layer is not None:
            identity = self.downsample_layer(x)
        
        # Add skip connection
        out += identity
        out = self.gelu2(out)
        
        return out

@ModelRegistry.register(ModelName.LINEAR_CLASSIFIER_DECODER)
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

@ModelRegistry.register(ModelName.CLS_TOKEN_CLASSIFIER_DECODER)
class CLS_Token_Classifier_Decoder(nn.Module):
    """
    CLS token based classifier decoder that uses self-attention to aggregate 
    quantized features for classification tasks.
    
    Expected input shape: (batch_size, 128, length_after_encoder)
    Output shape: (batch_size, num_classes)
    
    Uses a learnable CLS token that attends to all temporal positions 
    to create a global sequence representation for classification.
    """
    def __init__(
        self, 
        input_dim: int = 128,
        num_classes: int = 77, 
        dropout: float = 0.1,
        num_heads: int = 8,
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.num_heads = num_heads

        # Initialize learnable cls_token
        self.cls_token = nn.Parameter(torch.randn(1, 1, input_dim))

        # Self-attention for cls_token processing
        self.cls_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Normalization and dropout layers
        self.cls_norm = nn.LayerNorm(input_dim)
        self.cls_dropout = nn.Dropout(dropout)

        # Classification head - similar to your existing Linear_Classifier_Decoder
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

        # Initialize parameters
        self._reset_parameters()

    def _reset_parameters(self):
        """Initialize parameters with proper scaling."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Initialize cls_token with small random values
        nn.init.normal_(self.cls_token, std=0.02)

    def forward(self, x):
        """
        Forward pass with cls_token aggregation.
        
        Args:
            x: Input tensor of shape [batch_size, 128, length_after_encoder]
            
        Returns:
            Classification logits of shape [batch_size, num_classes]
        """
        # Input from quantizer: [batch_size, 128, length_after_encoder]
        # Transpose to sequence format: [batch_size, length_after_encoder, 128]
        x = x.transpose(1, 2)  # [B, L, D] where D=128
        
        B, L, D = x.shape

        # Expand cls_token for batch
        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, D]

        # Concatenate cls_token with input embeddings
        x_with_cls = torch.cat([cls_tokens, x], dim=1)  # [B, L+1, D]

        # Apply self-attention
        attn_out, _ = self.cls_attention(
            query=x_with_cls,
            key=x_with_cls,
            value=x_with_cls
        )  # [B, L+1, D]

        # Extract cls_token output (first token)
        cls_output = attn_out[:, 0, :]  # [B, D]

        # Apply normalization and dropout
        aggregated_features = self.cls_norm(cls_output)
        aggregated_features = self.cls_dropout(aggregated_features)

        # Apply classification head
        return self.classifier(aggregated_features)

@ModelRegistry.register(ModelName.RESNET_CLASSIFIER_DECODER)
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

@ModelRegistry.register(ModelName.ECG_TOKENIZER_QUANTIZER)
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

    def forward(
        self, 
        x: torch.Tensor, 
        return_all_codes: bool = False
    ):
        # The ResidualVQ layer returns (quantized, indices, commit_loss)
        quantizer_outputs = self.quantizer(x, return_all_codes=return_all_codes)
        return quantizer_outputs

@ModelRegistry.register(ModelName.CONV_DECODER)
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

@ModelRegistry.register(ModelName.ECG_TOKENIZER_WRAPPER)
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
        encoder_name: str = "Conv_Encoder", 
        quantizer_name: str = "ECG_Tokenizer_Quantizer", 
        decoder_name: str = "Linear_Decoder",
        num_quantizers: int = 8,
        codebook_size: int = 512,
        decoder_mode: DecoderMode = DecoderMode.LLM,
        num_classes: int = 77,
    ):
        super(ECG_Tokenizer_Wrapper, self).__init__()

        # Save the names for potential reference
        self.encoder_name: str = encoder_name
        self.quantizer_name: str = quantizer_name
        self.decoder_name: str = decoder_name
        
        # Use the DecoderMode enum instead of a string
        self.decoder_mode: DecoderMode = decoder_mode if isinstance(decoder_mode, DecoderMode) else DecoderMode(decoder_mode)

        # Retrieve the components from the registry using the provided names
        encoder_class = ModelRegistry.get(encoder_name)
        if encoder_class is None:
            raise ValueError(f"Encoder '{encoder_name}' not found in ModelRegistry")
        self.encoder = encoder_class()
        
        quantizer_class = ModelRegistry.get(quantizer_name)
        if quantizer_class is None:
            raise ValueError(f"Quantizer '{quantizer_name}' not found in ModelRegistry")
        self.quantizer = quantizer_class(
            num_quantizers=num_quantizers,
            codebook_size=codebook_size
        )
        
        # Initialize appropriate decoder based on mode
        decoder_class = ModelRegistry.get(decoder_name)
        if decoder_class is None:
            raise ValueError(f"Decoder '{decoder_name}' not found in ModelRegistry")
            
        if self.decoder_mode == DecoderMode.CLASSIFICATION and decoder_name in ["Linear_Classifier_Decoder", "CLS_Token_Classifier_Decoder"]:
            self.decoder = decoder_class(num_classes=num_classes)
        elif self.decoder_mode == DecoderMode.CLASSIFICATION and decoder_name == "ResNet_Classifier_Decoder":
            self.decoder = decoder_class(num_classes=num_classes)
        else:
            self.decoder = decoder_class()

    def forward(
        self, 
        x: torch.Tensor, 
        return_all_codes: bool = False
    ):
        features = self.encoder(x)
        
        quantizer_outputs = self.quantizer(
            features,
            return_all_codes=return_all_codes
        )
                
        # If no decoder, return quantized, indices, commit_loss
        if not self.decoder: # happens when we to use the quantized ecg embeddings as input to the LLM or other models
            return quantizer_outputs
        
        if return_all_codes:
            quantized, indices, commit_loss, all_codes = quantizer_outputs
            reconstructed_output = self.decoder(quantized)
            return reconstructed_output, indices, commit_loss, all_codes
        else:
            quantized, indices, commit_loss = quantizer_outputs
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
        layers: List[nn.Module] = [nn.Flatten()]
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