import torch
import torch.nn as nn
from typing import Optional, Dict, Any, Union, cast
from models.local_residual_vq import ResidualVQ
from vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize

from utils.registry import ModelRegistry
from utils.enums import DecoderMode, ModelName
from models.types import ModelT, ModelClassT
import math

try:
    from peft import LoraConfig, get_peft_model, TaskType
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

@ModelRegistry.register(ModelName.CONV_ENCODER)
class Conv_Encoder(nn.Module):
    """
    Encoder module that processes the input with convolutional layers to extract features.

    Expected input shape: (batch_size, 12, length)
    """
    def __init__(self, input_channels=12):
        """
        Args:
            input_channels: Number of input channels
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, 12, length)
        """
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
        """
        Args:
            input_channels: Number of input channels
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, 12, length)
        """
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
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1, downsample: bool = False):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for the convolution
            downsample: Whether to downsample the input
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, 12, length)
        """
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
        """
        Args:
            num_classes: Number of classes for classification
            dropout_rate: Dropout rate for regularization
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, 12, length)
        """
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
        """
        Args:
            input_dim: Dimension of the input
            num_classes: Number of classes for classification
            dropout: Dropout rate for regularization
            num_heads: Number of attention heads
        """
        super(CLS_Token_Classifier_Decoder, self).__init__()
        
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
        """
        Args:
            num_classes: Number of classes for classification
            dropout_rate: Dropout rate for regularization
        """
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
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, 12, length)
        """
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def _find_optimal_num_groups(num_channels):
    """
    Find the largest divisor of num_channels that is closest to a quarter of num_channels.
    
    Args:
    - num_channels (int): The number of channels in the input tensor.
    
    Returns:
    - int: The optimal number of groups.
    """
    target_num_groups = num_channels // 2
    best_diff = num_channels  # Initialize with the maximum possible difference
    best_divisor = 1
    for divisor in range(1, num_channels + 1):
        if num_channels % divisor == 0:
            diff = abs(divisor - target_num_groups)
            if diff < best_diff:
                best_diff = diff
                best_divisor = divisor
            elif diff == best_diff and divisor > best_divisor:
                best_divisor = divisor
    return best_divisor

def get_backbone_config(variant):
    configs = {
        'b0_v2': {'width_coefficient': 1.0, 'depth_coefficient': 1.0},
        'b1_v2': {'width_coefficient': 1.0, 'depth_coefficient': 1.1},
        'b2_v2': {'width_coefficient': 1.1, 'depth_coefficient': 1.2},
        's_v2':  {'width_coefficient': 1.0, 'depth_coefficient': 2.0},
        'm_v2':  {'width_coefficient': 1.1, 'depth_coefficient': 2.1},
    }
    return configs[variant]

def get_activation(name='relu'):
    activations = {
        'relu': nn.ReLU(),
        'swish': nn.SiLU(),
        'mish': nn.Mish(),
        'selu': nn.SELU(),
        'gelu': nn.GELU(),
        'leaky_relu': nn.LeakyReLU(0.01),
    }
    return activations[name]

class CustomNorm(nn.Module):
    def __init__(self, num_features, norm_type="batch"):
        """
        Initializes a custom normalization layer.
        
        Args:
        - num_features (int): Number of features in the input.
        - norm_type (str): Type of normalization ('batch', 'group', 'layer', 'instance').
        """
        super().__init__()
        if norm_type == "batch":
            self.norm = nn.BatchNorm1d(num_features)
        elif norm_type == "group":
            optimal_groups = _find_optimal_num_groups(num_features)
            self.norm = nn.GroupNorm(num_groups=optimal_groups, num_channels=num_features)
        elif norm_type == "layer":
            # Assuming 1D LayerNorm for simplicity; adjust as needed for your application
            self.norm = nn.LayerNorm(normalized_shape=[num_features])
        elif norm_type == "instance":
            self.norm = nn.InstanceNorm1d(num_features)
        else:
            raise ValueError(f"Unsupported norm_type {norm_type}")

    def forward(self, x):
        return self.norm(x)

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduced_dim, activation_func=nn.ReLU(inplace=True)):
        super(SEBlock, self).__init__()
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(in_channels, reduced_dim, 1),
            activation_func,
            nn.Conv1d(reduced_dim, in_channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        return x * self.se(x)

class StochasticDepth(nn.Module):
    def __init__(self, drop_prob):
        super(StochasticDepth, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.training and self.drop_prob > 0.:
            keep_prob = 1 - self.drop_prob
            shape = (x.shape[0],) + (1,) * (x.ndim - 1)
            random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
            random_tensor.floor_()
            return x.div(keep_prob) * random_tensor
        return x

class FusedMBConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, activation='relu', use_se=False, se_ratio=4, dropout_rate=0.0, stochastic_depth_prob=0.0, norm_type="batch"):
        super(FusedMBConv1d, self).__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        activation_func = get_activation(activation)

        self.fused_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=kernel_size // 2, bias=False),
            CustomNorm(out_channels, norm_type),
            activation_func,
        )

        self.stochastic_depth = StochasticDepth(stochastic_depth_prob) if self.use_residual else nn.Identity()
        self.se = SEBlock(out_channels, max(1, int(out_channels // se_ratio)), activation_func) if use_se else nn.Identity()
        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()

    def forward(self, x):
        identity = x if self.use_residual else None
        x = self.fused_conv(x)
        x = self.se(x)
        x = self.dropout(x)
        if self.use_residual:
            x = self.stochastic_depth(x) + identity
        return x

class MBConv1d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, expansion=1, activation='relu', use_se=False, se_ratio=4, dropout_rate=0.0, stochastic_depth_prob=0.0, norm_type="batch"):
        super(MBConv1d, self).__init__()
        self.use_residual = in_channels == out_channels and stride == 1
        activation_func = get_activation(activation)
        mid_channels = in_channels * expansion

        self.expand_conv = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, 1, bias=False),
            CustomNorm(mid_channels, norm_type),
            activation_func,
        ) if expansion > 1 else nn.Identity()

        self.depthwise_conv = nn.Sequential(
            nn.Conv1d(mid_channels, mid_channels, kernel_size, stride=stride, padding=kernel_size // 2, groups=mid_channels, bias=False),
            CustomNorm(mid_channels, norm_type),
            activation_func,
        )

        self.se = SEBlock(mid_channels, max(1, int(mid_channels // se_ratio)), activation_func) if use_se else nn.Identity()
        self.project_conv = nn.Sequential(
            nn.Conv1d(mid_channels, out_channels, 1, bias=False),
            CustomNorm(out_channels, norm_type),
        )

        self.dropout = nn.Dropout(p=dropout_rate) if dropout_rate > 0 else nn.Identity()
        self.stochastic_depth = StochasticDepth(stochastic_depth_prob) if self.use_residual else nn.Identity()

    def forward(self, x):
        identity = x if self.use_residual else None
        x = self.expand_conv(x)
        x = self.depthwise_conv(x)
        x = self.se(x)
        x = self.project_conv(x)
        x = self.dropout(x)
        if self.use_residual:
            x = self.stochastic_depth(x) + identity
        return x

@ModelRegistry.register(ModelName.EFFICIENTNETV2_CLASSIFIER_DECODER)
class EfficientNetV2_Classifier_Decoder(nn.Module):
    """
    EfficientNetV2 classifier decoder that generates class probabilities from quantized latent representation.
    
    Expected input shape: (batch_size, 128, length_after_encoder)
    Output shape: (batch_size, num_classes)
    """
    def __init__(
        self, 
        num_classes=77, 
        dropout_rate=0.3,
        variant='b0_v2',
        activation='swish', 
        use_se=True,
        norm_type="batch"
    ):
        """
        Args:
            num_classes: Number of classes for classification
            dropout_rate: Dropout rate for regularization
            variant: EfficientNet variant ('b0_v2', 'b1_v2', 's_v2', etc.)
            activation: Activation function to use
            use_se: Whether to use Squeeze-and-Excitation blocks
            norm_type: Type of normalization ('batch', 'group', 'layer', 'instance')
        """
        super(EfficientNetV2_Classifier_Decoder, self).__init__()
        
        # Get configuration for the variant
        config = get_backbone_config(variant)
        width_coefficient, depth_coefficient = config['width_coefficient'], config['depth_coefficient']
        
        # Configuration for decoder (adapted for 1D signals starting from 128 channels)
        base_channels = [128, 160, 192, 256, 320]  # Start from input 128 channels
        base_depths = [2, 2, 3, 3]  # Number of blocks per stage
        se_ratio = [4, 4, 4, 4]
        expansion_factors = [4, 6, 6, 6]
        kernel_sizes = [3, 3, 5, 3]
        strides = [1, 2, 1, 2]  # Downsample at stages 1 and 3
        
        # Apply scaling coefficients
        channels = [max(1, int(c * width_coefficient)) for c in base_channels]
        depths = [max(1, math.ceil(d * depth_coefficient)) for d in base_depths]
        
        # Stochastic depth probability
        stochastic_depth_prob = 0.2
        
        # Build the feature extraction layers
        self.features, final_channels = self._make_layers(
            channels, depths, kernel_sizes, strides, expansion_factors, 
            se_ratio, activation, stochastic_depth_prob, dropout_rate, 
            use_se, norm_type, variant
        )
        
        # Final conv to increase channels before pooling
        # Use the actual output channels from features instead of channels[-2]
        final_output_channels = channels[-1]  # Last channel count
        self.final_conv = nn.Sequential(
            nn.Conv1d(final_channels, final_output_channels, kernel_size=1, stride=1, padding=0, bias=False),
            CustomNorm(final_output_channels, norm_type),
            get_activation(activation)
        )
        
        # Global pooling and classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout_rate),
            nn.Linear(final_output_channels, num_classes)
        )

    def _make_layers(self, channels, depths, kernel_sizes, strides, expansion_factors, 
                    se_ratio, activation, stochastic_depth_prob, dropout_rate, 
                    use_se, norm_type, variant):
        """Build the EfficientNetV2 layers."""
        layers = []
        in_channels = channels[0]  # Start with 128 channels
        
        for i, (out_channels, num_blocks) in enumerate(zip(channels[1:], depths)):
            stride = strides[i]
            
            # Use FusedMBConv for early stages, MBConv for later stages
            use_fused = i < 2  # First 2 stages use FusedMBConv
            
            for j in range(num_blocks):
                if j > 0:  # Only the first block in each stage uses the defined stride
                    stride = 1
                
                if use_fused:
                    block = FusedMBConv1d(
                        in_channels, out_channels, kernel_sizes[i], stride, 
                        activation, use_se, se_ratio[i], dropout_rate, 
                        stochastic_depth_prob, norm_type
                    )
                else:
                    block = MBConv1d(
                        in_channels, out_channels, kernel_sizes[i], stride, 
                        expansion_factors[i], activation, use_se, se_ratio[i], 
                        dropout_rate, stochastic_depth_prob, norm_type
                    )
                
                layers.append(block)
                in_channels = out_channels  # Update for next block
                
        # Return both the layers and the final channel count
        return nn.Sequential(*layers), in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the EfficientNetV2 classifier.
        
        Args:
            x: Input tensor of shape (batch_size, 128, length_after_encoder)
            
        Returns:
            Class logits of shape (batch_size, num_classes)
        """
        # Apply EfficientNetV2 feature extraction
        x = self.features(x)
        
        # Final convolution
        x = self.final_conv(x)
        
        # Classification
        x = self.classifier(x)
        
        return x

class BasicBlock(nn.Module):
    """
    Basic residual block for 1D signals.
    """
    def __init__(self, in_channels, out_channels, stride=1):
        """
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            stride: Stride for the convolution
        """
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
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, 12, length)
        """
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

@ModelRegistry.register(ModelName.ECG_TOKENIZER_QUANTIZER_RVQ)
class ECG_Tokenizer_Quantizer_RVQ(nn.Module):
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
        """
        Args:
            num_quantizers: Number of quantizers
            codebook_size: Size of the codebook
        """
        super(ECG_Tokenizer_Quantizer_RVQ, self).__init__()
        # Adjust the latent dimension based on input timesteps.
        latent_dim = 82

        self.quantizer: ModelT = ResidualVQ(
            dim=latent_dim,
            num_quantizers=num_quantizers,
            codebook_size=codebook_size,
            commitment_weight=0.25,
            implicit_neural_codebook=False
        )

    def forward(
        self, 
        x: torch.Tensor, 
        return_all_codes: bool = False
    ):
        """
        Args:
            x: Input tensor of shape (batch_size, 12, length)
            return_all_codes: Whether to return all codes
        """
        # The ResidualVQ layer returns (quantized, indices, commit_loss)
        quantizer_outputs = self.quantizer(x, return_all_codes=return_all_codes)
        return quantizer_outputs

@ModelRegistry.register(ModelName.ECG_TOKENIZER_QUANTIZER_VANILLA)
class ECG_Tokenizer_Quantizer_Vanilla(nn.Module):
    """
    Baseline vanilla Vector Quantization that uses a single codebook (no residual quantizers).

    It wraps a single VectorQuantize layer from vector_quantize_pytorch, matching the interface
    of the residual variant for drop-in replacement within the wrapper.
    """
    def __init__(
        self,
        num_quantizers: int,  # kept for interface compatibility, must be 1 for vanilla VQ
        codebook_size: int
    ):
        super().__init__()
        if num_quantizers != 1:
            raise ValueError("ECG_Tokenizer_Quantizer_Vanilla expects num_quantizers == 1")

        latent_dim = 82

        # Use a single VectorQuantize layer with the same commitment weight default as RVQ
        self.quantizer = VectorQuantize(
            dim=latent_dim,
            codebook_size=codebook_size,
            codebook_dim=latent_dim,
            commitment_weight=0.25,
            accept_image_fmap=False
        )

    def forward(
        self,
        x: torch.Tensor,
        return_all_codes: bool = False
    ):
        # VectorQuantize returns (quantized, indices, commit_loss)
        quantized, indices, commit_loss = self.quantizer(x)

        if return_all_codes:
            all_codes = quantized.unsqueeze(0)
            return quantized, indices.unsqueeze(-1), commit_loss.unsqueeze(-1), all_codes
        return quantized, indices.unsqueeze(-1), commit_loss.unsqueeze(-1)

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
        """
        Args:
            num_quantizers: Number of quantizers
            codebook_size: Size of the codebook
        """
        super(ECG_Tokenizer_Quantizer, self).__init__()
        # Adjust the latent dimension based on input timesteps.
        latent_dim = 82

        self.quantizer: ModelT = ResidualVQ(
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
        """
        Args:
            x: Input tensor of shape (batch_size, 12, length)
            return_all_codes: Whether to return all codes
        """
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, 12, length)
        """
        for layer in self.decoder_layers:
            x = layer(x)
        return x   

@ModelRegistry.register(ModelName.ECG_TOKENIZER_WRAPPER)
@ModelRegistry.register(ModelName.ECG_TOKENIZER_LLM_FINETUNING)
@ModelRegistry.register(ModelName.ECG_TOKENIZER_LINEAR_PROBING)
@ModelRegistry.register(ModelName.LLAMA32_TOKENIZER_WRAPPER)
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
        huggingface_model_name: str = 'gpt2',
        llm_input_embedding_size: int = 768,
        adapter_name: str = "GPT2_SimpleEmbeddingAdapter",
        adapter_dropout: float = 0.2,
        use_lora: bool = False,
        lora_config: Optional[dict[str, Any]] = None
    ):
        """
        Args:
            encoder_name: Name of the encoder
            quantizer_name: Name of the quantizer
            decoder_name: Name of the decoder
            use_lora: Whether to apply LoRA to the LLM decoder
            lora_config: LoRA configuration dictionary
        """
        super(ECG_Tokenizer_Wrapper, self).__init__()

        # Save the names for potential reference
        self.encoder_name: str = encoder_name
        self.quantizer_name: str = quantizer_name
        self.decoder_name: str = decoder_name
        self.using_pretrained_weights: bool = False
        self.use_lora: bool = use_lora
        
        # Use the DecoderMode enum instead of a string
        self.decoder_mode: DecoderMode = decoder_mode if isinstance(decoder_mode, DecoderMode) else DecoderMode(decoder_mode)

        # Retrieve the components from the registry using the provided names
        encoder_class: ModelClassT = ModelRegistry.get(encoder_name)
        if encoder_class is None:
            raise ValueError(f"Encoder '{encoder_name}' not found in ModelRegistry")
        self.encoder: ModelT = encoder_class()
        
        quantizer_class: ModelClassT = ModelRegistry.get(quantizer_name)
        if quantizer_class is None:
            raise ValueError(f"Quantizer '{quantizer_name}' not found in ModelRegistry")
        quantizer_ctor = cast(Any, quantizer_class)
        self.quantizer = cast(nn.Module, quantizer_ctor(
            num_quantizers=num_quantizers,
            codebook_size=codebook_size
        ))
        
        decoder_class: ModelClassT = ModelRegistry.get(decoder_name)
        if decoder_class is None:
            raise ValueError(f"Decoder '{decoder_name}' not found in ModelRegistry")
        
        if self.decoder_mode == DecoderMode.LLM:
            try:
                quantized_feature_shape = (128, 82)
                decoder_ctor = cast(Any, decoder_class)
                self.decoder = cast(nn.Module, decoder_ctor(
                    huggingface_model_name=huggingface_model_name,
                    llm_input_embedding_size=llm_input_embedding_size,
                    quantized_feature_shape=quantized_feature_shape,
                    adapter_name=adapter_name,
                    adapter_dropout=adapter_dropout
                ))
                
                # Apply LoRA to the LLM if requested
                if use_lora and lora_config:
                    self._apply_lora(lora_config)
                    
            except TypeError as e:
                raise ValueError(
                    f"Decoder '{decoder_name}' does not support LLM mode parameters. "
                    f"For LLM mode, decoder must accept: huggingface_model_name, llm_input_embedding_size, "
                    f"quantized_feature_shape, adapter_name, and adapter_dropout. Error: {e}"
                )
        elif self.decoder_mode == DecoderMode.CLASSIFICATION:

            decoder_ctor = cast(Any, decoder_class)
            self.decoder = cast(nn.Module, decoder_ctor(num_classes=num_classes))

        elif self.decoder_mode == DecoderMode.RECONSTRUCTION:
            try:
                decoder_ctor = cast(Any, decoder_class)
                self.decoder = cast(nn.Module, decoder_ctor())
            except TypeError as e:
                raise ValueError(
                    f"Decoder '{decoder_name}' does not support reconstruction mode. "
                    f"For reconstruction, decoder should accept no parameters. Error: {e}"
                )
        else:
            raise ValueError(f"Unsupported decoder mode '{decoder_mode}' with decoder '{decoder_name}'")

    def _apply_lora(self, lora_config: dict[str, Any]):
        """Apply LoRA to the LLM component."""
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT library is not available. Please install it with: pip install peft")
        
        # Default target modules for common LLM architectures
        default_targets = {
            'gpt2': ['c_attn', 'c_proj'],
            'qwen2': ['q_proj', 'k_proj', 'v_proj', 'o_proj'],
            'llama': ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        }
        
        # Try to get the LLM model from the decoder
        llm_model = None
        if hasattr(self.decoder, 'llm_model'):
            llm_model = self.decoder.llm_model
        elif hasattr(self.decoder, 'llm'):
            llm_model = self.decoder.llm
        else:
            print("Warning: Could not find LLM model in decoder. LoRA not applied.")
            return
        
        # Determine model type and target modules
        model_type = llm_model.config.model_type.lower()
        target_modules = lora_config.get('target_modules') or default_targets.get(model_type, ['q_proj', 'v_proj'])
        
        lora_config_obj = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=lora_config.get('r', 16),
            lora_alpha=lora_config.get('alpha', 32),
            lora_dropout=lora_config.get('dropout', 0.05),
            target_modules=target_modules,
            bias=lora_config.get('bias', 'none')
        )
        
        # Apply LoRA to the LLM
        if hasattr(self.decoder, 'llm_model'):
            self.decoder.llm_model = get_peft_model(self.decoder.llm_model, lora_config_obj)
        elif hasattr(self.decoder, 'llm'):
            self.decoder.llm = get_peft_model(self.decoder.llm, lora_config_obj)
        
        print(f"Applied LoRA to LLM with config: {lora_config_obj}")

    def set_lora_inference_mode(self, inference_mode: bool = True):
        """Set LoRA inference mode to optimize for inference."""
        if not self.use_lora:
            return
            
        # Try to find LoRA model and set inference mode
        llm_model = None
        if hasattr(self.decoder, 'llm_model'):
            llm_model = self.decoder.llm_model
        elif hasattr(self.decoder, 'llm'):
            llm_model = self.decoder.llm
            
        if llm_model and hasattr(llm_model, 'peft_config'):
            for peft_config in llm_model.peft_config.values():
                peft_config.inference_mode = inference_mode
            print(f"Set LoRA inference mode to: {inference_mode}")

    def _load_state_dict(
        self, 
        state_dict: dict[str, torch.Tensor], 
        strict: bool = False
    ):
        """
        Args:
            state_dict: State dictionary to load
            strict: Whether to strictly enforce that the keys in state_dict match the keys in the model
        """
        """Load state dict selectively based on configuration."""
        print("Loading state dict...")
        
        # Check if this is a LoRA checkpoint by looking for LoRA-specific keys
        has_lora_keys = any('lora_A' in key or 'lora_B' in key or 'base_layer' in key for key in state_dict.keys())
        current_has_lora = any('lora_A' in key or 'lora_B' in key or 'base_layer' in key for key in self.state_dict().keys())
        
        if has_lora_keys and current_has_lora:
            # Both checkpoint and current model have LoRA - need to handle naming differences
            print("Loading LoRA checkpoint into LoRA model...")
            filtered_state_dict = {}
            current_state_dict = self.state_dict()
            
            for key, value in state_dict.items():
                # Handle different LoRA naming conventions
                if 'lora_A.weight' in key and 'lora_A.default.weight' not in key:
                    # Convert old naming to new naming: .lora_A.weight -> .lora_A.default.weight
                    new_key = key.replace('.lora_A.weight', '.lora_A.default.weight')
                    if new_key in current_state_dict:
                        filtered_state_dict[new_key] = value
                    else:
                        # If the new key doesn't exist, try the original
                        if key in current_state_dict:
                            filtered_state_dict[key] = value
                elif 'lora_B.weight' in key and 'lora_B.default.weight' not in key:
                    # Convert old naming to new naming: .lora_B.weight -> .lora_B.default.weight
                    new_key = key.replace('.lora_B.weight', '.lora_B.default.weight')
                    if new_key in current_state_dict:
                        filtered_state_dict[new_key] = value
                    else:
                        # If the new key doesn't exist, try the original
                        if key in current_state_dict:
                            filtered_state_dict[key] = value
                else:
                    # For all other keys, try direct mapping
                    if key in current_state_dict:
                        filtered_state_dict[key] = value
            
            self.load_state_dict(filtered_state_dict, strict=False)
        elif has_lora_keys and not current_has_lora:
            # Checkpoint has LoRA but current model doesn't - need to extract base weights
            print("Loading LoRA checkpoint into non-LoRA model...")
            filtered_state_dict = {}
            for key, value in state_dict.items():
                if 'base_layer.weight' in key:
                    # Extract base layer weights from LoRA
                    new_key = key.replace('.base_layer.weight', '.weight')
                    filtered_state_dict[new_key] = value
                elif 'lora_A' not in key and 'lora_B' not in key and 'base_layer' not in key:
                    # Keep non-LoRA weights as-is
                    filtered_state_dict[key] = value
            
            self.load_state_dict(filtered_state_dict, strict=strict)
        elif not has_lora_keys and current_has_lora:
            # Checkpoint doesn't have LoRA but current model does - load into base layers
            print("Loading non-LoRA checkpoint into LoRA model...")
            filtered_state_dict = {}
            current_state_dict = self.state_dict()
            
            for key, value in state_dict.items():
                # Try to map regular weights to LoRA base layer weights
                lora_key = key.replace('.weight', '.base_layer.weight')
                if lora_key in current_state_dict:
                    filtered_state_dict[lora_key] = value
                elif key in current_state_dict:
                    filtered_state_dict[key] = value
            
            self.load_state_dict(filtered_state_dict, strict=False)
        else:
            # Neither has LoRA - normal load
            print("Loading regular checkpoint into regular model...")
            self.load_state_dict(state_dict, strict=strict)

    def _load_pretrained_weights(
        self, 
        pretrained_state_dict: dict[str, torch.Tensor],
        freeze_pretrained_components: bool = True
    )->None:
        """
        Args:
            pretrained_state_dict: State dictionary to load
            freeze_pretrained_components: Whether to freeze the pretrained components
        """
        """Load pretrained weights selectively based on configuration."""
        print("Loading pretrained weights...")
        
        # Get current model's state dict for shape comparison
        current_state_dict = self.state_dict()
        
        # Filter state dict based on configuration
        filtered_state_dict = {}
        
        for key, value in pretrained_state_dict.items():
            should_load = False
            
            # Check encoder loading
            if key.startswith('encoder.'):
                should_load = True
                
            # Check quantizer loading (base VQ layers but skip MLPs)
            elif key.startswith('quantizer.'):
                # Load VQ layers but skip MLPs for retraining
                if 'mlps.' in key:
                    should_load = True
                else:
                    should_load = True
                            
            if should_load and key in current_state_dict:
                # Check if shapes match
                if current_state_dict[key].shape == value.shape:
                    filtered_state_dict[key] = value
                else:
                    print(f"Shape mismatch for {key}: current {current_state_dict[key].shape} vs pretrained {value.shape}")
            elif should_load:
                print(f"Key {key} not found in current model")
        
        # Load the filtered state dict
        self.load_state_dict(filtered_state_dict, strict=False)
        print(f"Loaded {len(filtered_state_dict)} parameters from pretrained model")
        
        # Freeze the loaded components (encoder and quantizer base)
        if freeze_pretrained_components:
            self._freeze_pretrained_components()
            
        self.using_pretrained_weights = True
        
    def _freeze_pretrained_components(self):
        """Freeze encoder and quantizer components (except MLPs)."""
        # Freeze encoder completely
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # Freeze quantizer VQ layers but keep MLPs trainable
        for name, param in self.quantizer.named_parameters():
            # if 'mlps.' not in name:  # Freeze everything except MLPs
            param.requires_grad = False
        
    def get_training_info(self) -> dict[str, Any]:
        """Get information about trainable vs frozen parameters."""
        trainable_params: int = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params: int = sum(p.numel() for p in self.parameters())
        
        # Component-wise breakdown
        encoder_trainable: int = sum(p.numel() for p in self.encoder.parameters() if p.requires_grad)
        encoder_total: int = sum(p.numel() for p in self.encoder.parameters())
        
        quantizer_trainable: int = sum(p.numel() for p in self.quantizer.parameters() if p.requires_grad)
        quantizer_total: int = sum(p.numel() for p in self.quantizer.parameters())
        
        decoder_trainable: int = sum(p.numel() for p in self.decoder.parameters() if p.requires_grad)
        decoder_total: int = sum(p.numel() for p in self.decoder.parameters())
        
        # MLPs specific info
        mlp_trainable: int = 0
        mlp_total: int = 0
        if (hasattr(self.quantizer, 'quantizer') and 
            hasattr(self.quantizer.quantizer, 'mlps') and 
            hasattr(self.quantizer.quantizer.mlps, 'parameters') and
            callable(getattr(self.quantizer.quantizer.mlps, 'parameters'))):
            try:
                mlp_trainable = sum(p.numel() for p in self.quantizer.quantizer.mlps.parameters() if p.requires_grad)
                mlp_total = sum(p.numel() for p in self.quantizer.quantizer.mlps.parameters())
            except (AttributeError, TypeError):
                # MLPs might exist but not be a proper PyTorch module
                mlp_trainable = 0
                mlp_total = 0
        
        return {
            "total_params": total_params,
            "trainable_params": trainable_params,
            "frozen_params": total_params - trainable_params,
            "trainable_ratio": trainable_params / total_params * 100,
            "encoder": {"trainable": encoder_trainable, "total": encoder_total},
            "quantizer": {"trainable": quantizer_trainable, "total": quantizer_total},
            "quantizer_mlps": {"trainable": mlp_trainable, "total": mlp_total},
            "decoder": {"trainable": decoder_trainable, "total": decoder_total}
        }

    def train(self, mode: bool = True):
        """
        Override train method to handle frozen components properly.
        Sets the model to training mode but keeps frozen components in eval mode.
        """
        # Call parent train method first
        super().train(mode)
        
        # If we're in training mode, set frozen components to eval mode
        if mode and self.using_pretrained_weights:
            self._set_frozen_components_to_eval()
        
        return self
    
    def _set_frozen_components_to_eval(self):
        """Set frozen components to eval mode to prevent BatchNorm updates."""
        # Check if encoder is frozen and set to eval mode
        encoder_frozen: bool = all(not p.requires_grad for p in self.encoder.parameters())
        if encoder_frozen:
            self.encoder.eval()
            
        # For quantizer, check if base layers are frozen
        if hasattr(self.quantizer, 'quantizer'):
            # Check if VQ layers (non-MLP parts) are frozen
            vq_params_frozen = True
            for name, param in self.quantizer.named_parameters():
                if 'mlps.' not in name and param.requires_grad:
                    vq_params_frozen = False
                    break
            
            if vq_params_frozen:
                # Set the entire quantizer to eval, then set MLPs back to train if they're trainable
                self.quantizer.eval()
                
                # Check if MLPs should be in training mode
                if (hasattr(self.quantizer.quantizer, 'mlps') and 
                    hasattr(self.quantizer.quantizer.mlps, 'parameters') and
                    callable(getattr(self.quantizer.quantizer.mlps, 'parameters'))):
                    try:
                        mlp_params_trainable = any(p.requires_grad for p in self.quantizer.quantizer.mlps.parameters())
                        if mlp_params_trainable:
                            self.quantizer.quantizer.mlps.train()
                    except (AttributeError, TypeError):
                        # MLPs might exist but not be a proper PyTorch module
                        pass
    
    def _check_frozen_components_status(self):
        """Debug method to check the training status of components."""
        print("Component training status:")
        print(f"  Encoder: {'TRAIN' if self.encoder.training else 'EVAL'}")
        print(f"  Quantizer: {'TRAIN' if self.quantizer.training else 'EVAL'}")
        
        # Safely check MLPs status
        mlps = getattr(getattr(self.quantizer, 'quantizer', None), 'mlps', None)
        if mlps is not None and hasattr(mlps, 'training') and isinstance(mlps, nn.Module):
            print(f"  Quantizer MLPs: {'TRAIN' if mlps.training else 'EVAL'}")
        else:
            print(f"  Quantizer MLPs: Not available")
            
        print(f"  Decoder: {'TRAIN' if self.decoder.training else 'EVAL'}")

    def forward(
        self, 
        ecg_signal: torch.Tensor, 
        return_all_codes: bool = False,
        # Additional parameters for LLM mode
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    )->Union[Dict[str, Any], tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]]:
        """
        Args:
            ecg_signal: Input tensor of shape (batch_size, 12, length)
            return_all_codes: Whether to return all codes
            input_ids: Input IDs for the LLM
            attention_mask: Attention mask for the LLM
            labels: Labels for the LLM
        """
        ecg_signal = ecg_signal.to(dtype=torch.float32)  # or torch.bfloat16 if you prefer
        features = self.encoder(ecg_signal)
        
        quantizer_outputs = self.quantizer(
            features,
            return_all_codes=return_all_codes
        )
                
        # If no decoder, return quantized, indices, commit_loss
        if not self.decoder:
            return quantizer_outputs
        
        if return_all_codes:
            quantized, indices, commit_loss, all_codes = quantizer_outputs
        else:
            quantized, indices, commit_loss = quantizer_outputs
            all_codes = None

        # Handle different decoder types
        if self.decoder_mode == DecoderMode.LLM:
            try:
                decoder_output = self.decoder(
                    quantized_features=quantized,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                if isinstance(decoder_output, dict):
                    return decoder_output
                else:
                    reconstructed_output = decoder_output
                    return {"logits": reconstructed_output, "indices": indices, "commit_loss": commit_loss}
            except TypeError as e:
                # If the decoder doesn't accept LLM parameters, fall back to basic call
                raise ValueError(
                    f"LLM decoder '{self.decoder_name}' does not accept expected LLM parameters "
                    f"(quantized_features, input_ids, attention_mask, labels). Error: {e}"
                )
        else:
            reconstructed_output = self.decoder(quantized)
            
        if return_all_codes:
            return reconstructed_output, indices, commit_loss, all_codes
        else:
            return reconstructed_output, indices, commit_loss, None

    @torch.no_grad()
    def generate_report(
        self,
        x: torch.Tensor,
        max_token_length: int = 512,
        **generate_kwargs
    ) -> torch.Tensor:
        """
        Generate a clinical report from ECG signal using any LLM decoder.
        Available when decoder_mode is LLM and decoder supports generation.
        
        Args:
            x: ECG signal tensor
            max_token_length: Maximum length of generated tokens
            **generate_kwargs: Additional arguments for generation
            
        Returns:
            Generated token IDs
        """
        if self.decoder_mode != DecoderMode.LLM:
            raise ValueError("generate_report() is only available in LLM mode")
        
        if self.decoder is None:
            raise ValueError("No decoder available for generation")
            
        # Ensure input is in the right dtype
        x = x.to(dtype=torch.float32)
        
        # Get quantized features
        features = self.encoder(x)
        quantized, _, _ = self.quantizer(features)
        
        # Check if decoder has a generate method
        if not hasattr(self.decoder, 'generate_report'):
            raise ValueError(f"Decoder '{self.decoder_name}' does not support text generation")
        
        return self.decoder.generate_report(
            quantized_features=quantized,
            max_token_length=max_token_length,
            **generate_kwargs
        )

    @torch.no_grad()
    def generate_report_with_question(
        self,
        x: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        max_token_length: int = 512,
        **generate_kwargs
    ) -> torch.Tensor:
        """Generate conditioned on a chat-formatted question prompt."""
        if self.decoder_mode != DecoderMode.LLM:
            raise ValueError("generate_report_with_question() is only available in LLM mode")
        if self.decoder is None:
            raise ValueError("No decoder available for generation")

        x = x.to(dtype=torch.float32)
        features = self.encoder(x)
        quantized, _, _ = self.quantizer(features)

        if not hasattr(self.decoder, 'generate_report_with_question'):
            raise ValueError(f"Decoder '{self.decoder_name}' does not support question-conditioned generation")

        return self.decoder.generate_report_with_question(
            quantized_features=quantized,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            max_token_length=max_token_length,
            **generate_kwargs
        )