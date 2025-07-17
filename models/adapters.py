import torch
import torch.nn as nn
from utils.registry import ModelRegistry
from utils.enums import AdapterName

@ModelRegistry.register(AdapterName.GPT2_LINEAR_ADAPTER)
class LinearAdapter(nn.Module):
    """Simple linear adapter that flattens 3D input and maps to GPT-2 embedding size."""
    
    def __init__(
        self,
        input_shape: tuple[int, int, int] = (8, 128, 160),
        output_size: int = 768,
        dropout: float = 0.0  # Set dropout > 0 to enable dropout regularization
    ):
        """
        Args:
            input_shape: 3D input dimensions (channels, height, width)
            output_size: Target embedding dimension for GPT-2
            dropout: Dropout probability for regularization
        """        
        super(LinearAdapter, self).__init__()
        # Calculate the flattened input size (e.g. 8 * 128 * 160 = 163840)
        self.flatten_dim: int = input_shape[0] * input_shape[1] * input_shape[2]
        
        self.adapter: nn.Sequential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(1024, output_size)
        )
    
    def forward(self, x):
        """Forward pass through linear adapter."""
        return self.adapter(x) 
    
@ModelRegistry.register(AdapterName.GPT2_EMBEDDING_ADAPTER)
class EmbeddingAdapter(nn.Module):
    """CNN-based adapter that processes 3D input through convolutional layers to GPT-2 embedding size."""

    def __init__(
        self, 
        input_shape: tuple[int, int, int] = (8, 128, 160), 
        output_size: int = 768,
        dropout: float = 0.2
    ):
        """
        Args:
            input_shape: 3D input dimensions (channels, height, width)
            output_size: Target embedding dimension for GPT-2
            dropout: Dropout probability for regularization
        """
        super(EmbeddingAdapter, self).__init__()
        self.input_shape: tuple[int, int, int] = input_shape
        self.output_size: int = output_size
        self.conv_layers: nn.Sequential = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape[0], 
                out_channels=32, 
                kernel_size=3, 
                stride=2, 
                padding=1
            ),  # -> (32, 64, 80)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, 
                out_channels=64, 
                kernel_size=3, 
                stride=2, 
                padding=1
            ),  # -> (64, 32, 40)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, 
                out_channels=128, 
                kernel_size=3, 
                stride=2, 
                padding=1
            ),  # -> (128, 16, 20)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()  # -> 128 x 1 x 1 = 128
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, output_size)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through CNN layers, global pooling, and FC layers."""
        x = x / 1024.0
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x  # Shape: (batch_size, 768)

@ModelRegistry.register(AdapterName.GPT2_SIMPLE_EMBEDDING_ADAPTER)
class SimpleEmbeddingAdapter(nn.Module):
    """Minimal adapter using global average pooling and single linear layer."""
    
    def __init__(
        self,
        input_shape: tuple[int, int, int] = (8, 128, 160),
        output_size: int = 768,
        dropout: float = 0.2
    ):
        """
        Args:
            input_shape: 3D input dimensions (channels, height, width)
            output_size: Target embedding dimension for GPT-2
            dropout: Dropout probability for regularization
        """
        super(SimpleEmbeddingAdapter, self).__init__()
        # Global average pooling over the height and width dimensions.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Map from the number of channels (first element in input_shape) to the desired output size.
        self.fc = nn.Linear(input_shape[0], output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through global average pooling and single linear layer."""
        # x shape: (batch, channels, height, width)
        x = self.global_pool(x)   # -> shape: (batch, channels, 1, 1)
        x = x.view(x.size(0), -1)   # -> shape: (batch, channels)
        # Map directly to GPT-2's embedding size.
        x = self.fc(x)
        x = self.dropout(x)
        return x  # Output shape: (batch, output_size)
    
@ModelRegistry.register(AdapterName.GPT2_SEQUENCE_ADAPTER)
class SequenceAdapter(nn.Module):
    """Adapter that processes 2D input through a sequence of operations to GPT-2 embedding size."""
    
    def __init__(
        self, 
        input_shape: tuple[int, int] = (128, 82), 
        output_size: int = 768, 
        dropout: float = 0.2, 
    ):
        """
        Args:
            input_shape: 2D input dimensions (sequence length, feature dimension)
            output_size: Target embedding dimension for GPT-2
            dropout: Dropout probability for regularization
        """
        super().__init__()
        
        # Extract the actual feature dimensions
        # NOTE: For ECG encoder output (batch, 128, 82):
        # - 128 = sequence length (temporal dimension)  
        # - 82 = feature dimension (quantized features)
        # GPT2Decoder may add extra dim: (batch, 1, 128, 82) -> handled in forward()
        seq_len = input_shape[0]  # 128 = sequence length
        channels = input_shape[1]  # 82 = feature dimension
        
        # Project channels to a smaller intermediate dimension first
        self.channel_projection = nn.Sequential(
            nn.Linear(channels, output_size // 2),
            nn.LayerNorm(output_size // 2),
            nn.GELU(),
            nn.Dropout(dropout)
        )
                
        # Use attention to aggregate sequence information more intelligently
        self.attention = nn.MultiheadAttention(
            embed_dim=output_size // 2, 
            num_heads=8, 
            dropout=dropout, 
            batch_first=True
        )
        self.attention_norm = nn.LayerNorm(output_size // 2)
        
        # Final projection to output size
        self.final_projection = nn.Sequential(
            nn.Linear(output_size // 2, output_size),
            nn.LayerNorm(output_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Learnable positional embeddings for sequence length
        self.positional_embedding = nn.Parameter(
            torch.randn(1, seq_len, output_size // 2) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through sequence adapter."""
        # Handle 4D input from GPT2Decoder: (batch, 1, seq_len, channels) -> (batch, seq_len, channels)
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)  # Remove the extra dimension
        
        # Project channels: (batch, seq_len, channels) -> (batch, seq_len, output_size//2)
        x = self.channel_projection(x)
        
        # Add positional embeddings
        x = x + self.positional_embedding

        # Apply self-attention to aggregate sequence information
        attended, _ = self.attention(x, x, x)
        x = self.attention_norm(attended + x)  # Residual connection
        
        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # (batch, output_size//2)
                
        # Final projection to GPT2 embedding size
        x = self.final_projection(x)  # (batch, output_size)

        return x 

# Register Llama3.2 adapters (reusing existing implementations with different names)
@ModelRegistry.register(AdapterName.LLAMA32_SEQUENCE_ADAPTER)
class Llama32SequenceAdapter(SequenceAdapter):
    """Llama3.2 sequence adapter - same as SequenceAdapter but registered for Llama3.2."""
    
    def __init__(
        self, 
        input_shape: tuple[int, int] = (128, 82), 
        output_size: int = 2048,  # Llama3.2-1B hidden size
        dropout: float = 0.2, 
    ):
        """
        Args:
            input_shape: 2D input dimensions (sequence length, feature dimension)
            output_size: Target embedding dimension for Llama3.2 (default 2048 for 1B model)
            dropout: Dropout probability for regularization
        """
        super().__init__(input_shape, output_size, dropout)

@ModelRegistry.register(AdapterName.LLAMA32_EMBEDDING_ADAPTER)
class Llama32EmbeddingAdapter(EmbeddingAdapter):
    """Llama3.2 embedding adapter - same as EmbeddingAdapter but registered for Llama3.2."""
    
    def __init__(
        self, 
        input_shape: tuple[int, int, int] = (8, 128, 160), 
        output_size: int = 2048,  # Llama3.2-1B hidden size
        dropout: float = 0.2
    ):
        """
        Args:
            input_shape: 3D input dimensions (channels, height, width)
            output_size: Target embedding dimension for Llama3.2 (default 2048 for 1B model)
            dropout: Dropout probability for regularization
        """
        super().__init__(input_shape, output_size, dropout)

@ModelRegistry.register(AdapterName.LLAMA32_SIMPLE_EMBEDDING_ADAPTER)
class Llama32SimpleEmbeddingAdapter(SimpleEmbeddingAdapter):
    """Llama3.2 simple embedding adapter - same as SimpleEmbeddingAdapter but registered for Llama3.2."""
    
    def __init__(
        self,
        input_shape: tuple[int, int, int] = (8, 128, 160),
        output_size: int = 2048,  # Llama3.2-1B hidden size
        dropout: float = 0.2
    ):
        """
        Args:
            input_shape: 3D input dimensions (channels, height, width)
            output_size: Target embedding dimension for Llama3.2 (default 2048 for 1B model)
            dropout: Dropout probability for regularization
        """
        super().__init__(input_shape, output_size, dropout)