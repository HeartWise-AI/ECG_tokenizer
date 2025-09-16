import torch
import torch.nn as nn
from typing import Optional
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
@ModelRegistry.register(AdapterName.LLAMA32_EMBEDDING_ADAPTER)
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
@ModelRegistry.register(AdapterName.LLAMA32_SIMPLE_EMBEDDING_ADAPTER)
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
@ModelRegistry.register(AdapterName.LLAMA32_SEQUENCE_ADAPTER)
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

@ModelRegistry.register(AdapterName.GPT2_SEQUENCE_TOKEN_ADAPTER)
@ModelRegistry.register(AdapterName.LLAMA32_SEQUENCE_TOKEN_ADAPTER)
class SequenceTokenAdapter(nn.Module):
    """
    Adapter that converts each of the 128 ECG positions into separate LLM tokens
    instead of compressing them into a single embedding.
    
    This preserves fine-grained spatial/temporal information and allows the LLM
    to attend to specific ECG regions during text generation.
    """
    
    def __init__(
        self, 
        input_shape: tuple[int, int] = (128, 82), 
        output_size: int = 2048, 
        dropout: float = 0.05,
        use_cross_attention: bool = True,
        num_attention_heads: int = 8,
        intermediate_dim: int = None
    ):
        """
        Initialize the sequence token adapter.
        
        Args:
            input_shape: (seq_len, feature_dim) = (128, 82) for ECG quantized features
            output_size: LLM embedding dimension (e.g., 2048 for Llama-3.2-1B)
            dropout: Dropout rate for regularization
            use_cross_attention: Whether to apply cross-attention between positions
            num_attention_heads: Number of attention heads for cross-attention
            intermediate_dim: Intermediate projection dimension (defaults to output_size // 2)
        """
        super().__init__()
        
        seq_len, feature_dim = input_shape  # 128, 82
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.output_size = output_size
        self.use_cross_attention = use_cross_attention
        self.num_tokens = seq_len  # For compatibility with decoder
        
        if intermediate_dim is None:
            intermediate_dim = output_size // 2
        
        # Project each position's features to LLM embedding size
        self.token_projection = nn.Sequential(
            # nn.Linear(feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim, output_size),
            nn.LayerNorm(output_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Learnable positional embeddings for each ECG position
        # This helps the model understand temporal/spatial relationships
        self.positional_embedding = nn.Parameter(
            torch.randn(1, seq_len, output_size) * 0.02
        )
        
        # Optional cross-attention for position refinement
        if use_cross_attention:
            self.cross_attention = nn.MultiheadAttention(
                embed_dim=output_size,
                num_heads=num_attention_heads,
                dropout=dropout,
                batch_first=True
            )
            self.attention_norm = nn.LayerNorm(output_size)
            self.attention_dropout = nn.Dropout(dropout)
        
        # Additional processing layers for better representation
        self.final_refinement = nn.Sequential(
            nn.Linear(output_size, output_size),
            nn.LayerNorm(output_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        text_embeddings: Optional[torch.Tensor] = None,
        text_attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Convert quantized ECG features to sequence of LLM tokens with optional cross-modal attention.

        Args:
            x: Quantized features [batch, 128, 82] or [batch, 1, 128, 82]
            text_embeddings: Optional text embeddings [batch, text_len, output_size] for cross-attention
            text_attention_mask: Optional mask [batch, text_len] indicating valid tokens

        Returns:
            Token embeddings [batch, 128, output_size] - one token per ECG position
        """
        # Handle 4D input from some decoder configurations
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)  # Remove singleton dimension

        if x.dim() != 3:
            raise ValueError(f"Expected 3D input [batch, seq_len, features], got {x.shape}")

        batch_size, seq_len, feature_dim = x.shape

        if seq_len != self.seq_len or feature_dim != self.feature_dim:
            raise ValueError(
                f"Input shape mismatch. Expected [{batch_size}, {self.seq_len}, {self.feature_dim}], "
                f"got [{batch_size}, {seq_len}, {feature_dim}]"
            )

        # Project each position independently: [batch, 128, 82] -> [batch, 128, output_size]
        token_embeddings = self.token_projection(x)
        token_embeddings = token_embeddings + self.positional_embedding

        key_padding_mask = None
        if text_attention_mask is not None:
            if text_attention_mask.dim() == 1:
                text_attention_mask = text_attention_mask.unsqueeze(0)
            if text_attention_mask.dim() != 2:
                raise ValueError("text_attention_mask must be [batch, seq] if provided")
            key_padding_mask = ~text_attention_mask.bool()

        # Apply cross-attention with text if provided, otherwise self-attention
        if self.use_cross_attention:
            if text_embeddings is not None:
                # Cross-modal attention: ECG queries attend to text keys/values
                attended, attention_weights = self.cross_attention(
                    query=token_embeddings,
                    key=text_embeddings,
                    value=text_embeddings,
                    key_padding_mask=key_padding_mask
                )
            else:
                # Self-attention among ECG positions
                attended, attention_weights = self.cross_attention(
                    query=token_embeddings,
                    key=token_embeddings,
                    value=token_embeddings
                )

            token_embeddings = self.attention_norm(
                token_embeddings + self.attention_dropout(attended)
            )

        token_embeddings = self.final_refinement(token_embeddings)

        return token_embeddings  # [batch, 128, output_size]
    
    def get_sequence_length(self) -> int:
        """Return the number of tokens this adapter produces."""
        return self.seq_len
    
    def get_token_info(self) -> dict:
        """Return information about the tokens produced by this adapter."""
        return {
            "num_tokens": self.seq_len,
            "token_dim": self.output_size,
            "token_type": "sequence",
            "description": f"Each of {self.seq_len} ECG positions becomes a separate token"
        }


@ModelRegistry.register(AdapterName.GPT2_SIMPLE_TOKEN_ADAPTER)
@ModelRegistry.register(AdapterName.LLAMA32_SIMPLE_TOKEN_ADAPTER)
class SimpleTokenAdapter(nn.Module):
    """
    Simple adapter that converts each of the 128 ECG positions into separate LLM tokens
    using a straightforward linear projection approach.
    
    This creates 128 tokens with simple linear projection from 82-dim features 
    to LLM embedding size (2048), without complex attention mechanisms.
    """
    
    def __init__(
        self, 
        input_shape: tuple[int, int] = (128, 82), 
        output_size: int = 2048,
        dropout: float = 0.1
    ):
        """
        Args:
            input_shape: (seq_len, feature_dim) = (128, 82) for ECG quantized features
            output_size: LLM embedding dimension (e.g., 2048 for Llama-3.2-1B)
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.seq_len, self.feature_dim = input_shape  # 128, 82
        self.output_size = output_size
        self.num_tokens = self.seq_len  # For compatibility with decoder (128 tokens)
        self.input_layernorm = nn.LayerNorm(self.feature_dim)
        
        # Simple linear projection for each position
        self.token_projection = nn.Sequential(
            nn.Linear(self.feature_dim, output_size),  # 82 -> 2048
            nn.LayerNorm(output_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert quantized ECG features to sequence of LLM tokens.
        
        Args:
            x: Quantized features [batch, 128, 82] or [batch, 1, 128, 82]
            
        Returns:
            Token embeddings [batch, 128, output_size] - one token per ECG position
        """

        if x.dim() == 4:
            x = x.squeeze(1)
        
        batch_size, seq_len, feature_dim = x.shape
        assert seq_len == self.seq_len, f"Expected sequence length {self.seq_len}, got {seq_len}"
        assert feature_dim == self.feature_dim, f"Expected feature dim {self.feature_dim}, got {feature_dim}"
        
        # CRITICAL: Normalize ECG features to [-1, 1] range to match LLM embedding scale
        # This prevents scale mismatch issues during training
        x_min = x.min(dim=-1, keepdim=True)[0]  # [batch, 128, 1]
        x_max = x.max(dim=-1, keepdim=True)[0]  # [batch, 128, 1]
        x_range = x_max - x_min
        
        # # Avoid division by zero for constant features
        # x_range = torch.clamp(x_range, min=1e-8)
        
        # # Normalize to [0, 1] then scale to [-1, 1]
        # x_normalized = (x - x_min) / x_range  # [0, 1]
        # x_scaled = 0.2 * x_normalized - 0.1   # [-0.1, 0.1]
        
        # # Apply linear projection to normalized features: [batch, 128, 82] -> [batch, 128, 2048]
        # token_embeddings = self.token_projection(x_scaled)
        
        # # Scale output to match typical LLM embedding magnitudes (~0.02 std)
        # # This ensures compatibility with pre-trained LLM weights
        # token_embeddings = token_embeddings * 0.02
        x = self.input_layernorm(x)
        token_embeddings = self.token_projection(x)
        
        return token_embeddings  # [batch, 128, output_size]
    
    def get_sequence_length(self) -> int:
        """Return the number of tokens this adapter produces."""
        return self.seq_len
    
    def get_token_info(self) -> dict:
        """Return information about the tokens produced by this adapter."""
        return {
            "num_tokens": self.seq_len,
            "token_dim": self.output_size,
            "token_type": "simple_sequence",
            "description": f"Simple linear projection: each of {self.seq_len} ECG positions -> separate token"
        } 