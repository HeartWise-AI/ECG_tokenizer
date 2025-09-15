import torch
import torch.nn as nn
from typing import Optional, Tuple
from utils.registry import ModelRegistry
from utils.enums import AdapterName

@ModelRegistry.register(AdapterName.LLAMA32_SEQUENCE_TOKEN_ADAPTER)
class CrossModalSequenceTokenAdapter(nn.Module):
    """
    Enhanced adapter that converts ECG positions into LLM tokens with cross-modal attention.
    
    This adapter enables bidirectional attention between ECG tokens and text tokens,
    allowing the model to learn rich cross-modal representations.
    """
    
    def __init__(
        self, 
        input_shape: tuple[int, int] = (128, 82), 
        output_size: int = 2048, 
        dropout: float = 0.05,
        use_cross_attention: bool = True,
        num_attention_heads: int = 8,
        intermediate_dim: int = None,
        num_cross_attention_layers: int = 2
    ):
        """
        Initialize the cross-modal sequence token adapter.
        
        Args:
            input_shape: (seq_len, feature_dim) = (128, 82) for ECG quantized features
            output_size: LLM embedding dimension (e.g., 2048 for Llama-3.2-1B)
            dropout: Dropout rate for regularization
            use_cross_attention: Whether to apply cross-attention
            num_attention_heads: Number of attention heads
            intermediate_dim: Intermediate projection dimension
            num_cross_attention_layers: Number of cross-attention layers
        """
        super().__init__()
        
        seq_len, feature_dim = input_shape  # 128, 82
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.output_size = output_size
        self.use_cross_attention = use_cross_attention
        self.num_tokens = seq_len
        self.num_cross_attention_layers = num_cross_attention_layers
        
        if intermediate_dim is None:
            intermediate_dim = output_size // 2
        
        # Initial projection of ECG features to embedding space
        self.token_projection = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, output_size),
            nn.LayerNorm(output_size)
        )
        
        # Learnable positional embeddings for ECG positions
        self.ecg_positional_embedding = nn.Parameter(
            torch.randn(1, seq_len, output_size) * 0.02
        )
        
        # Modal type embeddings to distinguish ECG from text
        self.ecg_modal_embedding = nn.Parameter(
            torch.randn(1, 1, output_size) * 0.02
        )
        
        # Stack of cross-attention layers
        if use_cross_attention:
            self.cross_attention_layers = nn.ModuleList([
                CrossAttentionLayer(
                    embed_dim=output_size,
                    num_heads=num_attention_heads,
                    dropout=dropout,
                    intermediate_dim=intermediate_dim * 2
                )
                for _ in range(num_cross_attention_layers)
            ])
        
        # Final refinement layer
        self.final_refinement = nn.Sequential(
            nn.Linear(output_size, output_size),
            nn.LayerNorm(output_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Store attention weights for visualization
        self.attention_weights = []
        
    def forward(
        self, 
        x: torch.Tensor,
        text_embeddings: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[list]]:
        """
        Convert ECG features to tokens with optional cross-modal attention.
        
        Args:
            x: ECG quantized features [batch, 128, 82]
            text_embeddings: Optional text token embeddings [batch, text_len, output_size]
            return_attention: Whether to return attention weights
            
        Returns:
            - ECG token embeddings [batch, 128, output_size]
            - Optional attention weights if return_attention=True
        """
        # Handle 4D input
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)
        
        batch_size, seq_len, feature_dim = x.shape
        
        # Project ECG features to embedding space
        ecg_embeddings = self.token_projection(x)
        
        # Add positional and modal embeddings
        ecg_embeddings = ecg_embeddings + self.ecg_positional_embedding
        ecg_embeddings = ecg_embeddings + self.ecg_modal_embedding
        
        # Apply cross-attention layers if text embeddings are provided
        attention_weights = []
        if self.use_cross_attention and text_embeddings is not None:
            for layer in self.cross_attention_layers:
                ecg_embeddings, attn_weights = layer(
                    query=ecg_embeddings,
                    key_value=text_embeddings,
                    return_attention=return_attention
                )
                if return_attention:
                    attention_weights.append(attn_weights)
        
        # Final refinement
        ecg_embeddings = self.final_refinement(ecg_embeddings)
        
        # Store attention weights for visualization
        if return_attention:
            self.attention_weights = attention_weights
        
        return ecg_embeddings, attention_weights if return_attention else None
    
    def get_sequence_length(self) -> int:
        """Return the number of tokens this adapter produces."""
        return self.seq_len
    
    def get_token_info(self) -> dict:
        """Return information about the tokens produced by this adapter."""
        return {
            "num_tokens": self.seq_len,
            "token_dim": self.output_size,
            "token_type": "cross_modal_sequence",
            "description": f"Cross-modal attention between {self.seq_len} ECG tokens and text"
        }


class CrossAttentionLayer(nn.Module):
    """
    A single cross-attention layer with feedforward network.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float,
        intermediate_dim: int
    ):
        super().__init__()
        
        # Multi-head cross-attention
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, intermediate_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(intermediate_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply cross-attention between query and key-value tensors.
        
        Args:
            query: Query tensor [batch, query_len, embed_dim]
            key_value: Key and value tensor [batch, kv_len, embed_dim]
            return_attention: Whether to return attention weights
            
        Returns:
            - Updated query tensor [batch, query_len, embed_dim]
            - Optional attention weights [batch, num_heads, query_len, kv_len]
        """
        # Cross-attention with residual connection
        residual = query
        query = self.norm1(query)
        
        if return_attention:
            attended, attn_weights = self.cross_attention(
                query=query,
                key=key_value,
                value=key_value,
                need_weights=True,
                average_attn_weights=False  # Keep per-head weights
            )
        else:
            attended, attn_weights = self.cross_attention(
                query=query,
                key=key_value,
                value=key_value,
                need_weights=False
            ), None
        
        query = residual + attended
        
        # Feedforward with residual connection
        residual = query
        query = self.norm2(query)
        query = residual + self.ffn(query)
        
        return query, attn_weights