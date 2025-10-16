"""ECG Bridge modules for connecting ECG tokens to LLM embedding space."""

import math
import torch
import torch.nn as nn
from typing import Optional, cast


class ECGCodeBridge(nn.Module):
    """Discrete ECG code bridge with learned-query resampler and RMSNorm projector."""

    uses_codes: bool = True

    def __init__(
        self,
        vocab_size: int,
        d_mid: int,
        d_model: int,
        num_output_tokens: int,
        num_heads: int = 8,
        num_special_tokens: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if d_mid % num_heads != 0:
            raise ValueError(
                f"d_mid ({d_mid}) must be divisible by num_heads ({num_heads}) for ECGCodeBridge."
            )

        self.vocab_size = vocab_size
        self.num_output_tokens = num_output_tokens
        self.num_heads = num_heads
        self.head_dim = d_mid // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.total_vocab = vocab_size + max(0, num_special_tokens)
        self.pad_id = vocab_size
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else None

        self.embed = nn.Embedding(self.total_vocab, d_mid)
        self.norm_in = nn.RMSNorm(d_mid)

        self.q = nn.Parameter(torch.randn(num_output_tokens, d_mid) * (1.0 / math.sqrt(d_mid)))
        self.k_proj = nn.Linear(d_mid, d_mid, bias=False)
        self.v_proj = nn.Linear(d_mid, d_mid, bias=False)

        self.proj = nn.Sequential(
            nn.Linear(d_mid, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.norm_out = nn.RMSNorm(d_model)

        # Gentle initialisation keeps gradients flowing without shocking the frozen LLM
        first_linear = cast(nn.Linear, self.proj[0])
        nn.init.normal_(first_linear.weight, std=0.02)
        nn.init.zeros_(first_linear.bias)

        final_linear = cast(nn.Linear, self.proj[-1])
        nn.init.normal_(final_linear.weight, std=1e-3)
        nn.init.zeros_(final_linear.bias)

    @property
    def num_tokens(self) -> int:
        return self.num_output_tokens

    def _build_positions(self, length: int, d_model: int, device: torch.device) -> torch.Tensor:
        position = torch.arange(length, device=device).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
            * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(length, d_model, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def _split_heads(self, tensor: torch.Tensor, batch: int) -> torch.Tensor:
        return tensor.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, ecg_ids: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Project discrete ECG ids into LLM embedding space."""

        if ecg_ids.dtype != torch.long:
            ecg_ids = ecg_ids.long()

        batch_size, seq_len = ecg_ids.shape
        device = ecg_ids.device

        x = self.embed(ecg_ids)
        if self.dropout is not None:
            x = self.dropout(x)

        pos = self._build_positions(seq_len, x.size(-1), device)
        x = x + pos.unsqueeze(0)

        x = self.norm_in(x)

        q = self.q.unsqueeze(0).expand(batch_size, -1, -1)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q_heads = self._split_heads(q, batch_size)
        k_heads = self._split_heads(k, batch_size)
        v_heads = self._split_heads(v, batch_size)

        scores = torch.matmul(q_heads, k_heads.transpose(-2, -1)) * self.scale

        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))

        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn, nan=0.0)

        if attn_mask is not None:
            attn = attn.masked_fill(attn_mask == 0, 0.0)

        resampled = torch.matmul(attn, v_heads)
        resampled = resampled.transpose(1, 2).contiguous().view(batch_size, self.num_output_tokens, -1)

        out = self.proj(resampled)
        out = self.norm_out(out)
        return out


class ECGProjectionBridge(nn.Module):
    """Lightweight projection bridge that maps ECG features into the LLM space."""

    uses_codes: bool = False

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_tokens: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.input_dim = input_dim
        self.d_model = d_model
        self.num_tokens = num_tokens

        self.feature_norm = nn.LayerNorm(input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.dropout = nn.Dropout(dropout if dropout and dropout > 0 else 0.0)
        self.positional_embedding = nn.Embedding(num_tokens, d_model)
        self.out_norm = nn.LayerNorm(d_model)

        # gentle init on final projection for stable fusion
        final_linear = cast(nn.Linear, self.mlp[-1])
        nn.init.normal_(final_linear.weight, std=1e-3)
        nn.init.zeros_(final_linear.bias)

    def _prepare_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 4:
            # Handle (batch, quantizers, seq, dim) by taking first quantizer
            features = features[:, 0]
        if features.dim() != 3:
            raise ValueError(
                f"ECGProjectionBridge expects [batch, seq, dim] features, got shape {features.shape}"
            )
        return features

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self._prepare_features(features)
        batch_size, seq_len, feat_dim = features.shape

        if feat_dim != self.input_dim:
            raise ValueError(
                f"Feature dim mismatch: expected {self.input_dim}, got {feat_dim}"
            )

        if seq_len != self.num_tokens:
            if seq_len < self.num_tokens:
                raise ValueError(
                    f"ECG sequence shorter than expected: {seq_len} < {self.num_tokens}"
                )
            sample_positions = torch.linspace(
                0,
                seq_len - 1,
                self.num_tokens,
                device=features.device,
                dtype=torch.float32
            )
            index = sample_positions.round().long().clamp(max=seq_len - 1)
            index = index.unsqueeze(0).expand(batch_size, -1)
            features = features.gather(
                dim=1,
                index=index.unsqueeze(-1).expand(batch_size, self.num_tokens, feat_dim)
            )
            seq_len = self.num_tokens

        x = self.feature_norm(features)
        x = self.mlp(x)
        x = self.dropout(x)

        positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
        pos_emb = self.positional_embedding(positions).unsqueeze(0)
        x = x + pos_emb
        return self.out_norm(x)

class PerceiverProjectionBridge(nn.Module):
    """Perceiver-style bridge that resamples continuous ECG features into the LLM space."""

    uses_codes: bool = False

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_output_tokens: int,
        num_heads: int = 8,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by num_heads ({num_heads})."
            )

        self.num_output_tokens = num_output_tokens

        self.feature_proj = nn.Linear(input_dim, d_model)
        self.queries = nn.Parameter(
            torch.randn(num_output_tokens, d_model) * (1.0 / math.sqrt(d_model))
        )

        self.norm_features = nn.LayerNorm(d_model)
        self.norm_queries = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_dropout = nn.Dropout(dropout)

        self.norm_mlp = nn.LayerNorm(d_model)
        mlp_hidden_dim = int(d_model * 4)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, d_model),
            nn.Dropout(dropout),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        final_mlp_linear = cast(nn.Linear, self.mlp[-2])
        nn.init.normal_(final_mlp_linear.weight, std=1e-3)
        nn.init.zeros_(final_mlp_linear.bias)

    @property
    def num_tokens(self) -> int:
        return self.num_output_tokens

    def _prepare_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 4:
            # Preserve information from each channel/quantizer by folding it into the sequence axis
            features = features.reshape(features.size(0), -1, features.size(-1))
        if features.dim() != 3:
            raise ValueError(
                f"PerceiverProjectionBridge expects [batch, seq, dim] features, got shape {features.shape}"
            )
        return features

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self._prepare_features(features)
        batch_size = features.shape[0]

        projected_features = self.feature_proj(features)
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)

        attn_output, _ = self.attn(
            query=self.norm_queries(queries),
            key=self.norm_features(projected_features),
            value=self.norm_features(projected_features),
        )
        queries = queries + self.attn_dropout(attn_output)

        mlp_output = self.mlp(self.norm_mlp(queries))
        prefix_embeddings = queries + mlp_output

        return prefix_embeddings
    
