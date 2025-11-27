"""ECG Bridge modules for connecting ECG tokens to LLM embedding space."""

import math
import re
import warnings
from typing import Callable, List, Optional, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import ModelRegistry
from utils.enums import BridgeName

__all__ = [
    "ECGCodeBridge",
    "ECGProjectionBridge",
    "PerceiverProjectionBridge",
    "LinearBridge",
    "EmbeddingBridge",
    "SimpleEmbeddingBridge",
    "SequenceBridge",
    "SequenceTokenBridge",
    "SimpleTokenBridge",
    "CrossModalSequenceTokenBridge",
    "CrossAttentionLayer",
    "calibrate_bridge_scale",
]


@ModelRegistry.register(BridgeName.LLAMA32_ECG_CODE_BRIDGE)
@ModelRegistry.register("ECGCodeBridge")
class ECGCodeBridge(nn.Module):
    """Discrete ECG code bridge with learned-query resampler and RMSNorm projector.

    Supports multi-codebook inputs where different residual VQ codebooks can be selected
    and averaged/combined before projection into LLM space.
    """

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
        num_codebooks: int = 1,
        codebook_offset: int = 0,
        original_num_codebooks: Optional[int] = None,
        debug_mask_logging: bool = False,
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
        self.debug_mask_logging = debug_mask_logging

        # Multi-codebook support
        # Note: num_codebooks is the TOTAL number of embedding tables to create
        # The bridge processes whatever codebooks it receives in the input
        self.num_codebooks = num_codebooks
        self.original_num_codebooks = (
            int(original_num_codebooks) if original_num_codebooks is not None else int(num_codebooks)
        )
        self.codebook_offset = int(codebook_offset or 0)

        # Create separate embedding tables for each codebook
        # This allows the model to learn different representations for each residual level
        self.embed_tables = nn.ModuleList([
            nn.Embedding(self.total_vocab, d_mid)
            for _ in range(num_codebooks)
        ])

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

        # Mixing strategy across residual codebooks
        self.mix_strategy: str = "softmax"
        self.mix_bias_last: bool = True
        self.register_parameter("mix_weights", None)
        if self.num_codebooks > 1:
            self.mix_weights = nn.Parameter(torch.zeros(self.num_codebooks))
        
        # Cache positional encodings (will be populated on first forward)
        self.register_buffer('cached_positions', None)
        self.cached_max_len = 0
        self._warned_all_masked = False
        self._selected_codebooks: List[int] = self._compute_selected_codebooks()

    def _compute_selected_codebooks(self) -> List[int]:
        """Compute which original codebooks are expected after slicing."""
        total = max(1, int(self.original_num_codebooks))
        keep = max(1, int(self.num_codebooks))
        if keep >= total:
            return list(range(total))

        offset = self.codebook_offset
        if offset < 0:
            offset = max(total - keep, 0)
        offset = max(0, min(offset, total - keep))
        return list(range(offset, offset + keep))

    def enable_mask_debug(self, enabled: bool = True) -> None:
        """Toggle mask-related debugging warnings during forward passes."""
        self.debug_mask_logging = bool(enabled)

    @property
    def num_tokens(self) -> int:
        return self.num_output_tokens

    def _get_positions(self, length: int, d_model: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """Get cached positional encodings, building and caching if needed.

        Args:
            length: Sequence length needed
            d_model: Model dimension
            device: Target device
            dtype: Target dtype (will be converted if cached dtype differs)
        """
        if self.cached_positions is None or length > self.cached_max_len:
            # Build positions for max(length, 2048) to avoid frequent rebuilding
            # Always cache in float32 for precision, convert on use if needed
            max_len = max(length, 2048)
            position = torch.arange(max_len, device=device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
                * (-math.log(10000.0) / d_model)
            )
            pe = torch.zeros(max_len, d_model, device=device, dtype=torch.float32)
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            self.register_buffer('cached_positions', pe)
            self.cached_max_len = max_len

        positions = self.cached_positions[:length]
        # Convert to target dtype if different
        if positions.dtype != dtype:
            positions = positions.to(dtype=dtype)
        return positions

    def _split_heads(self, tensor: torch.Tensor, batch: int) -> torch.Tensor:
        return tensor.view(batch, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def _load_from_state_dict(
        self,
        state_dict: dict[str, torch.Tensor],
        prefix: str,
        local_metadata,
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        """Allow loading checkpoints with different codebook counts by slicing/padding."""
        embed_pattern = re.compile(re.escape(prefix) + r"embed_tables\.(\d+)\.weight")
        selected_ckpt_indices: List[int] = []
        ckpt_indices = sorted({
            int(match.group(1))
            for key in list(state_dict.keys())
            if (match := embed_pattern.match(key)) is not None
        })

        if ckpt_indices:
            desired = self._selected_codebooks if self._selected_codebooks else list(range(self.num_codebooks))
            # Ensure desired indices exist in checkpoint; fall back to tail slice if needed
            available = [idx for idx in desired if idx in ckpt_indices]
            if len(available) < self.num_codebooks:
                if len(ckpt_indices) >= self.num_codebooks:
                    available = ckpt_indices[-self.num_codebooks:]
                else:
                    available = ckpt_indices[:]

            selected_ckpt_indices = available[:self.num_codebooks]

            # Re-map checkpoint weights to consecutive indices starting at 0
            keep_keys = {f"{prefix}embed_tables.{i}.weight" for i in range(self.num_codebooks)}
            for new_idx in range(self.num_codebooks):
                target_key = f"{prefix}embed_tables.{new_idx}.weight"
                if new_idx < len(selected_ckpt_indices):
                    old_idx = selected_ckpt_indices[new_idx]
                    source_key = f"{prefix}embed_tables.{old_idx}.weight"
                    if source_key in state_dict:
                        state_dict[target_key] = state_dict.pop(source_key)
                else:
                    # Remove any stale entries so default init is kept
                    state_dict.pop(target_key, None)

            # Drop any leftover embed table entries from the checkpoint
            for key in list(state_dict.keys()):
                if embed_pattern.match(key) and key not in keep_keys:
                    state_dict.pop(key, None)

            # Align embedding table shapes if vocab sizes differ
            for new_idx in range(self.num_codebooks):
                target_key = f"{prefix}embed_tables.{new_idx}.weight"
                tensor = state_dict.get(target_key)
                if tensor is None:
                    continue
                target_param = self.embed_tables[new_idx].weight
                if tensor.shape != target_param.shape and tensor.shape[1:] == target_param.shape[1:]:
                    resized = target_param.detach().clone()
                    rows = min(tensor.shape[0], target_param.shape[0])
                    resized[:rows] = tensor[:rows]
                    state_dict[target_key] = resized

        mix_key = f"{prefix}mix_weights"
        if mix_key in state_dict:
            tensor = state_dict[mix_key]
            expected = self.mix_weights.shape[0] if isinstance(self.mix_weights, torch.nn.Parameter) else self.num_codebooks
            if tensor.shape[0] != expected:
                if selected_ckpt_indices and max(selected_ckpt_indices) < tensor.shape[0]:
                    index_tensor = torch.tensor(
                        selected_ckpt_indices[:expected],
                        dtype=torch.long,
                        device=tensor.device,
                    )
                    state_dict[mix_key] = tensor.index_select(0, index_tensor)
                elif tensor.shape[0] >= expected:
                    state_dict[mix_key] = tensor[-expected:].clone()
                else:
                    padded = tensor.new_zeros(expected)
                    padded[-tensor.shape[0]:] = tensor
                    state_dict[mix_key] = padded
            elif selected_ckpt_indices and len(selected_ckpt_indices) == expected and tensor.shape[0] >= max(selected_ckpt_indices) + 1:
                index_tensor = torch.tensor(selected_ckpt_indices, dtype=torch.long, device=tensor.device)
                state_dict[mix_key] = tensor.index_select(0, index_tensor)

        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, ecg_ids: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Project discrete ECG ids into LLM embedding space using optimized SDPA.

        Args:
            ecg_ids: Codebook indices
                - Shape [batch, seq] for single codebook (backward compatible)
                - Shape [batch, seq, num_codebooks] for multiple codebooks
            attn_mask: Optional attention mask [batch, seq] where True = attend, False = mask

        Returns:
            Embeddings of shape [batch, num_output_tokens, d_model]
        """

        if ecg_ids.dtype != torch.long:
            ecg_ids = ecg_ids.long()

        device = ecg_ids.device

        # Handle both [batch, seq] (single codebook) and [batch, seq, num_codebooks]
        if ecg_ids.dim() == 2:
            # Single codebook - backward compatible
            batch_size, seq_len = ecg_ids.shape
            ecg_ids = ecg_ids.unsqueeze(-1)  # [batch, seq, 1]
        elif ecg_ids.dim() == 3:
            # Multi-codebook
            batch_size, seq_len, num_cb = ecg_ids.shape
            if num_cb != self.num_codebooks:
                # We'll reconcile below to allow flexible inputs
                pass
        else:
            raise ValueError(f"ecg_ids must be 2D or 3D, got shape {ecg_ids.shape}")

        # Ensure the last dimension matches the bridge's expected number of codebooks by
        # either trimming extras (keep the most recent residual levels) or padding with PAD ids.
        available_codebooks = ecg_ids.size(-1)
        if available_codebooks > self.num_codebooks:
            ecg_ids = ecg_ids[..., available_codebooks - self.num_codebooks:]
        elif available_codebooks < self.num_codebooks:
            pad_shape = list(ecg_ids.shape[:-1]) + [self.num_codebooks - available_codebooks]
            pad_tensor = torch.full(
                pad_shape,
                self.pad_id,
                dtype=ecg_ids.dtype,
                device=ecg_ids.device,
            )
            ecg_ids = torch.cat([ecg_ids, pad_tensor], dim=-1)
        # Update helper variables after reconciliation
        batch_size, seq_len, _ = ecg_ids.shape

        # Build default attention mask from padding if not provided
        # For multi-codebook, a position is valid if ANY codebook has a valid code
        if attn_mask is None:
            # Check if any codebook at each position has a valid (non-padding) code
            attn_mask = (ecg_ids != self.pad_id).any(dim=-1)
        else:
            if attn_mask.dim() not in {2, 4}:
                raise ValueError(
                    "attn_mask for ECGCodeBridge must be 2D (batch, seq_len) or 4D (batch, heads, q, k). "
                    f"Received shape {attn_mask.shape}."
                )
            if attn_mask.size(0) != batch_size:
                raise ValueError(
                    f"attn_mask batch size ({attn_mask.size(0)}) does not match input batch size ({batch_size})."
                )
            if attn_mask.dim() == 2 and attn_mask.size(1) != seq_len:
                raise ValueError(
                    f"attn_mask sequence length ({attn_mask.size(1)}) does not match input sequence length ({seq_len})."
                )

        if attn_mask.dtype != torch.bool:
            raise TypeError(
                "ECGCodeBridge expects boolean attention masks with True marking valid tokens. "
                "Please convert additive masks to boolean (e.g. `mask = mask > 0`) before calling the bridge."
            )

        # Ensure mask is boolean where True means "keep" / valid token
        attn_mask = attn_mask.to(device=device, dtype=torch.bool, copy=False)

        # Gather embeddings from each codebook and average
        embeds = []
        for level in range(self.num_codebooks):
            # Get codes for this codebook level
            ids = ecg_ids[..., level].clamp_min(0)  # Clamp negatives to 0
            # Look up embeddings
            embeds.append(self.embed_tables[level](ids))

        # Combine embeddings from different residual levels
        if self.num_codebooks == 1 or self.mix_strategy == "last":
            x = embeds[-1]
        elif self.mix_strategy == "mean":
            x = torch.stack(embeds, dim=0).mean(dim=0)
        else:
            stacked = torch.stack(embeds, dim=0)  # [num_codebooks, batch, seq, d_mid]
            mix_weights = getattr(self, "mix_weights", None)
            if mix_weights is None:
                x = stacked[-1]
            else:
                weights = mix_weights
                if self.mix_bias_last and weights.numel() > 0:
                    boost = torch.zeros_like(weights)
                    boost[-1] = 1.0
                    weights = weights + boost
                weights = weights.softmax(dim=0)  # [num_codebooks]
                x = torch.tensordot(weights, stacked, dims=([0], [0]))  # [batch, seq, d_mid]

        if self.dropout is not None:
            x = self.dropout(x)

        # Use cached positional encodings with matching dtype
        pos = self._get_positions(seq_len, x.size(-1), device, dtype=x.dtype)
        x = x + pos.unsqueeze(0)

        x = self.norm_in(x)

        # Prepare queries, keys, values
        q = self.q.unsqueeze(0).expand(batch_size, -1, -1)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Reshape for multi-head attention
        q_heads = self._split_heads(q, batch_size)  # (B, H, num_output_tokens, head_dim)
        k_heads = self._split_heads(k, batch_size)  # (B, H, seq_len, head_dim)
        v_heads = self._split_heads(v, batch_size)  # (B, H, seq_len, head_dim)

        # Prepare attention mask for SDPA
        # PyTorch SDPA expects either:
        #   - Boolean mask where True = masked position (do not attend)
        #   - Additive mask where 0.0 = attend, -inf = masked
        # Our input mask still uses True = attend, so invert when building the SDPA mask.
        attn_mask_4d = None
        if attn_mask.dim() == 2:
            attn_mask_4d = ~attn_mask[:, None, None, :]
        elif attn_mask.dim() == 4:
            attn_mask_4d = ~attn_mask

        # When using additive masks, use real negative infinity to avoid leakage with bf16/flash attention.
        additive_mask = None
        if attn_mask_4d is not None:
            if attn_mask_4d.dtype != torch.bool:
                attn_mask_4d = attn_mask_4d.to(dtype=torch.bool)
            additive_mask = torch.zeros(
                attn_mask_4d.shape,
                dtype=q_heads.dtype,
                device=device,
            )
            # Use -inf for proper masking; finfo.min can be unreliable with fused kernels
            # Cast -inf to the target dtype to avoid type errors
            neg_inf = float('-inf')
            additive_mask.masked_fill_(attn_mask_4d, neg_inf)

        total_slots = attn_mask.numel()
        valid_count = attn_mask.sum().item() if total_slots > 0 else 0

        if (
            self.debug_mask_logging
            and additive_mask is not None
            and total_slots > 0
            and valid_count == 0
            and not self._warned_all_masked
        ):
            warnings.warn(
                "ECGCodeBridge: attention mask removed all ECG tokens; generation will rely on text-only context.",
                RuntimeWarning,
                stacklevel=2,
            )
            self._warned_all_masked = True

        # Use scaled_dot_product_attention with Flash Attention when available
        # Note: SDPA automatically uses Flash Attention v2 when conditions are met
        dropout_p = 0.1 if self.training and self.dropout is not None else 0.0

        # Use attn_mask parameter with proper boolean semantics
        # PyTorch SDPA: attn_mask True = mask out, False = attend
        resampled_heads = F.scaled_dot_product_attention(
            q_heads,
            k_heads,
            v_heads,
            attn_mask=additive_mask,
            dropout_p=dropout_p,
            scale=self.scale,
            is_causal=False  # Explicitly set for clarity
        )

        # DEFENSIVE CHECK: Detect NaN/Inf in attention output
        if torch.isnan(resampled_heads).any() or torch.isinf(resampled_heads).any():
            # This should never happen with our fixed attention mask handling
            # If it does, it indicates a serious numerical issue
            warnings.warn(
                "ECGCodeBridge: NaN or Inf detected in attention output! "
                "This indicates numerical instability. "
                f"Input dtype: {ecg_ids.dtype}, Mask dtype: {attn_mask.dtype}, "
                f"Query dtype: {q_heads.dtype}, Has NaN: {torch.isnan(resampled_heads).any()}, "
                f"Has Inf: {torch.isinf(resampled_heads).any()}",
                RuntimeWarning,
                stacklevel=2
            )
            # Replace NaN/Inf with zeros to prevent propagation (emergency fallback)
            resampled_heads = torch.where(
                torch.isnan(resampled_heads) | torch.isinf(resampled_heads),
                torch.zeros_like(resampled_heads),
                resampled_heads
            )

        # Reshape back to (B, num_output_tokens, d_mid)
        resampled = resampled_heads.transpose(1, 2).contiguous().view(
            batch_size, self.num_output_tokens, -1
        )

        # Final projection and normalization
        out = self.proj(resampled)
        out = self.norm_out(out)

        # DEFENSIVE CHECK: Detect degenerate output (all zeros or very small values)
        # This can cause empty generations even without NaN/Inf
        output_norm = out.abs().mean().item()
        if output_norm < 1e-6:
            warnings.warn(
                f"ECGCodeBridge: Output has very small norm ({output_norm:.2e}). "
                "This may cause empty/poor text generation. "
                f"Attention mask had {valid_count}/{attn_mask.numel()} valid positions.",
                RuntimeWarning,
                stacklevel=2
            )

        # DEFENSIVE CHECK: Detect NaN/Inf in final output
        if torch.isnan(out).any() or torch.isinf(out).any():
            warnings.warn(
                "ECGCodeBridge: NaN or Inf in final output! "
                f"Has NaN: {torch.isnan(out).any()}, Has Inf: {torch.isinf(out).any()}",
                RuntimeWarning,
                stacklevel=2
            )
            # Emergency fallback: return zero embeddings
            out = torch.zeros_like(out)

        return out


@ModelRegistry.register(BridgeName.LLAMA32_ECG_PROJECTION_BRIDGE)
@ModelRegistry.register("ECGProjectionBridge")
class ECGProjectionBridge(nn.Module):
    """Lightweight projection bridge that maps ECG features into the LLM space."""

    uses_codes: bool = False

    def __init__(
        self,
        input_dim: int,
        d_model: int,
        num_tokens: int,
        dropout: float = 0.1,
        num_codebooks: int = 1,
        code_dim: Optional[int] = None,
        target_std: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.input_channels = int(input_dim)
        self.d_model = d_model
        self.num_tokens = num_tokens
        self.num_codebooks = max(1, int(num_codebooks))
        if code_dim is None:
            if self.num_codebooks > 1:
                if self.input_channels % self.num_codebooks != 0:
                    raise ValueError(
                        f"input_dim ({self.input_channels}) must be divisible by num_codebooks ({self.num_codebooks})"
                    )
                code_dim = self.input_channels // self.num_codebooks
            else:
                code_dim = self.input_channels
        self.code_dim = int(code_dim)
        if self.code_dim * self.num_codebooks != self.input_channels:
            raise ValueError(
                "Inconsistent channel configuration for ECGProjectionBridge: "
                f"input_channels={self.input_channels}, num_codebooks={self.num_codebooks}, code_dim={self.code_dim}"
            )

        self.token_gate = nn.Sequential(
            nn.Conv1d(self.code_dim * self.num_codebooks, self.num_codebooks, kernel_size=1, groups=1, bias=False),
            nn.GELU(),
        )

        self.feature_norm = nn.RMSNorm(self.code_dim)
        self.mlp = nn.Sequential(
            nn.Linear(self.code_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model)
        )
        self.dropout = nn.Dropout(dropout if dropout and dropout > 0 else 0.0)
        self.positional_embedding = nn.Embedding(num_tokens, d_model)
        self.out_norm = nn.RMSNorm(d_model)
        scale_init = float(target_std) if target_std is not None else 1.0
        self.output_scale = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))

        # Simple anti-aliasing stack: depthwise stride-2 conv followed by
        # a smoothing conv before final interpolation.
        self.downsample_conv = nn.Conv1d(
            self.code_dim,
            self.code_dim,
            kernel_size=5,
            stride=2,
            padding=2,
            groups=self.code_dim,
            bias=False,
        )
        self.prefilter_conv = nn.Conv1d(
            self.code_dim,
            self.code_dim,
            kernel_size=3,
            padding=1,
            groups=self.code_dim,
            bias=False,
        )
        self.resample_activation = nn.GELU()

        # gentle init on projections for stable fusion
        first_linear = cast(nn.Linear, self.mlp[0])
        nn.init.normal_(first_linear.weight, std=0.02)
        nn.init.zeros_(first_linear.bias)
        final_linear = cast(nn.Linear, self.mlp[-1])
        nn.init.normal_(final_linear.weight, std=1e-3)
        nn.init.zeros_(final_linear.bias)
        # nn.init.orthogonal_(final_linear.weight)
        # final_linear.weight.data.mul_(0.5) # Scale down slightly, but not to zero
        # nn.init.zeros_(final_linear.bias)

    def _prepare_features(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() == 4:
            # Accept [batch, num_codebooks, seq, dim] or [batch, num_codebooks, dim, seq]
            if features.size(1) == self.num_codebooks and features.size(3) == self.code_dim:
                features = features.permute(0, 1, 3, 2).contiguous()  # [B, Q, C, L]
            elif features.size(1) == self.num_codebooks and features.size(2) == self.code_dim:
                features = features.contiguous()
            else:
                raise ValueError(
                    f"Unsupported 4D feature layout for ECGProjectionBridge: {features.shape}"
                )
            features = features.view(features.size(0), self.num_codebooks * self.code_dim, features.size(-1))

        if features.dim() != 3:
            raise ValueError(
                f"ECGProjectionBridge expects a 3D tensor; received {features.shape}"
            )

        if features.size(1) == self.input_channels:
            return features
        if features.size(2) == self.input_channels:
            return features.transpose(1, 2).contiguous()

        raise ValueError(
            f"ECGProjectionBridge received features with incompatible shape {features.shape}; "
            f"expected channel dimension {self.input_channels}"
        )

    def _mix_quantizers(self, features: torch.Tensor) -> torch.Tensor:
        """Combine per-codebook channels using learned mixing weights."""
        batch, channels, length = features.shape
        if channels != self.input_channels:
            raise ValueError(
                f"Feature channels {channels} do not match expected {self.input_channels}"
            )
        reshaped = features.view(batch, self.num_codebooks, self.code_dim, length)
        if self.num_codebooks == 1:
            return reshaped.view(batch, self.code_dim, length)

        gate_logits = self.token_gate(reshaped.flatten(1, 2))  # [B, num_codebooks, length]
        weights = torch.softmax(gate_logits, dim=1).unsqueeze(2)  # [B, num_codebooks, 1, length]
        mixed = (reshaped * weights).sum(dim=1)  # [B, code_dim, length]
        return mixed

    def _resample_to_tokens(self, features: torch.Tensor) -> torch.Tensor:
        """Low-pass filter then interpolate to the required token length."""
        # Input: [batch, code_dim, length]
        if features.size(-1) == self.num_tokens:
            return features.transpose(1, 2).contiguous()

        x = features
        # Repeatedly apply stride-2 convolution until close to target length
        while x.size(-1) > self.num_tokens * 2:
            x = self.resample_activation(self.downsample_conv(x))

        # Apply smoothing before final interpolation
        x = self.resample_activation(self.prefilter_conv(x))
        x = F.interpolate(
            x,
            size=self.num_tokens,
            mode="linear",
            align_corners=False,
        )
        return x.transpose(1, 2).contiguous()

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        features = self._prepare_features(features)
        mixed = self._mix_quantizers(features)
        token_space = self._resample_to_tokens(mixed)  # [B, num_tokens, code_dim]

        x = self.feature_norm(token_space)
        x = self.mlp(x)
        x = self.dropout(x)

        positions = torch.arange(self.num_tokens, device=x.device, dtype=torch.long)
        pos_emb = self.positional_embedding(positions).unsqueeze(0)
        x = x + pos_emb
        x = self.out_norm(x)
        return x * self.output_scale


@ModelRegistry.register(BridgeName.GPT2_LINEAR_BRIDGE)
@ModelRegistry.register("LinearBridge")
class LinearBridge(nn.Module):
    """Simple linear bridge that flattens 3D input and maps to LLM embedding size."""
    
    def __init__(
        self,
        input_shape: tuple[int, int, int] = (8, 128, 160),
        output_size: int = 768,
        dropout: float = 0.0  # Set dropout > 0 to enable dropout regularization
    ):
        """
        Args:
            input_shape: 3D input dimensions (channels, height, width)
            output_size: Target embedding dimension for LLM
            dropout: Dropout probability for regularization
        """        
        super(LinearBridge, self).__init__()
        # Calculate the flattened input size (e.g. 8 * 128 * 160 = 163840)
        self.flatten_dim: int = input_shape[0] * input_shape[1] * input_shape[2]
        
        self.bridge: nn.Sequential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(1024, output_size)
        )
        # Backwards-compatible attribute name expected by existing tests/integrations
        self.adapter = self.bridge
    
    def forward(self, x):
        """Forward pass through linear bridge."""
        return self.bridge(x)


@ModelRegistry.register(BridgeName.GPT2_EMBEDDING_BRIDGE)
@ModelRegistry.register(BridgeName.LLAMA32_EMBEDDING_BRIDGE)
@ModelRegistry.register("EmbeddingBridge")
class EmbeddingBridge(nn.Module):
    """CNN-based bridge that processes 3D input through convolutional layers to LLM embedding size."""

    def __init__(
        self, 
        input_shape: tuple[int, int, int] = (8, 128, 160), 
        output_size: int = 768,
        dropout: float = 0.2
    ):
        """
        Args:
            input_shape: 3D input dimensions (channels, height, width)
            output_size: Target embedding dimension for LLM
            dropout: Dropout probability for regularization
        """
        super(EmbeddingBridge, self).__init__()
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
        return x  # Shape: (batch_size, output_size)


@ModelRegistry.register(BridgeName.ECG_PERCEIVER_BRIDGE)
@ModelRegistry.register("PerceiverProjectionBridge")
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

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if features.dim() != 3:
            raise ValueError(
                f"PerceiverProjectionBridge expects [batch, seq, dim] features, got shape {features.shape}"
            )

        x = self.feature_proj(features)
        x = self.norm_features(x)

        batch_size = x.size(0)
        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        queries = self.norm_queries(queries)

        attended, _ = self.attn(
            query=queries,
            key=x,
            value=x,
            need_weights=False,
        )
        attended = self.attn_dropout(attended)

        out = self.norm_mlp(attended + queries)
        out = out + self.mlp(out)
        return out


@ModelRegistry.register(BridgeName.GPT2_SIMPLE_EMBEDDING_BRIDGE)
@ModelRegistry.register(BridgeName.LLAMA32_SIMPLE_EMBEDDING_BRIDGE)
@ModelRegistry.register("SimpleEmbeddingBridge")
class SimpleEmbeddingBridge(nn.Module):
    """Minimal bridge using global average pooling and single linear layer."""
    
    def __init__(
        self,
        input_shape: tuple[int, int, int] = (8, 128, 160),
        output_size: int = 768,
        dropout: float = 0.2
    ):
        """
        Args:
            input_shape: 3D input dimensions (channels, height, width)
            output_size: Target embedding dimension for LLM
            dropout: Dropout probability for regularization
        """
        super(SimpleEmbeddingBridge, self).__init__()
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
        # Map directly to LLM's embedding size.
        x = self.fc(x)
        x = self.dropout(x)
        return x  # Output shape: (batch, output_size)


@ModelRegistry.register(BridgeName.GPT2_SEQUENCE_BRIDGE)
@ModelRegistry.register(BridgeName.LLAMA32_SEQUENCE_BRIDGE)
@ModelRegistry.register("SequenceBridge")
class SequenceBridge(nn.Module):
    """Bridge that processes 2D input through a sequence of operations to LLM embedding size."""
    
    def __init__(
        self,
        # input_shape: tuple[int, int] = (128, 82),
        input_shape: tuple[int, int] = (128, 1024),
        output_size: int = 768,
        dropout: float = 0.2,
        target_std: Optional[float] = None,
    ):
        """
        Args:
            input_shape: 2D input dimensions (sequence length, feature dimension)
            output_size: Target embedding dimension for LLM
            dropout: Dropout probability for regularization
        """
        super().__init__()
        
        # Extract the actual feature dimensions
        # NOTE: For ECG encoder output (batch, 128, 1024):
        # - 128 = sequence length (temporal dimension)  
        # - 1024 = feature dimension (quantized features)
        seq_len = input_shape[0]  # 128 = sequence length
        channels = input_shape[1]  # 1024 = feature dimension
        self.seq_len = seq_len
        self.feature_dim = channels
        scale_init = float(target_std) if target_std is not None else 1.0
        
        # Project channels to a smaller intermediate dimension first
        self.channel_projection = nn.Sequential(
            nn.Linear(channels, output_size // 2),
            nn.RMSNorm(output_size // 2),
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
        self.attention_norm = nn.RMSNorm(output_size // 2)
        
        # Final projection to output size
        self.final_linear = nn.Linear(output_size // 2, output_size)
        self.final_activation = nn.GELU()
        self.final_dropout = nn.Dropout(dropout)
        self.output_norm = nn.RMSNorm(output_size)
        self.output_scale = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))
        
        # Learnable positional embeddings for sequence length
        self.positional_embedding = nn.Parameter(
            torch.randn(1, seq_len, output_size // 2) * 0.02
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through sequence bridge."""
        # Handle common layouts: (batch, 1, seq, channels) or (batch, channels, seq)
        if x.dim() == 4 and x.size(1) == 1:
            x = x.squeeze(1)
        if x.dim() != 3:
            raise ValueError(f"SequenceBridge expects a 3D tensor, got {x.shape}")
        if x.size(2) == self.feature_dim and x.size(1) == self.seq_len:
            pass  # already (batch, seq, channels)
        elif x.size(1) == self.feature_dim and x.size(2) == self.seq_len:
            x = x.transpose(1, 2).contiguous()
        else:
            raise ValueError(
                f"SequenceBridge received tensor with incompatible shape {x.shape}; "
                f"expected seq_len={self.seq_len}, feature_dim={self.feature_dim}"
            )
        
        # Project channels: (batch, seq_len, channels) -> (batch, seq_len, output_size//2)
        x = self.channel_projection(x)
        
        # Add positional embeddings
        x = x + self.positional_embedding

        # Apply self-attention to aggregate sequence information
        attended, _ = self.attention(x, x, x)
        x = self.attention_norm(attended + x)  # Residual connection
        
        # Global average pooling over sequence dimension
        x = x.mean(dim=1)  # (batch, output_size//2)
                
        # Final projection to LLM embedding size
        x = self.final_dropout(self.final_activation(self.final_linear(x)))
        x = self.output_norm(x)

        return x * self.output_scale


@ModelRegistry.register(BridgeName.GPT2_SEQUENCE_TOKEN_BRIDGE)
@ModelRegistry.register(BridgeName.LLAMA32_SEQUENCE_TOKEN_BRIDGE)
@ModelRegistry.register("SequenceTokenBridge")
class SequenceTokenBridge(nn.Module):
    """
    Bridge that converts each of the 128 ECG positions into separate LLM tokens
    instead of compressing them into a single embedding.
    
    This preserves fine-grained spatial/temporal information and allows the LLM
    to attend to specific ECG regions during text generation.
    """
    
    def __init__(
        self, 
        # input_shape: tuple[int, int] = (128, 82), 
        input_shape: tuple[int, int] = (128, 1024), 
        output_size: int = 2048, 
        dropout: float = 0.05,
        use_cross_attention: bool = True,
        num_attention_heads: int = 8,
        intermediate_dim: int = None
    ):
        """
        Initialize the sequence token bridge.
        
        Args:
            input_shape: (seq_len, feature_dim) = (128, 1024) for ECG quantized features
            output_size: LLM embedding dimension (e.g., 2048 for Llama-3.2-1B)
            dropout: Dropout rate for regularization
            use_cross_attention: Whether to apply cross-attention between positions
            num_attention_heads: Number of attention heads for cross-attention
            intermediate_dim: Intermediate projection dimension (defaults to output_size // 2)
        """
        super().__init__()
        
        seq_len, feature_dim = input_shape  # 128, 1024
        self.seq_len = seq_len
        self.feature_dim = feature_dim
        self.output_size = output_size
        self.use_cross_attention = use_cross_attention
        self.num_tokens = seq_len  # For compatibility with decoder
        
        if intermediate_dim is None:
            intermediate_dim = output_size // 2
        
        # Project each position's features to LLM embedding size
        self.token_projection = nn.Sequential(
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
            x: Quantized features [batch, 128, 1024] or [batch, 1, 128, 1024]
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

        # Project each position independently: [batch, 128, 1024] -> [batch, 128, output_size]
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
        """Return the number of tokens this bridge produces."""
        return self.seq_len
    
    def get_token_info(self) -> dict:
        """Return information about the tokens produced by this bridge."""
        return {
            "num_tokens": self.seq_len,
            "token_dim": self.output_size,
            "token_type": "sequence",
            "description": f"Each of {self.seq_len} ECG positions becomes a separate token"
        }


@ModelRegistry.register(BridgeName.GPT2_SIMPLE_TOKEN_BRIDGE)
@ModelRegistry.register(BridgeName.LLAMA32_SIMPLE_TOKEN_BRIDGE)
@ModelRegistry.register("SimpleTokenBridge")
class SimpleTokenBridge(nn.Module):
    """
    Simple bridge that converts each of the 128 ECG positions into separate LLM tokens
    using a straightforward linear projection approach.
    
    This creates 128 tokens with simple linear projection from 1024-dim features 
    to LLM embedding size (2048), without complex attention mechanisms.
    """
    
    def __init__(
        self, 
        # input_shape: tuple[int, int] = (128, 82), 
        input_shape: tuple[int, int] = (128, 1024), 
        output_size: int = 2048,
        dropout: float = 0.1
    ):
        """
        Args:
            input_shape: (seq_len, feature_dim) = (128, 1024) for ECG quantized features
            output_size: LLM embedding dimension (e.g., 2048 for Llama-3.2-1B)
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.seq_len, self.feature_dim = input_shape  # 128, 1024
        self.output_size = output_size
        self.num_tokens = self.seq_len  # For compatibility with decoder (128 tokens)
        self.input_layernorm = nn.LayerNorm(self.feature_dim)
        
        # Simple linear projection for each position
        self.token_projection = nn.Sequential(
            nn.Linear(self.feature_dim, output_size),  # 1024 -> 2048
            nn.LayerNorm(output_size),
            nn.GELU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convert quantized ECG features to sequence of LLM tokens.
        
        Args:
            x: Quantized features [batch, 128, 1024] or [batch, 1, 128, 1024]
            
        Returns:
            Token embeddings [batch, 128, output_size] - one token per ECG position
        """

        if x.dim() == 4:
            x = x.squeeze(1)
        
        batch_size, seq_len, feature_dim = x.shape
        assert seq_len == self.seq_len, f"Expected sequence length {self.seq_len}, got {seq_len}"
        assert feature_dim == self.feature_dim, f"Expected feature dim {self.feature_dim}, got {feature_dim}"
        
        x = self.input_layernorm(x)
        token_embeddings = self.token_projection(x)
        
        return token_embeddings  # [batch, 128, output_size]
    
    def get_sequence_length(self) -> int:
        """Return the number of tokens this bridge produces."""
        return self.seq_len
    
    def get_token_info(self) -> dict:
        """Return information about the tokens produced by this bridge."""
        return {
            "num_tokens": self.seq_len,
            "token_dim": self.output_size,
            "token_type": "simple_sequence",
            "description": f"Simple linear projection: each of {self.seq_len} ECG positions -> separate token"
        }

@ModelRegistry.register(BridgeName.CROSS_MODAL_SEQUENCE_TOKEN_BRIDGE)
@ModelRegistry.register("CrossModalSequenceTokenBridge")
class CrossModalSequenceTokenBridge(nn.Module):
    """
    Enhanced bridge that converts ECG positions into LLM tokens with cross-modal attention.
    
    This bridge enables bidirectional attention between ECG tokens and text tokens,
    allowing the model to learn rich cross-modal representations.
    """
    
    def __init__(
        self, 
        # input_shape: tuple[int, int] = (128, 82), 
        input_shape: tuple[int, int] = (128, 1024), 
        output_size: int = 2048, 
        dropout: float = 0.05,
        use_cross_attention: bool = True,
        num_attention_heads: int = 8,
        intermediate_dim: int = None,
        num_cross_attention_layers: int = 2
    ):
        """
        Initialize the cross-modal sequence token bridge.
        
        Args:
            input_shape: (seq_len, feature_dim) = (128, 1024) for ECG quantized features
            output_size: LLM embedding dimension (e.g., 2048 for Llama-3.2-1B)
            dropout: Dropout rate for regularization
            use_cross_attention: Whether to apply cross-attention
            num_attention_heads: Number of attention heads
            intermediate_dim: Intermediate projection dimension
            num_cross_attention_layers: Number of cross-attention layers
        """
        super().__init__()
        
        seq_len, feature_dim = input_shape  # 128, 1024
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
        text_attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> tuple[torch.Tensor, Optional[list]]:
        """
        Convert ECG features to tokens with optional cross-modal attention.

        Args:
            x: ECG quantized features [batch, 128, 1024]
            text_embeddings: Optional text token embeddings [batch, text_len, output_size]
            text_attention_mask: Optional attention mask [batch, text_len] (1 = keep)
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

        key_padding_mask = None
        if text_attention_mask is not None:
            if text_attention_mask.dim() == 1:
                text_attention_mask = text_attention_mask.unsqueeze(0)
            if text_attention_mask.dim() != 2:
                raise ValueError("text_attention_mask must be [batch, seq] if provided")
            key_padding_mask = ~text_attention_mask.bool()

        # Apply cross-attention layers if text embeddings are provided
        attention_weights = []
        if self.use_cross_attention and text_embeddings is not None:
            for layer in self.cross_attention_layers:
                ecg_embeddings, attn_weights = layer(
                    query=ecg_embeddings,
                    key_value=text_embeddings,
                    key_padding_mask=key_padding_mask,
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
        """Return the number of tokens this bridge produces."""
        return self.seq_len
    
    def get_token_info(self) -> dict:
        """Return information about the tokens produced by this bridge."""
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
        key_padding_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply cross-attention between query and key-value tensors.

        Args:
            query: Query tensor [batch, query_len, embed_dim]
            key_value: Key and value tensor [batch, kv_len, embed_dim]
            key_padding_mask: Optional mask where True marks positions to ignore
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
                average_attn_weights=False,  # Keep per-head weights
                key_padding_mask=key_padding_mask
            )
        else:
            attended, attn_weights = self.cross_attention(
                query=query,
                key=key_value,
                value=key_value,
                need_weights=False,
                key_padding_mask=key_padding_mask
            ), None

        query = residual + attended

        # Feedforward with residual connection
        residual = query
        query = self.norm2(query)
        query = residual + self.ffn(query)

        return query, attn_weights


@torch.no_grad()
def calibrate_bridge_scale(
    model: torch.nn.Module,
    sample: Union[torch.Tensor, Callable[[], torch.Tensor], dict],
    *,
    attn_mask: Optional[torch.Tensor] = None,
    device: Optional[Union[str, torch.device]] = None,
) -> dict[str, float]:
    """
    Calibrate the bridge's output scale so ECG token embeddings match the LLM embedding RMS.

    Args:
        model: Wrapper containing a decoder with `llm_model`/`llm` and `bridge`.
        sample: Example batch passed through the bridge. Can be a tensor, a callable
            returning a tensor, or a dict containing `input`/`features`/`codes`.
        attn_mask: Optional attention mask to use when the bridge expects one.
        device: Device to run calibration on. Defaults to bridge parameter device.

    Returns:
        Dictionary with calibration diagnostics (target RMS, current RMS, applied scale).
    """
    decoder = getattr(model, "decoder", model)
    bridge = getattr(decoder, "bridge", None)
    if bridge is None:
        raise ValueError("Model does not expose a decoder bridge for calibration.")

    llm = getattr(decoder, "llm_model", None) or getattr(decoder, "llm", None)
    if llm is None or not hasattr(llm, "get_input_embeddings"):
        raise ValueError("Decoder LLM does not expose input embeddings for calibration.")

    embeddings = llm.get_input_embeddings().weight.detach()
    target_rms = embeddings.float().pow(2).mean().sqrt().item()

    bridge_param = next(bridge.parameters(), None)
    bridge_device = bridge_param.device if bridge_param is not None else embeddings.device
    target_device = torch.device(device) if device is not None else bridge_device

    if callable(sample):
        sample_batch = sample()
    elif isinstance(sample, dict):
        sample_batch = (
            sample.get("input")
            or sample.get("features")
            or sample.get("codes")
        )
        if sample_batch is None:
            raise ValueError(
                "Sample dict must contain 'input', 'features', or 'codes' tensor for calibration."
            )
        attn_mask = sample.get("attn_mask", attn_mask)
    else:
        sample_batch = sample

    if not isinstance(sample_batch, torch.Tensor):
        raise TypeError("Calibration sample must resolve to a torch.Tensor.")

    sample_batch = sample_batch.to(target_device)
    attn_mask_tensor = attn_mask.to(target_device) if isinstance(attn_mask, torch.Tensor) else None

    prev_mode = bridge.training
    bridge.eval()
    if getattr(bridge, "uses_codes", False):
        outputs = bridge(sample_batch, attn_mask=attn_mask_tensor)
    else:
        outputs = bridge(sample_batch)

    current_rms = outputs.float().pow(2).mean().sqrt().item()
    scale = target_rms / (current_rms + 1e-8)

    if hasattr(bridge, "output_scale") and isinstance(bridge.output_scale, torch.nn.Parameter):
        old_scale = bridge.output_scale.item()
        bridge.output_scale.mul_(scale)
        new_scale = bridge.output_scale.item()
        print(
            f"[BridgeCalib] output_scale adjusted: {old_scale:.6f} -> {new_scale:.6f} "
            f"(target_rms={target_rms:.6f}, current_rms={current_rms:.6f})"
        )
    else:
        warnings.warn(
            "Bridge does not expose an `output_scale` parameter; calibration skipped.",
            RuntimeWarning,
        )

    mix_params = getattr(bridge, "mix_weights", None)
    if isinstance(mix_params, torch.nn.Parameter):
        with torch.no_grad():
            mix_probs = mix_params.softmax(dim=0)
            formatted = ", ".join(f"{p:.4f}" for p in mix_probs.tolist())
            print(f"[BridgeCalib] mix_weights softmax: [{formatted}]")

    if prev_mode:
        bridge.train(True)

    return {
        "target_rms": target_rms,
        "current_rms": current_rms,
        "applied_scale": scale,
    }
