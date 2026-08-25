"""ECG Bridge modules for connecting ECG tokens to LLM embedding space."""

import math
import re
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union, cast, Literal, Sequence, TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.registry import ModelRegistry
from utils.enums import BridgeName

if TYPE_CHECKING:
    from transformers.models.bert.modeling_bert import BertLayer

__all__ = [
    "ECGCodeBridge",
    "ECGProjectionBridge",
    "PerceiverProjectionBridge",
    "ECGQFormerBridge",
    "ECGQFormerBridgeStage1",
    "LinearBridge",
    "EmbeddingBridge",
    "SimpleEmbeddingBridge",
    "SequenceBridge",
    "SequenceTokenBridge",
    "SimpleTokenBridge",
    "CrossModalSequenceTokenBridge",
    "CrossAttentionLayer",
    "calibrate_bridge_scale",
    "InstructionAwareECGQFormerBridge",
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


@ModelRegistry.register(BridgeName.LLAMA32_ECG_QFORMER_BRIDGE)
@ModelRegistry.register("ECGQFormerBridge")
class ECGQFormerBridge(nn.Module):
    """BLIP-2 style Q-Former bridge that resamples discrete ECG codes for LLM conditioning.

    The bridge consumes residual VQ code indices (up to 8 codebooks by default), mixes them
    dynamically per time-step, and produces both:
      • Prefix tokens projected into the LLM embedding space (shape [B, K, d_llm])
      • A pooled ECG embedding suited for SigLIP-style contrastive losses (shape [B, d_txt])
    """

    uses_codes: bool = True

    def __init__(
        self,
        vocab_size: int,
        num_codebooks: int,
        d_mid: int,
        d_llm: int,
        d_txt: int,
        num_steps: int = 128,
        num_query_tokens: int = 32,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_special_tokens: int = 4,
        bias_last_codebook: float = 0.5,
        codebook_dropout: float = 0.0,
        mix_strategy: str = "softmax",
        token_axis: str = "channel",
        codebook_dim: int = 82,
    ) -> None:
        super().__init__()

        if d_mid % num_heads != 0:
            raise ValueError(f"d_mid ({d_mid}) must be divisible by num_heads ({num_heads}).")
        if num_codebooks <= 0:
            raise ValueError("num_codebooks must be positive for ECGQFormerBridge.")
        if mix_strategy not in ("softmax", "sum", "concat_linear"):
            raise ValueError(
                f"mix_strategy must be 'softmax', 'sum', or 'concat_linear'; got '{mix_strategy}'."
            )
        if token_axis not in ("channel", "time"):
            raise ValueError(f"token_axis must be 'channel' or 'time'; got '{token_axis}'.")

        self.vocab_size = int(vocab_size)
        self.pad_id = int(vocab_size)
        self.num_codebooks = int(num_codebooks)
        self.num_steps = int(num_steps)
        self.num_query_tokens = int(num_query_tokens)
        self.bias_last_codebook = float(bias_last_codebook)
        self.codebook_dropout = float(codebook_dropout)
        self.mix_strategy = str(mix_strategy)
        self.token_axis = str(token_axis)
        self.codebook_dim = int(codebook_dim)

        total_vocab = self.vocab_size + max(1, int(num_special_tokens))
        self.embed_tables = nn.ModuleList([
            nn.Embedding(total_vocab, d_mid)
            for _ in range(self.num_codebooks)
        ])

        self.input_norm = nn.RMSNorm(d_mid)
        if self.token_axis == "time":
            # Step 3.2: kv positions are TIME slices, not channels. Codes are assigned
            # per channel (RVQ dim=codebook_dim quantises the time course), so the ids
            # carry no time axis — rebuild the continuous post-quant tensor z through the
            # FROZEN quantizer itself (get_output_from_indices; the x1_split RVQ uses
            # implicit neural codebooks, so levels >=1 are MLP-conditioned on the running
            # sum and no static table can reproduce them), transpose it time-major, and
            # project each time slice to d_mid with an identity init: at step 0 the kv
            # features ARE zᵀ (+PE) — the probe-validated view. The quantizer is attached
            # by reference via attach_quantizer() and is NOT part of this module's params.
            object.__setattr__(self, "_frozen_rvq", None)
            self.time_proj = nn.Linear(self.num_steps, d_mid)
            with torch.no_grad():
                self.time_proj.weight.zero_()
                self.time_proj.weight[: self.num_steps, : self.num_steps] = torch.eye(self.num_steps)
                self.time_proj.bias.zero_()
        elif self.mix_strategy == "softmax":
            self.mix_gate = nn.Sequential(
                nn.Linear(self.num_codebooks * d_mid, d_mid),
                nn.GELU(),
                nn.Linear(d_mid, self.num_codebooks),
            )
        elif self.mix_strategy == "concat_linear":
            # RVQ is additive: sum(e_d) is the vector the tokenizer's decoder consumes.
            # Initialise each d_mid x d_mid block to the identity so the projection starts
            # as the plain sum and can only diverge from it if the data asks for it.
            self.mix_proj = nn.Linear(self.num_codebooks * d_mid, d_mid)
            with torch.no_grad():
                self.mix_proj.weight.copy_(
                    torch.eye(d_mid).repeat(1, self.num_codebooks)
                )
                self.mix_proj.bias.zero_()

        self.register_buffer("time_pe_cache", torch.empty(0), persistent=False)

        query_init = torch.randn(self.num_query_tokens, d_mid) * (1.0 / math.sqrt(d_mid))
        self.queries = nn.Parameter(query_init)

        self.blocks = nn.ModuleList([
            _QFormerBlock(
                d_mid=d_mid,
                num_heads=num_heads,
                dropout=dropout,
            )
            for _ in range(int(num_layers))
        ])

        self.to_llm = nn.Linear(d_mid, d_llm)
        self.norm_out = nn.RMSNorm(d_llm)
        self.output_scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))

        self.pool_norm = nn.RMSNorm(d_mid)
        self.pool_gate = nn.Linear(d_mid, d_mid)
        self.to_txt = nn.Linear(d_mid, d_txt)

        self.dropout = nn.Dropout(dropout if dropout and dropout > 0 else 0.0)

    def attach_quantizer(self, rvq: nn.Module) -> None:
        """Attach the FROZEN ResidualVQ by reference (token_axis='time' only).

        Stored via object.__setattr__ so it is NOT registered as a submodule: its
        weights stay out of this bridge's state_dict/optimizer and are owned by the
        tokenizer that loaded them. A reference stays valid if the owner loads new
        weights in place afterwards."""
        if self.token_axis != "time":
            raise RuntimeError("attach_quantizer is only meaningful for token_axis='time'.")
        if not hasattr(rvq, "get_output_from_indices"):
            raise TypeError("attach_quantizer expects a ResidualVQ with get_output_from_indices.")
        cb = getattr(rvq, "codebooks", None)
        if cb is not None and int(cb.shape[-1]) != self.codebook_dim:
            raise ValueError(
                f"quantizer codebook_dim {int(cb.shape[-1])} != bridge codebook_dim {self.codebook_dim}."
            )
        object.__setattr__(self, "_frozen_rvq", rvq)

    @property
    def num_tokens(self) -> int:
        return self.num_query_tokens

    def _prepare_ids(self, ecg_ids: torch.Tensor) -> torch.Tensor:
        if ecg_ids.dim() == 2:
            ecg_ids = ecg_ids.unsqueeze(-1)
        elif ecg_ids.dim() != 3:
            raise ValueError(f"ECG ids must be 2D or 3D, got {ecg_ids.shape}.")

        batch, seq_len, depth = ecg_ids.shape
        if depth > self.num_codebooks:
            ecg_ids = ecg_ids[..., -self.num_codebooks:]
        elif depth < self.num_codebooks:
            pad = torch.full(
                (batch, seq_len, self.num_codebooks - depth),
                self.pad_id,
                dtype=ecg_ids.dtype,
                device=ecg_ids.device,
            )
            ecg_ids = torch.cat([ecg_ids, pad], dim=-1)
        return ecg_ids

    def _time_pe(self, length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self.time_pe_cache.numel() == 0 or self.time_pe_cache.size(0) < length or self.time_pe_cache.size(1) != dim:
            positions = torch.arange(length, device=device).unsqueeze(1)
            div_term = torch.exp(
                torch.arange(0, dim, 2, device=device, dtype=torch.float32)
                * (-math.log(10000.0) / dim)
            )
            pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
            pe[:, 0::2] = torch.sin(positions * div_term)
            pe[:, 1::2] = torch.cos(positions * div_term)
            self.time_pe_cache = pe
        return self.time_pe_cache[:length].to(device=device, dtype=dtype)

    def _embed_and_fuse(self, ecg_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed per-codebook ids and fuse them into one vector per position.

        Returns (mixed [B, L, d_mid] — normalised, dropout applied — and
        valid_positions [B, L], True where at least one codebook holds a real code).
        """
        if ecg_ids.dtype != torch.long:
            ecg_ids = ecg_ids.long()

        ecg_ids = self._prepare_ids(ecg_ids)
        batch_size, seq_len, _ = ecg_ids.shape

        original_ids = ecg_ids
        ids = ecg_ids.clamp_min(0)
        valid_levels = original_ids >= 0
        if self.pad_id is not None:
            valid_levels = valid_levels & (original_ids != self.pad_id)

        if self.token_axis == "time":
            rvq = getattr(self, "_frozen_rvq", None)
            if rvq is None:
                raise RuntimeError(
                    "token_axis='time' requires attach_quantizer(rvq) before the first forward."
                )
            # ids: [B, L(channels), num_codebooks] -> exact post-quant tensor z through the
            # frozen quantizer decode (handles implicit-neural-codebook levels correctly).
            # The INC decode materialises large per-level intermediates, so rebuild in
            # small batch chunks — z itself is tiny (B x 128 x 82).
            ids_safe = ids.clamp(max=self.vocab_size - 1)
            chunk = int(getattr(self, "rebuild_chunk", 64) or 64)
            with torch.no_grad():
                parts = [
                    rvq.get_output_from_indices(ids_safe[i:i + chunk])
                    for i in range(0, ids_safe.size(0), chunk)
                ]
                z = torch.cat(parts, dim=0)                      # [B, L, codebook_dim]
            z = z.to(dtype=self.time_proj.weight.dtype)
            zt = z.permute(0, 2, 1)                              # [B, D(time), L(channels)]
            mixed = self.time_proj(zt)
            time_pe = self._time_pe(self.codebook_dim, mixed.size(-1), mixed.device, mixed.dtype)
            mixed = mixed + time_pe.unsqueeze(0)
            mixed = self.input_norm(mixed)
            mixed = self.dropout(mixed)
            # every time slice of a full-length ECG is a valid kv position
            valid_positions = torch.ones(batch_size, self.codebook_dim,
                                         dtype=torch.bool, device=mixed.device)
            return mixed, valid_positions

        embeddings: List[torch.Tensor] = []
        for level, table in enumerate(self.embed_tables):
            level_ids = ids[..., level].clamp(max=table.num_embeddings - 1)
            embeddings.append(table(level_ids))
        x_stack = torch.stack(embeddings, dim=2)  # [B, L, num_codebooks, d_mid]

        time_pe = self._time_pe(seq_len, x_stack.size(-1), x_stack.device, x_stack.dtype)
        level_mask = valid_levels.unsqueeze(-1).to(x_stack.dtype)

        if self.mix_strategy == "softmax":
            x_stack = x_stack + time_pe.unsqueeze(0).unsqueeze(2)
            x_stack = x_stack * level_mask

            gate_input = x_stack.reshape(batch_size, seq_len, -1)
            gate_logits = self.mix_gate(gate_input)

            if self.bias_last_codebook:
                gate_logits[..., -1] = gate_logits[..., -1] + float(self.bias_last_codebook)

            if self.training and self.codebook_dropout > 0.0:
                drop_prob = torch.rand_like(gate_logits) < self.codebook_dropout
                drop_prob = drop_prob & valid_levels
                gate_logits = gate_logits.masked_fill(drop_prob, -1e4)

            mask_logits = (~valid_levels).to(gate_logits.dtype) * -1e4
            gate_logits = gate_logits + mask_logits

            weights = gate_logits.softmax(dim=-1)
            mixed = (weights.unsqueeze(-1) * x_stack).sum(dim=2)
        else:
            # Additive fusion: invalid levels are zeroed so they contribute nothing to the
            # sum, and the positional encoding is added once after fusion (the softmax gate
            # folds it in at weight-sum 1; an unscaled per-level add would inject it
            # num_codebooks times).
            x_stack = x_stack * level_mask
            if self.training and self.codebook_dropout > 0.0:
                keep = (
                    torch.rand(
                        batch_size, seq_len, self.num_codebooks, 1,
                        device=x_stack.device,
                    )
                    >= self.codebook_dropout
                )
                x_stack = x_stack * keep.to(x_stack.dtype)
            if self.mix_strategy == "sum":
                mixed = x_stack.sum(dim=2)
            else:
                mixed = self.mix_proj(x_stack.reshape(batch_size, seq_len, -1))
            mixed = mixed + time_pe.unsqueeze(0)

        valid_positions = valid_levels.any(dim=-1)
        mixed = mixed * valid_positions.unsqueeze(-1).to(mixed.dtype)
        mixed = self.input_norm(mixed)
        mixed = self.dropout(mixed)
        return mixed, valid_positions

    def forward(
        self,
        ecg_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mixed, valid_positions = self._embed_and_fuse(ecg_ids)
        batch_size = mixed.size(0)

        if attn_mask is not None and self.token_axis == "channel":
            if attn_mask.dim() > 2:
                raise ValueError("attn_mask for ECGQFormerBridge must be [batch, seq].")
            attn_mask_bool = attn_mask.to(dtype=torch.bool, device=mixed.device)
            valid_positions = valid_positions & attn_mask_bool

        key_padding_mask = ~valid_positions

        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        queries = queries.to(device=mixed.device, dtype=mixed.dtype)

        for block in self.blocks:
            queries = block(queries, mixed, key_padding_mask=key_padding_mask)

        prefix = self.to_llm(queries)
        prefix = self.norm_out(prefix)
        prefix = prefix * self.output_scale

        pooled = self.pool_norm(queries)
        gate = torch.sigmoid(self.pool_gate(pooled))
        gated = gate * pooled
        ecg_vec = self.to_txt(gated.mean(dim=1))
        ecg_vec = F.normalize(ecg_vec, p=2, dim=-1, eps=1e-6)

        return prefix, ecg_vec


class _QFormerBlock(nn.Module):
    """Transformer block that mirrors BLIP-2 Q-Former behaviour."""

    def __init__(self, d_mid: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        self.self_norm = nn.RMSNorm(d_mid)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_mid,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_norm_q = nn.RMSNorm(d_mid)
        self.cross_norm_kv = nn.RMSNorm(d_mid)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_mid,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.ffn_norm = nn.RMSNorm(d_mid)
        self.ffn = nn.Sequential(
            nn.Linear(d_mid, d_mid * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_mid * 4, d_mid),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout if dropout and dropout > 0 else 0.0)

    def forward(
        self,
        queries: torch.Tensor,
        kv: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = queries
        q_norm = self.self_norm(queries)
        self_out, _ = self.self_attn(q_norm, q_norm, q_norm, need_weights=False)
        queries = residual + self.dropout(self_out)

        residual = queries
        q_norm = self.cross_norm_q(queries)
        kv_norm = self.cross_norm_kv(kv)
        cross_out, _ = self.cross_attn(
            q_norm,
            kv_norm,
            kv_norm,
            need_weights=False,
            key_padding_mask=key_padding_mask,
        )
        queries = residual + self.dropout(cross_out)

        residual = queries
        ffn_out = self.ffn(self.ffn_norm(queries))
        queries = residual + ffn_out
        return queries


Stage1Mode = Literal["ETC", "ETM", "ETG"]


class _Stage1Block(nn.Module):
    """Shared self-attention over [queries || text] with optional cross-attn into ECG memory."""

    def __init__(
        self,
        d_mid: int,
        num_heads: int,
        dropout: float,
        *,
        do_cross: bool,
        bert_layer: Optional["BertLayer"] = None,
    ) -> None:
        super().__init__()
        self.do_cross = bool(do_cross)

        self.self_norm = nn.RMSNorm(d_mid)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=d_mid,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.cross_norm_q = nn.RMSNorm(d_mid)
        self.cross_norm_kv = nn.RMSNorm(d_mid)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_mid,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.ffn_norm = nn.RMSNorm(d_mid)
        self.ffn = nn.Sequential(
            nn.Linear(d_mid, d_mid * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_mid * 4, d_mid),
            nn.Dropout(dropout),
        )
        self.dropout = nn.Dropout(dropout if dropout and dropout > 0 else 0.0)

        if bert_layer is not None:
            self._init_from_bert_layer(bert_layer)

    def _init_from_bert_layer(self, bert_layer: "BertLayer") -> None:
        self_attention = bert_layer.attention.self
        self_output = bert_layer.attention.output
        intermediate = bert_layer.intermediate
        output = bert_layer.output

        hidden_size = self_attention.query.weight.size(0)
        if hidden_size != self.self_attn.embed_dim:
            raise ValueError(
                "BERT layer hidden size does not match Q-Former hidden size: "
                f"{hidden_size} vs {self.self_attn.embed_dim}"
            )

        with torch.no_grad():
            self.self_attn.in_proj_weight.copy_(
                torch.cat(
                    [self_attention.query.weight, self_attention.key.weight, self_attention.value.weight],
                    dim=0,
                )
            )
            self.self_attn.in_proj_bias.copy_(
                torch.cat(
                    [self_attention.query.bias, self_attention.key.bias, self_attention.value.bias],
                    dim=0,
                )
            )
            self.self_attn.out_proj.weight.copy_(self_output.dense.weight)
            self.self_attn.out_proj.bias.copy_(self_output.dense.bias)

            attn_norm = getattr(self_output, "LayerNorm", None)
            if attn_norm is None:
                attn_norm = getattr(self_attention, "layer_norm", None)
            if attn_norm is not None and hasattr(attn_norm, "weight"):
                self.self_norm.weight.copy_(attn_norm.weight)

            self.ffn[0].weight.copy_(intermediate.dense.weight)
            self.ffn[0].bias.copy_(intermediate.dense.bias)
            self.ffn[-2].weight.copy_(output.dense.weight)
            self.ffn[-2].bias.copy_(output.dense.bias)

            ffn_norm = getattr(output, "LayerNorm", None)
            if ffn_norm is not None and hasattr(ffn_norm, "weight"):
                self.ffn_norm.weight.copy_(ffn_norm.weight)

    def forward(
        self,
        tokens: torch.Tensor,
        kv: torch.Tensor,
        *,
        query_len: int,
        self_attn_mask: Optional[torch.Tensor],
        self_key_pad: Optional[torch.Tensor],
        kv_key_pad: Optional[torch.Tensor],
    ) -> torch.Tensor:
        residual = tokens
        norm_tokens = self.self_norm(tokens)
        self_out, _ = self.self_attn(
            norm_tokens,
            norm_tokens,
            norm_tokens,
            attn_mask=self_attn_mask,
            key_padding_mask=self_key_pad,
            need_weights=False,
        )
        tokens = residual + self.dropout(self_out)

        if self.do_cross and query_len > 0:
            q_tokens = tokens[:, :query_len, :]
            residual_q = q_tokens
            q_norm = self.cross_norm_q(q_tokens)
            kv_norm = self.cross_norm_kv(kv)
            cross_out, _ = self.cross_attn(
                q_norm,
                kv_norm,
                kv_norm,
                key_padding_mask=kv_key_pad,
                need_weights=False,
            )
            q_tokens = residual_q + self.dropout(cross_out)
            tokens = torch.cat([q_tokens, tokens[:, query_len:, :]], dim=1)

        residual = tokens
        ffn_out = self.ffn(self.ffn_norm(tokens))
        tokens = residual + ffn_out
        return tokens


@ModelRegistry.register("ECGQFormerBridgeStage1")
class ECGQFormerBridgeStage1(ECGQFormerBridge):
    """Extends ECGQFormerBridge with BLIP-2 style Stage-1 ECG↔text objectives."""

    def __init__(
        self,
        vocab_size: int,
        num_codebooks: int,
        d_mid: int,
        d_llm: int,
        d_txt: int,
        num_steps: int = 128,
        num_query_tokens: int = 32,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_special_tokens: int = 4,
        bias_last_codebook: float = 0.5,
        codebook_dropout: float = 0.0,
        mix_strategy: str = "softmax",
        token_axis: str = "channel",
        codebook_dim: int = 82,
        *,
        txt_vocab_size: int,
        txt_pad_id: int,
        txt_cls_id: Optional[int] = None,
        cross_every: int = 2,
        bert_layers: Optional[Sequence["BertLayer"]] = None,
    ) -> None:
        if txt_vocab_size <= 0:
            raise ValueError("txt_vocab_size must be positive for ECGQFormerBridgeStage1.")
        if cross_every <= 0:
            raise ValueError("cross_every must be >= 1 for ECGQFormerBridgeStage1.")

        super().__init__(
            vocab_size=vocab_size,
            num_codebooks=num_codebooks,
            d_mid=d_mid,
            d_llm=d_llm,
            d_txt=d_txt,
            num_steps=num_steps,
            num_query_tokens=num_query_tokens,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            num_special_tokens=num_special_tokens,
            bias_last_codebook=bias_last_codebook,
            codebook_dropout=codebook_dropout,
            mix_strategy=mix_strategy,
            token_axis=token_axis,
            codebook_dim=codebook_dim,
        )

        self.txt_pad_id = int(txt_pad_id)
        self.txt_cls_id = int(txt_cls_id) if txt_cls_id is not None else None
        self.stage1_cross_every = int(cross_every)

        self.txt_embed = nn.Embedding(txt_vocab_size, d_mid, padding_idx=self.txt_pad_id)
        self.lm_head = nn.Linear(d_mid, txt_vocab_size, bias=False)
        self.lm_head.weight = self.txt_embed.weight

        bert_layer_list: Optional[Sequence["BertLayer"]] = None
        if bert_layers is not None:
            bert_layer_list = list(bert_layers)
            if not bert_layer_list:
                bert_layer_list = None
            else:
                bert_hidden = bert_layer_list[0].attention.self.query.weight.size(0)
                if bert_hidden != d_mid:
                    raise ValueError(
                        "Text encoder layer hidden size does not match bridge hidden size: "
                        f"{bert_hidden} vs {d_mid}."
                    )

        stage1_blocks: List[_Stage1Block] = []
        for i in range(int(num_layers)):
            bert_layer_ref = None
            if bert_layer_list is not None and i < len(bert_layer_list):
                bert_layer_ref = bert_layer_list[i]
            stage1_blocks.append(
                _Stage1Block(
                    d_mid=d_mid,
                    num_heads=num_heads,
                    dropout=dropout,
                    do_cross=(i % self.stage1_cross_every == 0),
                    bert_layer=bert_layer_ref,
                )
            )
        self.stage1_blocks = nn.ModuleList(stage1_blocks)

        self.txt_pool_norm = nn.RMSNorm(d_mid)
        self.itm_head = nn.Linear(d_mid, 2)
        self.log_tau = nn.Parameter(torch.tensor(math.log(0.07), dtype=torch.float32))

    def temperature(self) -> torch.Tensor:
        return self.log_tau.exp()

    def _build_self_attn_mask(self, mode: Stage1Mode, query_len: int, text_len: int, *, device, dtype) -> Optional[torch.Tensor]:
        if text_len == 0:
            return None

        total = query_len + text_len
        mask = torch.zeros((total, total), device=device, dtype=torch.bool)

        if mode == "ETC":
            if query_len > 0 and text_len > 0:
                mask[:query_len, query_len:] = True
                mask[query_len:, :query_len] = True
            return mask

        if mode == "ETM":
            return None

        if mode == "ETG":
            if query_len > 0 and text_len > 0:
                mask[:query_len, query_len:] = True
            if text_len > 0:
                causal = torch.triu(torch.ones((text_len, text_len), device=device, dtype=torch.bool), diagonal=1)
                mask[query_len:, query_len:] = causal
            return mask

        raise ValueError(f"Unknown Stage1 mode: {mode}")

    def _pool_queries(self, queries: torch.Tensor) -> torch.Tensor:
        pooled = self.pool_norm(queries)
        gate = torch.sigmoid(self.pool_gate(pooled))
        pooled = (gate * pooled).mean(dim=1)
        vec = self.to_txt(pooled)
        return F.normalize(vec, p=2, dim=-1, eps=1e-6)

    def _pool_text(self, hidden: torch.Tensor, text_ids: torch.Tensor, text_pad: torch.Tensor) -> torch.Tensor:
        normed = self.txt_pool_norm(hidden)
        if self.txt_cls_id is not None:
            cls_mask = (text_ids == self.txt_cls_id)
            has_cls = cls_mask.any(dim=1)
            if has_cls.all():
                idx = cls_mask.float().argmax(dim=1)
                batch_idx = torch.arange(normed.size(0), device=normed.device)
                pooled = normed[batch_idx, idx, :]
            else:
                valid = ~text_pad
                weights = valid.unsqueeze(-1).to(normed.dtype)
                pooled = (normed * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        else:
            valid = ~text_pad
            weights = valid.unsqueeze(-1).to(normed.dtype)
            pooled = (normed * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)

        return F.normalize(pooled, p=2, dim=-1, eps=1e-6)

    def _siglip_loss(self, ecg_vec: torch.Tensor, txt_vec: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        scale = self.logit_scale.clamp(min=math.log(1.0), max=math.log(100.0)).exp()
        logits_ecg_to_txt = scale * ecg_vec @ txt_vec.t()
        logits_txt_to_ecg = logits_ecg_to_txt.t()
        labels = torch.arange(ecg_vec.size(0), device=ecg_vec.device)
        one_hot = F.one_hot(labels, ecg_vec.size(0)).to(ecg_vec.dtype)
        loss_e2t = F.binary_cross_entropy_with_logits(logits_ecg_to_txt, one_hot)
        loss_t2e = F.binary_cross_entropy_with_logits(logits_txt_to_ecg, one_hot)
        loss = 0.5 * (loss_e2t + loss_t2e)
        return loss, {"logit_scale": float(scale.detach().item())}

    def _ecg_to_hidden(
        self,
        ecg_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mixed, valid_positions = self._embed_and_fuse(ecg_ids)

        if attn_mask is not None and self.token_axis == "channel":
            if attn_mask.dim() > 2:
                raise ValueError("ecg attn_mask must be 2D for ECGQFormerBridgeStage1.")
            attn_mask_bool = attn_mask.to(dtype=torch.bool, device=mixed.device)
            valid_positions = valid_positions & attn_mask_bool

        key_padding_mask = ~valid_positions
        return mixed, key_padding_mask

    def forward_stage1(
        self,
        ecg_ids: torch.Tensor,
        text_ids: Optional[torch.Tensor],
        text_attn_mask: Optional[torch.Tensor],
        mode: Stage1Mode,
        *,
        ecg_attn_mask: Optional[torch.Tensor] = None,
        labels_for_etg: Optional[torch.Tensor] = None,
    ):
        memory, kv_key_pad = self._ecg_to_hidden(ecg_ids, ecg_attn_mask)
        batch_size = memory.size(0)
        query_len = self.num_query_tokens

        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        queries = queries.to(device=memory.device, dtype=memory.dtype)

        text_ids_tensor: Optional[torch.Tensor] = None
        text_pad: torch.Tensor
        if mode == "ETC":
            tokens = queries
            text_len = 0
            text_pad = torch.zeros((batch_size, 0), dtype=torch.bool, device=memory.device)
        else:
            if text_ids is None:
                raise ValueError("text_ids must be provided for ETM and ETG modes.")
            if text_ids.dtype != torch.long:
                text_ids = text_ids.long()
            text_ids_tensor = text_ids.to(memory.device)
            if text_attn_mask is None:
                text_attn_mask = (text_ids_tensor != self.txt_pad_id)
            text_pad = ~text_attn_mask.to(dtype=torch.bool, device=memory.device)
            text_tokens = self.txt_embed(text_ids_tensor).to(dtype=memory.dtype)
            tokens = torch.cat([queries, text_tokens], dim=1)
            text_len = text_tokens.size(1)

        self_key_pad = torch.cat(
            [
                torch.zeros((batch_size, query_len), dtype=torch.bool, device=memory.device),
                text_pad,
            ],
            dim=1,
        )

        attn_mask = self._build_self_attn_mask(
            mode,
            query_len,
            text_len,
            device=memory.device,
            dtype=memory.dtype,
        )

        for block in self.stage1_blocks:
            tokens = block(
                tokens,
                memory,
                query_len=query_len,
                self_attn_mask=attn_mask,
                self_key_pad=self_key_pad,
                kv_key_pad=kv_key_pad,
            )

        q_out = tokens[:, :query_len, :]
        t_out = tokens[:, query_len:, :]

        if mode == "ETC":
            ecg_vec = self._pool_queries(q_out)
            return ecg_vec, None

        if mode == "ETM":
            logits_per_query = self.itm_head(q_out)
            return logits_per_query.mean(dim=1)

        if mode == "ETG":
            if text_ids_tensor is None:
                raise ValueError("ETG mode requires text inputs.")
            lm_logits = self.lm_head(t_out)
            if labels_for_etg is None:
                if text_len <= 1:
                    raise ValueError("ETG requires at least two text tokens when labels are not provided.")
                lm_input = lm_logits[:, :-1, :]
                target = text_ids_tensor[:, 1:]
            else:
                if labels_for_etg.shape != lm_logits.shape[:2]:
                    raise ValueError("labels_for_etg shape must match text length.")
                lm_input = lm_logits
                target = labels_for_etg.to(device=lm_logits.device, dtype=torch.long)

            loss = F.cross_entropy(
                lm_input.reshape(-1, lm_input.size(-1)),
                target.reshape(-1),
                ignore_index=self.txt_pad_id,
            )
            return lm_logits, loss

        raise ValueError(f"Unknown Stage1 mode received: {mode}")


@ModelRegistry.register("InstructionAwareECGQFormerBridge")
class InstructionAwareECGQFormerBridge(ECGQFormerBridge):
    """Q-Former bridge that consumes Stage-1 checkpoints but exposes instruction-aware fusion.

    The module mirrors the Stage-1 architecture (including the shared self-attention blocks) so
    that we can load the pre-trained weights while bypassing the Stage-1 text embedding head.
    Instruction token embeddings are expected to already live in the Q-Former hidden dimension.
    """

    def __init__(
        self,
        vocab_size: int,
        num_codebooks: int,
        d_mid: int,
        d_llm: int,
        d_txt: int,
        num_steps: int = 128,
        num_query_tokens: int = 32,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.1,
        num_special_tokens: int = 4,
        bias_last_codebook: float = 0.5,
        codebook_dropout: float = 0.0,
        mix_strategy: str = "softmax",
        token_axis: str = "channel",
        codebook_dim: int = 82,
        *,
        cross_every: int = 2,
    ) -> None:
        super().__init__(
            vocab_size=vocab_size,
            num_codebooks=num_codebooks,
            d_mid=d_mid,
            d_llm=d_llm,
            d_txt=d_txt,
            num_steps=num_steps,
            num_query_tokens=num_query_tokens,
            num_layers=num_layers,
            num_heads=num_heads,
            dropout=dropout,
            num_special_tokens=num_special_tokens,
            bias_last_codebook=bias_last_codebook,
            codebook_dropout=codebook_dropout,
            mix_strategy=mix_strategy,
            token_axis=token_axis,
            codebook_dim=codebook_dim,
        )

        if cross_every <= 0:
            raise ValueError("cross_every must be >= 1 for InstructionAwareECGQFormerBridge.")

        self.stage1_cross_every = int(cross_every)
        self.stage1_blocks = nn.ModuleList([
            _Stage1Block(
                d_mid=d_mid,
                num_heads=num_heads,
                dropout=dropout,
                do_cross=(idx % self.stage1_cross_every == 0),
            )
            for idx in range(int(num_layers))
        ])
        # Light normalization so downstream callers can feed raw LLM embeddings if desired.
        self.instruction_norm = nn.RMSNorm(d_mid)

    def forward_instruction_hidden(
        self,
        ecg_ids: torch.Tensor,
        instruction_hidden: Optional[torch.Tensor],
        instruction_attention_mask: Optional[torch.Tensor] = None,
        *,
        ecg_attn_mask: Optional[torch.Tensor] = None,
        project_to_llm: bool = True,
        detach_soft_prompts: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Run the Q-Former with ECG codes plus instruction embeddings.

        Args:
            ecg_ids: Residual VQ code indices, shape [batch, seq, depth] or [batch, seq].
            instruction_hidden: Instruction embeddings already projected into d_mid.
            instruction_attention_mask: Optional mask where 1 denotes valid tokens.
            ecg_attn_mask: Optional mask over ECG timestep dimension.
            project_to_llm: Whether to map query outputs into the LLM embedding space.
            detach_soft_prompts: Detach the projected prompts (useful for inference).

        Returns:
            Dictionary containing:
              • token_embeddings: [B, num_query_tokens, d_llm] soft prompts.
              • query_hidden: [B, num_query_tokens, d_mid] Q-Former query states.
              • instruction_hidden: [B, T, d_mid] instruction states after fusion (if any tokens).
              • pooled_queries: [B, d_txt] pooled query representation.
        """
        memory, kv_key_pad = self._ecg_to_hidden(ecg_ids, ecg_attn_mask)
        batch_size = memory.size(0)
        query_len = self.num_query_tokens

        queries = self.queries.unsqueeze(0).expand(batch_size, -1, -1)
        queries = queries.to(device=memory.device, dtype=memory.dtype)

        if instruction_hidden is not None:
            if instruction_hidden.size(-1) != queries.size(-1):
                raise ValueError(
                    "instruction_hidden must match Q-Former hidden size: "
                    f"{instruction_hidden.size(-1)} vs {queries.size(-1)}."
                )
            instr = instruction_hidden.to(device=memory.device, dtype=memory.dtype)
            instr = self.instruction_norm(instr)
            text_len = instr.size(1)
            if instruction_attention_mask is None:
                instruction_attention_mask = torch.ones(
                    batch_size, text_len, device=memory.device, dtype=torch.long
                )
            text_pad = ~(instruction_attention_mask.to(dtype=torch.bool, device=memory.device))
        else:
            instr = torch.zeros(batch_size, 0, queries.size(-1), device=memory.device, dtype=memory.dtype)
            text_pad = torch.zeros(batch_size, 0, device=memory.device, dtype=torch.bool)
            text_len = 0

        tokens = torch.cat([queries, instr], dim=1)
        self_key_pad = torch.cat(
            [
                torch.zeros((batch_size, query_len), dtype=torch.bool, device=memory.device),
                text_pad,
            ],
            dim=1,
        )

        # Allow full self-attention between queries and instruction tokens.
        attn_mask = None

        for block in self.stage1_blocks:
            tokens = block(
                tokens,
                memory,
                query_len=query_len,
                self_attn_mask=attn_mask,
                self_key_pad=self_key_pad,
                kv_key_pad=kv_key_pad,
            )

        q_hidden = tokens[:, :query_len, :]
        instr_hidden = tokens[:, query_len:, :] if text_len > 0 else torch.zeros_like(instr)

        if project_to_llm:
            soft_prompts = self.to_llm(q_hidden)
            soft_prompts = self.norm_out(soft_prompts)
            soft_prompts = soft_prompts * self.output_scale
            if detach_soft_prompts:
                soft_prompts = soft_prompts.detach()
        else:
            soft_prompts = q_hidden

        pooled_queries = self.pool_norm(q_hidden)
        gate = torch.sigmoid(self.pool_gate(pooled_queries))
        pooled_queries = gate * pooled_queries
        pooled_queries = self.to_txt(pooled_queries.mean(dim=1))
        pooled_queries = F.normalize(pooled_queries, p=2, dim=-1, eps=1e-6)

        return {
            "token_embeddings": soft_prompts,
            "query_hidden": q_hidden,
            "instruction_hidden": instr_hidden,
            "pooled_queries": pooled_queries,
        }

    def _ecg_to_hidden(
        self,
        ecg_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        mixed, valid_positions = self._embed_and_fuse(ecg_ids)

        if attn_mask is not None and self.token_axis == "channel":
            if attn_mask.dim() > 2:
                raise ValueError("ecg attn_mask must be 2D for instruction-aware bridge.")
            attn_mask_bool = attn_mask.to(dtype=torch.bool, device=mixed.device)
            valid_positions = valid_positions & attn_mask_bool

        key_padding_mask = ~valid_positions
        return mixed, key_padding_mask

    def load_stage1_checkpoint(
        self,
        checkpoint_path: str,
        *,
        strict: bool = False,
        map_location: Union[str, torch.device] = "cpu",
    ) -> Dict[str, List[str]]:
        """Load weights from a Stage-1 checkpoint, adapting to tokenizer/depth changes."""
        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        if "model_state_dict" not in checkpoint:
            raise ValueError(f"Checkpoint at {checkpoint_path} is missing 'model_state_dict'.")
        stage1_state = checkpoint["model_state_dict"]
        current_state = self.state_dict()

        block_pattern = re.compile(r"^blocks\.(\d+)\.")
        stage1_block_pattern = re.compile(r"^stage1_blocks\.(\d+)\.")

        def _count_layers(pattern: re.Pattern[str]) -> int:
            indices = [
                int(match.group(1))
                for key in stage1_state.keys()
                if (match := pattern.match(key)) is not None
            ]
            return (max(indices) + 1) if indices else 0

        stage1_block_count = _count_layers(block_pattern)
        stage1_instruction_block_count = _count_layers(stage1_block_pattern)

        loadable_state: Dict[str, torch.Tensor] = {}
        partially_loaded_keys: List[str] = []
        missing_keys: List[str] = []
        reinitialized_keys: List[str] = []
        shape_mismatched_keys: List[Tuple[str, Tuple[int, ...], Tuple[int, ...]]] = []

        expected_reinit_prefixes: Tuple[str, ...] = ("to_llm", "norm_out", "instruction_norm")

        def _is_expected_reinit(key: str) -> bool:
            return any(key == prefix or key.startswith(prefix + ".") for prefix in expected_reinit_prefixes)

        for key, tensor in current_state.items():
            ckpt_tensor = stage1_state.get(key)
            if ckpt_tensor is None:
                block_match = block_pattern.match(key)
                if block_match is not None and int(block_match.group(1)) >= stage1_block_count:
                    continue

                instruction_match = stage1_block_pattern.match(key)
                if instruction_match is not None and int(instruction_match.group(1)) >= stage1_instruction_block_count:
                    continue

                if _is_expected_reinit(key):
                    reinitialized_keys.append(key)
                    continue

                missing_keys.append(key)
                continue

            if tensor.shape == ckpt_tensor.shape:
                loadable_state[key] = ckpt_tensor
                continue

            loaded = False
            if key.startswith("embed_tables.") and tensor.ndim >= 2 and tensor.shape[1:] == ckpt_tensor.shape[1:]:
                rows = min(tensor.shape[0], ckpt_tensor.shape[0])
                if rows > 0:
                    new_tensor = tensor.clone()
                    new_tensor[:rows] = ckpt_tensor[:rows]
                    loadable_state[key] = new_tensor
                    partially_loaded_keys.append(key)
                    loaded = True
            elif key == "queries" and tensor.shape[1:] == ckpt_tensor.shape[1:]:
                rows = min(tensor.shape[0], ckpt_tensor.shape[0])
                if rows > 0:
                    new_tensor = tensor.clone()
                    new_tensor[:rows] = ckpt_tensor[:rows]
                    loadable_state[key] = new_tensor
                    partially_loaded_keys.append(key)
                    loaded = True

            if loaded:
                continue

            if _is_expected_reinit(key):
                reinitialized_keys.append(key)
                continue

            shape_mismatched_keys.append((key, tuple(ckpt_tensor.shape), tuple(tensor.shape)))

        self.load_state_dict(loadable_state, strict=strict and not missing_keys)

        unexpected_keys = [key for key in stage1_state.keys() if key not in current_state]
        unused_checkpoint_keys = [key for key in stage1_state.keys() if key not in loadable_state]

        return {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "unused_checkpoint_keys": unused_checkpoint_keys,
            "partially_loaded_keys": partially_loaded_keys,
            "reinitialized_keys": sorted(set(reinitialized_keys)),
            "shape_mismatched_keys": shape_mismatched_keys,
            "stage1_block_count": stage1_block_count,
            "model_block_count": len(self.blocks),
            "stage1_instruction_block_count": stage1_instruction_block_count,
            "model_instruction_block_count": len(self.stage1_blocks),
        }

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
        # Stability and fusion controls
        softmax_temp: float = 1.0,
        mix_residual: float = 0.2,
        max_downsample_steps: int = 6,
        add_modality_embed: bool = True,
        pos_embedding_max_len: Optional[int] = None,
        add_cls_token: bool = False,
        gating_init: float = 1e-3,
        use_sinusoidal_pos_emb: bool = False,
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
        # Positional embedding with flexibility for varying token counts
        self.max_pos = int(pos_embedding_max_len) if pos_embedding_max_len is not None else max(int(num_tokens), 4096)
        self.use_sinusoidal_pos_emb = bool(use_sinusoidal_pos_emb)
        if not self.use_sinusoidal_pos_emb:
            self.positional_embedding = nn.Embedding(self.max_pos, d_model)
        else:
            self.register_buffer("_sin_pos_cache", None, persistent=False)
        self.out_norm = nn.RMSNorm(d_model)
        # Match typical LLM embedding scale (~0.02) unless overridden
        scale_init = float(target_std) if target_std is not None else 0.02
        self.output_scale = nn.Parameter(torch.tensor(scale_init, dtype=torch.float32))
        # Tiny residual gate for safer optimization
        self.alpha = nn.Parameter(torch.tensor(float(gating_init), dtype=torch.float32))
        # Optional modality/type embedding to flag ECG tokens
        self.modality_embed = nn.Parameter(torch.zeros(1, 1, d_model)) if add_modality_embed else None
        # Optional CLS token that can be prepended when desired
        self.add_cls_token = bool(add_cls_token)
        if self.add_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        # Mixing controls
        self.soft_temp = float(max(1e-6, softmax_temp))
        self.mix_residual = float(min(max(mix_residual, 0.0), 1.0))
        # Resampling guard
        self.max_downsample_steps = int(max(0, max_downsample_steps))

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

    def _sinusoidal_positions(self, length: int, dim: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Create sinusoidal positional embeddings [1, length, dim]."""
        position = torch.arange(length, device=device, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / dim))
        pe = torch.zeros(length, dim, device=device, dtype=torch.float32)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe.unsqueeze(0).to(dtype=dtype)

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
        weights = torch.softmax(gate_logits / self.soft_temp, dim=1).unsqueeze(2)  # [B, num_codebooks, 1, length]
        mixed = (reshaped * weights).sum(dim=1)  # [B, code_dim, length]
        # Residual path from uniform average to avoid collapse
        if self.mix_residual > 0:
            avg = reshaped.mean(dim=1, keepdim=False)
            mixed = (1.0 - self.mix_residual) * mixed + self.mix_residual * avg
        return mixed

    def _resample_to_tokens(self, features: torch.Tensor) -> torch.Tensor:
        """Low-pass filter then interpolate to the required token length."""
        # Input: [batch, code_dim, length]
        if features.size(-1) == self.num_tokens:
            return features.transpose(1, 2).contiguous()

        x = features
        # Repeatedly apply stride-2 convolution until close to target length, with a safety cap
        steps = 0
        while x.size(-1) > self.num_tokens * 2 and steps < self.max_downsample_steps:
            x = self.resample_activation(self.downsample_conv(x))
            steps += 1

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

        seq_len = x.size(1)
        if not self.use_sinusoidal_pos_emb:
            positions = torch.arange(seq_len, device=x.device, dtype=torch.long)
            pos_emb = self.positional_embedding(positions.clamp_max(self.max_pos - 1)).unsqueeze(0)
        else:
            pos_emb = self._sinusoidal_positions(seq_len, x.size(-1), x.device, x.dtype)
        x = x + pos_emb
        if self.modality_embed is not None:
            x = x + self.modality_embed
        x = self.out_norm(x)
        if self.add_cls_token:
            cls = self.cls_token.expand(x.size(0), -1, -1)
            x = torch.cat([cls, x], dim=1)
        # Apply scale and gentle residual gate
        return (x * self.output_scale) * torch.sigmoid(self.alpha)


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
        input_shape: tuple[int, int] = (128, 82),
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
        # NOTE: For ECG encoder output (batch, 128, 82):
        # - 128 = sequence length (temporal dimension)  
        # - 82 = feature dimension (quantized features)
        seq_len = input_shape[0]  # 128 = sequence length
        channels = input_shape[1]  # 82 = feature dimension
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
        input_shape: tuple[int, int] = (128, 82), 
        output_size: int = 2048, 
        dropout: float = 0.05,
        use_cross_attention: bool = True,
        num_attention_heads: int = 8,
        intermediate_dim: int = None
    ):
        """
        Initialize the sequence token bridge.
        
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
        input_shape: tuple[int, int] = (128, 82), 
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
        text_attention_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> tuple[torch.Tensor, Optional[list]]:
        """
        Convert ECG features to tokens with optional cross-modal attention.

        Args:
            x: ECG quantized features [batch, 128, 82]
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

    if isinstance(outputs, tuple):
        outputs = outputs[0]
    elif isinstance(outputs, dict):
        outputs = outputs.get("token_embeddings") or outputs.get("embeddings")

    if not isinstance(outputs, torch.Tensor):
        raise TypeError(
            "Bridge output must be a tensor (or tuple/dict containing one) for calibration."
        )

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
