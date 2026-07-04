from __future__ import annotations

import torch
import torch.nn.functional as F


def normalize_ecg_embeddings(
    ecg_embeddings: torch.Tensor,
    target_std: float | None,
) -> torch.Tensor:
    """Rescale ECG token embeddings to match the LLM embedding std."""
    if target_std is None:
        return ecg_embeddings
    original_dtype = ecg_embeddings.dtype
    work = ecg_embeddings.float()
    with torch.no_grad():
        cur_std = work.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
    scaled = work / cur_std * float(target_std)
    return scaled.to(dtype=original_dtype)


def build_inputs_with_ecg_prefix(
    llm_model,
    text_input_ids: torch.Tensor,
    text_attention_mask: torch.Tensor,
    ecg_token_embeddings: torch.Tensor,
    *,
    target_embedding_std: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Concatenate ECG embeddings in front of token embeddings for LLM input.

    Returns inputs_embeds and combined attention mask. text_input_ids must exclude ECG tokens.
    """
    device = text_input_ids.device
    ecg_token_embeddings = normalize_ecg_embeddings(ecg_token_embeddings, target_embedding_std)

    text_embeddings = llm_model.get_input_embeddings()(text_input_ids)
    if text_embeddings.dtype != ecg_token_embeddings.dtype:
        ecg_token_embeddings = ecg_token_embeddings.to(text_embeddings.dtype)
    inputs_embeds = torch.cat([ecg_token_embeddings, text_embeddings], dim=1)

    ecg_mask = torch.ones(
        (text_attention_mask.size(0), ecg_token_embeddings.size(1)),
        dtype=text_attention_mask.dtype,
        device=device,
    )
    attention_mask = torch.cat([ecg_mask, text_attention_mask], dim=1)
    return inputs_embeds, attention_mask


def pad_labels_for_ecg_prefix(
    labels: torch.Tensor,
    ecg_prefix_length: int,
) -> torch.Tensor:
    """Add -100 padding in front of labels to ignore ECG tokens during CE loss."""
    if ecg_prefix_length <= 0:
        return labels
    if labels.dim() != 2:
        raise ValueError(f"Expected 2D labels for padding, got shape {labels.shape}")
    pad_shape = (ecg_prefix_length, 0, 0, 0)  # pad along sequence dimension (columns)
    return F.pad(labels, pad_shape, value=-100)
