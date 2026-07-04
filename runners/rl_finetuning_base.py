"""Shared base class for GRPO and DPO finetuning runners.

Contains common methods for ECG encoding, log-probability computation,
label expansion, and tokenizer access.
"""

from typing import Any, Callable, Optional

import torch
from torch.utils.data import DataLoader

from runners.base_runner import BaseRunner
from utils.enums import RunMode


class RLFinetuningRunnerBase(BaseRunner):
    """Base class with shared logic for GRPO and DPO training runners."""

    def __init__(
        self,
        config,
        wandb_wrapper=None,
        model=None,
        ref_model=None,
        train_dataloader: DataLoader | None = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
    ):
        super().__init__(config, wandb_wrapper)
        self.model = model
        self.ref_model = ref_model
        self.train_dataloader = train_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.global_step = 0
        self.start_time = None

    # ------------------------------------------------------------------
    # Token / sequence log-probability helpers
    # ------------------------------------------------------------------

    def _token_logp(self, logits: torch.Tensor, labels: torch.Tensor) -> tuple:
        """Compute per-token log-probabilities and mask.

        Returns:
            token_logp: [B, seq_len-1] per-token log-probs (0 where masked)
            mask: [B, seq_len-1] boolean mask for valid (non -100) tokens
        """
        shift_logits = logits[:, :-1, :]
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        shift_labels = shift_labels.clamp_min(0)
        log_probs = torch.log_softmax(shift_logits, dim=-1)
        token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
        token_logp = token_logp * mask
        return token_logp, mask

    def _sequence_logp(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute per-sequence log-probabilities (sum of token log-probs)."""
        token_logp, mask = self._token_logp(logits, labels)
        return token_logp.sum(dim=-1)

    def _sequence_nll(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Negative log-likelihood per token, averaged over sequences. Reuses _sequence_logp."""
        shift_labels = labels[:, 1:]
        mask = shift_labels != -100
        token_counts = mask.sum(dim=-1).clamp_min(1)
        seq_logp = self._sequence_logp(logits, labels)
        nll = -seq_logp / token_counts
        return nll.mean()

    # ------------------------------------------------------------------
    # ECG encoding (cacheable per model per step)
    # ------------------------------------------------------------------

    def _encode_ecg(
        self,
        model,
        ecg_signal: torch.Tensor,
        prompt_input_ids: Optional[torch.Tensor],
        prompt_attention_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Run encoder + quantizer + bridge to produce ECG embeddings.

        This is deterministic for the same model weights and signal, so the
        result can be cached and reused across multiple forward passes within
        the same training step.
        """
        ecg_signal = ecg_signal.to(dtype=torch.float32)
        features = model.encoder(ecg_signal)
        quantized, indices, _ = model.quantizer(features)
        quantized_codes = model._extract_primary_codes(
            indices, model.num_codebooks_kept, model.codebook_offset
        )
        ecg_embeddings, _ = model.decoder._compute_ecg_embeddings(
            quantized,
            quantized_codes,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
        )
        return ecg_embeddings

    # ------------------------------------------------------------------
    # Label expansion with ECG injection
    # ------------------------------------------------------------------

    def _expand_labels_with_ecg(
        self,
        decoder,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        ecg_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        merged = decoder._inject_ecg_after_image_token(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            ecg_embeddings=ecg_embeddings,
            embed_layer=decoder.llm_model.get_input_embeddings(),
        )
        if merged is not None:
            _, _, _, labels_out = merged[:4]
            if labels_out is None:
                raise ValueError("Failed to expand labels with ECG injection.")
            return labels_out

        prefix_len = int(ecg_embeddings.size(1))
        if prefix_len <= 0:
            return labels
        ignore_pad = torch.full(
            (labels.size(0), prefix_len),
            -100,
            dtype=labels.dtype,
            device=labels.device,
        )
        return torch.cat([ignore_pad, labels], dim=1)

    # ------------------------------------------------------------------
    # Forward pass (with optional pre-computed ECG embeddings)
    # ------------------------------------------------------------------

    def _forward_logits_and_labels(
        self,
        model,
        ecg_signal: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
        prompt_input_ids: Optional[torch.Tensor],
        prompt_attention_mask: Optional[torch.Tensor],
        ecg_embeddings: Optional[torch.Tensor] = None,
    ) -> tuple:
        """Forward pass returning logits and expanded labels.

        Args:
            ecg_embeddings: Pre-computed ECG embeddings from ``_encode_ecg``.
                If None, they will be computed from ``ecg_signal``.
        """
        if ecg_embeddings is None:
            ecg_embeddings = self._encode_ecg(
                model, ecg_signal, prompt_input_ids, prompt_attention_mask
            )

        outputs = model.decoder(
            quantized_features=None,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            quantized_codes=None,
            ecg_embeddings=ecg_embeddings,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
        )
        logits = outputs["logits"]
        labels_expanded = self._expand_labels_with_ecg(
            decoder=model.decoder,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            ecg_embeddings=ecg_embeddings,
        )
        return logits, labels_expanded

    # ------------------------------------------------------------------
    # Tokenizer access
    # ------------------------------------------------------------------

    def _get_text_tokenizer(self):
        if self.model is None:
            return None
        if hasattr(self.model, "_get_text_tokenizer"):
            try:
                return self.model._get_text_tokenizer()
            except Exception:
                pass
        decoder = getattr(self.model, "decoder", None)
        return getattr(decoder, "tokenizer", None)

    # ------------------------------------------------------------------
    # Abstract from BaseRunner (kept as not-implemented here)
    # ------------------------------------------------------------------

    def _run_epoch(
        self,
        mode: RunMode,
        epoch: int,
        dataloader: DataLoader,
        step_fn: Callable,
    ) -> dict[str, float]:
        raise NotImplementedError
