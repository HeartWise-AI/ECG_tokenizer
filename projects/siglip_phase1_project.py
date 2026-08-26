from __future__ import annotations

import inspect
import os
import csv
import re
from collections import Counter
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, cast

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from data.siglip_dataset import SiglipBatchCollatorInfoNCE, SiglipDataset, get_siglip_dataloader
from models.decoder.medgemma_decoder import MedGemmaDecoder
from projects.base_project import BaseProject
from runners.siglip_phase1_runner import SiglipPhase1Runner
from utils.config.siglip_phase1_config import SiglipPhase1Config
from utils.enums import ProjectName, RunMode, BridgeName
from utils.registry import ModelRegistry, ProjectRegistry
from utils.ddp import DistributedUtils
from utils.debug import log_once, ensure_dir

from transformers import AutoModel, AutoTokenizer

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def build_alpha_map_from_effective_num(
    samples,
    all_text_ids: list[str],
    beta: float,
    clip_min: float,
    clip_max: float,
) -> tuple[dict[str, float], dict[str, int]]:
    """Return (alpha_map, positive_counts_by_text_id)."""
    pos_counts = Counter()
    for sample in samples:
        for text_id, label, _ in sample.targets:
            if label > 0:
                pos_counts[text_id] += 1

    alpha: dict[str, float] = {}
    for text_id in all_text_ids:
        count = pos_counts.get(text_id, 0)
        effective = (1.0 - (beta ** max(count, 1))) / (1.0 - beta)
        value = 1.0 / effective
        value = max(clip_min, min(clip_max, value))
        alpha[text_id] = float(value)

    total_pos = sum(pos_counts.get(text_id, 0) for text_id in all_text_ids)
    denom = sum(alpha[text_id] * pos_counts.get(text_id, 0) for text_id in all_text_ids) or 1.0
    scale = (total_pos / denom) if total_pos > 0 else 1.0
    for text_id in alpha:
        alpha[text_id] = float(alpha[text_id] * scale)

    return alpha, dict(pos_counts)


def select_tail_class_ids(
    pos_counts: dict[str, int],
    all_text_ids: list[str],
    mode: str = "topN",
    top_n: int = 50,
    min_positives: int = 10,
    include_regex: list[str] | None = None,
    exclude_regex: list[str] | None = None,
) -> list[str]:
    if not pos_counts:
        return []

    compiled_include = [re.compile(pattern) for pattern in include_regex or []]
    compiled_exclude = [re.compile(pattern) for pattern in exclude_regex or []]

    def _eligible(text_id: str) -> bool:
        if compiled_exclude and any(pattern.search(text_id) for pattern in compiled_exclude):
            return False
        if compiled_include and not any(pattern.search(text_id) for pattern in compiled_include):
            return False
        return pos_counts.get(text_id, 0) > 0

    filtered = [text_id for text_id in all_text_ids if _eligible(text_id)]
    if not filtered:
        return []

    mode = (mode or "").lower()
    key_fn = lambda tid: pos_counts.get(tid, 0)
    if mode == "threshold":
        threshold = max(1, int(min_positives))
        selected = [text_id for text_id in filtered if pos_counts.get(text_id, 0) <= threshold]
        return sorted(selected, key=key_fn)

    filtered.sort(key=key_fn)
    return filtered[: max(0, int(top_n))]


def _pool_token_embeddings(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    embedding_layer: nn.Module,
) -> torch.Tensor:
    """Mean-pool input-embedding vectors over non-pad tokens.

    Shared by the fixed text-bank path (`_prepare_text_embeddings`) and the
    Phase A-v2 per-report path so both use *identical* embedding semantics.
    Returns L2-normalized [B, H] float embeddings.
    """
    mask = attention_mask.unsqueeze(-1).to(dtype=torch.float32)
    token_embeds = embedding_layer(input_ids)  # [B, T, H]
    summed = (token_embeds.float() * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp_min(1.0)
    pooled = summed / counts
    return F.normalize(pooled, dim=-1)


class ReportTextEmbedder:
    """Embed free-text reports with the SAME mechanism used for the text bank.

    Wraps the tokenizer + frozen input-embedding layer of the text tower and
    exposes a single ``embed(reports) -> [B, H]`` call. Tokenization is cached
    per unique report string (reports repeat heavily, e.g. "Sinus rhythm"), so
    the per-step cost is dominated by the (cheap) embedding-lookup + mean-pool,
    not by the tokenizer.
    """

    def __init__(
        self,
        tokenizer: Any,
        embedding_layer: nn.Module,
        max_length: int = 256,
        device: Optional[torch.device] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.embedding_layer = embedding_layer
        self.max_length = int(max_length)
        self.device = device or next(embedding_layer.parameters()).device
        for param in self.embedding_layer.parameters():
            param.requires_grad = False
        self._tok_cache: Dict[str, tuple[list[int], int]] = {}

    def to(self, device: torch.device) -> "ReportTextEmbedder":
        self.embedding_layer = self.embedding_layer.to(device)
        self.device = device
        return self

    def _tokenize_one(self, text: str) -> tuple[list[int], int]:
        cached = self._tok_cache.get(text)
        if cached is not None:
            return cached
        ids = self.tokenizer(
            text if text else " ",
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_length,
        )["input_ids"]
        if not ids:
            # Guarantee at least one token so mean-pool is well defined.
            pad_id = self.tokenizer.pad_token_id or 0
            ids = [int(pad_id)]
        entry = (list(ids), len(ids))
        self._tok_cache[text] = entry
        return entry

    @torch.no_grad()
    def embed(self, reports: Sequence[str]) -> torch.Tensor:
        """Return L2-normalized [B, H] embeddings for a list of report strings."""
        tokenized = [self._tokenize_one(str(r)) for r in reports]
        max_len = max((length for _, length in tokenized), default=1)
        max_len = max(max_len, 1)
        pad_id = int(self.tokenizer.pad_token_id or 0)
        batch_ids = torch.full((len(tokenized), max_len), pad_id, dtype=torch.long)
        attn = torch.zeros((len(tokenized), max_len), dtype=torch.long)
        for row, (ids, length) in enumerate(tokenized):
            batch_ids[row, :length] = torch.tensor(ids, dtype=torch.long)
            attn[row, :length] = 1
        batch_ids = batch_ids.to(self.device)
        attn = attn.to(self.device)
        return _pool_token_embeddings(batch_ids, attn, self.embedding_layer)


class SiglipBridgeWrapper(nn.Module):
    """Wrap a standard bridge to provide pooled embeddings and temperature control."""

    class _RMSNorm(nn.Module):
        def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
            super().__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.eps = eps

        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            norm_x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return norm_x * self.weight

    def __init__(
        self,
        bridge: nn.Module,
        hidden_size: int,
        temperature_init: float = 0.07,
    ) -> None:
        super().__init__()
        self.bridge = bridge
        self.hidden_size = hidden_size
        self.uses_codes = bool(getattr(bridge, "uses_codes", False))
        self.target_embedding_std: float | None = None

        # Pooling components mirror the original SiglipECGBridge behaviour
        self.pool_norm = self._RMSNorm(hidden_size)
        self.pool_gate = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size),
        )
        init_tau = torch.log(torch.tensor(float(temperature_init), dtype=torch.float32))
        self.log_tau = nn.Parameter(init_tau, requires_grad=True)
        final_linear = cast(nn.Linear, self.out_proj[-1])
        nn.init.normal_(final_linear.weight, std=1e-5)
        nn.init.zeros_(final_linear.bias)

    def forward(
        self,
        *,
        features: Optional[torch.Tensor] = None,
        codes: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the underlying bridge and pool its sequence outputs."""
        if self.uses_codes:
            if codes is None:
                raise ValueError("Bridge expects code indices but none were provided.")
            bridge_input = codes
        else:
            if features is None:
                raise ValueError("Bridge expects continuous features but none were provided.")
            bridge_input = features

        bridge_output = self.bridge(bridge_input, **kwargs)

        pooled_from_bridge: Optional[torch.Tensor] = None

        if isinstance(bridge_output, tuple):
            token_embeddings = bridge_output[0]
            if len(bridge_output) > 1:
                pooled_from_bridge = bridge_output[1]
        elif isinstance(bridge_output, dict):
            if "token_embeddings" in bridge_output:
                token_embeddings = bridge_output["token_embeddings"]
            elif "embeddings" in bridge_output:
                token_embeddings = bridge_output["embeddings"]
            else:
                raise ValueError(
                    "Bridge returned a dictionary without recognised embedding keys: "
                    f"{list(bridge_output.keys())}"
                )
            if "pooled" in bridge_output:
                pooled_from_bridge = bridge_output["pooled"]
        else:
            token_embeddings = bridge_output

        if token_embeddings.dim() == 4 and token_embeddings.size(1) == 1:
            token_embeddings = token_embeddings.squeeze(1)

        if token_embeddings.dim() != 3:
            raise ValueError(
                "Expected bridge output with shape [batch, seq, hidden], "
                f"got {token_embeddings.shape}"
            )

        if self.target_embedding_std is not None:
            original_dtype = token_embeddings.dtype
            work = token_embeddings.float()
            with torch.no_grad():
                cur_std = work.std(dim=(1, 2), keepdim=True).clamp_min(1e-6)
            work = work / cur_std * float(self.target_embedding_std)
            token_embeddings = work.to(dtype=original_dtype)

        if pooled_from_bridge is not None:
            pooled = F.normalize(pooled_from_bridge, dim=-1)
        else:
            mean_pool = token_embeddings.mean(dim=1)
            gated = torch.sigmoid(self.pool_gate(self.pool_norm(mean_pool))) * mean_pool
            pooled = self.out_proj(gated)
            pooled = F.normalize(pooled, dim=-1)

        return pooled, token_embeddings

    def temperature(self) -> torch.Tensor:
        return torch.clamp(torch.exp(self.log_tau), 0.03, 0.2)


@ProjectRegistry.register(ProjectName.SIGLIP_PHASE1)
class SiglipPhase1Project(BaseProject):
    """Project orchestrating SigLIP Phase-1 alignment training."""

    def __init__(self, config: SiglipPhase1Config, wandb_wrapper) -> None:
        super().__init__(config, wandb_wrapper)
        self.config = config

    def _apply_decoder_unfreeze_policy(self, decoder: MedGemmaDecoder) -> None:
        num_layers = getattr(self.config, "llm_unfreeze_last_n_layers", None)
        extra_patterns = list(getattr(self.config, "llm_unfreeze_additional_param_patterns", []) or [])
        include_lm_head = bool(getattr(self.config, "llm_unfreeze_lm_head", False))

        if (not num_layers or num_layers <= 0) and not extra_patterns and not include_lm_head:
            return

        llm_model = getattr(decoder, "llm_model", None)
        if llm_model is None:
            print("[SigLIP] Decoder does not expose an llm_model; unable to apply unfreeze policy.")
            return

        layer_pattern = re.compile(r"\.(layers|h|block|blocks)\.(\d+)\.")
        layer_params: dict[int, list[nn.Parameter]] = {}

        for name, param in llm_model.named_parameters():
            match = layer_pattern.search(name)
            if match:
                layer_idx = int(match.group(2))
                layer_params.setdefault(layer_idx, []).append(param)

        target_layers: list[int] = []
        if num_layers and num_layers > 0:
            if layer_params:
                max_idx = max(layer_params.keys())
                target_layers = [idx for idx in layer_params if idx >= max_idx - num_layers + 1]
                for idx in target_layers:
                    for param in layer_params[idx]:
                        param.requires_grad = True
                target_layers.sort()
                print(f"[SigLIP] Unfroze last {len(target_layers)} LLM transformer layers: {target_layers}")
            else:
                print("[SigLIP] Warning: could not identify transformer blocks in llm_model; "
                      "layer-specific unfreeze skipped.")

        if include_lm_head:
            lm_head = getattr(llm_model, "lm_head", None)
            if lm_head is None and hasattr(llm_model, "get_output_embeddings"):
                lm_head = llm_model.get_output_embeddings()
            if lm_head is not None:
                for param in lm_head.parameters():
                    param.requires_grad = True
                print("[SigLIP] Unfroze LLM output head for fine-tuning.")
            else:
                print("[SigLIP] Warning: LLM model lacks an lm_head module; cannot unfreeze head.")

        if extra_patterns:
            matched = False
            for name, param in llm_model.named_parameters():
                if any(pattern in name for pattern in extra_patterns):
                    param.requires_grad = True
                    matched = True
            if matched:
                print(f"[SigLIP] Unfroze additional LLM parameters matching patterns: {extra_patterns}")
            else:
                print(f"[SigLIP] Warning: no LLM parameters matched extra unfreeze patterns {extra_patterns}.")

    def run(self) -> None:
        super().run()

    # BaseProject interface -------------------------------------------------
    def _setup_inference_objects(self) -> Dict[str, Any]:  # pragma: no cover - not used
        raise NotImplementedError("Inference mode is not implemented for SigLIP Phase-1")

    def _setup_extraction_objects(self) -> Dict[str, Any]:  # pragma: no cover - not used
        raise NotImplementedError("Embedding extraction is not implemented for SigLIP Phase-1")

    def _setup_validation_objects(self) -> Dict[str, Any]:  # pragma: no cover - SigLIP validates in-loop during training
        raise NotImplementedError("Standalone validation mode is not implemented for SigLIP Phase-1 (validation runs in-loop during train)")

    def _setup_test_objects(self) -> Dict[str, Any]:  # pragma: no cover - SigLIP validates in-loop during training
        raise NotImplementedError("Standalone test mode is not implemented for SigLIP Phase-1")

    def _setup_training_objects(self) -> Dict[str, Any]:
        device = torch.device("cuda", self.config.device) if torch.cuda.is_available() else torch.device("cpu")

        output_dir = self.config.output_dir
        log_once(f"Initial output directory: {output_dir}", rank=int(self.config.device))
        if DistributedUtils.dist.is_available() and DistributedUtils.dist.is_initialized():
            dir_list = [output_dir]
            DistributedUtils.dist.broadcast_object_list(dir_list, src=0)
            output_dir = dir_list[0]
            self.config.output_dir = output_dir
            log_once(f"Broadcasted output directory: {output_dir}", rank=int(self.config.device))

        if output_dir:
            ensure_dir(output_dir)
        else:
            fallback = os.path.join(
                self.config.base_checkpoint_path,
                self.config.pipeline_project,
                "tmp_runs",
                f"rank{self.config.device}"
            )
            ensure_dir(fallback)
            self.config.output_dir = fallback
            output_dir = fallback
            log_once(f"Using fallback output directory: {fallback}", rank=int(self.config.device))

        train_dataset = SiglipDataset(
            mapping_csv=self.config.mapping_csv,
            text_bank_csv=self.config.text_bank_csv,
            split=self.config.mapping_split,
            expected_waveform_length=self.config.waveform_length,
            num_leads=self.config.num_leads,
            normalize_waveforms=self.config.normalize_waveforms,
            lead_stats=self.config.lead_stats,
            qa_positive_weight_multiplier=self.config.qa_positive_weight_multiplier,
            signal_path_column="ecg_id",
            shuffle=True,
            shuffle_seed=self.config.seed,
        )

        alpha_map, pos_counts = build_alpha_map_from_effective_num(
            train_dataset.samples,
            train_dataset.all_text_ids,
            beta=float(self.config.alpha_beta),
            clip_min=float(self.config.alpha_clip_min),
            clip_max=float(self.config.alpha_clip_max),
        )

        if self.config.tail_enable:
            tails = select_tail_class_ids(
                pos_counts,
                train_dataset.all_text_ids,
                mode=self.config.tail_select_mode,
                top_n=int(self.config.tail_top_n),
                min_positives=int(self.config.tail_min_positives),
                include_regex=getattr(self.config, "tail_include_regex", None),
                exclude_regex=getattr(self.config, "tail_exclude_regex", None),
            )
        else:
            tails = []
        self.config.tail_class_ids = tails

        tail_alpha_boost = float(getattr(self.config, "tail_alpha_boost", 1.0) or 1.0)
        if tails and abs(tail_alpha_boost - 1.0) > 1e-6:
            for tid in tails:
                if tid in alpha_map:
                    alpha_map[tid] = float(alpha_map[tid] * tail_alpha_boost)
        self.config.class_pos_weight_map = alpha_map

        artifacts_root = self.config.output_dir or "."
        artifacts_dir = os.path.join(artifacts_root, "artifacts")
        ensure_dir(artifacts_dir)
        alpha_csv_path = os.path.join(artifacts_dir, self.config.alpha_csv_name)

        with open(alpha_csv_path, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["text_id", "text", "pos_count", "alpha", "is_tail"])
            for text_id in train_dataset.all_text_ids:
                writer.writerow(
                    [
                        text_id,
                        train_dataset.text_lookup.get(text_id, ""),
                        pos_counts.get(text_id, 0),
                        alpha_map.get(text_id, 1.0),
                        1 if text_id in tails else 0,
                    ]
                )

        if getattr(self.config, "is_ref_device", True):
            alphas = list(alpha_map.values())
            alphas_sorted = sorted(alphas) if alphas else [1.0]
            print(
                f"[SigLIP] α map built: count={len(alphas)} "
                f"min={alphas_sorted[0]:.3f} med={alphas_sorted[len(alphas_sorted) // 2]:.3f} "
                f"max={alphas_sorted[-1]:.3f}. "
                f"CSV -> {alpha_csv_path}"
            )
            if tails:
                print(
                    "[SigLIP] Tail selection: "
                    f"{len(tails)} labels (mode={self.config.tail_select_mode}, "
                    f"top_n={self.config.tail_top_n}, min_pos={self.config.tail_min_positives})"
                )

        if (
            self.wandb_wrapper
            and self.wandb_wrapper.is_initialized()
            and getattr(self.config, "is_ref_device", True)
        ):
            import wandb

            negs_per_ecg = getattr(self.config, "negatives_per_ecg", None)
            if negs_per_ecg is None:
                negs_per_ecg = getattr(self.config, "implicit_negatives_per_batch", None)
            if negs_per_ecg is None:
                negs_per_ecg = getattr(self.config, "k_impl", None)
            negs_per_ecg = int(negs_per_ecg) if negs_per_ecg is not None else None

            self.wandb_wrapper.log(
                {
                    "cfg/focal_infonce": bool(self.config.focal_infonce),
                    "cfg/focal_gamma_pos": float(self.config.focal_gamma_pos),
                    "cfg/focal_gamma_neg": float(self.config.focal_gamma_neg),
                    "cfg/focal_alpha_default": float(self.config.focal_alpha_default),
                    "cfg/alpha_beta": float(self.config.alpha_beta),
                    "cfg/alpha_clip_min": float(self.config.alpha_clip_min),
                    "cfg/alpha_clip_max": float(self.config.alpha_clip_max),
                    "cfg/negatives_per_ecg": negs_per_ecg if negs_per_ecg is not None else -1,
                    "cfg/tail_enable": bool(self.config.tail_enable),
                    "cfg/tail_select_mode": str(self.config.tail_select_mode),
                    "cfg/tail_top_n": int(self.config.tail_top_n),
                    "cfg/tail_min_positives": int(self.config.tail_min_positives),
                    "cfg/tail_alpha_boost": float(getattr(self.config, "tail_alpha_boost", 1.0) or 1.0),
                    "cfg/tail_count": float(len(tails)),
                }
            )

            alphas_np = np.array([alpha_map[text_id] for text_id in train_dataset.all_text_ids], dtype=float)
            tail_alphas = np.array([alpha_map[tid] for tid in tails if tid in alpha_map], dtype=float)
            alpha_payload = {
                "alpha/histogram": wandb.Histogram(alphas_np),
                "alpha/tail_alpha_boost": float(getattr(self.config, "tail_alpha_boost", 1.0) or 1.0),
                "alpha/tail_alpha_median": float(np.median(tail_alphas)) if tail_alphas.size else 0.0,
            }
            self.wandb_wrapper.log(alpha_payload)

            preview_ids = sorted(
                train_dataset.all_text_ids,
                key=lambda text_id: pos_counts.get(text_id, 0),
            )[:200]
            table = wandb.Table(columns=["text_id", "text", "pos_count", "alpha", "is_tail"])
            for text_id in preview_ids:
                table.add_data(
                    text_id,
                    train_dataset.text_lookup.get(text_id, ""),
                    int(pos_counts.get(text_id, 0)),
                    float(alpha_map.get(text_id, 1.0)),
                    int(text_id in tails),
                )
            self.wandb_wrapper.log({"alpha/table_preview": table})

        implicit_negatives_per_batch = getattr(self.config, "implicit_negatives_per_batch", None)
        if implicit_negatives_per_batch is None:
            implicit_negatives_per_batch = getattr(self.config, "k_impl", None)
        if implicit_negatives_per_batch is None:
            implicit_negatives_per_batch = 128
        implicit_negatives_per_batch = max(0, int(implicit_negatives_per_batch))
        implicit_negatives_per_row = getattr(self.config, "implicit_negatives_per_row", None)
        if implicit_negatives_per_row is not None:
            implicit_negatives_per_row = max(0, int(implicit_negatives_per_row))
        negatives_mode_raw = getattr(self.config, "negatives_mode", "per_batch")
        negatives_mode = str(negatives_mode_raw or "per_batch").lower()
        if negatives_mode not in {"per_batch", "per_row"}:
            log_once(
                f"Invalid negatives_mode '{negatives_mode_raw}' supplied; defaulting to per_batch.",
                rank=int(self.config.device),
            )
            negatives_mode = "per_batch"

        max_hardneg_per_group = 3
        if int(getattr(self.config, "max_hardneg_per_group", 3)) != max_hardneg_per_group:
            log_once(
                "Overriding max_hardneg_per_group to 3 for SigLIP Phase-1.",
                rank=int(self.config.device),
            )
        log_once(
            f"Implicit negatives configured: mode={negatives_mode}, per_batch={implicit_negatives_per_batch}, "
            f"per_row={implicit_negatives_per_row}",
            rank=int(self.config.device),
        )
        self.config.negatives_mode = negatives_mode
        self.config.implicit_negatives_per_batch = implicit_negatives_per_batch
        self.config.implicit_negatives_per_row = implicit_negatives_per_row
        self.config.max_hardneg_per_group = max_hardneg_per_group
        collate_fn = SiglipBatchCollatorInfoNCE(
            text_lookup=train_dataset.text_lookup,
            all_text_ids=train_dataset.all_text_ids,
            implicit_negatives_per_batch=implicit_negatives_per_batch,
            implicit_negatives_per_row=implicit_negatives_per_row,
            negatives_mode=negatives_mode,
            implicit_weight=1.0,
            seed=self.config.seed,
            max_hardneg_per_group=max_hardneg_per_group,
        )

        if (
            self.wandb_wrapper
            and self.wandb_wrapper.is_initialized()
            and getattr(self.config, "is_ref_device", True)
        ):
            self.wandb_wrapper.log(
                {
                    "cfg/negatives_mode": str(negatives_mode),
                    "cfg/implicit_negatives_per_batch": float(implicit_negatives_per_batch),
                    "cfg/implicit_negatives_per_row": float(implicit_negatives_per_row or 0),
                }
            )

        train_loader = get_siglip_dataloader(
            dataset=train_dataset,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            num_replicas=self.config.world_size,
            rank=self.config.device,
            shuffle=True,
            collate_fn=collate_fn,
        )
        log_once(
            f"Loaded train dataset with {len(train_dataset)} ECGs, text ids={len(train_dataset.all_text_ids)}",
            rank=int(self.config.device),
        )

        validation_loader: Optional[torch.utils.data.DataLoader] = None
        if self.config.validation_mapping_csv:
            val_dataset = SiglipDataset(
                mapping_csv=self.config.validation_mapping_csv,
                text_bank_csv=self.config.validation_text_bank_csv or self.config.text_bank_csv,
                split=self.config.validation_mapping_split or self.config.mapping_split,
                expected_waveform_length=self.config.waveform_length,
                num_leads=self.config.num_leads,
                normalize_waveforms=self.config.normalize_waveforms,
                lead_stats=self.config.lead_stats,
                qa_positive_weight_multiplier=self.config.qa_positive_weight_multiplier,
                signal_path_column="ecg_id",
                shuffle=False,
                shuffle_seed=self.config.seed,
            )
            val_collate = SiglipBatchCollatorInfoNCE(
                text_lookup=val_dataset.text_lookup,
                all_text_ids=val_dataset.all_text_ids,
                implicit_negatives_per_batch=implicit_negatives_per_batch,
                implicit_negatives_per_row=implicit_negatives_per_row,
                negatives_mode=negatives_mode,
                implicit_weight=1.0,
                seed=self.config.seed,
                max_hardneg_per_group=max_hardneg_per_group,
            )
            validation_loader = get_siglip_dataloader(
                dataset=val_dataset,
                batch_size=self.config.batch_size,
                num_workers=self.config.num_workers,
                num_replicas=self.config.world_size,
                rank=self.config.device,
                shuffle=False,
                collate_fn=val_collate,
            )
            log_once(
                f"Loaded validation dataset with {len(val_dataset)} ECGs",
                rank=int(self.config.device),
            )

        if not self.config.text_embedding_cache_path:
            artifacts_root = output_dir
            artifacts_dir = os.path.join(artifacts_root, "artifacts") if artifacts_root else None
            if artifacts_dir:
                ensure_dir(artifacts_dir)
                self.config.text_embedding_cache_path = os.path.join(artifacts_dir, "text_embeddings.pt")
                log_once(f"Embedding cache path set to {self.config.text_embedding_cache_path}", rank=int(self.config.device))

        if self.config.checkpoint_dir:
            ensure_dir(self.config.checkpoint_dir)
        elif output_dir:
            ckpt_dir = os.path.join(output_dir, "checkpoints")
            ensure_dir(ckpt_dir)
            self.config.checkpoint_dir = ckpt_dir
        log_once(f"Checkpoint directory: {self.config.checkpoint_dir}", rank=int(self.config.device))

        model_name_cfg = (
            getattr(self.config, "text_encoder_model_name", None)
            or getattr(self.config, "medgemma_model_name", None)
            or "google/medgemma-4b-it"
        )
        text_embeddings, text_id_to_idx = self._prepare_text_embeddings(
            text_bank_path=self.config.text_bank_csv,
            cache_path=self.config.text_embedding_cache_path,
            model_name=model_name_cfg,
        )

        if text_embeddings.shape[1] != self.config.bridge_hidden_size:
            raise ValueError(
                f"bridge_hidden_size={self.config.bridge_hidden_size} does not match text embedding dim {text_embeddings.shape[1]}"
            )

        # Phase A-v2: per-ECG free-text report contrastive target. Build the
        # shared text embedder only when requested; "bank" mode is untouched.
        report_text_embedder: Optional[ReportTextEmbedder] = None
        contrastive_text_mode = str(
            getattr(self.config, "contrastive_text_mode", "bank") or "bank"
        ).lower()
        if contrastive_text_mode == "report":
            report_text_embedder = self._build_report_text_embedder(
                model_name=model_name_cfg,
                device=device,
            )

        encoder, quantizer, tokenizer_cfg = self._load_tokenizer_components(device=device)
        dummy = torch.zeros(1, self.config.num_leads, self.config.waveform_length, device=device)
        with torch.no_grad():
            encoder_features = encoder(dummy)
            bridge_features = encoder_features
            bridge_codes = None
            use_quantized_inputs = bool(
                quantizer is not None and self.config.use_quantized_bridge_inputs
            )
            total_codebooks = None
            if use_quantized_inputs:
                quantized_outputs = quantizer(encoder_features, return_all_codes=True)
                if len(quantized_outputs) == 4:
                    quantized, indices, _, all_codes = quantized_outputs
                    total_codebooks = all_codes.size(0)
                else:
                    quantized, indices, _ = quantized_outputs
                    all_codes = None

                kept = getattr(self.config, "num_codebooks_kept", None)
                if total_codebooks is None:
                    total_codebooks = indices.size(-1)
                if kept is None or kept <= 0 or kept > total_codebooks:
                    kept = total_codebooks
                requested_offset = getattr(self.config, "codebook_offset", 0)
                max_valid_offset = max(total_codebooks - kept, 0)
                if requested_offset < 0:
                    offset = max_valid_offset
                else:
                    upper_bound = max(total_codebooks - 1, 0)
                    offset = max(0, min(requested_offset, upper_bound))
                    if offset > max_valid_offset:
                        offset = max_valid_offset
                self.config.codebook_offset = int(offset)
                end = min(offset + kept, total_codebooks)

                bridge_codes = indices[..., offset:end].long()
                if bridge_codes.size(-1) == 1:
                    bridge_codes = bridge_codes.squeeze(-1)

                if all_codes is not None:
                    selected = all_codes[offset:end]  # [kept, batch, seq, dim]
                    selected = selected.permute(1, 0, 3, 2).contiguous()  # [batch, kept, dim, seq]
                    bridge_features = selected.view(
                        selected.size(0),
                        selected.size(1) * selected.size(2),
                        selected.size(3),
                    )
                    self.config.bridge_num_codebooks = int(selected.size(1))
                    self.config.bridge_code_dim = int(selected.size(2))
                else:
                    if quantized.dim() != 3:
                        raise ValueError(
                            f"Quantized features must be [batch, seq, dim]; received {quantized.shape}"
                        )
                    bridge_features = quantized.permute(0, 2, 1).contiguous()
                    self.config.bridge_num_codebooks = 1
                    self.config.bridge_code_dim = int(bridge_features.size(1))
            else:
                if bridge_features.dim() != 3:
                    raise ValueError(
                        f"Encoder features must be [batch, seq, dim]; received {bridge_features.shape}"
                    )
                bridge_features = bridge_features.permute(0, 2, 1).contiguous()
                self.config.bridge_num_codebooks = 1
                self.config.bridge_code_dim = int(bridge_features.size(1))
            self.config.bridge_input_channels = int(bridge_features.size(1))
            if (
                getattr(self.config, "is_ref_device", True)
                and self.wandb_wrapper
                and self.wandb_wrapper.is_initialized()
            ):
                self.wandb_wrapper.log(
                    {
                        "cfg/effective_num_codebooks": int(getattr(self.config, "bridge_num_codebooks", -1)),
                        "cfg/bridge_input_channels": int(self.config.bridge_input_channels),
                    }
                )
            if getattr(self.config, "is_ref_device", True):
                kept = getattr(self.config, "num_codebooks_kept", None)
                offset = getattr(self.config, "codebook_offset", 0)
                total = total_codebooks
                print(
                    "[SigLIP] Bridge input sample: features"
                    f" {tuple(bridge_features.shape)} | codes {None if bridge_codes is None else tuple(bridge_codes.shape)}"
                    f" | codebooks kept={kept if kept is not None else 'all'}"
                    f" (offset={offset}, total={total})"
                )

        if bridge_features.dim() != 3:
            raise ValueError(
                "Bridge features must have shape [batch, channels, length]; "
                f"received {bridge_features.shape}"
            )

        seq_len = bridge_features.shape[2]
        if seq_len > self.config.bridge_max_seq_len:
            raise ValueError(
                f"Bridge sequence length {seq_len} exceeds bridge_max_seq_len={self.config.bridge_max_seq_len}"
            )

        bridge_wrapper, raw_bridge = self._build_bridge(
            bridge_features=bridge_features,
            bridge_codes=bridge_codes,
            device=device,
            configured_name=self.config.bridge_name,
            tokenizer_config=tokenizer_cfg,
        )

        decoder = self._build_decoder(raw_bridge=raw_bridge, device=device)
        decoder.bridge = raw_bridge
        decoder.num_ecg_tokens = getattr(raw_bridge, "num_tokens", decoder.num_ecg_tokens)

        optimizer_settings = self.config.optimizer
        base_lr = float(self.config.lr)
        weight_decay = float(self.config.weight_decay)
        component_scales = getattr(self.config, "component_lr_scales", None)
        optimizer_type = "AdamW"

        if isinstance(optimizer_settings, dict):
            optimizer_type = str(optimizer_settings.get("type", optimizer_type))
            base_lr = float(optimizer_settings.get("lr", base_lr))
            weight_decay = float(optimizer_settings.get("weight_decay", weight_decay))
            component_scales = optimizer_settings.get("component_lr_scales", component_scales)
        elif isinstance(optimizer_settings, str):
            optimizer_type = optimizer_settings or optimizer_type
        elif optimizer_settings is not None:
            optimizer_type = str(optimizer_settings)

        optimizer_cls = getattr(optim, optimizer_type, None)
        if optimizer_cls is None:
            raise ValueError(f"Unsupported optimizer type '{optimizer_type}' for SigLIP Phase-1 project.")

        param_groups = []
        assigned_param_ids: set[int] = set()

        def _add_group(params_iterable: Iterable[nn.Parameter], scale: float, group_name: str | None) -> None:
            params: list[nn.Parameter] = []
            for param in params_iterable:
                if not isinstance(param, nn.Parameter):
                    continue
                if not param.requires_grad:
                    continue
                pid = id(param)
                if pid in assigned_param_ids:
                    continue
                assigned_param_ids.add(pid)
                params.append(param)
            if not params:
                return
            param_groups.append(
                {
                    "params": params,
                    "lr": base_lr * float(scale),
                    "weight_decay": weight_decay,
                    "name": group_name or "",
                }
            )

        if component_scales:
            resolved_scales = dict(component_scales)
            recognized: set[str] = set()

            shared_scale = resolved_scales.get("shared_bridge")
            if shared_scale is not None:
                recognized.add("shared_bridge")
                _add_group(raw_bridge.parameters(), shared_scale, "shared_bridge")

            projector_scale = resolved_scales.get("siglip_projector")
            if projector_scale is not None:
                recognized.add("siglip_projector")
                wrapper_params = [
                    param
                    for name, param in bridge_wrapper.named_parameters()
                    if not name.startswith("bridge.")
                ]
                _add_group(wrapper_params, projector_scale, "siglip_projector")

            decoder_scale = resolved_scales.get("llm_decoder")
            if decoder_scale is not None:
                recognized.add("llm_decoder")
                if decoder is not None:
                    _add_group(decoder.parameters(), decoder_scale, "llm_decoder")

            unknown_components = set(resolved_scales.keys()) - recognized
            if unknown_components:
                raise ValueError(
                    f"Unknown component(s) specified in component_lr_scales: {sorted(unknown_components)}"
                )

        remaining_params: list[nn.Parameter] = []
        for module in (bridge_wrapper, decoder):
            if module is None:
                continue
            for param in module.parameters():
                if not param.requires_grad:
                    continue
                if id(param) in assigned_param_ids:
                    continue
                assigned_param_ids.add(id(param))
                remaining_params.append(param)

        if param_groups and remaining_params:
            _add_group(remaining_params, 1.0, "default")
        elif not param_groups:
            param_groups = [
                {
                    "params": [param for param in bridge_wrapper.parameters() if param.requires_grad],
                    "lr": base_lr,
                    "weight_decay": weight_decay,
                    "name": "default",
                }
            ]

        # Phase A: add the encoder as its own (low-LR) param group so gradients
        # from the contrastive loss reach the raw signal encoder. Gated on the
        # train_encoder flag; default runs are unchanged.
        if bool(getattr(self.config, "train_encoder", False)):
            encoder_lr = getattr(self.config, "encoder_lr", None)
            if encoder_lr is None or float(encoder_lr) <= 0.0:
                encoder_lr = base_lr * 0.1
            encoder_lr = float(encoder_lr)
            encoder_params = [
                param
                for param in encoder.parameters()
                if param.requires_grad and id(param) not in assigned_param_ids
            ]
            for param in encoder_params:
                assigned_param_ids.add(id(param))
            if encoder_params:
                param_groups.append(
                    {
                        "params": encoder_params,
                        "lr": encoder_lr,
                        "weight_decay": weight_decay,
                        "name": "encoder",
                    }
                )
                if getattr(self.config, "is_ref_device", True):
                    n_enc = sum(p.numel() for p in encoder_params)
                    print(
                        f"[SigLIP] train_encoder=True -> added encoder param group "
                        f"({len(encoder_params)} tensors, {n_enc/1e6:.2f}M params) at lr={encoder_lr:.2e}."
                    )
            else:
                print("[SigLIP] WARNING: train_encoder=True but no trainable encoder params found.")

        optimizer = optimizer_cls(param_groups, lr=base_lr, weight_decay=weight_decay)

        runner_kwargs = {
            "encoder": encoder,
            "quantizer": quantizer,
            "bridge": bridge_wrapper,
            "decoder": decoder,
            "train_dataloader": train_loader,
            "optimizer": optimizer,
            "config": self.config,
            "text_embeddings": text_embeddings,
            "text_id_to_idx": text_id_to_idx,
            "text_lookup": train_dataset.text_lookup,
            "validation_dataloader": validation_loader,
            "use_quantized_inputs": use_quantized_inputs,
            "tokenizer_config": tokenizer_cfg,
            "report_text_embedder": report_text_embedder,
        }
        return runner_kwargs

    # ------------------------------------------------------------------
    def _load_tokenizer_components(
        self,
        device: torch.device,
    ) -> tuple[nn.Module, Optional[nn.Module], Dict[str, Any]]:
        checkpoint_path = self.config.pretrained_encoder_checkpoint
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            raise FileNotFoundError(
                f"Pretrained encoder checkpoint not found: {checkpoint_path}. Phase-1 requires a pretrained tokenizer."
            )

        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
        ckpt_config = checkpoint.get("config")

        encoder_name = self._resolve_config_value("encoder_name", ckpt_config)
        if encoder_name is None:
            raise ValueError(
                "Unable to determine encoder_name from checkpoint or config. "
                "Set `encoder_name` or ensure the checkpoint stores it."
            )

        quantizer_name = self._resolve_config_value("quantizer_name", ckpt_config)
        num_quantizers = self._resolve_config_value("num_quantizers", ckpt_config)
        codebook_size = self._resolve_config_value("codebook_size", ckpt_config)

        if num_quantizers is not None:
            self.config.num_quantizers = num_quantizers
        if codebook_size is not None:
            self.config.codebook_size = codebook_size

        print("[SigLIP] Loaded checkpoint:", checkpoint_path)
        if ckpt_config is not None:
            print("[SigLIP] Pretrained tokenizer config:", ckpt_config)
        else:
            print("[SigLIP] Pretrained tokenizer config: <missing>")

        train_encoder = bool(getattr(self.config, "train_encoder", False))
        encoder_class = ModelRegistry.get(encoder_name)
        encoder: nn.Module = encoder_class().to(device)
        if train_encoder:
            encoder.train()
        else:
            encoder.eval()

        encoder_state = self._extract_module_state(state_dict, "encoder")
        if not encoder_state:
            raise ValueError("Checkpoint does not contain encoder weights (keys missing 'encoder').")

        missing_encoder = encoder.load_state_dict(encoder_state, strict=False)
        if missing_encoder.missing_keys or missing_encoder.unexpected_keys:
            raise ValueError(
                "Encoder state_dict mismatch. Missing keys: "
                f"{missing_encoder.missing_keys}; Unexpected keys: {missing_encoder.unexpected_keys}."
            )
        print("[SigLIP] Encoder weights loaded successfully.")
        if train_encoder:
            for param in encoder.parameters():
                param.requires_grad = True
            print("[SigLIP] train_encoder=True -> encoder is TRAINABLE (warm-started from checkpoint).")
        else:
            for param in encoder.parameters():
                param.requires_grad = False

        quantizer: Optional[nn.Module] = None
        quantizer_state = self._extract_module_state(state_dict, "quantizer")
        if quantizer_state and quantizer_name:
            if num_quantizers is None or codebook_size is None:
                raise ValueError(
                    "Quantizer weights found but num_quantizers/codebook_size are missing in config or checkpoint."
                )
            quantizer_class = ModelRegistry.get(str(quantizer_name))
            quantizer = quantizer_class(
                num_quantizers=int(num_quantizers),
                codebook_size=int(codebook_size),
            ).to(device)
            quantizer.eval()
            missing_quantizer = quantizer.load_state_dict(quantizer_state, strict=False)
            if missing_quantizer.missing_keys or missing_quantizer.unexpected_keys:
                raise ValueError(
                    "Quantizer state_dict mismatch. Missing keys: "
                    f"{missing_quantizer.missing_keys}; Unexpected keys: {missing_quantizer.unexpected_keys}."
                )
            print("[SigLIP] Quantizer weights loaded successfully.")
            for param in quantizer.parameters():
                param.requires_grad = False
        elif quantizer_name:
            print(
                "[SigLIP] Quantizer name provided but weights missing in checkpoint; proceeding without quantizer."
            )

        metadata = {
            "encoder_name": encoder_name,
            "quantizer_name": quantizer_name,
            "num_quantizers": num_quantizers,
            "codebook_size": codebook_size,
        }
        return encoder, quantizer, metadata

    def _resolve_config_value(self, key: str, ckpt_config: Any, default: Any = None) -> Any:
        """Resolve config precedence: explicit config overrides checkpoint config."""
        config_value = getattr(self.config, key, None)
        if config_value is not None:
            return config_value
        if ckpt_config is None:
            return default
        if isinstance(ckpt_config, dict):
            return ckpt_config.get(key, default)
        return getattr(ckpt_config, key, default)

    def _extract_module_state(self, state_dict: Dict[str, torch.Tensor], module_key: str) -> Dict[str, torch.Tensor]:
        """Extract submodule weights regardless of DDP/module prefixes."""
        extracted: Dict[str, torch.Tensor] = {}
        token = f"{module_key}."
        for key, value in state_dict.items():
            if token in key:
                sub_key = key.split(token, 1)[1]
                extracted[sub_key] = value
        return extracted

    def _build_bridge(
        self,
        bridge_features: torch.Tensor,
        bridge_codes: Optional[torch.Tensor],
        device: torch.device,
        configured_name: Optional[str],
        tokenizer_config: Optional[Dict[str, Any]],
    ) -> tuple[SiglipBridgeWrapper, nn.Module]:
        bridge_name = self._resolve_bridge_name(configured_name)
        bridge_class = ModelRegistry.get(bridge_name)
        uses_codes = bool(getattr(bridge_class, "uses_codes", False))

        if uses_codes:
            if bridge_codes is None:
                raise ValueError(
                    f"Bridge '{bridge_name}' expects discrete codes but none were provided."
                )
            codes = bridge_codes
            if codes.dim() == 2:
                seq_len = codes.shape[1]
                num_codebooks = 1
            elif codes.dim() == 3:
                seq_len = codes.shape[1]
                num_codebooks = codes.shape[2]
            else:
                raise ValueError(f"Unexpected code tensor shape {codes.shape} for bridge '{bridge_name}'")

            vocab_size = getattr(self.config, "codebook_size", None)
            if vocab_size is None:
                raise ValueError(
                    "codebook_size must be set in config or checkpoint metadata to use ECGCodeBridge."
                )

            if bridge_name == "ECGQFormerBridge" or bridge_name == BridgeName.LLAMA32_ECG_QFORMER_BRIDGE.value:
                # BLIP-2 style: fixed number of query tokens (default 32), configurable via `num_query_tokens`.
                cfg_q = getattr(self.config, "num_query_tokens", None)
                num_query_tokens = int(cfg_q) if cfg_q is not None else 32
                kwargs = dict(
                    vocab_size=int(vocab_size),
                    num_codebooks=int(num_codebooks),
                    d_mid=self.config.bridge_hidden_size,
                    d_llm=self.config.bridge_hidden_size,
                    d_txt=self.config.bridge_hidden_size,
                    num_steps=int(seq_len),
                    num_query_tokens=num_query_tokens,
                    num_layers=int(self.config.bridge_num_layers),
                    num_heads=int(self.config.bridge_num_heads),
                    dropout=float(self.config.bridge_dropout),
                    num_special_tokens=int(getattr(self.config, "bridge_num_special_tokens", 0)),
                    bias_last_codebook=float(getattr(self.config, "bridge_bias_last_codebook", 0.5)),
                    codebook_dropout=float(getattr(self.config, "bridge_codebook_dropout", 0.0)),
                    mix_strategy=str(getattr(self.config, "bridge_mix_strategy", "softmax") or "softmax"),
                )
            else:
                kwargs = dict(
                    vocab_size=int(vocab_size),
                    d_mid=self.config.bridge_hidden_size,
                    d_model=self.config.bridge_hidden_size,
                    num_output_tokens=seq_len,
                    num_heads=self.config.bridge_num_heads,
                    num_special_tokens=int(getattr(self.config, "bridge_num_special_tokens", 0)),
                    dropout=self.config.bridge_dropout,
                    num_codebooks=int(num_codebooks),
                )
        else:
            if bridge_features.dim() != 3:
                raise ValueError(
                    f"Bridge features must have shape [batch, channels, length]; received {bridge_features.shape}"
                )
            seq_len = bridge_features.shape[2]
            feature_dim = bridge_features.shape[1]
            kwargs = self._bridge_kwargs_from_signature(bridge_class, seq_len, feature_dim)

        raw_bridge = bridge_class(**kwargs).to(device)
        wrapper = SiglipBridgeWrapper(
            bridge=raw_bridge,
            hidden_size=self.config.bridge_hidden_size,
            temperature_init=self.config.temperature_init,
        ).to(device)
        target_std = getattr(self.config, "target_embedding_std", None)
        if target_std is not None:
            wrapper.target_embedding_std = float(target_std)
        return wrapper, raw_bridge

    def _resolve_bridge_name(self, configured_name: Optional[str]) -> str:
        """Map legacy bridge names onto the standard registry."""
        if configured_name:
            try:
                ModelRegistry.get(configured_name)
                return configured_name
            except ValueError as exc:
                raise ValueError(
                    f"Bridge '{configured_name}' is not registered."
                ) from exc

        # Default bridge when none specified
        ModelRegistry.get("SequenceTokenBridge")
        return "SequenceTokenBridge"

    def _bridge_kwargs_from_signature(
        self,
        bridge_class: type[nn.Module],
        seq_len: int,
        feature_dim: int,
    ) -> Dict[str, Any]:
        signature = inspect.signature(bridge_class)
        kwargs: Dict[str, Any] = {}
        params = signature.parameters

        if "input_shape" in params:
            kwargs["input_shape"] = (seq_len, feature_dim)
        if "input_dim" in params:
            kwargs["input_dim"] = feature_dim
        if "input_channels" in params:
            kwargs["input_channels"] = feature_dim
        if "d_model" in params:
            kwargs["d_model"] = self.config.bridge_hidden_size
        if "num_tokens" in params:
            kwargs["num_tokens"] = seq_len
        if "output_size" in params:
            kwargs["output_size"] = self.config.bridge_hidden_size
        if "dropout" in params:
            kwargs["dropout"] = self.config.bridge_dropout
        if "use_cross_attention" in params:
            use_cross_attention = getattr(self.config, "bridge_use_cross_attention", None)
            kwargs["use_cross_attention"] = True if use_cross_attention is None else bool(use_cross_attention)
        if "num_attention_heads" in params:
            heads = getattr(self.config, "bridge_num_heads", None)
            if heads is None:
                raise ValueError("bridge_num_heads must be set for the selected bridge")
            kwargs["num_attention_heads"] = int(heads)
        if "intermediate_dim" in params:
            intermediate = getattr(self.config, "bridge_intermediate_dim", None)
            if intermediate is None:
                intermediate = self.config.bridge_hidden_size // 2
            kwargs["intermediate_dim"] = int(intermediate)
        if "num_codebooks" in params:
            num_codebooks = getattr(self.config, "bridge_num_codebooks", None)
            if num_codebooks is None:
                num_codebooks = getattr(self.config, "num_codebooks_kept", None) or 1
            kwargs["num_codebooks"] = int(num_codebooks)
        if "code_dim" in params:
            code_dim = getattr(self.config, "bridge_code_dim", None)
            if code_dim is not None:
                kwargs["code_dim"] = int(code_dim)
        if "target_std" in params:
            target_std = getattr(self.config, "target_embedding_std", None)
            if target_std is not None:
                kwargs["target_std"] = float(target_std)

        return kwargs

    def _build_decoder(
        self,
        raw_bridge: nn.Module,
        device: torch.device,
    ) -> MedGemmaDecoder:
        dtype_str = str(getattr(self.config, "dtype", "bf16") or "bf16").lower()
        if dtype_str in {"bf16", "bfloat16"}:
            torch_dtype = torch.bfloat16
        elif dtype_str in {"fp16", "float16", "half"}:
            torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        num_visual_tokens = getattr(raw_bridge, "num_tokens", None)
        if num_visual_tokens is None:
            raise ValueError("Shared bridge does not expose `num_tokens`; required for decoder integration.")

        model_name_cfg = (
            getattr(self.config, "text_encoder_model_name", None)
            or getattr(self.config, "medgemma_model_name", None)
            or "google/medgemma-4b-it"
        )
        decoder = MedGemmaDecoder(
            huggingface_model_name=model_name_cfg,
            llm_input_embedding_size=self.config.bridge_hidden_size,
            bridge_name=self.config.bridge_name,
            quantized_feature_shape=(num_visual_tokens, getattr(self.config, "bridge_hidden_size", 2560)),
            num_quantizers=getattr(self.config, "num_quantizers", 8) or 8,
            ecg_codebook_size=getattr(self.config, "codebook_size", 512) or 512,
            num_visual_tokens=num_visual_tokens,
            bridge_mid_dim=self.config.bridge_hidden_size,
            bridge_num_heads=self.config.bridge_num_heads,
            bridge_dropout=self.config.bridge_dropout,
            bridge_num_special_tokens=getattr(self.config, "bridge_num_special_tokens", 0),
            torch_dtype=torch_dtype,
            prefix_tuning=False,
        ).to(device)

        decoder.bridge = raw_bridge
        decoder.num_ecg_tokens = num_visual_tokens
        decoder.freeze_llm_parameters()
        self._apply_decoder_unfreeze_policy(decoder)
        decoder.eval()
        return decoder

    def _prepare_text_embeddings(
        self,
        text_bank_path: str,
        cache_path: Optional[str],
        model_name: str,
    ) -> tuple[torch.Tensor, Dict[str, int]]:
        def _load_cache_if_valid(path: Optional[str]) -> Optional[tuple[torch.Tensor, Dict[str, int]]]:
            if not path or not os.path.exists(path):
                return None
            try:
                cache = torch.load(path, map_location="cpu")
            except Exception as exc:  # pragma: no cover - defensive
                if self.config.is_ref_device:
                    print(f"[SigLIP] Failed to load text embedding cache at {path}: {exc}. Regenerating.")
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                return None

            target_std = cache.get("target_std")
            if target_std is None:
                if self.config.is_ref_device:
                    print(f"[SigLIP] Detected legacy text embedding cache at {path} without target_std. Regenerating.")
                    try:
                        os.remove(path)
                    except OSError:
                        pass
                return None

            embeddings = cache["embeddings"]
            text_id_to_idx = cache["text_id_to_idx"]
            self.config.target_embedding_std = float(target_std)
            return embeddings, text_id_to_idx

        cached = _load_cache_if_valid(cache_path)
        if cached is not None:
            return cached

        if cache_path and not self.config.is_ref_device:
            DistributedUtils.sync_process_group(self.config.world_size, self.config.device)
            cached = _load_cache_if_valid(cache_path)
            if cached is not None:
                return cached
            raise RuntimeError(
                "[SigLIP] Text embedding cache missing or stale after regeneration on reference device."
            )

        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        text_df = pd.read_csv(text_bank_path)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.bos_token is not None:
                tokenizer.pad_token = tokenizer.bos_token
            else:
                raise ValueError("Tokenizer does not define a pad/eos token required for embedding pooling.")

        text_model = AutoModel.from_pretrained(model_name, torch_dtype=torch.float32, trust_remote_code=True)
        text_model.eval()
        text_model.to("cpu")
        embedding_layer = text_model.get_input_embeddings()
        try:
            target_std = float(embedding_layer.weight.float().std().item())
        except Exception:
            target_std = None
        self.config.target_embedding_std = target_std

        texts = text_df["text"].tolist()
        text_ids = text_df["text_id"].tolist()

        all_embeddings = []
        with torch.no_grad():
            for start in range(0, len(texts), 64):
                chunk = texts[start:start + 64]
                inputs = tokenizer(
                    chunk,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=256,
                    add_special_tokens=False,
                )
                pooled = _pool_token_embeddings(
                    inputs["input_ids"],
                    inputs["attention_mask"],
                    embedding_layer,
                )
                all_embeddings.append(pooled.cpu())
        embeddings = torch.cat(all_embeddings, dim=0)
        embeddings = F.normalize(embeddings.float(), dim=-1)
        text_id_to_idx = {tid: idx for idx, tid in enumerate(text_ids)}

        if cache_path and self.config.is_ref_device:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            torch.save(
                {
                    "embeddings": embeddings,
                    "text_id_to_idx": text_id_to_idx,
                    "target_std": self.config.target_embedding_std,
                },
                cache_path,
            )
        DistributedUtils.sync_process_group(self.config.world_size, self.config.device)

        if cache_path and not self.config.is_ref_device:
            cached = _load_cache_if_valid(cache_path)
            if cached is not None:
                return cached
            raise RuntimeError(
                "[SigLIP] Text embedding cache still unavailable after regeneration."
            )
        return embeddings, text_id_to_idx

    def _build_report_text_embedder(
        self,
        model_name: str,
        device: torch.device,
    ) -> "ReportTextEmbedder":
        """Build the per-report text embedder used by contrastive_text_mode='report'.

        Uses the SAME tokenizer + input-embedding layer / mean-pool semantics as
        the fixed text bank (`_prepare_text_embeddings`), so report and bank
        embeddings live in one shared space.
        """
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            elif tokenizer.bos_token is not None:
                tokenizer.pad_token = tokenizer.bos_token
            else:
                raise ValueError(
                    "Tokenizer does not define a pad/eos token required for report pooling."
                )
        text_model = AutoModel.from_pretrained(
            model_name, torch_dtype=torch.float32, trust_remote_code=True
        )
        text_model.eval()
        embedding_layer = text_model.get_input_embeddings()
        # Detach the embedding layer from the full text model so we keep only the
        # lightweight lookup table resident (the rest is garbage-collected).
        embedder = ReportTextEmbedder(
            tokenizer=tokenizer,
            embedding_layer=embedding_layer,
            max_length=256,
            device=device,
        ).to(device)
        if getattr(self.config, "is_ref_device", True):
            print(
                f"[SigLIP] contrastive_text_mode=report -> built ReportTextEmbedder "
                f"from '{model_name}' (embedding dim={embedding_layer.weight.shape[1]})."
            )
        return embedder
