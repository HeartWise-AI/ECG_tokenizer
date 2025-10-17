from __future__ import annotations

import csv
import glob
import math
import os
import random
import time
from typing import Any, Dict, Optional, List, Tuple, Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler
from torch.amp import autocast
from torch.optim import Optimizer
from torch.utils.data import DataLoader

import pandas as pd
import wandb
from tqdm.auto import tqdm

from runners.base_runner import BaseRunner
from utils.enums import RunMode, RunnerName
from utils.registry import RunnerRegistry
from utils.config.siglip_phase1_config import SiglipPhase1Config
from utils.ddp import DistributedUtils
from utils.metrics.siglip_metrics import compute_alignment, compute_recall_at_many
from utils.debug import ensure_dir


@RunnerRegistry.register(RunnerName.SIGLIP_PHASE1)
class SiglipPhase1Runner(BaseRunner):
    """Runner implementing SigLIP-style BCE alignment for ECG embeddings."""

    def __init__(
        self,
        encoder: nn.Module,
        bridge: nn.Module,
        train_dataloader: DataLoader,
        optimizer: Optimizer,
        config: SiglipPhase1Config,
        text_embeddings: torch.Tensor,
        text_id_to_idx: Dict[str, int],
        text_lookup: Dict[str, str],
        quantizer: Optional[nn.Module] = None,
        use_quantized_inputs: bool = True,
        tokenizer_config: Optional[Dict[str, Any]] = None,
        wandb_wrapper=None,
        validation_dataloader: Optional[DataLoader] = None,
        scaler: Optional[GradScaler] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        super().__init__(config=config, wandb_wrapper=wandb_wrapper)
        self.encoder = encoder
        self.quantizer = quantizer
        self.bridge = bridge
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = scaler
        self.text_embeddings = text_embeddings  # [U, H]
        self.text_id_to_idx = text_id_to_idx
        self.text_lookup = text_lookup
        self.use_quantized_inputs = bool(use_quantized_inputs and quantizer is not None)
        self.tokenizer_config = tokenizer_config or {}
        self.grad_accum = max(1, int(config.gradient_accumulation_steps))
        loss_type = str(getattr(config, "loss_type", "infonce") or "infonce").lower()
        if loss_type not in {"infonce", "siglip_bce"}:
            raise ValueError(f"Unsupported loss_type '{loss_type}' for SiglipPhase1Runner.")
        self.loss_type = loss_type

        # Focal-InfoNCE configuration
        self.class_pos_weight_map = dict(getattr(config, "class_pos_weight_map", {}) or {})
        self.focal_gamma_pos = float(getattr(config, "focal_gamma_pos", 0.0) or 0.0)
        self.focal_gamma_neg = float(getattr(config, "focal_gamma_neg", 0.0) or 0.0)
        self.use_focal_infonce = bool(getattr(config, "focal_infonce", False))
        self.focal_alpha_default = float(getattr(config, "focal_alpha_default", 1.0) or 1.0)
        self.focal_detach_weights = bool(getattr(config, "focal_detach_weights", False))
        self._base_focal_detach = self.focal_detach_weights
        self._base_focal_gamma_neg = self.focal_gamma_neg

        # Tail recall tracking
        self.tail_enable = bool(getattr(config, "tail_enable", True))
        self.tail_class_ids = list(getattr(config, "tail_class_ids", []) or [])

        self.global_step = 0
        self.device = torch.device("cuda", config.device) if torch.cuda.is_available() else torch.device("cpu")
        self._init_mutual_exclusion_groups(text_lookup)

        dtype_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
        }
        self.amp_dtype = dtype_map.get(config.dtype.lower(), None)
        self.use_autocast = self.amp_dtype is not None
        if self.amp_dtype == torch.float16 and self.scaler is None:
            self.scaler = GradScaler()
        self.grad_clip = 1.0

        self.encoder.eval()
        for param in self.encoder.parameters():
            param.requires_grad = False

        if self.quantizer is not None:
            self.quantizer.eval()
            for param in self.quantizer.parameters():
                param.requires_grad = False

        self.bridge.train()
        self.text_embeddings = F.normalize(self.text_embeddings, dim=-1)
        self.text_embeddings = self.text_embeddings.to(self.device)

        self.best_checkpoint_path: Optional[str] = None
        self.best_metric_value: float = float("inf")
        self.best_metric_epoch: Optional[int] = None
        self.best_metric_name: Optional[str] = None
        self.last_checkpoint_path: Optional[str] = None
        self.last_checkpoint_epoch: Optional[int] = None

        self._initialize_checkpoint_state()

    def _init_mutual_exclusion_groups(self, text_lookup: Dict[str, str]) -> None:
        """Pre-compute mutually exclusive text groups for training constraints."""
        existing_ids = set(text_lookup.keys())
        hard_groups: List[List[str]] = []
        exclusive_groups: List[List[str]] = []

        rhythm_label_group = [
            "LBL_afib",
            "LBL_atrial_flutter",
            "LBL_atrial_tachycardia_100_bpm",
            "LBL_junctional_rhythm",
            "LBL_supraventricular_tachycardia",
            "LBL_ventricular_tachycardia",
        ]
        label_group_filtered = [tid for tid in rhythm_label_group if tid in existing_ids]
        if len(label_group_filtered) >= 2:
            hard_groups.append(label_group_filtered)

        rhythm_qa_group = [
            "QA_afib_yes",
            "QA_atrial_flutter_yes",
            "QA_atrial_tachycardia_100_bpm_yes",
            "QA_junctional_rhythm_yes",
            "QA_supraventricular_tachycardia_yes",
            "QA_ventricular_tachycardia_yes",
        ]
        qa_group_filtered = [tid for tid in rhythm_qa_group if tid in existing_ids]
        if len(qa_group_filtered) >= 2:
            hard_groups.append(qa_group_filtered)
            exclusive_groups.append(qa_group_filtered)

        seen_pairs: set[tuple[str, str]] = set()
        for tid in existing_ids:
            if not tid.startswith("QA_") or not tid.endswith("_yes"):
                continue
            base = tid[:-3]  # strip 'yes'
            no_tid = base + "no"
            if no_tid not in existing_ids:
                continue
            pair_key = (tid, no_tid)
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            hard_groups.append([tid, no_tid])
            exclusive_groups.append([tid, no_tid])

        self.hard_negative_groups: List[List[str]] = hard_groups
        self.mutual_exclusive_groups: List[List[str]] = exclusive_groups
        self.hard_negative_weight: float = float(getattr(self.config, "w_hardneg", 0.0) or 0.0)
        self.group_loss_weight: float = float(getattr(self.config, "mutual_exclusion_aux_weight", 0.0) or 0.0)

    def _apply_group_hard_negatives(
        self,
        labels: torch.Tensor,
        weights: torch.Tensor,
        text_ids: List[str],
    ) -> torch.Tensor:
        # Collator already encodes the candidate mask; keep weights unchanged.
        return weights

    def _compute_group_auxiliary_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        text_ids: List[str],
    ) -> torch.Tensor:
        if self.group_loss_weight <= 0 or not self.mutual_exclusive_groups:
            return logits.new_zeros(())

        col_lookup = {tid: idx for idx, tid in enumerate(text_ids)}
        loss_accum = logits.new_zeros(())
        valid_groups = 0

        for group in self.mutual_exclusive_groups:
            cols = [col_lookup[tid] for tid in group if tid in col_lookup]
            if len(cols) < 2:
                continue
            g_idx = torch.tensor(cols, device=logits.device, dtype=torch.long)
            group_logits = logits.index_select(1, g_idx)
            group_labels = labels.index_select(1, g_idx)
            pos_counts = (group_labels > 0.5).sum(dim=1)
            valid_mask = pos_counts == 1
            if not torch.any(valid_mask):
                continue
            target = group_labels[valid_mask].argmax(dim=1).to(dtype=torch.long)
            ce = F.cross_entropy(group_logits[valid_mask], target, reduction="mean")
            loss_accum = loss_accum + ce
            valid_groups += 1

        if valid_groups == 0:
            return logits.new_zeros(())

        return loss_accum * (self.group_loss_weight / valid_groups)

    def _alpha_for_text_ids(self, text_ids: List[str]) -> torch.Tensor:
        values = [float(self.class_pos_weight_map.get(tid, self.focal_alpha_default)) for tid in text_ids]
        return torch.tensor(values, device=self.device, dtype=torch.float32)

    def _infonce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        alpha_col: Optional[torch.Tensor],
        gamma_pos: float,
        gamma_neg: float,
        detach_weights: bool = False,
    ) -> torch.Tensor:
        """Multi-positive InfoNCE with optional focal parameters and alpha weights."""
        pos_mask = (labels > 0.5) & mask
        valid_rows = pos_mask.any(dim=1)
        if not torch.any(valid_rows):
            return logits.new_zeros(())

        logits_sel = logits[valid_rows]
        mask_sel = mask[valid_rows]
        pos_mask = pos_mask[valid_rows]
        neg_mask = (~pos_mask) & mask_sel

        neg_inf = torch.finfo(logits_sel.dtype).min
        logits_masked = logits_sel.masked_fill(~mask_sel, neg_inf)
        probs = torch.softmax(logits_masked, dim=1)
        probs_detached = probs.detach() if detach_weights else probs

        if alpha_col is None:
            alpha = logits_sel.new_ones((1, logits_sel.size(1)))
        else:
            alpha = alpha_col.view(1, -1).to(logits_sel.dtype)
        alpha = alpha.expand_as(logits_sel)

        if gamma_pos > 0.0:
            pos_weights = alpha * (1.0 - probs_detached).clamp_min(1e-6).pow(gamma_pos)
        else:
            pos_weights = alpha
        pos_weights = torch.where(pos_mask, pos_weights, torch.zeros_like(pos_weights)).clamp_min(1e-12)

        if gamma_neg > 0.0:
            denom_weights = torch.where(
                neg_mask,
                probs_detached.clamp_min(1e-6).pow(gamma_neg),
                torch.ones_like(probs_detached),
            )
        else:
            denom_weights = torch.ones_like(probs)

        logits_denom = logits_sel + torch.log(denom_weights.clamp_min(1e-12))
        logits_denom = logits_denom.masked_fill(~mask_sel, neg_inf)
        denom = torch.logsumexp(logits_denom, dim=1)

        logits_num = logits_sel + torch.log(pos_weights)
        logits_num = logits_num.masked_fill(~pos_mask, neg_inf)
        numer = torch.logsumexp(logits_num, dim=1)

        return (denom - numer).mean().to(logits.dtype)

    def _siglip_bce_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        alpha_col: Optional[torch.Tensor],
        gamma_pos: float,
        gamma_neg: float,
        detach_weights: bool = False,
    ) -> torch.Tensor:
        """Per-candidate sigmoid BCE variant inspired by SigLIP."""
        pos_mask = (labels > 0.5) & mask
        valid_rows = pos_mask.any(dim=1)
        if not torch.any(valid_rows):
            return logits.new_zeros(())

        logits_sel = logits[valid_rows]
        mask_sel = mask[valid_rows]
        pos_mask = pos_mask[valid_rows]
        neg_mask = (~pos_mask) & mask_sel

        probs = torch.sigmoid(logits_sel)
        probs_detached = probs.detach() if detach_weights else probs

        if alpha_col is None:
            alpha = logits_sel.new_ones((1, logits_sel.size(1)))
        else:
            alpha = alpha_col.view(1, -1).to(logits_sel.dtype)
        alpha = alpha.expand_as(logits_sel)

        pos_weights = alpha
        neg_weights = alpha
        if gamma_pos > 0.0:
            pos_weights = pos_weights * (1.0 - probs_detached).clamp_min(1e-6).pow(gamma_pos)
        if gamma_neg > 0.0:
            neg_weights = neg_weights * probs_detached.clamp_min(1e-6).pow(gamma_neg)

        pos_loss = torch.zeros_like(probs[:, 0])
        neg_loss = torch.zeros_like(probs[:, 0])

        if torch.any(pos_mask):
            pos_term = -(pos_weights * pos_mask) * torch.log(probs.clamp_min(1e-6))
            pos_counts = pos_mask.sum(dim=1).clamp_min(1.0)
            pos_loss = pos_term.sum(dim=1) / pos_counts

        if torch.any(neg_mask):
            neg_term = -(neg_weights * neg_mask) * torch.log((1.0 - probs).clamp_min(1e-6))
            neg_counts = neg_mask.sum(dim=1).clamp_min(1.0)
            neg_loss = neg_term.sum(dim=1) / neg_counts

        total_loss = pos_loss + neg_loss
        return total_loss.mean().to(logits.dtype)

    def _collect_local_yes_no_pairs(self, text_ids: List[str]) -> List[tuple[int, int]]:
        index_map = {tid: idx for idx, tid in enumerate(text_ids)}
        pairs: List[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for tid in text_ids:
            if tid.endswith("yes") or tid.endswith("no"):
                base = tid[:-3]
                yes_tid = base + "yes"
                no_tid = base + "no"
                if yes_tid in index_map and no_tid in index_map:
                    pair = (index_map[yes_tid], index_map[no_tid])
                    key = tuple(sorted(pair))
                    if key not in seen:
                        seen.add(key)
                        pairs.append(pair)
        return pairs

    def _collapse_yes_no_logits(self, logits: torch.Tensor, text_ids: List[str]) -> torch.Tensor:
        pairs = self._collect_local_yes_no_pairs(text_ids)
        if not pairs:
            return logits
        collapsed = logits.clone()
        floor_value = torch.finfo(collapsed.dtype).min
        for yes_idx, no_idx in pairs:
            yes_scores = collapsed[:, yes_idx]
            no_scores = collapsed[:, no_idx]
            choose_yes = yes_scores >= no_scores
            collapsed[~choose_yes, yes_idx] = floor_value
            collapsed[choose_yes, no_idx] = floor_value
        return collapsed

    @torch.no_grad()
    def _accumulate_tail_recall_at_k(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        text_ids: List[str],
        ks: tuple[int, ...],
        accum: dict,
    ) -> None:
        if not self.tail_enable or not self.tail_class_ids:
            return

        if logits.numel() == 0 or mask.numel() == 0:
            return

        collapsed = self._collapse_yes_no_logits(logits, text_ids)
        neg_inf = torch.finfo(collapsed.dtype).min
        masked_logits = collapsed.masked_fill(~mask, neg_inf)

        col_lookup = {tid: idx for idx, tid in enumerate(text_ids)}
        tail_cols = {tid: col_lookup[tid] for tid in self.tail_class_ids if tid in col_lookup}
        if not tail_cols:
            return

        max_k = max(ks)
        top_k = min(max_k, masked_logits.size(1))
        if top_k <= 0:
            return

        _, top_indices = torch.topk(masked_logits, k=top_k, dim=1)

        labels_with_pos: set[str] = accum.setdefault("labels_with_pos", set())
        accum.setdefault("total_pos_examples", 0)
        for tid, col in tail_cols.items():
            pos_rows_full = labels[:, col] > 0.5
            n_pos_full = int(pos_rows_full.sum().item())
            pos_rows = pos_rows_full & mask[:, col]
            n_pos_available = int(pos_rows.sum().item())

            availability = accum.setdefault("tail_availability", {})
            avail_pos, avail_total = availability.get(tid, (0, 0))
            availability[tid] = (avail_pos + n_pos_available, avail_total + n_pos_full)

            if n_pos_available == 0:
                continue
            labels_with_pos.add(tid)
            accum["total_pos_examples"] = int(accum.get("total_pos_examples", 0)) + n_pos_available
            rows_idx = torch.nonzero(pos_rows, as_tuple=False).squeeze(1)
            total = int(rows_idx.numel())
            if total == 0:
                continue
            for k in ks:
                key_hits = f"tail_hits@{k}"
                key_total = f"tail_total@{k}"
                key_label = f"tail_hits_per_label@{k}"
                slice_width = min(k, top_indices.size(1))
                picked = top_indices[rows_idx, :slice_width]
                hits = int((picked == col).any(dim=1).sum().item())
                accum[key_hits] = accum.get(key_hits, 0) + hits
                accum[key_total] = accum.get(key_total, 0) + total
                per_label = accum.setdefault(key_label, {})
                label_hits, label_total = per_label.get(tid, (0, 0))
                per_label[tid] = (label_hits + hits, label_total + total)

    def _log_alpha_statistics(self) -> None:
        if not self.class_pos_weight_map:
            return
        if not (self.wandb_wrapper and self.wandb_wrapper.is_initialized()):
            return
        if not getattr(self.config, "is_ref_device", True):
            return

        alpha_vals = list(self.class_pos_weight_map.values())
        if not alpha_vals:
            return
        alpha_sorted = sorted(alpha_vals)
        median = alpha_sorted[len(alpha_sorted) // 2]
        payload = {
            "alpha/stats_min": float(alpha_sorted[0]),
            "alpha/stats_median": float(median),
            "alpha/stats_max": float(alpha_sorted[-1]),
        }
        self.wandb_wrapper.log(payload)

    @staticmethod
    def _compute_grad_norm(
        parameters: Iterable[torch.nn.Parameter],
        norm_type: float = 2.0,
    ) -> float:
        total = 0.0
        for param in parameters:
            if param.grad is None:
                continue
            grad = param.grad.data
            param_norm = grad.norm(norm_type)
            total += float(param_norm.item() ** norm_type)
        if total <= 0.0:
            return 0.0
        return float(total ** (1.0 / norm_type))

    def _log_grad_norms(
        self,
        bridge_norm: Optional[float],
        adapter_norm: Optional[float],
        optimizer_step: int,
    ) -> None:
        if bridge_norm is None:
            return
        if not self.config.is_ref_device:
            return
        if not (self.wandb_wrapper and self.wandb_wrapper.is_initialized()):
            return

        payload: Dict[str, float] = {
            "train/grad_norm_bridge": float(bridge_norm),
            "trainer/optimizer_step": float(optimizer_step),
        }
        if adapter_norm is not None:
            payload["train/grad_norm_medgemma_adapter"] = float(adapter_norm)
        self.wandb_wrapper.log(payload)

    def _initialize_checkpoint_state(self) -> None:
        """Load existing checkpoint metadata so we can resume best/last tracking."""
        checkpoint_dir = self.config.checkpoint_dir
        if not checkpoint_dir:
            return

        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
        except OSError:
            return

        best_pattern = os.path.join(checkpoint_dir, "siglip_phase1_best_epoch_*.pt")
        for path in glob.glob(best_pattern):
            try:
                data = torch.load(path, map_location="cpu")
            except Exception:
                continue

            metric_value = data.get("metric_value", data.get("loss"))
            metric_name = data.get("metric_name")
            epoch = data.get("epoch")
            if metric_value is None or epoch is None:
                continue

            metric_value = float(metric_value)
            if math.isnan(metric_value):
                continue

            if metric_value < self.best_metric_value:
                self.best_metric_value = metric_value
                self.best_checkpoint_path = path
                self.best_metric_epoch = int(epoch)
                self.best_metric_name = metric_name

        last_pattern = os.path.join(checkpoint_dir, "siglip_phase1_last_epoch_*.pt")
        latest_epoch = -1
        for path in glob.glob(last_pattern):
            epoch = self._extract_epoch_from_filename(path)
            if epoch is None:
                try:
                    data = torch.load(path, map_location="cpu")
                    epoch = data.get("epoch")
                except Exception:
                    continue

            if epoch is None:
                continue

            epoch = int(epoch)
            if epoch > latest_epoch:
                latest_epoch = epoch
                self.last_checkpoint_path = path
                self.last_checkpoint_epoch = epoch

        if self.best_checkpoint_path is None:
            self.best_metric_value = float("inf")

    def _extract_epoch_from_filename(self, path: str) -> Optional[int]:
        name = os.path.splitext(os.path.basename(path))[0]
        tokens = name.split("_")
        for idx, token in enumerate(tokens):
            if token == "epoch" and idx + 1 < len(tokens):
                next_token = tokens[idx + 1]
                if next_token.isdigit():
                    return int(next_token)
        return None

    def _get_checkpoint_metric(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]],
    ) -> tuple[float, str]:
        if val_metrics is not None and "loss" in val_metrics:
            return float(val_metrics["loss"]), "val_loss"
        return float(train_metrics.get("loss", float("inf"))), "train_loss"

    def _save_checkpoint_file(
        self,
        checkpoint_path: str,
        epoch: int,
        loss: float,
        payload: Dict[str, Any],
    ) -> bool:
        try:
            temperature = float(self.bridge.temperature().detach().cpu())
        except Exception:
            temperature = None

        save_kwargs = dict(payload)
        try:
            if temperature is not None:
                self._save_checkpoint(
                    model=self.bridge,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    loss=loss,
                    checkpoint_path=checkpoint_path,
                    temperature=temperature,
                    **save_kwargs,
                )
            else:
                self._save_checkpoint(
                    model=self.bridge,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    loss=loss,
                    checkpoint_path=checkpoint_path,
                    **save_kwargs,
                )
            return True
        except Exception as exc:
            print(f"[{self.__class__.__name__}] Failed to save checkpoint {checkpoint_path}: {exc}")
            return False

    def _cleanup_checkpoint(self, path: Optional[str]) -> None:
        if not path or not os.path.exists(path):
            return
        try:
            os.remove(path)
            print(f"[{self.__class__.__name__}] Removed checkpoint: {path}")
        except OSError as exc:
            print(f"[{self.__class__.__name__}] Failed to remove checkpoint {path}: {exc}")

    def _maybe_save_epoch_checkpoints(
        self,
        epoch: int,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]],
    ) -> None:
        if not self.config.is_ref_device or not self.config.checkpoint_dir:
            return

        metric_value, metric_name = self._get_checkpoint_metric(train_metrics, val_metrics)
        if math.isnan(metric_value):
            return

        checkpoint_dir = self.config.checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        train_metrics_payload = dict(train_metrics)
        val_metrics_payload = dict(val_metrics) if val_metrics is not None else None

        last_filename = f"siglip_phase1_last_epoch_{epoch}.pt"
        last_path = os.path.join(checkpoint_dir, last_filename)

        payload = {
            "metric_value": metric_value,
            "metric_name": metric_name,
            "is_best": False,
            "is_last": True,
            "train_metrics": train_metrics_payload,
        }
        if val_metrics_payload is not None:
            payload["val_metrics"] = val_metrics_payload

        last_saved = self._save_checkpoint_file(
            checkpoint_path=last_path,
            epoch=epoch,
            loss=float(train_metrics.get("loss", metric_value)),
            payload=payload,
        )

        if last_saved:
            previous_last = self.last_checkpoint_path
            self.last_checkpoint_path = last_path
            self.last_checkpoint_epoch = epoch
            if previous_last and previous_last != last_path:
                self._cleanup_checkpoint(previous_last)

        if metric_value >= self.best_metric_value:
            return

        best_filename = f"siglip_phase1_best_epoch_{epoch}.pt"
        best_path = os.path.join(checkpoint_dir, best_filename)

        best_payload = dict(payload)
        best_payload.update({
            "is_best": True,
            "is_last": False,
        })

        best_saved = self._save_checkpoint_file(
            checkpoint_path=best_path,
            epoch=epoch,
            loss=float(train_metrics.get("loss", metric_value)),
            payload=best_payload,
        )

        if best_saved:
            previous_best = self.best_checkpoint_path
            self.best_checkpoint_path = best_path
            self.best_metric_epoch = epoch
            self.best_metric_value = metric_value
            self.best_metric_name = metric_name
            if previous_best and previous_best != best_path:
                self._cleanup_checkpoint(previous_best)

    def train(self) -> None:  # type: ignore[override]
        num_epochs = self.config.num_epochs
        for epoch in range(1, num_epochs + 1):
            if self.use_focal_infonce:
                if epoch == 1:
                    self.focal_detach_weights = True
                    self.focal_gamma_neg = 0.0
                elif epoch == 2:
                    self.focal_detach_weights = self._base_focal_detach
                    self.focal_gamma_neg = max(self._base_focal_gamma_neg, 0.25)

            collate_fn = getattr(self.train_dataloader, "collate_fn", None)
            if hasattr(collate_fn, "reseed"):
                rank = int(getattr(self.config, "device", 0))
                try:
                    if DistributedUtils.dist.is_available() and DistributedUtils.dist.is_initialized():
                        rank = int(DistributedUtils.dist.get_rank())
                except Exception:
                    rank = int(getattr(self.config, "device", 0))
                base_seed = int(self.config.seed)
                collate_fn.reseed(base_seed + 997 * epoch + rank)
            epoch_metrics = self._run_epoch(
                mode=RunMode.TRAIN,
                epoch=epoch,
                dataloader=self.train_dataloader,
            )

            if self.scheduler is not None:
                try:
                    self.scheduler.step()
                except Exception:
                    pass

            val_metrics = self._evaluate(self.validation_dataloader, epoch=epoch) if self.validation_dataloader else None
            analysis_rows = None
            best_indices = None
            worst_indices = None
            random_index = None
            if val_metrics is not None:
                analysis_rows = val_metrics.pop("analysis_rows", None)
                best_indices = val_metrics.pop("best_indices", None)
                worst_indices = val_metrics.pop("worst_indices", None)
                random_index = val_metrics.pop("random_index", None)

            if self.config.is_ref_device:
                alignment = epoch_metrics.get("alignment", 0.0)
                print(
                    f"Epoch {epoch}/{num_epochs} - loss: {epoch_metrics['loss']:.4f} "
                    f"pos_prob: {epoch_metrics['pos_prob']:.4f} neg_prob: {epoch_metrics['neg_prob']:.4f} "
                    f"alignment: {alignment:.4f} U: {epoch_metrics.get('avg_candidate_vocab', 0.0):.1f}"
                )
                train_payload = {f"train/{k}": v for k, v in epoch_metrics.items()}
                train_payload["trainer/epoch"] = float(epoch)
                train_payload["trainer/step"] = float(self.global_step)
                self._log_metrics(train_payload)
                self._log_alpha_statistics()

                if val_metrics is not None:
                    print(
                        f"    Validation - loss: {val_metrics['loss']:.4f} pos_prob: {val_metrics['pos_prob']:.4f} "
                        f"neg_prob: {val_metrics['neg_prob']:.4f} recall@5: {val_metrics['recall_at_5']:.4f} "
                        f"U: {val_metrics.get('avg_candidate_vocab', 0.0):.1f}"
                    )
                    val_payload = {f"val/{k}": v for k, v in val_metrics.items()}
                    val_payload["trainer/epoch"] = float(epoch)
                    self._log_metrics(val_payload)

                if analysis_rows:
                    self._log_retrieval_artifacts(
                        epoch=epoch,
                        rows=analysis_rows,
                        best_indices=best_indices or [],
                        worst_indices=worst_indices or [],
                        random_index=random_index,
                    )

                self._maybe_save_epoch_checkpoints(
                    epoch=epoch,
                    train_metrics=epoch_metrics,
                    val_metrics=val_metrics,
                )

    def inference(self):  # type: ignore[override]
        raise NotImplementedError("Inference is not implemented for SigLIP Phase-1 runner")

    def _run_epoch(
        self,
        mode: RunMode,
        epoch: int,
        dataloader: DataLoader,
        step_fn=None,
    ) -> Dict[str, float]:  # type: ignore[override]
        del step_fn  # Unused but kept for BaseRunner compatibility
        start = time.time()
        total_loss = 0.0
        total_pairs = 0
        pos_mass_sum = 0.0
        neg_mass_sum = 0.0
        row_counter = 0
        total_masked_candidates = 0.0
        total_rows = 0
        candidate_vocab_sum = 0.0
        candidate_vocab_count = 0

        self.bridge.train(mode == RunMode.TRAIN)

        self.optimizer.zero_grad(set_to_none=True)

        iterator = dataloader
        if self.config.is_ref_device:
            iterator = tqdm(
                dataloader,
                desc=f"Epoch {epoch}/{self.config.num_epochs}",
                leave=False,
                dynamic_ncols=True,
            )

        for step, batch in enumerate(iterator, start=1):
            metrics = self._training_step(batch)
            total_loss += metrics["loss_sum"]
            total_pairs += metrics["num_pairs"]
            pos_mass_sum += metrics["pos_prob_sum"]
            neg_mass_sum += metrics["neg_prob_sum"]
            row_counter += metrics["row_count"]
            total_masked_candidates += metrics.get("candidate_mask_sum", 0.0)
            total_rows += metrics.get("row_count", 0)
            candidate_vocab_sum += metrics.get("candidate_vocab", 0.0)
            candidate_vocab_count += 1

            if self.config.log_every_steps > 0 and step % self.config.log_every_steps == 0:
                avg_loss = total_loss / max(1, total_pairs)
                avg_pos = pos_mass_sum / max(1, row_counter)
                avg_neg = neg_mass_sum / max(1, row_counter)
                current_vocab = metrics.get("candidate_vocab", 0)
                alignment_avg = avg_pos - avg_neg
                if isinstance(iterator, tqdm):
                    iterator.set_postfix(
                        loss=f"{avg_loss:.4f}",
                        pos=f"{avg_pos:.3f}",
                        neg=f"{avg_neg:.3f}",
                        align=f"{alignment_avg:.3f}",
                        U=int(current_vocab),
                    )
                else:
                    print(
                        f"Epoch {epoch} Step {step}: loss={avg_loss:.4f} "
                        f"pos_prob={avg_pos:.4f} neg_prob={avg_neg:.4f} "
                        f"U={int(current_vocab)}"
                    )
                if self.config.is_ref_device:
                    running_avg_candidates = total_masked_candidates / max(1, total_rows)
                    running_vocab = candidate_vocab_sum / max(1, candidate_vocab_count)
                    self._log_metrics(
                        {
                            "train/loss": float(avg_loss),
                            "train/pos_prob": float(avg_pos),
                            "train/neg_prob": float(avg_neg),
                            "train/alignment": float(compute_alignment(avg_pos, avg_neg)),
                            "train/row_pos_mass": float(avg_pos),
                            "train/row_neg_mass": float(avg_neg),
                            "train/avg_candidates": float(running_avg_candidates),
                            "train/avg_candidate_vocab": float(running_vocab),
                            "train/temperature": float(self.bridge.temperature().detach().cpu()),
                            "trainer/step": float(self.global_step),
                        }
                    )

        epoch_loss = total_loss / max(1, total_pairs)
        epoch_pos = pos_mass_sum / max(1, row_counter)
        epoch_neg = neg_mass_sum / max(1, row_counter)
        avg_masked_candidates = total_masked_candidates / max(1, total_rows)
        avg_candidate_vocab = candidate_vocab_sum / max(1, candidate_vocab_count)

        duration = time.time() - start
        if self.config.is_ref_device:
            print(f"Epoch {epoch} completed in {duration/60:.2f} min")

        return {
            "loss": float(epoch_loss),
            "pos_prob": float(epoch_pos),
            "neg_prob": float(epoch_neg),
            "row_pos_mass": float(epoch_pos),
            "row_neg_mass": float(epoch_neg),
            "alignment": compute_alignment(epoch_pos, epoch_neg),
            "temperature": float(self.bridge.temperature().detach().cpu()),
            "avg_candidates": float(avg_masked_candidates),
            "avg_candidate_vocab": float(avg_candidate_vocab),
        }

    def _training_step(self, batch: dict) -> Dict[str, float]:
        signals: torch.Tensor = batch["signals"].to(self.device)
        labels: torch.Tensor = batch["labels"].to(self.device)
        weights: torch.Tensor = batch["weights"].to(self.device)
        text_ids: List[str] = batch["text_ids"]

        text_indices = [self.text_id_to_idx[tid] for tid in text_ids]
        text_vecs = self.text_embeddings[text_indices]
        text_vecs = F.normalize(text_vecs, dim=-1)

        bridge_inputs = self._compute_bridge_inputs(signals)

        weights_mask = self._apply_group_hard_negatives(labels, weights, text_ids) > 0
        if self.use_autocast:
            autocast_ctx = autocast("cuda", dtype=self.amp_dtype)
        else:
            autocast_ctx = torch.cuda.amp.autocast(enabled=False)

        with autocast_ctx:
            pooled, _ = self.bridge(**bridge_inputs)
            pooled = F.normalize(pooled, dim=-1)
            logits = pooled @ text_vecs.T
            logits = logits / self.bridge.temperature()
            mask = weights_mask
            alpha_col = self._alpha_for_text_ids(text_ids) if self.use_focal_infonce else None
            gamma_pos = self.focal_gamma_pos if self.use_focal_infonce else 0.0
            gamma_neg = self.focal_gamma_neg if self.use_focal_infonce else 0.0
            if self.loss_type == "siglip_bce":
                loss = self._siglip_bce_loss(
                    logits=logits,
                    labels=labels,
                    mask=mask,
                    alpha_col=alpha_col,
                    gamma_pos=gamma_pos,
                    gamma_neg=gamma_neg,
                    detach_weights=self.focal_detach_weights,
                )
            else:
                loss = self._infonce_loss(
                    logits=logits,
                    labels=labels,
                    mask=mask,
                    alpha_col=alpha_col,
                    gamma_pos=gamma_pos,
                    gamma_neg=gamma_neg,
                    detach_weights=self.focal_detach_weights,
                )
            if self.mutual_exclusive_groups and self.group_loss_weight > 0:
                loss = loss + self._compute_group_auxiliary_loss(logits, labels, text_ids)
            loss = loss / self.grad_accum

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        grad_norm_bridge: Optional[float] = None
        grad_norm_adapter: Optional[float] = None

        self.global_step += 1
        if self.global_step % self.grad_accum == 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            bridge_params = [param for param in self.bridge.parameters() if param.requires_grad]
            grad_norm_bridge = self._compute_grad_norm(bridge_params)
            inner_adapter = getattr(self.bridge, "bridge", None)
            if inner_adapter is not None:
                adapter_params = [param for param in inner_adapter.parameters() if param.requires_grad]
                if adapter_params:
                    grad_norm_adapter = self._compute_grad_norm(adapter_params)
            torch.nn.utils.clip_grad_norm_(bridge_params, self.grad_clip)
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            optimizer_step = self.global_step // self.grad_accum
            self._log_grad_norms(grad_norm_bridge, grad_norm_adapter, optimizer_step)

        logits_detached = logits.detach()
        with torch.no_grad():
            metrics_logits = self._collapse_yes_no_logits(logits_detached, text_ids)
            neg_inf = float("-inf")
            masked_logits = metrics_logits.masked_fill(~mask, neg_inf).to(torch.float32)
            probs = torch.softmax(masked_logits, dim=1)
            pos_mask = (labels > 0.5) & mask
            neg_mask = (~pos_mask) & mask
            row_pos_mass = (probs * pos_mask).sum(dim=1)
            row_neg_mass = (probs * neg_mask).sum(dim=1)
            pos_prob_sum = row_pos_mass.sum().item()
            neg_prob_sum = row_neg_mass.sum().item()
            row_count = mask.size(0)
            valid_rows = int(pos_mask.any(dim=1).sum().item())

        loss_numerator = float(loss.detach().cpu()) * self.grad_accum * max(valid_rows, 1)

        return {
            "loss_sum": loss_numerator,
            "num_pairs": max(valid_rows, 1),
            "pos_prob_sum": pos_prob_sum,
            "neg_prob_sum": neg_prob_sum,
            "row_count": row_count,
            "candidate_mask_sum": float(mask.sum().item()),
            "candidate_vocab": len(text_ids),
        }

    @torch.no_grad()
    def _evaluate(self, dataloader: Optional[DataLoader], epoch: Optional[int] = None) -> Optional[Dict[str, float]]:
        if dataloader is None:
            return None

        self.bridge.eval()
        total_loss = 0.0
        total_pairs = 0
        pos_mass_sum = 0.0
        neg_mass_sum = 0.0
        row_counter = 0
        total_masked_candidates = 0.0
        total_rows = 0
        candidate_vocab_sum = 0.0
        candidate_vocab_count = 0
        recall_keys = (1, 5, 10, 25)
        recall_sums = {k: 0.0 for k in recall_keys}
        recall_counts = {k: 0 for k in recall_keys}
        tail_ks = (1, 5, 10)
        tail_accum: dict = {}

        analysis_rows: list[dict] = []
        best_samples: list[Tuple[int, float]] = []
        worst_samples: list[Tuple[int, float]] = []
        analysis_weight_threshold = float(getattr(self.config, "analysis_log_min_weight", 1.0))

        iterator = dataloader
        if self.config.is_ref_device:
            iterator = tqdm(
                dataloader,
                desc="Validation",
                leave=False,
                dynamic_ncols=True,
            )

        for batch in iterator:
            signals = batch["signals"].to(self.device)
            labels = batch["labels"].to(self.device)
            weights = batch["weights"].to(self.device)
            text_ids: List[str] = batch["text_ids"]
            ecg_ids: List[str] = batch.get("ecg_ids", [""] * len(signals))

            text_indices = [self.text_id_to_idx[tid] for tid in text_ids]
            text_vecs = self.text_embeddings[text_indices]
            text_vecs = F.normalize(text_vecs, dim=-1)

            bridge_inputs = self._compute_bridge_inputs(signals)
            pooled, _ = self.bridge(**bridge_inputs)
            pooled = F.normalize(pooled, dim=-1)
            logits = pooled @ text_vecs.T
            logits = logits / self.bridge.temperature()

            weights_mask = self._apply_group_hard_negatives(labels, weights, text_ids) > 0
            total_masked_candidates += float(weights_mask.sum().item())
            total_rows += weights_mask.size(0)
            candidate_vocab_sum += len(text_ids)
            candidate_vocab_count += 1
            self._accumulate_tail_recall_at_k(
                logits=logits.detach(),
                labels=labels,
                mask=weights_mask,
                text_ids=text_ids,
                ks=tail_ks,
                accum=tail_accum,
            )
            alpha_col = self._alpha_for_text_ids(text_ids) if self.use_focal_infonce else None
            infonce_loss = self._infonce_loss(
                logits=logits,
                labels=labels,
                mask=weights_mask,
                alpha_col=alpha_col,
                gamma_pos=self.focal_gamma_pos if self.use_focal_infonce else 0.0,
                gamma_neg=self.focal_gamma_neg if self.use_focal_infonce else 0.0,
                detach_weights=self.focal_detach_weights,
            )
            group_loss = self._compute_group_auxiliary_loss(logits, labels, text_ids) if self.mutual_exclusive_groups and self.group_loss_weight > 0 else logits.new_zeros(())
            batch_loss = infonce_loss + group_loss

            metrics_logits = self._collapse_yes_no_logits(logits, text_ids)
            neg_inf = float("-inf")
            masked_logits = metrics_logits.masked_fill(~weights_mask, neg_inf).to(torch.float32)
            probs = torch.softmax(masked_logits, dim=1)
            pos_mask = (labels > 0.5) & weights_mask
            neg_mask = (~pos_mask) & weights_mask

            row_pos_mass = (probs * pos_mask).sum(dim=1)
            row_neg_mass = (probs * neg_mask).sum(dim=1)
            pos_mass_sum += row_pos_mass.sum().item()
            neg_mass_sum += row_neg_mass.sum().item()
            row_counter += weights_mask.size(0)

            valid_rows = int(pos_mask.any(dim=1).sum().item())
            total_loss += batch_loss.item() * max(valid_rows, 1)
            total_pairs += max(valid_rows, 1)

            logits_for_metrics = masked_logits
            batch_recalls = compute_recall_at_many(logits_for_metrics, labels, ks=recall_keys)
            for k, (value_sum, count) in batch_recalls.items():
                recall_sums[k] += value_sum
                recall_counts[k] += count

            log_top_k = min(20, metrics_logits.size(1))
            for row in range(logits.size(0)):
                row_mask = weights_mask[row]
                if not torch.any(row_mask):
                    continue

                pos_cols = torch.nonzero(pos_mask[row]).flatten()
                if pos_cols.numel() == 0:
                    continue

                row_probs = probs[row]
                pos_probs_row = row_probs[pos_cols]
                neg_probs_row = row_probs[neg_mask[row]]
                pos_mass_value = float(row_probs[pos_mask[row]].sum().item())
                neg_mass_value = float(row_probs[neg_mask[row]].sum().item())
                if neg_probs_row.numel() > 0:
                    alignment_row = pos_probs_row.mean().item() - neg_probs_row.mean().item()
                else:
                    alignment_row = pos_probs_row.mean().item()

                finite_mask = torch.isfinite(metrics_logits[row]) & row_mask
                if not torch.any(finite_mask):
                    continue
                finite_idx = torch.nonzero(finite_mask).flatten()
                finite_scores = metrics_logits[row][finite_idx]
                k = min(log_top_k, finite_scores.numel())
                if k == 0:
                    continue
                _, top_local = torch.topk(finite_scores, k)
                top_idx = finite_idx[top_local]
                top_pred_ids = [text_ids[idx] for idx in top_idx.tolist()]
                top_pred_probs = row_probs[top_idx].tolist()
                top_pred_texts = [self.text_lookup.get(tid, tid) for tid in top_pred_ids]

                gt_ids = [text_ids[idx] for idx in pos_cols.tolist()]
                gt_probs = row_probs[pos_cols].tolist()
                gt_texts = [self.text_lookup.get(tid, tid) for tid in gt_ids]

                explicit_cols = torch.nonzero(row_mask).flatten()
                explicit_all_ids: List[str] = []
                explicit_all_texts: List[str] = []
                explicit_all_probs: List[float] = []
                explicit_neg_ids: List[str] = []
                explicit_neg_texts: List[str] = []
                explicit_neg_probs: List[float] = []
                for col in explicit_cols.tolist():
                    text_id = text_ids[col]
                    text_prob = float(probs[row, col].item())
                    text_value = self.text_lookup.get(text_id, text_id)
                    explicit_all_ids.append(text_id)
                    explicit_all_texts.append(text_value)
                    explicit_all_probs.append(text_prob)
                    if labels[row, col] <= 0.5:
                        explicit_neg_ids.append(text_id)
                        explicit_neg_texts.append(text_value)
                        explicit_neg_probs.append(text_prob)

                analysis_rows.append(
                    {
                        "ecg_id": ecg_ids[row],
                        "alignment": alignment_row,
                        "row_pos_mass": pos_mass_value,
                        "row_neg_mass": neg_mass_value,
                        "ground_truth_ids": gt_ids,
                        "ground_truth_probs": gt_probs,
                        "ground_truth_texts": gt_texts,
                        "explicit_all_ids": explicit_all_ids,
                        "explicit_all_probs": explicit_all_probs,
                        "explicit_all_texts": explicit_all_texts,
                        "explicit_neg_ids": explicit_neg_ids,
                        "explicit_neg_probs": explicit_neg_probs,
                        "explicit_neg_texts": explicit_neg_texts,
                        "top_pred_ids": top_pred_ids,
                        "top_pred_probs": top_pred_probs,
                        "top_pred_texts": top_pred_texts,
                    }
                )
                index = len(analysis_rows) - 1
                best_samples.append((index, alignment_row))
                worst_samples.append((index, alignment_row))
        avg_loss = total_loss / max(1, total_pairs)
        avg_pos = pos_mass_sum / max(1, row_counter)
        avg_neg = neg_mass_sum / max(1, row_counter)
        avg_masked_candidates = total_masked_candidates / max(1, total_rows)
        avg_candidate_vocab = candidate_vocab_sum / max(1, candidate_vocab_count)
        recall_metrics = {
            k: (recall_sums[k] / recall_counts[k]) if recall_counts[k] > 0 else 0.0
            for k in recall_keys
        }

        self.bridge.train()

        metrics = {
            "loss": float(avg_loss),
            "pos_prob": float(avg_pos),
            "neg_prob": float(avg_neg),
            "row_pos_mass": float(avg_pos),
            "row_neg_mass": float(avg_neg),
            "alignment": compute_alignment(avg_pos, avg_neg),
            "temperature": float(self.bridge.temperature().detach().cpu()),
            "avg_candidates": float(avg_masked_candidates),
            "avg_candidate_vocab": float(avg_candidate_vocab),
        }
        for k, value in recall_metrics.items():
            metrics[f"recall_at_{k}"] = float(value)

        tail_rows: list[tuple[str, float, int, int]] = []
        coverage_min = 1.0
        coverage_warnings: list[tuple[str, float, int, int]] = []
        if self.tail_enable and self.tail_class_ids:
            labels_with_pos = tail_accum.get("labels_with_pos", set())
            metrics["tail/labels_with_pos"] = float(len(labels_with_pos))
            metrics["tail/labels_selected"] = float(len(self.tail_class_ids))
            metrics["tail/coverage"] = (
                float(len(labels_with_pos)) / max(1.0, float(len(self.tail_class_ids)))
            )
            metrics["tail/positives_seen"] = float(tail_accum.get("total_pos_examples", 0))
            for k in tail_ks:
                hits_key = f"tail_hits@{k}"
                total_key = f"tail_total@{k}"
                hits = tail_accum.get(hits_key, 0)
                total = tail_accum.get(total_key, 0)
                metrics[f"tail/recall@{k}"] = float(hits / total) if total > 0 else 0.0

            per_label = tail_accum.get("tail_hits_per_label@5", {})
            availability = tail_accum.get("tail_availability", {})
            for tid in self.tail_class_ids:
                hits, total = per_label.get(tid, (0, 0))
                recall = (hits / total) if total > 0 else 0.0
                tail_rows.append((tid, recall, hits, total))
                avail_pos, avail_total = availability.get(tid, (0, 0))
                if avail_total > 0:
                    coverage = avail_pos / max(avail_total, 1)
                    coverage_min = min(coverage_min, coverage)
                    if coverage < 0.999:
                        coverage_warnings.append((tid, coverage, avail_pos, avail_total))
            tail_rows.sort(key=lambda item: item[1])
            if tail_rows:
                metrics["tail/coverage_min"] = float(coverage_min)

            if getattr(self.config, "is_ref_device", True):
                worst = [entry for entry in tail_rows if entry[3] > 0][:10]
                if worst:
                    print("[SigLIP] Tail recall@5 (worst 10):")
                    for tid, recall, hits, total in worst:
                        text = self.text_lookup.get(tid, tid) or tid
                        print(f"   {tid:<40} recall@5={recall:.3f} ({hits}/{total})  '{text[:70]}'")
                if coverage_warnings:
                    print("[SigLIP] Tail availability gaps detected:")
                    for tid, coverage, avail_pos, avail_total in sorted(coverage_warnings, key=lambda x: x[1]):
                        text = self.text_lookup.get(tid, tid) or tid
                        print(f"   {tid:<40} coverage={coverage:.3f} ({avail_pos}/{avail_total}) '{text[:70]}'")

            if (
                self.wandb_wrapper
                and self.wandb_wrapper.is_initialized()
                and getattr(self.config, "is_ref_device", True)
            ):
                payload = {
                    "val/tail/recall@1": metrics.get("tail/recall@1", 0.0),
                    "val/tail/recall@5": metrics.get("tail/recall@5", 0.0),
                    "val/tail/recall@10": metrics.get("tail/recall@10", 0.0),
                }
                payload.update(
                    {
                        "val/tail/coverage": metrics.get("tail/coverage", 0.0),
                        "val/tail/labels_with_pos": metrics.get("tail/labels_with_pos", 0.0),
                        "val/tail/labels_selected": metrics.get("tail/labels_selected", 0.0),
                        "val/tail/positives_seen": metrics.get("tail/positives_seen", 0.0),
                    }
                )
                if tail_rows:
                    payload["val/tail/coverage_min"] = float(metrics.get("tail/coverage_min", 1.0))
                table = wandb.Table(columns=["text_id", "text", "recall@5", "hits", "total", "coverage"])
                for tid, recall, hits, total in tail_rows[:50]:
                    avail_pos, avail_total = availability.get(tid, (0, 0))
                    coverage = (avail_pos / max(avail_total, 1)) if avail_total > 0 else 1.0
                    table.add_data(
                        tid,
                        self.text_lookup.get(tid, tid),
                        float(recall),
                        int(hits),
                        int(total),
                        float(coverage),
                    )
                payload["val/tail/recall_table@5_top50worst"] = table
                self.wandb_wrapper.log(payload)

            if getattr(self.config, "output_dir", None) and epoch is not None:
                artifact_dir = os.path.join(self.config.output_dir, "artifacts")
                ensure_dir(artifact_dir)
                rank = int(getattr(self.config, "device", 0))
                rank_suffix = ""
                if not getattr(self.config, "is_ref_device", True):
                    rank_suffix = f"_rank{rank}"
                csv_path = os.path.join(
                    artifact_dir,
                    f"val_epoch_{epoch}_tail_recall_at5{rank_suffix}.csv",
                )
                with open(csv_path, "w", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["text_id", "text", "recall@5", "hits", "total", "coverage"])
                    for tid, recall, hits, total in tail_rows:
                        avail_pos, avail_total = availability.get(tid, (0, 0))
                        coverage = (avail_pos / max(avail_total, 1)) if avail_total > 0 else 1.0
                        writer.writerow(
                            [
                                tid,
                                self.text_lookup.get(tid, tid),
                                float(recall),
                                int(hits),
                                int(total),
                                float(coverage),
                            ]
                        )
                print(f"[SigLIP] Saved tail recall@5 CSV: {csv_path}")
        else:
            metrics["tail/labels_with_pos"] = 0.0
            metrics["tail/labels_selected"] = float(len(self.tail_class_ids or [])) if self.tail_class_ids else 0.0
            metrics["tail/coverage"] = 0.0
            metrics["tail/positives_seen"] = 0.0

        if analysis_rows and self.config.is_ref_device:
            best_samples.sort(key=lambda x: x[1], reverse=True)
            worst_samples.sort(key=lambda x: x[1])
            top_best = best_samples[:5]
            top_worst = worst_samples[:5]
            random_idx = random.randrange(len(analysis_rows))
            metrics["analysis_rows"] = analysis_rows
            metrics["best_indices"] = top_best
            metrics["worst_indices"] = top_worst
            metrics["random_index"] = random_idx
        return metrics

    def _compute_bridge_inputs(self, signals: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract frozen tokenizer features or codes for bridge consumption."""
        with torch.no_grad():
            encoder_feats = self.encoder(signals)
            features: torch.Tensor
            codes = None

            if self.quantizer is not None and self.use_quantized_inputs:
                keep = getattr(self.config, "num_codebooks_kept", None)
                offset_cfg = getattr(self.config, "codebook_offset", 0)
                quantized_outputs = self.quantizer(encoder_feats, return_all_codes=True)
                if len(quantized_outputs) == 4:
                    quantized, indices, _, all_codes = quantized_outputs
                else:
                    quantized, indices, _ = quantized_outputs
                    all_codes = None

                total_codebooks = indices.size(-1)
                if keep is None or keep <= 0 or keep > total_codebooks:
                    keep = total_codebooks
                max_valid_offset = max(total_codebooks - keep, 0)
                if offset_cfg < 0:
                    offset = max_valid_offset
                else:
                    upper_bound = max(total_codebooks - 1, 0)
                    offset = max(0, min(offset_cfg, upper_bound))
                    if offset > max_valid_offset:
                        offset = max_valid_offset
                self.config.codebook_offset = int(offset)
                end = min(offset + keep, total_codebooks)

                codes = indices[..., offset:end].long()
                if codes.size(-1) == 1:
                    codes = codes.squeeze(-1)

                if all_codes is not None:
                    selected = all_codes[offset:end]  # [keep, batch, seq, dim]
                    selected = selected.permute(1, 0, 3, 2).contiguous()  # [batch, keep, dim, seq]
                    batch, kept, dim, seq_len = selected.shape
                    features = selected.view(batch, kept * dim, seq_len)
                else:
                    if quantized.dim() != 3:
                        raise ValueError(f"Quantized features should be [batch, seq, dim]; got {quantized.shape}")
                    features = quantized.permute(0, 2, 1).contiguous()
            else:
                if encoder_feats.dim() != 3:
                    raise ValueError(f"Encoder features should be [batch, seq, dim]; got {encoder_feats.shape}")
                features = encoder_feats.permute(0, 2, 1).contiguous()

        if features.dim() != 3:
            raise AssertionError(
                f"Bridge expects features shaped [batch, channels, length]; got {features.shape}"
            )
        expected_channels = getattr(self.config, "bridge_input_channels", None)
        if expected_channels is not None and features.size(1) != int(expected_channels):
            raise AssertionError(
                f"Bridge channel dimension mismatch: expected {expected_channels}, got {features.size(1)}"
            )

        return {"features": features, "codes": codes}

    def _log_retrieval_artifacts(
        self,
        epoch: int,
        rows: List[dict],
        best_indices: List[Tuple[int, float]],
        worst_indices: List[Tuple[int, float]],
        random_index: Optional[int],
    ) -> None:
        if not rows:
            return

        artifact_dir = None
        rank_suffix = ""
        if self.config.output_dir:
            artifact_dir = os.path.join(self.config.output_dir, "artifacts")
            ensure_dir(artifact_dir)
            rank = int(getattr(self.config, "device", 0))
            if not getattr(self.config, "is_ref_device", True):
                rank_suffix = f"_rank{rank}"

        def _format_list(values: List[str]) -> str:
            return "\n".join(values)

        def _format_probs(values: List[float]) -> str:
            return ", ".join(f"{v:.3f}" for v in values)

        def _format_all_labels(row: dict) -> Tuple[str, str, str]:
            all_ids = row.get("explicit_all_ids") or row.get("ground_truth_ids", [])
            all_texts = row.get("explicit_all_texts") or row.get("ground_truth_texts", [])
            all_probs = row.get("explicit_all_probs") or row.get("ground_truth_probs", [])
            gt_ids = set(row.get("ground_truth_ids", []))
            neg_ids = set(row.get("explicit_neg_ids", []))
            formatted_texts: List[str] = []
            formatted_ids: List[str] = []
            for idx, text in enumerate(all_texts):
                if idx < len(all_ids):
                    text_id = all_ids[idx]
                    prefix = "-" if text_id in neg_ids else "+" if text_id in gt_ids else "?"
                else:
                    text_id = ""
                    prefix = "+" if text in row.get("ground_truth_texts", []) else "?"
                formatted_texts.append(f"{prefix} {text}")
                if text_id:
                    formatted_ids.append(f"{prefix} {text_id}")
            return _format_list(formatted_texts), _format_list(formatted_ids), _format_probs(all_probs)

        table_samples = []
        categories = [("best", best_indices[:3]), ("worst", worst_indices[:3])]
        if random_index is not None and 0 <= random_index < len(rows):
            categories.append(("random", [(random_index, rows[random_index]["alignment"]) ]))

        for label, entries in categories:
            for idx, score in entries:
                if 0 <= idx < len(rows):
                    row = rows[idx]
                    all_labels_str, all_ids_str, all_probs_str = _format_all_labels(row)
                    table_samples.append(
                        {
                            "category": label,
                            "ecg_id": row["ecg_id"],
                            "alignment": score,
                            "ground_truth_pos": _format_list(row.get("ground_truth_texts", [])),
                            "ground_truth_pos_probs": _format_probs(row.get("ground_truth_probs", [])),
                            "ground_truth_neg": _format_list(row.get("explicit_neg_texts", [])),
                            "ground_truth_neg_probs": _format_probs(row.get("explicit_neg_probs", [])),
                            "ground_truth_all_ids": all_ids_str,
                            "ground_truth_all": all_labels_str,
                            "ground_truth_all_probs": all_probs_str,
                            "top_pred": _format_list(row.get("top_pred_texts", [])),
                            "top_pred_probs": _format_probs(row.get("top_pred_probs", [])),
                        }
                    )

        if artifact_dir:
            df_rows = []
            for row in rows:
                all_labels_str, all_ids_str, all_probs_str = _format_all_labels(row)
                df_rows.append(
                    {
                        "ecg_id": row["ecg_id"],
                        "alignment": row["alignment"],
                        "ground_truth_pos_texts": _format_list(row.get("ground_truth_texts", [])),
                        "ground_truth_pos_probs": _format_probs(row.get("ground_truth_probs", [])),
                        "ground_truth_neg_texts": _format_list(row.get("explicit_neg_texts", [])),
                        "ground_truth_neg_probs": _format_probs(row.get("explicit_neg_probs", [])),
                        "ground_truth_all_ids": all_ids_str,
                        "ground_truth_all_texts": all_labels_str,
                        "ground_truth_all_probs": all_probs_str,
                        "top_pred_texts": _format_list(row.get("top_pred_texts", [])),
                        "top_pred_probs": _format_probs(row.get("top_pred_probs", [])),
                    }
                )
            df = pd.DataFrame(df_rows)
            csv_path = os.path.join(artifact_dir, f"val_epoch_{epoch}_retrieval{rank_suffix}.csv")
            df.to_csv(csv_path, index=False)
            print(f"[SigLIP] Saved retrieval CSV: {csv_path}")

        if (
            table_samples
            and self.wandb_wrapper
            and self.wandb_wrapper.is_initialized()
            and getattr(self.config, "is_ref_device", True)
        ):
            table = wandb.Table(columns=[
                "category",
                "ecg_id",
                "alignment",
                "ground_truth_pos",
                "ground_truth_pos_probs",
                "ground_truth_neg",
                "ground_truth_neg_probs",
                "ground_truth_all_ids",
                "ground_truth_all",
                "ground_truth_all_probs",
                "top_pred",
                "top_pred_probs",
            ])
            for sample in table_samples:
                table.add_data(
                    sample["category"],
                    sample["ecg_id"],
                    f"{sample['alignment']:.4f}",
                    sample["ground_truth_pos"],
                    sample["ground_truth_pos_probs"],
                    sample["ground_truth_neg"],
                    sample["ground_truth_neg_probs"],
                    sample["ground_truth_all_ids"],
                    sample["ground_truth_all"],
                    sample["ground_truth_all_probs"],
                    sample["top_pred"],
                    sample["top_pred_probs"],
                )
            self.wandb_wrapper.log({f"val/retrieval_samples_epoch_{epoch}": table})
