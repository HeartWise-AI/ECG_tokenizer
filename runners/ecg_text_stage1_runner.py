from __future__ import annotations

import math
import os
import random
from contextlib import nullcontext
from typing import Any, Dict, List, Optional, Tuple, Sequence, Set, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast
from torch.cuda.amp import GradScaler
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import csv
from transformers import PreTrainedTokenizerBase
from sklearn.metrics import roc_auc_score
import torch.distributed as dist

try:
    from torch.distributed.nn import functional as dist_nn_f
except (ImportError, AttributeError):  # pragma: no cover - optional dependency
    dist_nn_f = None

from runners.base_runner import BaseRunner
from utils.enums import RunMode, RunnerName
from utils.registry import RunnerRegistry
from utils.config.ecg_text_stage1_config import ECGTextStage1Config
from utils.metrics.siglip_metrics import compute_recall_at_many
from utils.debug import ensure_dir

# dataset_generation now lives in a separate repo (/volume/ECG_Dataset_QA).
# Ensure it is importable regardless of PYTHONPATH.
import sys as _sys
if "/volume/ECG_Dataset_QA" not in _sys.path:
    _sys.path.insert(0, "/volume/ECG_Dataset_QA")
from dataset_generation.siglip_shared import SIGLIP_TARGETED_HARD_NEGATIVES


@RunnerRegistry.register(RunnerName.ECG_TEXT_STAGE1)
class ECGTextStage1Runner(BaseRunner):
    """Runner implementing Stage-1 ECG ↔ text alignment losses (ETC, ETM, ETG)."""

    def __init__(
        self,
        encoder: nn.Module,
        quantizer: nn.Module,
        bridge: nn.Module,
        text_encoder: nn.Module,
        train_dataloader: DataLoader,
        optimizer: Optimizer,
        config: ECGTextStage1Config,
        wandb_wrapper=None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
        validation_dataloader: Optional[DataLoader] = None,
        text_bank: Optional[Tuple[List[str], List[str], torch.Tensor]] = None,
    ) -> None:
        super().__init__(config=config, wandb_wrapper=wandb_wrapper)
        self.encoder = encoder
        self.quantizer = quantizer
        self.bridge = bridge
        self.text_encoder = text_encoder
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.config = config
        self.best_checkpoint_path: Optional[str] = None
        self.last_checkpoint_path: Optional[str] = None
        self.tail_enable = bool(getattr(config, "tail_enable", False))
        tail_ids = getattr(config, "tail_class_ids", None) or []
        # Keep tail IDs for metric reporting even when weighting is disabled.
        self.tail_class_ids: Set[str] = {str(text_id) for text_id in tail_ids if text_id}

        self.bridge.train()

        txt_lr_mult = float(getattr(config, "text_encoder_lr_mult", 0.0) or 0.0)
        if txt_lr_mult == 0.0:
            self.text_encoder.eval()
            for param in self.text_encoder.parameters():
                param.requires_grad = False
        else:
            self.text_encoder.train()

        self.tokenizer: PreTrainedTokenizerBase = getattr(self.text_encoder, "tokenizer")
        if self.tokenizer is None:
            raise ValueError("Text encoder must expose a tokenizer attribute for Stage-1 training.")
        if "[DEC]" not in self.tokenizer.get_vocab():
            raise ValueError("[DEC] must exist in the tokenizer vocabulary for Stage-1 training.")
        self.dec_token_id = int(self.tokenizer.convert_tokens_to_ids("[DEC]"))

        self.device = torch.device("cuda", config.device) if torch.cuda.is_available() else torch.device("cpu")
        self.bridge.to(self.device)
        self.text_encoder.to(self.device)
        self.encoder.to(self.device).eval()
        self.quantizer.to(self.device).eval()

        for param in self.encoder.parameters():
            param.requires_grad = False
        for param in self.quantizer.parameters():
            param.requires_grad = False

        dtype_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
        }
        self.amp_dtype = dtype_map.get(str(config.dtype).lower(), None)
        self.use_autocast = self.device.type == "cuda" and self.amp_dtype is not None
        self.scaler: Optional[GradScaler]
        if self.amp_dtype == torch.float16 and self.device.type == "cuda":
            self.scaler = GradScaler()
        else:
            self.scaler = None

        self.bank_negative_samples = int(max(0, getattr(config, "siglip_bank_negatives", 0)))
        self.text_bank_ids: List[str] = []
        self.text_bank_texts: List[str] = []
        self.text_bank_embeddings: Optional[torch.Tensor] = None
        self.text_bank_index: Dict[str, int] = {}
        self.bank_indices_tensor: Optional[torch.Tensor] = None
        refresh_steps_cfg = getattr(config, "bank_refresh_every_steps", None)
        if refresh_steps_cfg is None:
            refresh_steps_cfg = 256
        self.bank_refresh_every_steps = max(0, int(refresh_steps_cfg))
        if self.bank_negative_samples > 0 and self.bank_refresh_every_steps == 0:
            self.bank_refresh_every_steps = 256
        self.bank_refresh_batch_size = max(1, int(getattr(config, "bank_refresh_batch_size", 2048)))
        self._last_bank_refresh_step = -1
        self.text_lookup: Dict[str, str] = {}
        self._text_id_set: Set[str] = set()
        self._rhythm_candidates: List[str] = []
        self._family_hard_negs: Dict[str, List[str]] = {}
        self._max_hardneg_per_group = int(getattr(config, "max_hardneg_per_group", 3) or 3)
        self.tail_alpha_boost = float(getattr(config, "tail_alpha_boost", 1.0) or 1.0)
        self._hardneg_rng = random.Random(int(self.config.seed) * 1231 + int(self.config.device))
        if text_bank is not None:
            bank_ids, bank_texts, bank_embeddings = text_bank
            if bank_embeddings is not None and bank_embeddings.numel() > 0:
                self.text_bank_ids = list(bank_ids)
                self.text_bank_texts = list(bank_texts)
                bank_tensor = bank_embeddings.to(self.device)
                self.text_bank_embeddings = self._cast_bank_embeddings(bank_tensor)
                self.text_bank_index = {tid: idx for idx, tid in enumerate(self.text_bank_ids)}
                self.bank_indices_tensor = torch.arange(self.text_bank_embeddings.size(0), device=self.device, dtype=torch.long)
                self.text_lookup = {tid: txt for tid, txt in zip(self.text_bank_ids, self.text_bank_texts)}
            else:
                self.bank_negative_samples = 0
        else:
            self.bank_negative_samples = 0
            self.bank_refresh_every_steps = 0
        self._update_text_bank_metadata()

        self._neg_sampler = random.Random(int(self.config.seed) * 997 + int(self.config.device))

        self.grad_accum = max(1, int(config.gradient_accumulation_steps))
        self.grad_clip = 1.0
        self.max_text_length = int(config.max_text_length)
        self.etc_weight = float(config.etc_weight)
        self.etm_weight = float(config.etm_weight)
        self.etg_weight = float(config.etg_weight)
        self.etg_delay_steps = max(0, int(getattr(config, "etg_delay_steps", 0)))
        self.etg_warmup_steps = max(0, int(getattr(config, "etg_warmup_steps", 0)))
        self.etm_hard_neg_k = max(0, int(getattr(config, "etm_hard_neg_k", 1)))
        self.log_every_steps = max(1, int(config.log_every_steps))
        self.checkpoint_every = max(1, int(config.checkpoint_every))
        self.etm_warmup_steps = max(0, int(getattr(config, "etm_warmup_steps", 0)))

        self.use_focal_infonce = bool(getattr(config, "focal_infonce", False))
        self.focal_gamma_pos = float(getattr(config, "focal_gamma_pos", 0.0) or 0.0)
        self.focal_gamma_neg = float(getattr(config, "focal_gamma_neg", 0.0) or 0.0)
        self.focal_alpha_default = float(getattr(config, "focal_alpha_default", 1.0) or 1.0)
        self.focal_detach_weights = bool(getattr(config, "focal_detach_weights", False))
        self.class_pos_weight_map = dict(getattr(config, "class_pos_weight_map", {}) or {})

        self._logit_scale_warn_threshold = float(getattr(config, "logit_scale_warn_threshold", 100.0) or 100.0)
        self._logit_scale_warned = False

        self.global_step = 0

        if self.tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer used for Stage-1 runner must define pad_token_id.")

    # ------------------------------------------------------------------
    def train(self) -> None:  # type: ignore[override]
        total_epochs = int(self.config.num_epochs)
        if total_epochs <= 0:
            raise ValueError("num_epochs must be positive for Stage-1 training.")

        best_loss = float("inf")
        last_metric_for_selection = float("inf")
        last_epoch = -1
        for epoch in range(total_epochs):
            metrics, _ = self._run_epoch(
                mode=RunMode.TRAIN,
                epoch=epoch,
                dataloader=self.train_dataloader,
                step_fn=self._train_step,
                generated_samples=[],
            )
            if self.scheduler is not None:
                self.scheduler.step()

            epoch_log = {f"train_epoch/{k}": v for k, v in metrics.items()}
            epoch_log["trainer/epoch"] = float(epoch)
            self._log_metrics(epoch_log)

            if self.config.is_ref_device:
                train_msg = (
                    f"[Stage1][Epoch {epoch + 1}/{total_epochs}] "
                    f"loss={metrics['loss']:.4f} "
                    f"etc={metrics['loss_etc']:.4f} "
                    f"etm={metrics['loss_etm']:.4f} "
                    f"etg={metrics['loss_etg']:.4f} "
                    f"etm_acc={metrics['etm_acc']:.3f}"
                )
                if "siglip_recall_at_1" in metrics:
                    train_msg += f" r@1={metrics['siglip_recall_at_1']:.4f}"
                if "siglip_recall_at_5" in metrics:
                    train_msg += f" r@5={metrics['siglip_recall_at_5']:.4f}"
                if "siglip_tail_recall_at_5" in metrics:
                    train_msg += f" tail@5={metrics['siglip_tail_recall_at_5']:.4f}"
                if "etm_auroc" in metrics:
                    train_msg += f" etm_auc={metrics['etm_auroc']:.4f}"
                if "etg_next_acc" in metrics:
                    train_msg += f" next={metrics['etg_next_acc']:.3f}"
                if "etg_copy_rate" in metrics:
                    train_msg += f" copy={metrics['etg_copy_rate']:.3f}"
                print(
                    train_msg
                )

            val_metrics, val_samples = self._run_validation_epoch(epoch)
            metric_for_selection = metrics["loss"]
            if val_metrics is not None:
                metric_for_selection = val_metrics["loss"]
                val_log = {f"val_epoch/{k}": v for k, v in val_metrics.items()}
                val_log["trainer/epoch"] = float(epoch)
                self._log_metrics(val_log)
                if self.config.is_ref_device:
                    val_msg = (
                        f"[Stage1][Val {epoch + 1}/{total_epochs}] "
                        f"loss={val_metrics['loss']:.4f} "
                        f"etc={val_metrics['loss_etc']:.4f} "
                        f"etm={val_metrics['loss_etm']:.4f} "
                        f"etg={val_metrics['loss_etg']:.4f} "
                        f"etm_acc={val_metrics['etm_acc']:.3f}"
                    )
                    if "alignment_score" in val_metrics:
                        val_msg += f" align={val_metrics['alignment_score']:.4f}"
                    if "siglip_recall_at_1" in val_metrics:
                        val_msg += f" r@1={val_metrics['siglip_recall_at_1']:.4f}"
                    if "siglip_recall_at_5" in val_metrics:
                        val_msg += f" r@5={val_metrics['siglip_recall_at_5']:.4f}"
                    if "siglip_tail_recall_at_5" in val_metrics:
                        val_msg += f" tail@5={val_metrics['siglip_tail_recall_at_5']:.4f}"
                    if "etm_auroc" in val_metrics:
                        val_msg += f" etm_auc={val_metrics['etm_auroc']:.4f}"
                    if "etg_next_acc" in val_metrics:
                        val_msg += f" next={val_metrics['etg_next_acc']:.3f}"
                    if "etg_copy_rate" in val_metrics:
                        val_msg += f" copy={val_metrics['etg_copy_rate']:.3f}"
                    print(
                        val_msg
                    )
                self._log_metrics({f"val/{k}": v for k, v in val_metrics.items()})
                if val_samples:
                    preview = []
                    for item in val_samples[:5]:
                        if isinstance(item, dict):
                            preview.append((item.get("reference", ""), item.get("prediction", "")))
                        else:
                            preview.append(item)
                    self._log_text_table(preview, key="val/sample_texts")

            if metric_for_selection < best_loss:
                best_loss = metric_for_selection
                self._maybe_save_checkpoint(
                    epoch,
                    metric_for_selection,
                    samples=val_samples,
                    metrics=val_metrics,
                    best_only=True,
                )

            # Save last checkpoint every epoch (not just at the end) to handle interruptions
            self._maybe_save_checkpoint(
                epoch,
                metric_for_selection,
                samples=val_samples,
                metrics=val_metrics,
                best_only=False,
            )

            last_metric_for_selection = metric_for_selection
            last_epoch = epoch

        if self.best_checkpoint_path is not None and self.config.is_ref_device:
            print(f"[Stage1] Best checkpoint: {self.best_checkpoint_path} (loss={best_loss:.4f})")

        if last_epoch >= 0:
            self._maybe_save_checkpoint(
                last_epoch,
                last_metric_for_selection,
                samples=val_samples,
                metrics=val_metrics,
                best_only=False,
            )

    # ------------------------------------------------------------------
    def inference(self) -> None:  # type: ignore[override]
        raise NotImplementedError("Inference mode is not implemented for Stage-1 runner.")

    # ------------------------------------------------------------------
    def _run_epoch(
        self,
        mode: RunMode,
        epoch: int,
        dataloader: DataLoader,
        step_fn,
        generated_samples: List[Tuple[str, str]],
    ) -> Tuple[Dict[str, float], List[Tuple[str, str]]]:
        assert mode == RunMode.TRAIN, "Stage-1 runner currently supports training mode only."
        self.bridge.train()

        total_steps = 0
        sum_loss = 0.0
        sum_etc = 0.0
        sum_etm = 0.0
        sum_etg = 0.0
        sum_acc = 0.0
        sum_pos_prob = 0.0
        sum_neg_prob = 0.0
        sum_candidate_count = 0.0
        sum_candidate_vocab = 0.0
        sum_candidate_count = 0.0
        sum_candidate_vocab = 0.0
        recall_hits = {1: 0.0, 5: 0.0}  # fallback
        recall_counts = {1: 0, 5: 0}  # fallback
        recall_hits_weighted = {1: 0.0, 5: 0.0}
        recall_weight_total = 0.0
        track_tail = bool(self.tail_class_ids)
        tail_hits = {1: 0.0, 5: 0.0} if track_tail else None  # fallback
        tail_counts = {1: 0, 5: 0} if track_tail else None  # fallback
        tail_hits_weighted = {1: 0.0, 5: 0.0} if track_tail else None
        tail_weight_total = 0.0
        etm_auroc_sum = 0.0
        etm_auroc_count = 0
        etg_next_sum = 0.0
        etg_next_count = 0
        etg_copy_sum = 0.0
        etg_copy_count = 0
        siglip_stats_keys = [
            "siglip_pos_mass",
            "siglip_neg_mass",
            "siglip_candidate_count",
            "siglip_duplicate_fraction",
            "siglip_random_r1",
            "siglip_positive_prob",
            "siglip_loss",
            "siglip_loss_raw",
            "siglip_loss_mismatch",
            "logit_scale",
            "temperature",
        ]
        siglip_weighted_totals = {key: 0.0 for key in siglip_stats_keys}
        siglip_weighted_counts = {key: 0.0 for key in siglip_stats_keys}
        siglip_fallback_totals = {key: 0.0 for key in siglip_stats_keys}
        siglip_fallback_counts = {key: 0 for key in siglip_stats_keys}

        self.optimizer.zero_grad(set_to_none=True)

        total_batches = len(dataloader)
        progress_iter = enumerate(dataloader)
        progress = tqdm(
            progress_iter,
            total=total_batches,
            desc=f"Epoch {epoch + 1}",
            leave=False,
            disable=not self.config.is_ref_device,
        )
        for step_idx, batch in progress:
            metrics = step_fn(batch, step_idx, step_idx == total_batches - 1, generated_samples)
            total_steps += 1
            sum_loss += metrics["loss"]
            sum_etc += metrics["loss_etc"]
            sum_etm += metrics["loss_etm"]
            sum_etg += metrics["loss_etg"]
            sum_acc += metrics["etm_acc"]
            sum_pos_prob += metrics.get("siglip_pos_mass", 0.0)
            sum_neg_prob += metrics.get("siglip_neg_mass", 0.0)
            sum_candidate_count += metrics.get("siglip_candidate_count", 0.0)
            sum_candidate_vocab += float(metrics.get("candidate_vocab", len(self.text_bank_ids)))
            weight = float(metrics.get("siglip_batch_size", 0.0))
            if weight > 0:
                recall_weight_total += weight
                for k in (1, 5):
                    hits_key = f"siglip_recall_at_{k}_hits"
                    if hits_key in metrics:
                        recall_hits_weighted[k] += float(metrics[hits_key])
                    else:
                        recall_hits_weighted[k] += float(metrics.get(f"siglip_recall_at_{k}", 0.0) * weight)
                if track_tail and tail_hits_weighted is not None:
                    tail_weight = float(metrics.get("tail_recall_count", 0.0))
                    if tail_weight > 0:
                        tail_weight_total += tail_weight
                        tail_hits_weighted[1] += float(
                            metrics.get("tail_recall_at_1_hits", metrics.get("siglip_tail_recall_at_1", 0.0) * tail_weight)
                        )
                        tail_hits_weighted[5] += float(
                            metrics.get("tail_recall_at_5_hits", metrics.get("siglip_tail_recall_at_5", 0.0) * tail_weight)
                        )
                for key in siglip_stats_keys:
                    if key in metrics:
                        siglip_weighted_totals[key] += float(metrics[key]) * weight
                        siglip_weighted_counts[key] += weight
            else:
                for k in (1, 5):
                    siglip_key = f"siglip_recall_at_{k}"
                    if siglip_key in metrics:
                        recall_hits[k] += float(metrics[siglip_key])
                        recall_counts[k] += 1
                        continue
                    recall_key = f"recall_at_{k}"
                    recall_count_key = f"{recall_key}_count"
                    count_value = metrics.get(recall_count_key, 0)
                    if count_value:
                        recall_hits[k] += metrics.get(recall_key, 0.0) * float(count_value)
                        recall_counts[k] += int(count_value)
                    if track_tail and tail_hits is not None and tail_counts is not None:
                        tail_key = f"tail_recall_at_{k}"
                        tail_count_key = f"{tail_key}_count"
                        tail_count_value = metrics.get(tail_count_key, 0)
                        if tail_count_value:
                            tail_hits[k] += metrics.get(tail_key, 0.0) * float(tail_count_value)
                            tail_counts[k] += int(tail_count_value)
                for key in siglip_stats_keys:
                    if key in metrics:
                        siglip_fallback_totals[key] += float(metrics[key])
                        siglip_fallback_counts[key] += 1
            etm_auroc_value = metrics.get("etm_auroc")
            if etm_auroc_value is not None and not math.isnan(etm_auroc_value):
                etm_auroc_sum += float(etm_auroc_value)
                etm_auroc_count += 1
            etg_next_value = metrics.get("etg_next_acc")
            if etg_next_value is not None and not math.isnan(etg_next_value):
                etg_next_sum += float(etg_next_value)
                etg_next_count += 1
            etg_copy_value = metrics.get("etg_copy_rate")
            if etg_copy_value is not None and not math.isnan(etg_copy_value):
                etg_copy_sum += float(etg_copy_value)
                etg_copy_count += 1

            if self.config.is_ref_device:
                progress.set_postfix({
                    "loss": f"{metrics['loss']:.4f}",
                    "ETC": f"{metrics['loss_etc']:.4f}",
                    "ETM": f"{metrics['loss_etm']:.4f}",
                    "ETG": f"{metrics['loss_etg']:.4f}",
                })

            if self.global_step % self.log_every_steps == 0:
                payload: Dict[str, float] = {
                    "train/epoch": epoch,
                    "train/loss": metrics["loss"],
                    "train/loss_etc": metrics["loss_etc"],
                    "train/loss_etm": metrics["loss_etm"],
                    "train/loss_etg": metrics["loss_etg"],
                    "train/etm_acc": metrics["etm_acc"],
                    "train/logit_scale": metrics["logit_scale"],
                    "train/etg_weight_eff": metrics["etg_weight_eff"],
                    "train/etm_weight_eff": metrics.get("etm_weight_eff", self.etm_weight),
                }
                if "temperature" in metrics:
                    payload["train/temperature"] = metrics["temperature"]
                pos_mass = metrics.get("siglip_pos_mass")
                neg_mass = metrics.get("siglip_neg_mass")
                if pos_mass is not None:
                    payload["train/siglip_pos_mass"] = float(pos_mass)
                if neg_mass is not None:
                    payload["train/siglip_neg_mass"] = float(neg_mass)
                if "siglip_candidate_count" in metrics:
                    payload["train/siglip_candidate_count"] = metrics["siglip_candidate_count"]
                if "siglip_duplicate_fraction" in metrics:
                    payload["train/siglip_duplicate_fraction"] = metrics["siglip_duplicate_fraction"]
                if "siglip_positive_prob" in metrics:
                    payload["train/siglip_positive_prob"] = metrics["siglip_positive_prob"]
                for k in (1, 5):
                    siglip_key = f"siglip_recall_at_{k}"
                    if siglip_key in metrics:
                        payload[f"train/{siglip_key}"] = metrics[siglip_key]
                tail_weight = metrics.get("tail_recall_count", 0.0)
                if tail_weight:
                    payload["train/siglip_tail_recall_at_1"] = metrics.get("siglip_tail_recall_at_1", 0.0)
                    payload["train/siglip_tail_recall_at_5"] = metrics.get("siglip_tail_recall_at_5", 0.0)
                if etm_auroc_value is not None and not math.isnan(etm_auroc_value):
                    payload["train/etm_auroc"] = float(etm_auroc_value)
                if etg_next_value is not None and not math.isnan(etg_next_value):
                    payload["train/etg_next_acc"] = float(etg_next_value)
                if etg_copy_value is not None and not math.isnan(etg_copy_value):
                    payload["train/etg_copy_rate"] = float(etg_copy_value)
                if "siglip_pos_mass" in metrics:
                    payload["train/siglip_pos_mass"] = metrics["siglip_pos_mass"]
                if "siglip_neg_mass" in metrics:
                    payload["train/siglip_neg_mass"] = metrics["siglip_neg_mass"]
                if "siglip_candidate_count" in metrics:
                    payload["train/siglip_candidate_count"] = metrics["siglip_candidate_count"]
                if "siglip_duplicate_fraction" in metrics:
                    payload["train/siglip_duplicate_fraction"] = metrics["siglip_duplicate_fraction"]
                if "siglip_random_r1" in metrics:
                    payload["train/siglip_random_r1"] = metrics["siglip_random_r1"]
                if "siglip_positive_prob" in metrics:
                    payload["train/siglip_positive_prob"] = metrics["siglip_positive_prob"]
                if "siglip_loss" in metrics:
                    payload["train/siglip_loss"] = metrics["siglip_loss"]
                self._log_metrics(payload)

        if total_steps == 0:
            raise RuntimeError("Training dataloader yielded zero batches.")

        if self._is_distributed():
            reduce_device = self.device if self.device.type == "cuda" else torch.device("cpu")
            recall_vec = torch.tensor(
                [recall_hits_weighted[1], recall_hits_weighted[5], recall_weight_total],
                device=reduce_device,
                dtype=torch.float32,
            )
            dist.all_reduce(recall_vec, op=dist.ReduceOp.SUM)
            recall_hits_weighted[1], recall_hits_weighted[5], recall_weight_total = recall_vec.tolist()

            if track_tail and tail_hits_weighted is not None:
                tail_vec = torch.tensor(
                    [tail_hits_weighted[1], tail_hits_weighted[5], tail_weight_total],
                    device=reduce_device,
                    dtype=torch.float32,
                )
                dist.all_reduce(tail_vec, op=dist.ReduceOp.SUM)
                tail_hits_weighted[1], tail_hits_weighted[5], tail_weight_total = tail_vec.tolist()

            if siglip_weighted_totals:
                stat_keys = list(siglip_weighted_totals.keys())
                totals_tensor = torch.tensor(
                    [siglip_weighted_totals[key] for key in stat_keys],
                    device=reduce_device,
                    dtype=torch.float32,
                )
                counts_tensor = torch.tensor(
                    [siglip_weighted_counts[key] for key in stat_keys],
                    device=reduce_device,
                    dtype=torch.float32,
                )
                dist.all_reduce(totals_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
                for idx, key in enumerate(stat_keys):
                    siglip_weighted_totals[key] = float(totals_tensor[idx].item())
                    siglip_weighted_counts[key] = float(counts_tensor[idx].item())

        epoch_metrics = {
            "loss": sum_loss / total_steps,
            "loss_etc": sum_etc / total_steps,
            "loss_etm": sum_etm / total_steps,
            "loss_etg": sum_etg / total_steps,
            "etm_acc": sum_acc / total_steps,
        }
        if recall_weight_total > 0:
            epoch_metrics["siglip_recall_at_1"] = recall_hits_weighted[1] / recall_weight_total
            epoch_metrics["siglip_recall_at_5"] = recall_hits_weighted[5] / recall_weight_total
        else:
            if recall_counts[1] > 0:
                epoch_metrics["siglip_recall_at_1"] = recall_hits[1] / recall_counts[1]
            if recall_counts[5] > 0:
                epoch_metrics["siglip_recall_at_5"] = recall_hits[5] / recall_counts[5]
        if track_tail:
            if tail_weight_total > 0 and tail_hits_weighted is not None:
                epoch_metrics["siglip_tail_recall_at_1"] = tail_hits_weighted[1] / tail_weight_total
                epoch_metrics["siglip_tail_recall_at_5"] = tail_hits_weighted[5] / tail_weight_total
            elif tail_counts and tail_counts[1] > 0:
                epoch_metrics["siglip_tail_recall_at_1"] = tail_hits[1] / tail_counts[1]
            if track_tail and tail_counts and tail_counts[5] > 0 and "siglip_tail_recall_at_5" not in epoch_metrics:
                epoch_metrics["siglip_tail_recall_at_5"] = tail_hits[5] / tail_counts[5]
        if etm_auroc_count > 0:
            epoch_metrics["etm_auroc"] = etm_auroc_sum / float(etm_auroc_count)
        if etg_next_count > 0:
            epoch_metrics["etg_next_acc"] = etg_next_sum / float(etg_next_count)
        if etg_copy_count > 0:
            epoch_metrics["etg_copy_rate"] = etg_copy_sum / float(etg_copy_count)
        for key in siglip_stats_keys:
            weighted_total = siglip_weighted_totals[key]
            weighted_count = siglip_weighted_counts[key]
            fallback_total = siglip_fallback_totals[key]
            fallback_count = siglip_fallback_counts[key]
            if weighted_count > 0:
                epoch_metrics[key] = weighted_total / weighted_count
            elif fallback_count > 0:
                epoch_metrics[key] = fallback_total / float(fallback_count)

        return epoch_metrics, generated_samples

    def _run_validation_epoch(
        self,
        epoch: int,
    ) -> Tuple[Optional[Dict[str, float]], List[Dict[str, Any]]]:
        if self.validation_dataloader is None:
            return None, []

        self.bridge.eval()
        self.text_encoder.eval()

        total_steps = 0
        sum_loss = 0.0
        sum_etc = 0.0
        sum_etm = 0.0
        sum_etg = 0.0
        sum_acc = 0.0
        recall_hits = {1: 0.0, 5: 0.0}  # fallback
        recall_counts = {1: 0, 5: 0}  # fallback
        recall_hits_weighted = {1: 0.0, 5: 0.0}
        recall_weight_total = 0.0
        track_tail = bool(self.tail_class_ids)
        tail_hits = {1: 0.0, 5: 0.0} if track_tail else None  # fallback
        tail_counts = {1: 0, 5: 0} if track_tail else None  # fallback
        tail_hits_weighted = {1: 0.0, 5: 0.0} if track_tail else None
        tail_weight_total = 0.0
        etm_auroc_sum = 0.0
        etm_auroc_count = 0
        etg_next_sum = 0.0
        etg_next_count = 0
        etg_copy_sum = 0.0
        etg_copy_count = 0
        sum_pos_prob = 0.0
        sum_neg_prob = 0.0
        sum_candidate_count = 0.0
        sum_candidate_vocab = 0.0
        siglip_stats_keys = [
            "siglip_pos_mass",
            "siglip_neg_mass",
            "siglip_candidate_count",
            "siglip_duplicate_fraction",
            "siglip_random_r1",
            "siglip_positive_prob",
            "siglip_loss",
            "siglip_loss_raw",
            "siglip_loss_mismatch",
            "logit_scale",
        ]
        siglip_weighted_totals = {key: 0.0 for key in siglip_stats_keys}
        siglip_weighted_counts = {key: 0.0 for key in siglip_stats_keys}
        siglip_fallback_totals = {key: 0.0 for key in siglip_stats_keys}
        siglip_fallback_counts = {key: 0 for key in siglip_stats_keys}
        collected: List[Dict[str, Any]] = []
        tail_accum: Dict[str, Any] = {} if track_tail else {}

        progress_iter = enumerate(self.validation_dataloader)
        progress = tqdm(
            progress_iter,
            total=len(self.validation_dataloader),
            desc=f"Val {epoch + 1}",
            leave=False,
            disable=not self.config.is_ref_device,
        )

        with torch.no_grad():
            for step_idx, batch in progress:
                metrics, samples, tail_info = self._validation_step(batch)
                total_steps += 1
                sum_loss += metrics["loss"]
                sum_etc += metrics["loss_etc"]
                sum_etm += metrics["loss_etm"]
                sum_etg += metrics["loss_etg"]
                sum_acc += metrics["etm_acc"]
                sum_pos_prob += metrics.get("siglip_pos_mass", 0.0)
                sum_neg_prob += metrics.get("siglip_neg_mass", 0.0)
                sum_candidate_count += metrics.get("siglip_candidate_count", 0.0)
                sum_candidate_vocab += float(metrics.get("candidate_vocab", len(self.text_bank_ids)))
                if track_tail and tail_info:
                    labels_set = tail_accum.setdefault("labels_with_pos", set())
                    labels_set.update(tail_info.get("labels_with_pos", set()))
                    tail_accum["total_pos_examples"] = int(tail_accum.get("total_pos_examples", 0)) + int(
                        tail_info.get("total_pos_examples", 0)
                    )
                    for key in ("tail_hits@1", "tail_total@1", "tail_hits@5", "tail_total@5"):
                        tail_accum[key] = int(tail_accum.get(key, 0)) + int(tail_info.get(key, 0))
                    for key in ("tail_hits_per_label@1", "tail_hits_per_label@5"):
                        per_label_global = tail_accum.setdefault(key, {})
                        for tid, (hits, total) in tail_info.get(key, {}).items():
                            ghits, gtotal = per_label_global.get(tid, (0, 0))
                            per_label_global[tid] = (ghits + hits, gtotal + total)
                    availability_global = tail_accum.setdefault("tail_availability", {})
                    for tid, (avail_pos, avail_total) in tail_info.get("tail_availability", {}).items():
                        gpos, gtotal = availability_global.get(tid, (0, 0))
                        availability_global[tid] = (gpos + avail_pos, gtotal + avail_total)
                weight = float(metrics.get("siglip_batch_size", 0.0))
                if weight > 0:
                    recall_weight_total += weight
                    for k in (1, 5):
                        hits_key = f"siglip_recall_at_{k}_hits"
                        if hits_key in metrics:
                            recall_hits_weighted[k] += float(metrics[hits_key])
                        else:
                            recall_hits_weighted[k] += float(metrics.get(f"siglip_recall_at_{k}", 0.0) * weight)
                    if track_tail and tail_hits_weighted is not None:
                        tail_weight = float(metrics.get("tail_recall_count", 0.0))
                        if tail_weight > 0:
                            tail_weight_total += tail_weight
                            tail_hits_weighted[1] += float(
                                metrics.get("tail_recall_at_1_hits", metrics.get("siglip_tail_recall_at_1", 0.0) * tail_weight)
                            )
                            tail_hits_weighted[5] += float(
                                metrics.get("tail_recall_at_5_hits", metrics.get("siglip_tail_recall_at_5", 0.0) * tail_weight)
                            )
                    for key in siglip_stats_keys:
                        if key in metrics:
                            siglip_weighted_totals[key] += float(metrics[key]) * weight
                            siglip_weighted_counts[key] += weight
                else:
                    for k in (1, 5):
                        siglip_key = f"siglip_recall_at_{k}"
                        if siglip_key in metrics:
                            recall_hits[k] += float(metrics[siglip_key])
                            recall_counts[k] += 1
                            continue
                        recall_key = f"recall_at_{k}"
                        recall_count_key = f"{recall_key}_count"
                        count_value = metrics.get(recall_count_key, 0)
                        if count_value:
                            recall_hits[k] += metrics.get(recall_key, 0.0) * float(count_value)
                            recall_counts[k] += int(count_value)
                        if track_tail and tail_hits is not None and tail_counts is not None:
                            tail_key = f"tail_recall_at_{k}"
                            tail_count_key = f"{tail_key}_count"
                            tail_count_value = metrics.get(tail_count_key, 0)
                            if tail_count_value:
                                tail_hits[k] += metrics.get(tail_key, 0.0) * float(tail_count_value)
                                tail_counts[k] += int(tail_count_value)
                    for key in siglip_stats_keys:
                        if key in metrics:
                            siglip_fallback_totals[key] += float(metrics[key])
                            siglip_fallback_counts[key] += 1
                etm_auroc_value = metrics.get("etm_auroc")
                if etm_auroc_value is not None and not math.isnan(etm_auroc_value):
                    etm_auroc_sum += float(etm_auroc_value)
                    etm_auroc_count += 1
                etg_next_value = metrics.get("etg_next_acc")
                if etg_next_value is not None and not math.isnan(etg_next_value):
                    etg_next_sum += float(etg_next_value)
                    etg_next_count += 1
                etg_copy_value = metrics.get("etg_copy_rate")
                if etg_copy_value is not None and not math.isnan(etg_copy_value):
                    etg_copy_sum += float(etg_copy_value)
                    etg_copy_count += 1
                collected.extend(samples)

                if self.config.is_ref_device:
                    progress.set_postfix({
                        "loss": f"{metrics['loss']:.4f}",
                        "ETC": f"{metrics['loss_etc']:.4f}",
                        "ETM": f"{metrics['loss_etm']:.4f}",
                        "ETG": f"{metrics['loss_etg']:.4f}",
                    })

        self.bridge.train()
        self.text_encoder.train()

        if total_steps == 0:
            return None, collected

        if self._is_distributed():
            reduce_device = self.device if self.device.type == "cuda" else torch.device("cpu")
            recall_vec = torch.tensor(
                [recall_hits_weighted[1], recall_hits_weighted[5], recall_weight_total],
                device=reduce_device,
                dtype=torch.float32,
            )
            dist.all_reduce(recall_vec, op=dist.ReduceOp.SUM)
            recall_hits_weighted[1], recall_hits_weighted[5], recall_weight_total = recall_vec.tolist()

            if track_tail and tail_hits_weighted is not None:
                tail_vec = torch.tensor(
                    [tail_hits_weighted[1], tail_hits_weighted[5], tail_weight_total],
                    device=reduce_device,
                    dtype=torch.float32,
                )
                dist.all_reduce(tail_vec, op=dist.ReduceOp.SUM)
                tail_hits_weighted[1], tail_hits_weighted[5], tail_weight_total = tail_vec.tolist()

            if siglip_weighted_totals:
                stat_keys = list(siglip_weighted_totals.keys())
                totals_tensor = torch.tensor(
                    [siglip_weighted_totals[key] for key in stat_keys],
                    device=reduce_device,
                    dtype=torch.float32,
                )
                counts_tensor = torch.tensor(
                    [siglip_weighted_counts[key] for key in stat_keys],
                    device=reduce_device,
                    dtype=torch.float32,
                )
                dist.all_reduce(totals_tensor, op=dist.ReduceOp.SUM)
                dist.all_reduce(counts_tensor, op=dist.ReduceOp.SUM)
                for idx, key in enumerate(stat_keys):
                    siglip_weighted_totals[key] = float(totals_tensor[idx].item())
                    siglip_weighted_counts[key] = float(counts_tensor[idx].item())

        tail_rows: List[Tuple[str, float, int, int]] = []
        coverage_warnings: List[Tuple[str, float, int, int]] = []
        availability: Dict[str, Tuple[int, int]] = {}

        averages = {
            "loss": sum_loss / total_steps,
            "loss_etc": sum_etc / total_steps,
            "loss_etm": sum_etm / total_steps,
            "loss_etg": sum_etg / total_steps,
            "etm_acc": sum_acc / total_steps,
        }
        if total_steps > 0:
            averages["siglip_pos_mass"] = sum_pos_prob / total_steps
            averages["siglip_neg_mass"] = sum_neg_prob / total_steps
            averages["siglip_candidate_count"] = sum_candidate_count / total_steps
            averages["siglip_candidate_vocab"] = sum_candidate_vocab / total_steps
        if recall_weight_total > 0:
            averages["siglip_recall_at_1"] = recall_hits_weighted[1] / recall_weight_total
            averages["siglip_recall_at_5"] = recall_hits_weighted[5] / recall_weight_total
        else:
            if recall_counts[1] > 0:
                averages["siglip_recall_at_1"] = recall_hits[1] / recall_counts[1]
            if recall_counts[5] > 0:
                averages["siglip_recall_at_5"] = recall_hits[5] / recall_counts[5]
        if track_tail:
            if tail_weight_total > 0 and tail_hits_weighted is not None:
                averages["siglip_tail_recall_at_1"] = tail_hits_weighted[1] / tail_weight_total
                averages["siglip_tail_recall_at_5"] = tail_hits_weighted[5] / tail_weight_total
            elif tail_counts and tail_counts[1] > 0:
                averages["siglip_tail_recall_at_1"] = tail_hits[1] / tail_counts[1]
            if track_tail and tail_counts and tail_counts[5] > 0 and "siglip_tail_recall_at_5" not in averages:
                averages["siglip_tail_recall_at_5"] = tail_hits[5] / tail_counts[5]
            labels_selected = len(self.tail_class_ids)
            labels_with_pos = tail_accum.get("labels_with_pos", set()) if tail_accum else set()
            positives_seen = int(tail_accum.get("total_pos_examples", 0)) if tail_accum else 0
            coverage = float(len(labels_with_pos)) / max(1.0, float(labels_selected)) if labels_selected else 0.0
            averages["tail/labels_with_pos"] = float(len(labels_with_pos))
            averages["tail/labels_selected"] = float(labels_selected)
            averages["tail/coverage"] = float(coverage)
            averages["tail/positives_seen"] = float(positives_seen)
            averages.setdefault("tail/recall@1", 0.0)
            averages.setdefault("tail/recall@5", 0.0)
            tail_hits1_total = int(tail_accum.get("tail_total@1", 0)) if tail_accum else 0
            tail_hits1 = int(tail_accum.get("tail_hits@1", 0)) if tail_accum else 0
            if tail_hits1_total > 0:
                averages["tail/recall@1"] = float(tail_hits1 / tail_hits1_total)
            tail_hits5_total = int(tail_accum.get("tail_total@5", 0)) if tail_accum else 0
            tail_hits5 = int(tail_accum.get("tail_hits@5", 0)) if tail_accum else 0
            if tail_hits5_total > 0:
                averages["tail/recall@5"] = float(tail_hits5 / tail_hits5_total)
            availability = tail_accum.get("tail_availability", {}) if tail_accum else {}
            per_label_5 = tail_accum.get("tail_hits_per_label@5", {}) if tail_accum else {}
            coverage_min = 1.0
            for tid in self.tail_class_ids:
                hits, total = per_label_5.get(tid, (0, 0))
                recall = (hits / total) if total > 0 else 0.0
                tail_rows.append((tid, float(recall), int(hits), int(total)))
                avail_pos, avail_total = availability.get(tid, (0, 0))
                if avail_total > 0:
                    coverage_label = avail_pos / max(avail_total, 1)
                    coverage_min = min(coverage_min, coverage_label)
                    if coverage_label < 0.999:
                        coverage_warnings.append((tid, float(coverage_label), int(avail_pos), int(avail_total)))
            if tail_rows:
                tail_rows.sort(key=lambda item: item[1])
                averages["tail/coverage_min"] = float(coverage_min)
            else:
                averages.setdefault("tail/coverage_min", 0.0)
        else:
            averages.setdefault("tail/labels_selected", float(len(self.tail_class_ids)))
            averages.setdefault("tail/labels_with_pos", 0.0)
            averages.setdefault("tail/coverage", 0.0)
            averages.setdefault("tail/positives_seen", 0.0)
            averages.setdefault("tail/recall@1", 0.0)
            averages.setdefault("tail/recall@5", 0.0)
            averages.setdefault("tail/coverage_min", 0.0)
        if etm_auroc_count > 0:
            averages["etm_auroc"] = etm_auroc_sum / float(etm_auroc_count)
        if etg_next_count > 0:
            averages["etg_next_acc"] = etg_next_sum / float(etg_next_count)
        if etg_copy_count > 0:
            averages["etg_copy_rate"] = etg_copy_sum / float(etg_copy_count)
        for key in siglip_stats_keys:
            weighted_total = siglip_weighted_totals[key]
            weighted_count = siglip_weighted_counts[key]
            fallback_total = siglip_fallback_totals[key]
            fallback_count = siglip_fallback_counts[key]
            if weighted_count > 0:
                averages[key] = weighted_total / weighted_count
            elif fallback_count > 0:
                averages[key] = fallback_total / float(fallback_count)

        if track_tail:
            if getattr(self.config, "is_ref_device", True):
                worst = [entry for entry in tail_rows if entry[3] > 0][:10]
                if worst:
                    print("[Stage1] Tail recall@5 (worst 10):")
                    for tid, recall, hits, total in worst:
                        text = self.text_lookup.get(tid, tid)
                        print(f"   {tid:<40} recall@5={recall:.3f} ({hits}/{total})  '{text[:70] if text else tid}'")
                if coverage_warnings:
                    print("[Stage1] Tail availability gaps detected:")
                    for tid, coverage_value, avail_pos, avail_total in sorted(coverage_warnings, key=lambda x: x[1]):
                        text = self.text_lookup.get(tid, tid)
                        print(
                            f"   {tid:<40} coverage={coverage_value:.3f} ({avail_pos}/{avail_total}) '{text[:70] if text else tid}'"
                        )

            if (
                self.wandb_wrapper
                and self.wandb_wrapper.is_initialized()
                and getattr(self.config, "is_ref_device", True)
                and tail_rows
            ):
                try:
                    import wandb  # type: ignore
                except ImportError:
                    wandb = None  # type: ignore
                if wandb is not None:
                    payload = {
                        "val/tail/recall@1": averages.get("tail/recall@1", 0.0),
                        "val/tail/recall@5": averages.get("tail/recall@5", 0.0),
                        "val/tail/coverage": averages.get("tail/coverage", 0.0),
                        "val/tail/labels_with_pos": averages.get("tail/labels_with_pos", 0.0),
                        "val/tail/labels_selected": averages.get("tail/labels_selected", 0.0),
                        "val/tail/positives_seen": averages.get("tail/positives_seen", 0.0),
                    }
                    if "tail/coverage_min" in averages:
                        payload["val/tail/coverage_min"] = averages.get("tail/coverage_min", 0.0)
                    table = wandb.Table(columns=["text_id", "text", "recall@5", "hits", "total", "coverage"])
                    top_worst = tail_rows[:50]
                    for tid, recall, hits, total in top_worst:
                        avail_pos, avail_total = availability.get(tid, (0, 0))
                        coverage_value = (avail_pos / max(avail_total, 1)) if avail_total > 0 else 1.0
                        table.add_data(
                            tid,
                            self.text_lookup.get(tid, tid),
                            float(recall),
                            int(hits),
                            int(total),
                            float(coverage_value),
                        )
                    payload["val/tail/recall_table@5_top50worst"] = table
                    self.wandb_wrapper.log(payload)

            if (
                getattr(self.config, "output_dir", None)
                and getattr(self.config, "is_ref_device", True)
                and tail_rows
            ):
                artifact_dir = os.path.join(self.config.output_dir, "artifacts")
                ensure_dir(artifact_dir)
                rank = int(getattr(self.config, "device", 0))
                csv_name = f"val_epoch_{epoch + 1}_tail_recall_at5_rank{rank}.csv"
                csv_path = os.path.join(artifact_dir, csv_name)
                with open(csv_path, "w", newline="", encoding="utf-8") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["text_id", "text", "recall@5", "hits", "total", "coverage"])
                    for tid, recall, hits, total in tail_rows:
                        avail_pos, avail_total = availability.get(tid, (0, 0))
                        coverage_value = (avail_pos / max(avail_total, 1)) if avail_total > 0 else 1.0
                        writer.writerow(
                            [
                                tid,
                                self.text_lookup.get(tid, tid),
                                float(recall),
                                int(hits),
                                int(total),
                                float(coverage_value),
                            ]
                        )
                print(f"[Stage1] Saved tail recall@5 CSV: {csv_path}")

        # Optionally persist ETG subset samples as CSV
        if bool(getattr(self.config, "validate_with_etg", False)) and bool(getattr(self.config, "validate_etg_save", True)):
            etg_rows = [s for s in collected if isinstance(s, dict) and s.get("_etg_record")]
            if etg_rows and getattr(self.config, "output_dir", None) and getattr(self.config, "is_ref_device", True):
                artifact_dir = os.path.join(self.config.output_dir, "artifacts")
                ensure_dir(artifact_dir)
                rank = int(getattr(self.config, "device", 0))
                csv_path = os.path.join(artifact_dir, f"val_epoch_{epoch + 1}_etg_subset_rank{rank}.csv")
                with open(csv_path, "w", newline="", encoding="utf-8") as handle:
                    writer = csv.DictWriter(handle, fieldnames=[
                        "ecg_id", "report_text", "generated_report", "etg_loss", "etg_next_acc", "etg_copy_rate"
                    ])
                    writer.writeheader()
                    for row in etg_rows:
                        writer.writerow({
                            "ecg_id": row.get("ecg_id", ""),
                            "report_text": row.get("report_text", ""),
                            "generated_report": row.get("generated_report", ""),
                            "etg_loss": row.get("etg_loss", float("nan")),
                            "etg_next_acc": row.get("etg_next_acc", float("nan")),
                            "etg_copy_rate": row.get("etg_copy_rate", float("nan")),
                        })
                if self.config.is_ref_device:
                    print(f"[Stage1] Saved ETG subset CSV: {csv_path}")

        return averages, collected

    # ------------------------------------------------------------------
    def _compute_siglip_recalls(
        self,
        logits: torch.Tensor,
        positive_ids: Sequence[str],
    ) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        if logits.numel() == 0:
            return metrics

        batch_size = logits.size(0)
        device = logits.device
        labels = torch.eye(batch_size, device=device, dtype=logits.dtype)
        recall_stats = compute_recall_at_many(logits, labels, ks=(1, 5))

        for k, (value_sum, count) in recall_stats.items():
            key = f"recall_at_{k}"
            metrics[key] = float(value_sum / count) if count > 0 else 0.0
            metrics[f"{key}_count"] = int(count)

        if self.tail_class_ids and positive_ids and len(positive_ids) == batch_size:
            tail_mask = torch.tensor(
                [str(pid) in self.tail_class_ids for pid in positive_ids],
                device=device,
                dtype=torch.bool,
            )
            if torch.any(tail_mask):
                tail_logits = logits[tail_mask]
                tail_labels = labels[tail_mask]
                tail_stats = compute_recall_at_many(tail_logits, tail_labels, ks=(1, 5))
                for k, (value_sum, count) in tail_stats.items():
                    key = f"tail_recall_at_{k}"
                    metrics[key] = float(value_sum / count) if count > 0 else 0.0
                    metrics[f"{key}_count"] = int(count)

        return metrics

    @staticmethod
    def _compute_binary_auroc(targets: torch.Tensor, probs: torch.Tensor) -> float:
        if targets.numel() == 0:
            return float("nan")
        unique = torch.unique(targets)
        if unique.numel() < 2:
            return float("nan")
        try:
            return float(
                roc_auc_score(
                    targets.detach().cpu().numpy(),
                    probs.detach().cpu().numpy(),
                )
            )
        except ValueError:
            return float("nan")

    def _align_token_tensor_shapes(
        self,
        pos_input_ids: torch.Tensor,
        pos_attn_mask: torch.Tensor,
        neg_input_ids: torch.Tensor,
        neg_attn_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        pos_input_ids = pos_input_ids.contiguous()
        pos_attn_mask = pos_attn_mask.contiguous()
        neg_input_ids = neg_input_ids.contiguous()
        neg_attn_mask = neg_attn_mask.contiguous()
        target_len = max(pos_input_ids.size(1), neg_input_ids.size(1))
        pad_token_id = int(self.tokenizer.pad_token_id)
        if pos_input_ids.size(1) < target_len:
            pad = target_len - pos_input_ids.size(1)
            pos_input_ids = F.pad(pos_input_ids, (0, pad), value=pad_token_id)
            pos_attn_mask = F.pad(pos_attn_mask, (0, pad), value=0)
        if neg_input_ids.size(1) < target_len:
            pad = target_len - neg_input_ids.size(1)
            neg_input_ids = F.pad(neg_input_ids, (0, pad), value=pad_token_id)
            neg_attn_mask = F.pad(neg_attn_mask, (0, pad), value=0)
        return (
            pos_input_ids.contiguous(),
            pos_attn_mask.contiguous(),
            neg_input_ids.contiguous(),
            neg_attn_mask.contiguous(),
        )

    def _select_hard_neg_ids_from_sim(
        self,
        sim_matrix: torch.Tensor,
        positive_ids: List[str],
        k: int = 1,
    ) -> List[str]:
        """Select hard negative text IDs based on similarity matrix."""
        batch_size = sim_matrix.size(0)
        hard_neg_ids = []
        
        for i in range(batch_size):
            # Get similarities for this ECG to all texts in the batch
            sims = sim_matrix[i]  # Shape: (batch_size,)
            
            # Exclude self (the positive)
            sims[i] = -float('inf')
            
            # Get top-k most similar (hardest negatives)
            if k >= sims.size(0):
                k_actual = sims.size(0) - 1
            else:
                k_actual = k
            
            if k_actual > 0:
                topk_indices = torch.topk(sims, k=k_actual, dim=0).indices
                # Pick the first one
                neg_idx = int(topk_indices[0].item())
                if neg_idx < len(positive_ids):
                    hard_neg_ids.append(positive_ids[neg_idx])
                else:
                    # Fallback to random from bank
                    hard_neg_ids.append(self.text_bank_ids[0] if self.text_bank_ids else "")
            else:
                # Fallback
                hard_neg_ids.append(self.text_bank_ids[0] if self.text_bank_ids else "")
        
        return hard_neg_ids

    def _select_etm_hard_negatives(
        self,
        sim_matrix: torch.Tensor,
        pos_input_ids: torch.Tensor,
        pos_attn_mask: torch.Tensor,
        neg_input_ids: torch.Tensor,
        neg_attn_mask: torch.Tensor,
        positive_ids: Sequence[str],
        ecg_ids: Optional[Sequence[Any]],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        batch_size = pos_input_ids.size(0)
        if self.etm_hard_neg_k <= 0 or batch_size <= 1:
            return neg_input_ids, neg_attn_mask

        device = sim_matrix.device
        sim_for_hard_neg = sim_matrix.detach().clone().contiguous()
        sim_for_hard_neg.fill_diagonal_(-1e4)

        # Mask same-label pairs so they are never sampled as negatives.
        label_groups: Dict[str, List[int]] = {}
        for idx, pos_id in enumerate(positive_ids):
            key = str(pos_id) if pos_id is not None else ""
            if not key:
                continue
            label_groups.setdefault(key, []).append(idx)
        for indices in label_groups.values():
            if len(indices) <= 1:
                continue
            idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
            sim_for_hard_neg[idx_tensor.unsqueeze(1), idx_tensor.unsqueeze(0)] = -1e4

        # Mask same-ECG pairs (when metadata is provided).
        if ecg_ids is not None:
            ecg_lookup: List[str] = []
            for i in range(batch_size):
                if i < len(ecg_ids):
                    ecg_lookup.append("" if ecg_ids[i] is None else str(ecg_ids[i]))
                else:
                    ecg_lookup.append("")
            ecg_groups: Dict[str, List[int]] = {}
            for idx, ecg_id in enumerate(ecg_lookup):
                if not ecg_id:
                    continue
                ecg_groups.setdefault(ecg_id, []).append(idx)
            for indices in ecg_groups.values():
                if len(indices) <= 1:
                    continue
                idx_tensor = torch.tensor(indices, device=device, dtype=torch.long)
                sim_for_hard_neg[idx_tensor.unsqueeze(1), idx_tensor.unsqueeze(0)] = -1e4

        valid_mask = sim_for_hard_neg > (-1e4 + 1e-6)
        hard_indices = torch.zeros(batch_size, dtype=torch.long, device=device)
        fallback_mask = torch.ones(batch_size, dtype=torch.bool, device=device)

        for row in range(batch_size):
            valid_idx = torch.nonzero(valid_mask[row], as_tuple=False).squeeze(1)
            if valid_idx.numel() == 0:
                continue
            top_k = min(self.etm_hard_neg_k, int(valid_idx.numel()))
            if top_k <= 0:
                continue
            row_scores = sim_for_hard_neg[row, valid_idx].contiguous()
            topk_rel = torch.topk(row_scores, k=top_k, dim=0).indices
            candidates = valid_idx[topk_rel]
            choice = torch.randint(0, top_k, (), device=device)
            hard_indices[row] = candidates[choice]
            fallback_mask[row] = False

        hard_neg_input_ids = pos_input_ids[hard_indices].clone()
        hard_neg_attn_mask = pos_attn_mask[hard_indices].clone()

        if fallback_mask.any():
            fallback_rows = torch.nonzero(fallback_mask, as_tuple=False).squeeze(1)
            if fallback_rows.numel() > 0:
                hard_neg_input_ids[fallback_rows] = neg_input_ids[fallback_rows].contiguous()
                hard_neg_attn_mask[fallback_rows] = neg_attn_mask[fallback_rows].contiguous()

        return hard_neg_input_ids.contiguous(), hard_neg_attn_mask.contiguous()

    @staticmethod
    def _is_distributed() -> bool:
        return dist.is_available() and dist.is_initialized()

    def _gather_tensor(self, tensor: torch.Tensor, *, with_grad: bool) -> torch.Tensor:
        if not self._is_distributed():
            return tensor
        world_size = dist.get_world_size()
        if world_size <= 1:
            return tensor

        # Track the first-dimension sizes on every rank so we can detect mismatches.
        if tensor.dim() == 0:
            tensor = tensor.unsqueeze(0)
        size_tensor = torch.tensor([tensor.shape[0]], device=tensor.device, dtype=torch.long)
        size_gather = [torch.zeros_like(size_tensor) for _ in range(world_size)]
        dist.all_gather(size_gather, size_tensor)
        part_sizes = [int(entry.item()) for entry in size_gather]
        sizes_uniform = all(sz == part_sizes[0] for sz in part_sizes)

        if sizes_uniform and with_grad and dist_nn_f is not None:
            try:
                gathered = dist_nn_f.all_gather(tensor.contiguous())
                if isinstance(gathered, torch.Tensor):
                    return gathered
                return torch.cat(list(gathered), dim=0)
            except RuntimeError:
                pass

        if sizes_uniform:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            try:
                dist.all_gather(gather_list, tensor.detach().contiguous())
                return torch.cat(gather_list, dim=0)
            except RuntimeError:
                pass

        # Fall back to object-based gather so variable-length tensors do not deadlock.
        tensor_cpu = tensor.detach().cpu()
        gather_objs: List[Optional[torch.Tensor]] = [None for _ in range(world_size)]
        dist.all_gather_object(gather_objs, tensor_cpu)

        restored: List[torch.Tensor] = []
        for obj, size in zip(gather_objs, part_sizes):
            if not isinstance(obj, torch.Tensor):
                continue
            slice_view = obj[:size]
            restored.append(slice_view.to(device=tensor.device, dtype=tensor.dtype))

        if restored:
            return torch.cat(restored, dim=0)
        return tensor

    def _gather_batch_metadata(self, local_batch: int, device: torch.device) -> Tuple[List[int], int]:
        if not self._is_distributed():
            return [local_batch], 0
        size_tensor = torch.tensor([local_batch], device=device, dtype=torch.long)
        world_size = dist.get_world_size()
        sizes_tensor = [torch.zeros_like(size_tensor) for _ in range(world_size)]
        dist.all_gather(sizes_tensor, size_tensor)
        sizes = [int(entry.item()) for entry in sizes_tensor]
        offset = sum(sizes[: dist.get_rank()])
        return sizes, offset

    def _cast_bank_embeddings(self, tensor: torch.Tensor) -> torch.Tensor:
        target_dtype = self.amp_dtype if (self.use_autocast and self.amp_dtype is not None) else tensor.dtype
        casted = tensor.to(device=self.device, dtype=target_dtype)
        return F.normalize(casted, dim=-1)

    def _sample_bank_negatives(self, exclude: Set[str], count: int) -> List[str]:
        if not self.text_bank_ids or self.bank_negative_samples <= 0 or count <= 0:
            return []
        pool_len = len(self.text_bank_ids)
        if pool_len == 0:
            return []
        samples: List[str] = []
        attempts = 0
        target = min(count, pool_len)
        while len(samples) < target and attempts < pool_len * 3:
            tid = self.text_bank_ids[self._neg_sampler.randrange(pool_len)]
            if tid in exclude or tid in samples:
                attempts += 1
                continue
            samples.append(tid)
        if len(samples) < target:
            for tid in self.text_bank_ids:
                if tid in exclude or tid in samples:
                    continue
                samples.append(tid)
                if len(samples) >= target:
                    break
        return samples

    def _update_text_bank_metadata(self) -> None:
        self._text_id_set = {str(tid) for tid in self.text_bank_ids if tid}
        if not self._text_id_set:
            self._rhythm_candidates = []
            self._family_hard_negs = {}
            return
        rhythm_tokens = [
            "afib",
            "atrial_flutter",
            "atrial_tachycardia",
            "junctional_rhythm",
            "supraventricular_tachycardia",
            "ventricular_tachycardia",
            "ventricular_fibrillation",
            "sinus_rhythm",
            "ventricular_paced",
        ]
        self._rhythm_candidates = [
            tid for tid in self._text_id_set if any(token in tid for token in rhythm_tokens)
        ]
        family: Dict[str, List[str]] = {}
        for key, values in SIGLIP_TARGETED_HARD_NEGATIVES.items():
            if key not in self._text_id_set:
                continue
            filtered = [val for val in values if val in self._text_id_set]
            if filtered:
                family[key] = filtered
        self._family_hard_negs = family

    @staticmethod
    def _counterpart(text_id: str) -> Optional[str]:
        if not text_id:
            return None
        if text_id.endswith("yes"):
            return text_id[:-3] + "no"
        if text_id.endswith("no"):
            return text_id[:-2] + "yes"
        return None

    def _sample_rhythm_negatives(self, exclude: Sequence[str]) -> List[str]:
        if not self._rhythm_candidates or self._max_hardneg_per_group <= 0:
            return []
        exclude_set = {str(tid) for tid in exclude if tid}
        pool = [tid for tid in self._rhythm_candidates if tid not in exclude_set]
        if not pool:
            return []
        sample_size = min(self._max_hardneg_per_group, len(pool))
        if sample_size <= 0:
            return []
        if sample_size >= len(pool):
            return pool
        return self._hardneg_rng.sample(pool, sample_size)

    def _monitor_logit_scale(self, scale: float) -> None:
        if not getattr(self.config, "is_ref_device", True):
            return
        threshold = float(self._logit_scale_warn_threshold)
        if threshold <= 0:
            return
        if scale >= threshold and not self._logit_scale_warned:
            print(
                f"[Stage1] logit_scale reached {scale:.2f} "
                f"(temperature≈{1.0 / max(scale, 1e-8):.3f})."
            )
            self._logit_scale_warned = True
        elif scale < threshold * 0.9:
            self._logit_scale_warned = False

    def _maybe_refresh_text_bank(self) -> None:
        if self.bank_refresh_every_steps <= 0:
            return
        if not self.text_bank_texts or self.text_bank_embeddings is None:
            return
        if self.global_step <= 0:
            return
        if self.global_step == self._last_bank_refresh_step:
            return
        if self.global_step % self.bank_refresh_every_steps != 0:
            return
        self._refresh_text_bank()
        self._last_bank_refresh_step = self.global_step

    def _refresh_text_bank(self) -> None:
        if not self.text_bank_texts:
            return
        was_training = self.text_encoder.training
        self.text_encoder.eval()
        new_embeddings: List[torch.Tensor] = []
        batch_size = int(self.bank_refresh_batch_size)
        if batch_size <= 0:
            batch_size = 2048
        with torch.no_grad():
            for start in range(0, len(self.text_bank_texts), batch_size):
                chunk = self.text_bank_texts[start : start + batch_size]
                tokens = self.tokenizer(
                    chunk,
                    padding=True,
                    truncation=True,
                    max_length=self.max_text_length,
                    return_tensors="pt",
                )
                input_ids = tokens["input_ids"].to(self.device)
                attn_mask = tokens["attention_mask"].to(self.device)
                embeddings = self.text_encoder(input_ids, attn_mask)
                new_embeddings.append(F.normalize(embeddings, dim=-1))
        if not new_embeddings:
            if was_training:
                self.text_encoder.train()
            return
        bank_tensor = torch.cat(new_embeddings, dim=0)
        self.text_bank_embeddings = self._cast_bank_embeddings(bank_tensor)
        self.bank_indices_tensor = torch.arange(self.text_bank_embeddings.size(0), device=self.device, dtype=torch.long)
        if was_training:
            self.text_encoder.train()
        else:
            self.text_encoder.eval()
        if self.text_bank_ids:
            self.text_lookup = {tid: txt for tid, txt in zip(self.text_bank_ids, self.text_bank_texts)}
        self._update_text_bank_metadata()

    def _prepend_dec_token(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        max_length: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.dec_token_id is None:
            return input_ids.contiguous(), attention_mask.contiguous()
        batch, seq_len = input_ids.size()
        if seq_len + 1 > max_length:
            truncate_len = max_length - 1
            input_ids = input_ids[:, :truncate_len]
            attention_mask = attention_mask[:, :truncate_len]
        dec_col = torch.full((batch, 1), self.dec_token_id, dtype=input_ids.dtype, device=input_ids.device)
        dec_mask = torch.ones((batch, 1), dtype=attention_mask.dtype, device=attention_mask.device)
        input_ids = torch.cat([dec_col, input_ids], dim=1).contiguous()
        attention_mask = torch.cat([dec_mask, attention_mask], dim=1).contiguous()
        return input_ids, attention_mask

    def _siglip_loss_with_global(
        self,
        ecg_vec: torch.Tensor,
        text_features: torch.Tensor,
        positive_ids: Sequence[str],
        positive_texts: Sequence[str],
        negative_ids: Optional[Sequence[Any]] = None,
        negative_texts: Optional[Sequence[Any]] = None,
        ecg_ids: Optional[Sequence[Any]] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float], List[Dict[str, Any]]]:
        batch_size = ecg_vec.size(0)
        if batch_size == 0:
            zero = ecg_vec.new_zeros(())
            metrics: Dict[str, float] = {
                "siglip_recall_at_1_hits": 0.0,
                "siglip_recall_at_5_hits": 0.0,
                "siglip_batch_size": 0.0,
            }
            return zero, metrics, []

        batch_id_map: Dict[str, int] = {pid: idx for idx, pid in enumerate(positive_ids) if pid}
        ecg_ids_list: Optional[List[str]] = None
        ecg_groups: Dict[str, List[int]] = {}
        if ecg_ids is not None:
            ecg_ids_list = [str(eid) for eid in ecg_ids]
            for idx, ecg_identifier in enumerate(ecg_ids_list):
                ecg_groups.setdefault(ecg_identifier, []).append(idx)

        explicit_neg_id_rows: List[List[str]] = [[] for _ in range(batch_size)]
        explicit_neg_text_map: Dict[str, str] = {}
        if negative_ids is not None:
            for row_idx in range(batch_size):
                if row_idx >= len(negative_ids):
                    break
                entry = negative_ids[row_idx]
                if isinstance(entry, (list, tuple, set)):
                    iterable = entry
                elif entry is None or entry == "":
                    iterable = []
                else:
                    iterable = [entry]
                explicit_neg_id_rows[row_idx] = [str(val) for val in iterable if str(val)]
        if negative_texts is not None:
            for row_idx in range(min(batch_size, len(negative_texts))):
                entry = negative_texts[row_idx]
                if isinstance(entry, (list, tuple)):
                    texts_iter = entry
                elif entry is None:
                    texts_iter = []
                else:
                    texts_iter = [entry]
                ids_row = explicit_neg_id_rows[row_idx]
                for local_idx, text_val in enumerate(texts_iter):
                    if local_idx >= len(ids_row):
                        break
                    explicit_neg_text_map[ids_row[local_idx]] = str(text_val)

        candidate_ids_list: List[List[str]] = []
        candidate_texts_list: List[List[str]] = []
        candidate_vecs_list: List[torch.Tensor] = []
        positive_indices_list: List[List[int]] = []
        row_weights: List[float] = []

        for i in range(batch_size):
            pos_id = positive_ids[i]
            pos_text = str(positive_texts[i]) if i < len(positive_texts) else ""
            pos_vec = text_features[i]
            candidate_ids: List[str] = []
            candidate_texts: List[str] = []
            candidate_vecs: List[torch.Tensor] = []
            positive_positions: List[int] = []

            seen: Set[str] = set()

            if pos_id:
                candidate_ids.append(pos_id)
                candidate_texts.append(pos_text)
                candidate_vecs.append(pos_vec.to(dtype=ecg_vec.dtype))
                positive_positions.append(0)
                seen.add(pos_id)

            co_positive_ids: Set[str] = set()
            current_ecg = ecg_ids_list[i] if ecg_ids_list is not None and i < len(ecg_ids_list) else None
            if current_ecg and current_ecg in ecg_groups:
                for other_idx in ecg_groups[current_ecg]:
                    if other_idx == i:
                        continue
                    other_id = positive_ids[other_idx]
                    if not other_id or other_id in seen:
                        continue
                    other_text = str(positive_texts[other_idx]) if other_idx < len(positive_texts) else ""
                    other_vec = text_features[other_idx]
                    candidate_ids.append(other_id)
                    candidate_texts.append(other_text)
                    candidate_vecs.append(other_vec.to(dtype=ecg_vec.dtype))
                    positive_positions.append(len(candidate_ids) - 1)
                    seen.add(other_id)
                    co_positive_ids.add(other_id)

            explicit_ids = explicit_neg_id_rows[i] if i < len(explicit_neg_id_rows) else []
            neg_candidates: List[str] = []
            neg_candidates.extend(explicit_ids)

            if pos_id:
                counterpart = self._counterpart(pos_id)
                if counterpart:
                    neg_candidates.append(counterpart)
                neg_candidates.extend(self._family_hard_negs.get(pos_id, []))

            for explicit in explicit_ids:
                counterpart = self._counterpart(explicit)
                if counterpart:
                    neg_candidates.append(counterpart)
                neg_candidates.extend(self._family_hard_negs.get(explicit, []))

            neg_candidates.extend(self._sample_rhythm_negatives([pos_id] + explicit_ids))

            for j, other_id in enumerate(positive_ids):
                if j == i or not other_id:
                    continue
                if current_ecg is not None and ecg_ids_list is not None and j < len(ecg_ids_list):
                    if ecg_ids_list[j] == current_ecg:
                        continue
                neg_candidates.append(other_id)

            # ensure co-positive ids are tracked as seen
            if co_positive_ids:
                seen.update(co_positive_ids)
            final_neg_ids: List[str] = []
            for nid in neg_candidates:
                if not nid or nid == pos_id or nid in seen:
                    continue
                seen.add(nid)
                final_neg_ids.append(nid)

            if self.bank_negative_samples > 0:
                remaining = max(0, self.bank_negative_samples - len(final_neg_ids))
                if remaining > 0:
                    extra = self._sample_bank_negatives(seen, remaining)
                    for nid in extra:
                        if nid and nid not in seen:
                            seen.add(nid)
                            final_neg_ids.append(nid)

            if not final_neg_ids:
                fallback = self._sample_bank_negatives(seen, 1)
                for nid in fallback:
                    if nid and nid not in seen:
                        seen.add(nid)
                        final_neg_ids.append(nid)

            for nid in final_neg_ids:
                if not nid or nid == pos_id:
                    continue
                vec: Optional[torch.Tensor] = None
                text_value = self.text_lookup.get(nid, explicit_neg_text_map.get(nid, ""))
                if self.text_bank_embeddings is not None and nid in self.text_bank_index:
                    idx_bank = self.text_bank_index[nid]
                    vec = self.text_bank_embeddings[idx_bank]
                    if not text_value:
                        text_value = self.text_bank_texts[idx_bank]
                elif nid in batch_id_map:
                    other_idx = batch_id_map[nid]
                    if other_idx == i:
                        continue
                    vec = text_features[other_idx]
                    if not text_value:
                        text_value = str(positive_texts[other_idx])
                if vec is None:
                    continue
                candidate_ids.append(nid)
                candidate_texts.append(text_value)
                candidate_vecs.append(vec.to(device=ecg_vec.device, dtype=ecg_vec.dtype))

            if len(candidate_ids) < 2:
                fallback = self._sample_bank_negatives(set(candidate_ids), 1)
                for nid in fallback:
                    if nid in self.text_bank_index:
                        idx_bank = self.text_bank_index[nid]
                        candidate_ids.append(nid)
                        candidate_texts.append(self.text_bank_texts[idx_bank])
                        bank_vec = self.text_bank_embeddings[idx_bank]
                        candidate_vecs.append(bank_vec.to(device=ecg_vec.device, dtype=ecg_vec.dtype))
                        break

            if len(candidate_ids) < 2:
                candidate_ids.append(f"{pos_id}_neg")
                candidate_texts.append("<synthetic>")
                candidate_vecs.append((-pos_vec).to(dtype=ecg_vec.dtype))

            stacked = torch.stack(candidate_vecs, dim=0).to(device=ecg_vec.device, dtype=ecg_vec.dtype)
            candidate_ids_list.append(candidate_ids)
            candidate_texts_list.append(candidate_texts)
            candidate_vecs_list.append(stacked)
            if positive_positions:
                positive_indices_list.append(positive_positions)
            else:
                fallback_indices = [0] if candidate_ids else []
                positive_indices_list.append(fallback_indices)
            if self.tail_enable and self.tail_class_ids and pos_id in self.tail_class_ids:
                row_weights.append(self.tail_alpha_boost)
            else:
                row_weights.append(1.0)

        max_candidates = max(vecs.size(0) for vecs in candidate_vecs_list)
        feature_dim = ecg_vec.size(1)
        candidate_tensor = ecg_vec.new_zeros((batch_size, max_candidates, feature_dim))
        candidate_mask = torch.zeros((batch_size, max_candidates), dtype=torch.bool, device=ecg_vec.device)

        for i, vecs in enumerate(candidate_vecs_list):
            size = vecs.size(0)
            candidate_tensor[i, :size] = vecs
            candidate_mask[i, :size] = True

        logits = torch.bmm(ecg_vec.unsqueeze(1), candidate_tensor.transpose(1, 2)).squeeze(1)
        logits = logits.masked_fill(~candidate_mask, -1e4)

        logit_scale_scalar = 1.0
        temperature_value = float("nan")
        temperature_attr = getattr(self.bridge, "temperature", None)
        if callable(temperature_attr):
            temperature = temperature_attr()
        else:
            temperature = temperature_attr
        temperature_value = 1.0
        if temperature is not None:
            if torch.is_tensor(temperature):
                temperature_tensor = temperature.to(device=logits.device, dtype=logits.dtype)
            else:
                temperature_tensor = torch.tensor(
                    float(temperature),
                    device=logits.device,
                    dtype=logits.dtype,
                )
            temperature_tensor = temperature_tensor.clamp_min(1e-6)
            logits = logits / temperature_tensor
            temperature_value = float(temperature_tensor.detach().item())
            logit_scale_scalar = 1.0 / max(temperature_value, 1e-8)

        self._monitor_logit_scale(logit_scale_scalar)

        pos_mask = torch.zeros_like(candidate_mask)
        for row_idx, pos_indices in enumerate(positive_indices_list):
            for pos_idx in pos_indices:
                if pos_idx < candidate_mask.size(1):
                    pos_mask[row_idx, pos_idx] = candidate_mask[row_idx, pos_idx]

        valid_rows_mask = candidate_mask.any(dim=1)
        valid_rows = int(valid_rows_mask.sum().item())
        if valid_rows == 0:
            zero = ecg_vec.new_zeros(())
            metrics = {
                "siglip_recall_at_1_hits": 0.0,
                "siglip_recall_at_5_hits": 0.0,
                "siglip_batch_size": 0.0,
                "logit_scale": logit_scale_scalar,
                "siglip_loss": 0.0,
                "siglip_loss_raw": 0.0,
                "siglip_loss_mismatch": 0.0,
                "siglip_candidate_count": 0.0,
                "siglip_pos_mass": 0.0,
                "siglip_neg_mass": 0.0,
                "siglip_positive_prob": 0.0,
                "siglip_duplicate_fraction": 0.0,
                "siglip_random_r1": 0.0,
            }
            if not math.isnan(temperature_value):
                metrics["temperature"] = temperature_value
            return zero, metrics, [{} for _ in range(batch_size)]

        work_dtype = torch.float32 if logits.dtype in (torch.float16, torch.bfloat16) else logits.dtype
        logits_work = logits.to(work_dtype)
        candidate_mask_sel = candidate_mask[valid_rows_mask]
        pos_mask_sel = pos_mask[valid_rows_mask]
        logits_sel = logits_work[valid_rows_mask]
        neg_mask_sel = (~pos_mask_sel) & candidate_mask_sel

        if not pos_mask_sel.any(dim=1).all():
            raise RuntimeError("ETC loss requires at least one positive candidate per row.")
        if not neg_mask_sel.any(dim=1).all():
            raise RuntimeError("ETC loss requires at least one negative candidate per row.")

        neg_inf = torch.finfo(logits_work.dtype).min
        logits_masked = logits_sel.masked_fill(~candidate_mask_sel, neg_inf)
        probs_sel = torch.softmax(logits_masked, dim=1)
        pos_mass_sel = (probs_sel * pos_mask_sel).sum(dim=1).clamp_min(1e-12)
        base_loss_per_row = -pos_mass_sel.log()

        alpha_mat: Optional[torch.Tensor] = None
        if self.use_focal_infonce:
            alpha_mat = logits_work.new_full((batch_size, max_candidates), float(self.focal_alpha_default))
            for row_idx, ids in enumerate(candidate_ids_list):
                if not ids:
                    continue
                limit = min(len(ids), max_candidates)
                alpha_mat[row_idx, :limit] = float(self.focal_alpha_default)
                alpha_mat[row_idx, 0] = float(self.class_pos_weight_map.get(ids[0], self.focal_alpha_default))
            alpha_mat = alpha_mat[valid_rows_mask]

        probs_detached = probs_sel.detach() if self.focal_detach_weights else probs_sel
        if alpha_mat is None:
            alpha_weights = torch.ones_like(logits_sel)
        else:
            alpha_weights = alpha_mat.to(logits_sel.dtype)

        if self.use_focal_infonce and self.focal_gamma_pos > 0.0:
            pos_weights = alpha_weights * (1.0 - probs_detached).clamp_min(1e-6).pow(self.focal_gamma_pos)
        else:
            pos_weights = alpha_weights
        pos_weights = torch.where(pos_mask_sel, pos_weights, torch.zeros_like(pos_weights)).clamp_min(1e-12)

        if self.use_focal_infonce and self.focal_gamma_neg > 0.0:
            denom_weights = torch.where(
                neg_mask_sel,
                probs_detached.clamp_min(1e-6).pow(self.focal_gamma_neg),
                torch.ones_like(probs_sel),
            )
        else:
            denom_weights = torch.ones_like(probs_sel)

        logits_denom = logits_sel + torch.log(denom_weights.clamp_min(1e-12))
        logits_denom = logits_denom.masked_fill(~candidate_mask_sel, neg_inf)
        denom_val = torch.logsumexp(logits_denom, dim=1)

        logits_num = logits_sel + torch.log(pos_weights)
        logits_num = logits_num.masked_fill(~pos_mask_sel, neg_inf)
        numer_val = torch.logsumexp(logits_num, dim=1)

        loss_infonce_per_row = denom_val - numer_val
        loss_per_row = base_loss_per_row if not self.use_focal_infonce else loss_infonce_per_row
        row_weights_tensor = torch.tensor(row_weights, device=loss_per_row.device, dtype=loss_per_row.dtype)
        row_weights_sel = row_weights_tensor[valid_rows_mask]
        weight_sum = row_weights_sel.sum().clamp_min(1.0)
        weighted_loss = loss_per_row * row_weights_sel
        loss = weighted_loss.sum() / weight_sum
        if logits_work.dtype != logits.dtype:
            loss = loss.to(logits.dtype)
        info_loss_weighted = (base_loss_per_row * row_weights_sel).sum() / weight_sum
        raw_loss_weighted = (loss_per_row * row_weights_sel).sum() / weight_sum
        loss_mismatch = (loss_per_row - base_loss_per_row).abs().mean()

        probs_full = logits_work.new_zeros((batch_size, max_candidates))
        probs_full[valid_rows_mask] = probs_sel
        probs_full = probs_full.to(logits.dtype)

        recall_at_1_hits = 0.0
        recall_at_5_hits = 0.0
        tail_hits1 = 0.0
        tail_hits5 = 0.0
        tail_count = 0.0
        pos_mass_total = 0.0
        neg_mass_total = 0.0
        positive_prob_total = 0.0
        positive_prob_max_total = 0.0
        candidate_count_total = 0.0
        random_r1_total = 0.0
        dup_fraction_total = 0.0
        candidate_details: List[Dict[str, Any]] = []

        for i in range(batch_size):
            if not valid_rows_mask[i]:
                candidate_details.append({})
                continue

            valid_count = int(candidate_mask[i].sum().item())
            candidate_ids = candidate_ids_list[i][:valid_count]
            candidate_texts = candidate_texts_list[i][:valid_count]
            prob_row = probs_full[i, :valid_count]

            pos_indices = [idx for idx in positive_indices_list[i] if idx < valid_count]
            if not pos_indices:
                pos_indices = [0]
            pos_tensor = prob_row[pos_indices]
            pos_prob_sum = float(pos_tensor.detach().sum().item())
            pos_prob_primary = float(pos_tensor[0].detach().item()) if pos_tensor.numel() > 0 else 0.0
            pos_prob_max_val = float(pos_tensor.detach().max().item()) if pos_tensor.numel() > 0 else 0.0
            pos_mass_total += pos_prob_sum
            total_prob_row = float(prob_row.detach().sum().item())
            neg_mass_total += float(max(0.0, total_prob_row - pos_prob_sum))
            positive_prob_total += pos_prob_primary
            positive_prob_max_total += pos_prob_max_val
            candidate_count_total += valid_count
            random_r1_total += 1.0 / valid_count if valid_count > 0 else 0.0
            unique_ids = len(set(candidate_ids))
            dup_fraction_total += 1.0 - (unique_ids / max(1.0, float(valid_count)))

            sorted_local = torch.argsort(prob_row, descending=True)
            seen: Set[str] = set()
            top_entries: List[Tuple[str, str, float]] = []
            for idx_local in sorted_local.tolist():
                tid = candidate_ids[idx_local]
                if tid in seen:
                    continue
                seen.add(tid)
                top_entries.append((tid, candidate_texts[idx_local], float(prob_row[idx_local].detach().item())))
                if len(top_entries) == 5:
                    break

            primary_pos_id = candidate_ids[pos_indices[0]] if pos_indices and pos_indices[0] < len(candidate_ids) else (candidate_ids[0] if candidate_ids else "")
            positive_id_set = {candidate_ids[idx_local] for idx_local in pos_indices if idx_local < len(candidate_ids)}
            if top_entries and any(entry[0] in positive_id_set for entry in top_entries[:1]):
                recall_at_1_hits += 1.0
            if any(entry[0] in positive_id_set for entry in top_entries):
                recall_at_5_hits += 1.0

            if self.tail_class_ids:
                for tail_pos_id in positive_id_set:
                    if tail_pos_id not in self.tail_class_ids:
                        continue
                    tail_count += 1.0
                    if top_entries and any(entry[0] == tail_pos_id for entry in top_entries[:1]):
                        tail_hits1 += 1.0
                    if any(entry[0] == tail_pos_id for entry in top_entries):
                        tail_hits5 += 1.0

            candidate_probs = [
                float(prob_row[idx_local].detach().item())
                for idx_local in range(valid_count)
            ]
            candidate_details.append({
                "top": top_entries,
                "candidate_count": valid_count,
                "pos_prob": pos_prob_primary,
                "pos_prob_sum": pos_prob_sum,
                "pos_ids": [candidate_ids[idx_local] for idx_local in pos_indices if idx_local < len(candidate_ids)],
                "ids": list(candidate_ids),
                "texts": list(candidate_texts),
                "probs": candidate_probs,
                "positive_indices": pos_indices,
            })

        valid_rows_f = max(1.0, float(valid_rows))
        metrics = {
            "siglip_recall_at_1_hits": float(recall_at_1_hits),
            "siglip_recall_at_5_hits": float(recall_at_5_hits),
            "siglip_batch_size": float(valid_rows),
            "siglip_recall_at_1": float(recall_at_1_hits / valid_rows_f),
            "siglip_recall_at_5": float(recall_at_5_hits / valid_rows_f),
            "siglip_pos_mass": float(pos_mass_total / valid_rows_f),
            "siglip_neg_mass": float(neg_mass_total / valid_rows_f),
            "siglip_candidate_count": float(candidate_count_total / valid_rows_f),
            "siglip_duplicate_fraction": float(dup_fraction_total / valid_rows_f),
            "siglip_positive_prob": float(positive_prob_total / valid_rows_f),
            "siglip_positive_prob_max": float(positive_prob_max_total / valid_rows_f),
            "siglip_random_r1": float(random_r1_total / valid_rows_f),
            "logit_scale": float(logit_scale_scalar),
            "siglip_loss": float(info_loss_weighted.detach().item()),
            "siglip_loss_raw": float(raw_loss_weighted.detach().item()),
            "siglip_loss_mismatch": float(loss_mismatch.detach().item()),
        }
        if not math.isnan(temperature_value):
            metrics["temperature"] = float(temperature_value)
        if tail_count > 0:
            tail_count_f = float(tail_count)
            metrics["tail_recall_at_1_hits"] = float(tail_hits1)
            metrics["tail_recall_at_5_hits"] = float(tail_hits5)
            metrics["tail_recall_count"] = tail_count_f
            metrics["siglip_tail_recall_at_1"] = float(tail_hits1 / tail_count_f)
            metrics["siglip_tail_recall_at_5"] = float(tail_hits5 / tail_count_f)

        return loss, metrics, candidate_details

    def _compute_etg_teacher_forcing_metrics(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Dict[str, float]:
        pad_id = self.tokenizer.pad_token_id
        mask = attention_mask.bool()
        if mask.size(1) > 0:
            mask = mask.clone()
            mask[:, 0] = False  # exclude [DEC]

        total = mask.sum().item()
        if total == 0:
            return {}

        preds = logits.argmax(dim=-1)

        copy_hits = ((preds == input_ids) & mask).float().sum().item()
        copy_rate = copy_hits / total if total > 0 else 0.0

        targets = input_ids.roll(-1, dims=1)
        if pad_id is not None:
            targets[:, -1] = pad_id
            next_mask = mask & (targets != pad_id)
        else:
            next_mask = mask.clone()
            if next_mask.size(1) > 0:
                next_mask[:, -1] = False
        next_total = next_mask.sum().item()
        next_acc = 0.0
        if next_total > 0:
            next_hits = ((preds == targets) & next_mask).float().sum().item()
            next_acc = next_hits / next_total

        return {
            "etg_copy_rate": float(copy_rate),
            "etg_next_acc": float(next_acc),
        }

    # ------------------------------------------------------------------
    def _train_step(
        self,
        batch: Dict[str, Any],
        step_idx: int,
        is_last_batch: bool,
        generated_samples: List[Tuple[str, str]],
    ) -> Dict[str, float]:
        self._maybe_refresh_text_bank()
        signals: torch.Tensor = batch["signal"].to(self.device, non_blocking=True)
        batch_size = signals.size(0)  # True batch size = number of ECGs
        
        # NEW FORMAT: Lists of lists (one list per ECG)
        positive_texts_lists = batch.get("positive_texts_lists", [])
        positive_ids_lists = batch.get("positive_ids_lists", [])
        negative_texts_lists = batch.get("negative_texts_lists", [])
        negative_ids_lists = batch.get("negative_ids_lists", [])
        
        # OPTIMIZATION: Collect ALL unique positive texts across all ECGs before encoding
        unique_pos_texts: Dict[str, str] = {}  # text_id -> text
        for ecg_pos_ids, ecg_pos_texts in zip(positive_ids_lists, positive_texts_lists):
            if isinstance(ecg_pos_ids, (list, tuple)) and isinstance(ecg_pos_texts, (list, tuple)):
                for tid, txt in zip(ecg_pos_ids, ecg_pos_texts):
                    tid_str = str(tid)
                    if tid_str and tid_str not in unique_pos_texts:
                        unique_pos_texts[tid_str] = str(txt)
        
        # Encode only unique positive texts (much more efficient!)
        # Instead of encoding 960 texts (96 ECGs × 10 positives), encode ~50-100 unique texts
        pos_text_embeddings: Dict[str, torch.Tensor] = {}
        if unique_pos_texts:
            pos_texts_to_encode = list(unique_pos_texts.values())
            pos_ids_to_encode = list(unique_pos_texts.keys())
            
            pos_tokens = self.tokenizer(
                pos_texts_to_encode,
                padding=True,
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt",
            )
            pos_input_ids_unique = pos_tokens["input_ids"].to(self.device)
            pos_attn_mask_unique = pos_tokens["attention_mask"].to(self.device)
            
            autocast_ctx = (
                autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_autocast)
                if self.use_autocast
                else nullcontext()
            )
            with autocast_ctx:
                pos_embeddings = self.text_encoder(pos_input_ids_unique, pos_attn_mask_unique)
                pos_embeddings = F.normalize(pos_embeddings, dim=-1)
            
            # Cache embeddings by text_id for fast lookup
            for tid, emb in zip(pos_ids_to_encode, pos_embeddings):
                pos_text_embeddings[tid] = emb
        
        # Flatten lists for loss computation (create one entry per positive per ECG)
        # This is for compatibility with existing loss functions
        positive_texts: List[str] = []
        positive_ids: List[str] = []
        text_features_list: List[torch.Tensor] = []
        negative_id_pool: List[List[str]] = []
        ecg_indices: List[int] = []  # Track which ECG each entry came from
        
        default_dtype = torch.float32
        if pos_text_embeddings:
            default_dtype = next(iter(pos_text_embeddings.values())).dtype
        
        for ecg_idx in range(batch_size):
            ecg_pos_ids = positive_ids_lists[ecg_idx] if ecg_idx < len(positive_ids_lists) else []
            ecg_pos_texts = positive_texts_lists[ecg_idx] if ecg_idx < len(positive_texts_lists) else []
            ecg_neg_ids = negative_ids_lists[ecg_idx] if ecg_idx < len(negative_ids_lists) else []
            
            if not isinstance(ecg_pos_ids, (list, tuple)):
                ecg_pos_ids = [ecg_pos_ids]
            if not isinstance(ecg_pos_texts, (list, tuple)):
                ecg_pos_texts = [ecg_pos_texts]
            if not isinstance(ecg_neg_ids, (list, tuple)):
                ecg_neg_ids = [ecg_neg_ids]
            
            # Create one entry per positive for this ECG
            for pos_id, pos_text in zip(ecg_pos_ids, ecg_pos_texts):
                pos_id_str = str(pos_id)
                positive_ids.append(pos_id_str)
                positive_texts.append(str(pos_text))
                negative_id_pool.append([str(nid) for nid in ecg_neg_ids])
                ecg_indices.append(ecg_idx)
                
                # Look up cached embedding
                if pos_id_str in pos_text_embeddings:
                    text_features_list.append(pos_text_embeddings[pos_id_str])
                else:
                    # Fallback (shouldn't happen)
                    text_features_list.append(torch.zeros(768, device=self.device, dtype=default_dtype))
        
        # Stack text features
        if text_features_list:
            text_features = torch.stack(text_features_list, dim=0)
        else:
            text_features = torch.zeros((0, 768), device=self.device, dtype=default_dtype)
        
        codes = self._compute_codes(signals)

        if (
            self.global_step == 0
            and step_idx == 0
            and getattr(self.config, "is_ref_device", True)
            and positive_texts
        ):
            pos_id_dbg = positive_ids[0] if positive_ids else ""
            neg_ids_dbg = ", ".join(negative_id_pool[0][:5]) if negative_id_pool else ""
            print(
                "[Stage1][Debug] First batch example:\n"
                f"   ECG batch_size={batch_size} (unique ECGs)\n"
                f"   Flattened samples={len(positive_ids)} (one per positive)\n"
                f"   Unique texts encoded={len(unique_pos_texts)}\n"
                f"   First pos_id={pos_id_dbg}\n"
                f"   First pos_text={positive_texts[0][:100]}\n"
                f"   First neg_ids={neg_ids_dbg}"
            )
        
        # For ETM/ETG losses, we need to tokenize texts (not just embeddings)
        # Tokenize positives for ETM
        pos_tokens = self.tokenizer(
            positive_texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        pos_input_ids = pos_tokens["input_ids"].to(self.device).contiguous()
        pos_attn_mask = pos_tokens["attention_mask"].to(self.device).contiguous()
        
        # Sample one negative per positive for ETM
        negative_texts_for_etm = []
        for neg_pool in negative_id_pool:
            if neg_pool:
                # Sample one negative from pool
                neg_id = neg_pool[0] if neg_pool else ""
                negative_texts_for_etm.append(self.text_lookup.get(neg_id, ""))
            else:
                negative_texts_for_etm.append("")
        
        # Tokenize negatives for ETM  
        neg_tokens = self.tokenizer(
            negative_texts_for_etm,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        neg_input_ids = neg_tokens["input_ids"].to(self.device).contiguous()
        neg_attn_mask = neg_tokens["attention_mask"].to(self.device).contiguous()
        
        # For ETG, encode reports
        report_raw = batch.get("report")
        if isinstance(report_raw, Sequence):
            report_texts = [str(r) if r is not None else "" for r in report_raw]
        else:
            report_texts = positive_texts
        
        report_tokens = self.tokenizer(
            report_texts,
            padding=True,
            truncation=True,
            max_length=self.max_text_length,
            return_tensors="pt",
        )
        report_input_ids = report_tokens["input_ids"].to(self.device).contiguous()
        report_attn_mask = report_tokens["attention_mask"].to(self.device).contiguous()
        report_input_ids_etg, report_attn_mask_etg = self._prepend_dec_token(
            report_input_ids,
            report_attn_mask,
            self.max_text_length,
        )
        
        # Align tensor shapes for ETM
        (
            pos_input_ids,
            pos_attn_mask,
            neg_input_ids,
            neg_attn_mask,
        ) = self._align_token_tensor_shapes(
            pos_input_ids,
            pos_attn_mask,
            neg_input_ids,
            neg_attn_mask,
        )

        codes = codes.to(self.device).contiguous()

        autocast_ctx = (
            autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_autocast)
            if self.use_autocast
            else nullcontext()
        )

        current_step = self.global_step
        effective_etg_weight = 0.0
        if current_step >= self.etg_delay_steps:
            if self.etg_warmup_steps > 0:
                progress = (current_step - self.etg_delay_steps + 1) / self.etg_warmup_steps
                progress = float(max(0.0, min(1.0, progress)))
            else:
                progress = 1.0
            effective_etg_weight = self.etg_weight * progress

        compute_etg = current_step >= self.etg_delay_steps
        etm_weight_eff = self.etm_weight
        if self.etm_warmup_steps > 0 and current_step < self.etm_warmup_steps:
            warm_frac = float(current_step + 1) / float(self.etm_warmup_steps)
            etm_weight_eff = self.etm_weight * min(1.0, warm_frac)

        with autocast_ctx:
            # Compute ECG embeddings for all unique ECGs (batch_size = 96)
            ecg_vec_all, _ = self.bridge.forward_stage1(
                codes,
                None,
                None,
                mode="ETC",
            )
            ecg_vec_all = F.normalize(ecg_vec_all, p=2, dim=-1, eps=1e-6).contiguous()
            
            # Expand ECG embeddings to match flattened positives using ecg_indices
            # ecg_vec[i] corresponds to the ECG for positive_ids[i]
            ecg_vec = torch.stack([ecg_vec_all[idx] for idx in ecg_indices], dim=0)

            # text_features already computed and normalized above from cached embeddings!
            # No need to call text_encoder again

            # Build ecg_ids list for loss
            ecg_ids_raw = batch.get("ecg_id", [""] * batch_size)
            ecg_ids_for_loss = [str(ecg_ids_raw[idx]) if idx < len(ecg_ids_raw) else "" for idx in ecg_indices]
            
            loss_etc, siglip_stats, _ = self._siglip_loss_with_global(
                ecg_vec,
                text_features,
                positive_ids,
                positive_texts,
                negative_ids=negative_id_pool,
                negative_texts=None,  # Use pre-encoded bank
                ecg_ids=ecg_ids_for_loss,
            )
            sim_matrix = ecg_vec @ text_features.t()
            sim_matrix = sim_matrix.contiguous()

            siglip_metrics = dict(siglip_stats)

            # Hard negative mining from in-batch similarities (using already-tokenized texts)
            hard_neg_input_ids, hard_neg_attn_mask = self._select_etm_hard_negatives(
                sim_matrix=sim_matrix,
                pos_input_ids=pos_input_ids,
                pos_attn_mask=pos_attn_mask,
                neg_input_ids=neg_input_ids,
                neg_attn_mask=neg_attn_mask,
                positive_ids=positive_ids,
                ecg_ids=ecg_ids_for_loss,
            )

            # For ETM, we need ECG codes for each flattened positive
            # Stack codes corresponding to each positive using ecg_indices
            codes_for_positives = torch.stack([codes[idx] for idx in ecg_indices], dim=0)
            flattened_batch_size = len(positive_ids)
            
            codes_etm = torch.cat([codes_for_positives, codes_for_positives], dim=0).contiguous()
            text_ids_etm = torch.cat([pos_input_ids, hard_neg_input_ids], dim=0).contiguous()
            mask_etm = torch.cat([pos_attn_mask, hard_neg_attn_mask], dim=0).contiguous()
            logits_etm = self.bridge.forward_stage1(
                codes_etm,
                text_ids_etm,
                mask_etm,
                mode="ETM",
            )
            targets_etm = torch.cat(
                [
                    torch.ones(flattened_batch_size, dtype=torch.long, device=self.device),
                    torch.zeros(flattened_batch_size, dtype=torch.long, device=self.device),
                ],
                dim=0,
            )
            loss_etm = F.cross_entropy(logits_etm, targets_etm)
            etm_acc = (logits_etm.argmax(dim=-1) == targets_etm).float().mean()
            probs_detached = torch.softmax(logits_etm.detach(), dim=-1)
            etm_pos_probs = probs_detached[:, 1]
            etm_probs_for_auc = etm_pos_probs
            if flattened_batch_size > 0 and etm_pos_probs.size(0) >= flattened_batch_size * 2:
                pos_mean_val = float(etm_pos_probs[:flattened_batch_size].mean().item())
                neg_mean_val = float(etm_pos_probs[flattened_batch_size:].mean().item())
                if math.isfinite(pos_mean_val) and math.isfinite(neg_mean_val) and pos_mean_val < neg_mean_val:
                    etm_probs_for_auc = 1.0 - etm_pos_probs
            targets_detached = targets_etm.detach()
            probs_for_auc = etm_probs_for_auc.detach()
            if self._is_distributed():
                targets_detached = self._gather_tensor(targets_detached, with_grad=False)
                probs_for_auc = self._gather_tensor(probs_for_auc, with_grad=False)
            etm_auroc = self._compute_binary_auroc(targets_detached, probs_for_auc)

            if compute_etg:
                lm_logits, loss_etg = self.bridge.forward_stage1(
                    codes,
                    report_input_ids_etg,
                    report_attn_mask_etg,
                    mode="ETG",
                )
                etg_metrics = self._compute_etg_teacher_forcing_metrics(
                    lm_logits.detach(),
                    report_input_ids_etg,
                    report_attn_mask_etg,
                )
            else:
                lm_logits = None
                loss_etg = loss_etc.new_zeros(())
                etg_metrics = {}

            total_loss = (
                self.etc_weight * loss_etc
                + etm_weight_eff * loss_etm
                + effective_etg_weight * loss_etg
            )

        loss_to_backprop = total_loss / self.grad_accum

        if self.scaler is not None:
            self.scaler.scale(loss_to_backprop).backward()
        else:
            loss_to_backprop.backward()

        should_step = ((step_idx + 1) % self.grad_accum == 0) or is_last_batch

        if should_step:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            all_params = [p for group in self.optimizer.param_groups for p in group.get("params", []) if p is not None]
            if all_params:
                torch.nn.utils.clip_grad_norm_(all_params, self.grad_clip)
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)

        # training samples are not logged; validation handles qualitative logging

        is_first_step = self.global_step == 0
        metrics_out: Dict[str, float] = {
            "loss": float(total_loss.detach().item()),
            "loss_etc": float(loss_etc.detach().item()),
            "loss_etm": float(loss_etm.detach().item()),
            "loss_etg": float(loss_etg.detach().item()),
            "etm_acc": float(etm_acc.detach().item()),
            "logit_scale": siglip_stats["logit_scale"],
            "etg_weight_eff": float(effective_etg_weight),
            "etm_weight_eff": float(etm_weight_eff),
            "etm_auroc": float(etm_auroc),
        }
        metrics_out.update(siglip_metrics)
        metrics_out.update(etg_metrics)
        if is_first_step and self.config.is_ref_device:
            candidate_count = metrics_out.get("siglip_candidate_count", float("nan"))
            rand_r1 = metrics_out.get("siglip_random_r1", float("nan"))
            pos_prob = metrics_out.get("siglip_positive_prob", float("nan"))
            etc_loss = metrics_out.get("siglip_loss", float("nan"))
            scale_val = metrics_out.get("logit_scale", float("nan"))
            print(
                f"[Stage1][ETC sanity] candidates≈{candidate_count:.1f} "
                f"rand@1≈{rand_r1:.4f} pos_prob≈{pos_prob:.4f} "
                f"etc_loss≈{etc_loss:.4f} scale≈{scale_val:.4f}"
            )
        self.global_step += 1
        return metrics_out

    # ------------------------------------------------------------------
    def _compute_codes(self, signals: torch.Tensor) -> torch.Tensor:
        if self.quantizer is None:
            raise ValueError("Stage-1 runner requires a quantizer to produce discrete codes.")

        with torch.no_grad():
            encoder_feats = self.encoder(signals)
            quantized_outputs = self.quantizer(encoder_feats, return_all_codes=True)

            if len(quantized_outputs) == 4:
                _, indices, _, _ = quantized_outputs
            else:
                _, indices, _ = quantized_outputs

            keep = getattr(self.config, "num_codebooks_kept", None)
            offset_cfg = getattr(self.config, "codebook_offset", 0)
            total_codebooks = indices.size(-1)
            if keep is None or keep <= 0 or keep > total_codebooks:
                keep = total_codebooks
            max_valid_offset = max(total_codebooks - keep, 0)
            if offset_cfg < 0:
                offset = max_valid_offset
            else:
                offset = max(0, min(offset_cfg, max_valid_offset))
            # Persist resolved values for visibility
            self.config.codebook_offset = int(offset)
            self.config.bridge_num_codebooks = int(keep if keep is not None else total_codebooks)
            end = min(offset + keep, total_codebooks)
            codes = indices[..., offset:end].long()
            if codes.size(-1) == 1:
                codes = codes.squeeze(-1)

            # Suppressed verbose codebook slice diagnostics
        return codes

    def _validation_step(self, batch: Dict[str, Any]) -> Tuple[Dict[str, float], List[Dict[str, Any]], Dict[str, Any]]:
        signals: torch.Tensor = batch["signal"].to(self.device, non_blocking=True)
        batch_size = signals.size(0)
        # Prepare containers for any qualitative samples collected during validation
        samples: List[Dict[str, Any]] = []
        
        # NEW FORMAT: Lists of lists (one list per ECG)
        positive_texts_lists = batch.get("positive_texts_lists", [])
        positive_ids_lists = batch.get("positive_ids_lists", [])
        negative_ids_lists = batch.get("negative_ids_lists", [])
        
        # Collect ALL unique positive texts across all ECGs before encoding
        unique_pos_texts: Dict[str, str] = {}
        for ecg_pos_ids, ecg_pos_texts in zip(positive_ids_lists, positive_texts_lists):
            if isinstance(ecg_pos_ids, (list, tuple)) and isinstance(ecg_pos_texts, (list, tuple)):
                for tid, txt in zip(ecg_pos_ids, ecg_pos_texts):
                    tid_str = str(tid)
                    if tid_str and tid_str not in unique_pos_texts:
                        unique_pos_texts[tid_str] = str(txt)
        
        # Encode only unique positive texts
        pos_text_embeddings: Dict[str, torch.Tensor] = {}
        if unique_pos_texts:
            pos_texts_to_encode = list(unique_pos_texts.values())
            pos_ids_to_encode = list(unique_pos_texts.keys())
            
            pos_tokens = self.tokenizer(
                pos_texts_to_encode,
                padding=True,
                truncation=True,
                max_length=self.max_text_length,
                return_tensors="pt",
            )
            pos_input_ids_unique = pos_tokens["input_ids"].to(self.device)
            pos_attn_mask_unique = pos_tokens["attention_mask"].to(self.device)
            
            with torch.no_grad():
                pos_embeddings = self.text_encoder(pos_input_ids_unique, pos_attn_mask_unique)
                pos_embeddings = F.normalize(pos_embeddings, dim=-1)
            
            for tid, emb in zip(pos_ids_to_encode, pos_embeddings):
                pos_text_embeddings[tid] = emb
        
        # Flatten for loss computation
        positive_texts: List[str] = []
        positive_ids: List[str] = []
        text_features_list: List[torch.Tensor] = []
        negative_id_pool: List[List[str]] = []
        ecg_indices: List[int] = []
        
        default_dtype = torch.float32
        if pos_text_embeddings:
            default_dtype = next(iter(pos_text_embeddings.values())).dtype
        
        for ecg_idx in range(batch_size):
            ecg_pos_ids = positive_ids_lists[ecg_idx] if ecg_idx < len(positive_ids_lists) else []
            ecg_pos_texts = positive_texts_lists[ecg_idx] if ecg_idx < len(positive_texts_lists) else []
            ecg_neg_ids = negative_ids_lists[ecg_idx] if ecg_idx < len(negative_ids_lists) else []
            
            if not isinstance(ecg_pos_ids, (list, tuple)):
                ecg_pos_ids = [ecg_pos_ids]
            if not isinstance(ecg_pos_texts, (list, tuple)):
                ecg_pos_texts = [ecg_pos_texts]
            if not isinstance(ecg_neg_ids, (list, tuple)):
                ecg_neg_ids = [ecg_neg_ids]
            
            for pos_id, pos_text in zip(ecg_pos_ids, ecg_pos_texts):
                pos_id_str = str(pos_id)
                positive_ids.append(pos_id_str)
                positive_texts.append(str(pos_text))
                negative_id_pool.append([str(nid) for nid in ecg_neg_ids])
                ecg_indices.append(ecg_idx)
                
                if pos_id_str in pos_text_embeddings:
                    text_features_list.append(pos_text_embeddings[pos_id_str])
                else:
                    text_features_list.append(torch.zeros(768, device=self.device, dtype=default_dtype))
        
        if text_features_list:
            text_features = torch.stack(text_features_list, dim=0)
        else:
            text_features = torch.zeros((0, 768), device=self.device, dtype=default_dtype)
        
        codes = self._compute_codes(signals)
        codes = codes.to(self.device).contiguous()

        autocast_ctx = (
            autocast(self.device.type, dtype=self.amp_dtype, enabled=self.use_autocast)
            if self.use_autocast
            else nullcontext()
        )

        tail_info: Dict[str, Any] = {}

        with torch.no_grad():
            with autocast_ctx:
                # Compute ECG embeddings for all unique ECGs
                ecg_vec_all, _ = self.bridge.forward_stage1(
                    codes,
                    None,
                    None,
                    mode="ETC",
                )
                ecg_vec_all = F.normalize(ecg_vec_all, p=2, dim=-1, eps=1e-6).contiguous()
                
                # Expand ECG embeddings to match flattened positives
                ecg_vec = torch.stack([ecg_vec_all[idx] for idx in ecg_indices], dim=0)
                
                # text_features already computed earlier!
                
                # Build ecg_ids list
                ecg_ids_raw = batch.get("ecg_id", [""] * batch_size)
                ecg_ids_for_loss = [str(ecg_ids_raw[idx]) if idx < len(ecg_ids_raw) else "" for idx in ecg_indices]
                
                loss_etc, siglip_stats, candidate_details = self._siglip_loss_with_global(
                    ecg_vec,
                    text_features,
                    positive_ids,
                    positive_texts,
                    negative_ids=negative_id_pool,
                    negative_texts=None,  # Use pre-encoded bank
                    ecg_ids=ecg_ids_for_loss,
                )
                siglip_metrics = dict(siglip_stats)
                sim_matrix = (ecg_vec @ text_features.t()).contiguous()

                # Compute ETM for validation (match training)
                # Tokenize positive texts
                pos_tokens = self.tokenizer(
                    positive_texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_text_length,
                    return_tensors="pt",
                )
                pos_input_ids = pos_tokens["input_ids"].to(self.device).contiguous()
                pos_attn_mask = pos_tokens["attention_mask"].to(self.device).contiguous()
                
                # Sample one negative per positive for ETM
                negative_texts_for_etm = []
                for neg_pool in negative_id_pool:
                    if neg_pool:
                        neg_id = neg_pool[0] if neg_pool else ""
                        negative_texts_for_etm.append(self.text_lookup.get(neg_id, ""))
                    else:
                        negative_texts_for_etm.append("")
                
                # Tokenize negatives
                neg_tokens = self.tokenizer(
                    negative_texts_for_etm,
                    padding=True,
                    truncation=True,
                    max_length=self.max_text_length,
                    return_tensors="pt",
                )
                neg_input_ids = neg_tokens["input_ids"].to(self.device).contiguous()
                neg_attn_mask = neg_tokens["attention_mask"].to(self.device).contiguous()
                
                # Align tensor shapes for ETM
                (
                    pos_input_ids,
                    pos_attn_mask,
                    neg_input_ids,
                    neg_attn_mask,
                ) = self._align_token_tensor_shapes(
                    pos_input_ids,
                    pos_attn_mask,
                    neg_input_ids,
                    neg_attn_mask,
                )
                
                # Hard negative mining
                hard_neg_input_ids, hard_neg_attn_mask = self._select_etm_hard_negatives(
                    sim_matrix=sim_matrix,
                    pos_input_ids=pos_input_ids,
                    pos_attn_mask=pos_attn_mask,
                    neg_input_ids=neg_input_ids,
                    neg_attn_mask=neg_attn_mask,
                    positive_ids=positive_ids,
                    ecg_ids=ecg_ids_for_loss,
                )
                
                # Stack codes for each positive using ecg_indices
                codes_for_positives = torch.stack([codes[idx] for idx in ecg_indices], dim=0)
                flattened_batch_size = len(positive_ids)
                
                # ETM forward pass
                codes_etm = torch.cat([codes_for_positives, codes_for_positives], dim=0).contiguous()
                text_ids_etm = torch.cat([pos_input_ids, hard_neg_input_ids], dim=0).contiguous()
                mask_etm = torch.cat([pos_attn_mask, hard_neg_attn_mask], dim=0).contiguous()
                logits_etm = self.bridge.forward_stage1(
                    codes_etm,
                    text_ids_etm,
                    mask_etm,
                    mode="ETM",
                )
                targets_etm = torch.cat(
                    [
                        torch.ones(flattened_batch_size, dtype=torch.long, device=self.device),
                        torch.zeros(flattened_batch_size, dtype=torch.long, device=self.device),
                    ],
                    dim=0,
                )
                loss_etm = F.cross_entropy(logits_etm, targets_etm)
                etm_acc = (logits_etm.argmax(dim=-1) == targets_etm).float().mean()
                
                # Compute AUROC
                probs_detached = torch.softmax(logits_etm.detach(), dim=-1)
                etm_pos_probs = probs_detached[:, 1]
                etm_probs_for_auc = etm_pos_probs
                if flattened_batch_size > 0 and etm_pos_probs.size(0) >= flattened_batch_size * 2:
                    pos_mean_val = float(etm_pos_probs[:flattened_batch_size].mean().item())
                    neg_mean_val = float(etm_pos_probs[flattened_batch_size:].mean().item())
                    if math.isfinite(pos_mean_val) and math.isfinite(neg_mean_val) and pos_mean_val < neg_mean_val:
                        etm_probs_for_auc = 1.0 - etm_pos_probs
                targets_detached = targets_etm.detach()
                probs_for_auc = etm_probs_for_auc.detach()
                if self._is_distributed():
                    targets_detached = self._gather_tensor(targets_detached, with_grad=False)
                    probs_for_auc = self._gather_tensor(probs_for_auc, with_grad=False)
                etm_auroc = self._compute_binary_auroc(targets_detached, probs_for_auc)
                
                # Optional ETG for validation (teacher forcing) on a subset
                loss_etg = loss_etc.new_zeros(())
                etg_metrics: Dict[str, float] = {}
                any_etg_rows = False
                if bool(getattr(self.config, "validate_with_etg", False)):
                    try:
                        report_raw = batch.get("report")
                        if isinstance(report_raw, Sequence):
                            report_texts = [str(r) if r is not None else "" for r in report_raw]
                        else:
                            # Fallback: reuse positive_texts when reports are missing
                            report_texts = positive_texts
                        rep_tokens = self.tokenizer(
                            report_texts,
                            padding=True,
                            truncation=True,
                            max_length=self.max_text_length,
                            return_tensors="pt",
                        )
                        rep_input_ids = rep_tokens["input_ids"].to(self.device).contiguous()
                        rep_attn_mask = rep_tokens["attention_mask"].to(self.device).contiguous()
                        rep_ids_etg, rep_mask_etg = self._prepend_dec_token(
                            rep_input_ids,
                            rep_attn_mask,
                            self.max_text_length,
                        )
                        frac = float(getattr(self.config, "validate_etg_fraction", 0.10) or 0.10)
                        num_sel = max(1, int(math.ceil(batch_size * max(0.0, min(1.0, frac)))))
                        sel_indices = list(range(0, batch_size, max(1, batch_size // num_sel)))[:num_sel]
                        if sel_indices:
                            codes_sel = codes[sel_indices]
                            rep_ids_sel = rep_ids_etg[sel_indices]
                            rep_mask_sel = rep_mask_etg[sel_indices]
                            lm_logits, loss_etg_tensor = self.bridge.forward_stage1(
                                codes_sel,
                                rep_ids_sel,
                                rep_mask_sel,
                                mode="ETG",
                            )
                            loss_etg = loss_etg_tensor
                            etg_stats = self._compute_etg_teacher_forcing_metrics(
                                lm_logits.detach(),
                                rep_ids_sel,
                                rep_mask_sel,
                            )
                            etg_metrics = {
                                "etg_next_acc": float(etg_stats.get("etg_next_acc", float("nan"))),
                                "etg_copy_rate": float(etg_stats.get("etg_copy_rate", float("nan"))),
                            }
                            # Decode greedy argmax predictions from teacher forcing logits for preview
                            try:
                                argmax_ids = lm_logits.argmax(dim=-1)
                                pad_id = int(self.tokenizer.pad_token_id) if self.tokenizer.pad_token_id is not None else None
                            except Exception:
                                argmax_ids = None
                                pad_id = None
                            # Attach sample rows (include generated text when available)
                            for local_idx, gidx in enumerate(sel_indices):
                                generated = ""
                                if argmax_ids is not None:
                                    mask = rep_mask_sel[local_idx].bool().clone()
                                    if mask.numel() > 0:
                                        # Drop the leading [DEC] from decoding
                                        mask[0] = False
                                    pred_ids_full = argmax_ids[local_idx]
                                    if pad_id is not None:
                                        # Exclude any pad positions from decoding
                                        mask = mask & (rep_ids_sel[local_idx] != pad_id)
                                    try:
                                        pred_ids = pred_ids_full[mask].tolist()
                                        generated = self.tokenizer.decode(pred_ids, skip_special_tokens=True)
                                    except Exception:
                                        generated = ""
                                samples.append({
                                    "ecg_id": str(batch.get("ecg_id", [""] * batch_size)[gidx]) if isinstance(batch.get("ecg_id"), Sequence) else "",
                                    "report_text": report_texts[gidx] if gidx < len(report_texts) else "",
                                    "generated_report": generated,
                                    "etg_loss": float(loss_etg.detach().item()),
                                    "etg_next_acc": etg_metrics.get("etg_next_acc", float("nan")),
                                    "etg_copy_rate": etg_metrics.get("etg_copy_rate", float("nan")),
                                    "_etg_record": True,
                                })
                                any_etg_rows = True
                    except Exception:
                        # Fail-soft: keep ETG at zero if anything goes wrong
                        pass

                # Fallback: if ETG is enabled but we could not append any rows (e.g., very short
                # tokenized reports), add a minimal preview row so that CSVs are not empty.
                if bool(getattr(self.config, "validate_with_etg", False)) and not any_etg_rows:
                    try:
                        gidx = 0
                        fallback_report = ""
                        report_raw_fb = batch.get("report")
                        if isinstance(report_raw_fb, Sequence) and len(report_raw_fb) > 0:
                            fallback_report = str(report_raw_fb[0]) if report_raw_fb[0] is not None else ""
                        if not fallback_report and positive_texts:
                            fallback_report = positive_texts[0]
                        samples.append({
                            "ecg_id": str(batch.get("ecg_id", [""] * batch_size)[gidx]) if isinstance(batch.get("ecg_id"), Sequence) else "",
                            "report_text": fallback_report,
                            "generated_report": "",
                            "etg_loss": float(loss_etg.detach().item()) if isinstance(loss_etg, torch.Tensor) else float("nan"),
                            "etg_next_acc": float("nan"),
                            "etg_copy_rate": float("nan"),
                            "_etg_record": True,
                        })
                    except Exception:
                        pass
                
                total_loss = (
                    self.etc_weight * loss_etc
                    + self.etm_weight * loss_etm
                    + self.etg_weight * loss_etg
                )
                # Retrieval samples disabled in this path; keep metrics only
                retrieval_stats = {}

        metrics_out: Dict[str, float] = {
            "loss": float(total_loss.detach().item()),
            "loss_etc": float(loss_etc.detach().item()),
            "loss_etm": float(loss_etm.detach().item()),
            "loss_etg": float(loss_etg.detach().item()),
            "etm_acc": float(etm_acc.detach().item()),
            "etm_auroc": float(etm_auroc),
            "logit_scale": siglip_stats.get("logit_scale", float("nan")),
        }
        # Include ETG metrics if available
        if etg_metrics:
            metrics_out.update(etg_metrics)
        metrics_out.update(siglip_metrics)
        if retrieval_stats:
            loss_batch_size = float(siglip_metrics.get("siglip_batch_size", 0.0))
            retrieval_batch_size = float(retrieval_stats.get("batch_size", 0.0))
            if loss_batch_size <= 0.0 and retrieval_batch_size > 0.0:
                metrics_out["siglip_batch_size"] = retrieval_batch_size
            metrics_out["siglip_retrieval_batch_size"] = retrieval_batch_size
            batch_size_retrieval = max(1.0, retrieval_batch_size)
            pos_mass_total = float(retrieval_stats.get("pos_mass_total", 0.0))
            neg_mass_total = float(retrieval_stats.get("neg_mass_total", 0.0))
            alignment_total = float(retrieval_stats.get("alignment_total", 0.0))
            candidate_count_total = float(retrieval_stats.get("candidate_count_total", 0.0))
            duplicate_fraction_total = float(retrieval_stats.get("duplicate_fraction_total", 0.0))
            positive_prob_total = float(retrieval_stats.get("positive_prob_total", 0.0))
            random_r1_total = float(retrieval_stats.get("random_r1_total", 0.0))
            metrics_out["siglip_pos_mass"] = pos_mass_total / batch_size_retrieval
            metrics_out["siglip_neg_mass"] = neg_mass_total / batch_size_retrieval
            metrics_out["alignment_score"] = alignment_total / batch_size_retrieval
            metrics_out["siglip_candidate_count"] = candidate_count_total / batch_size_retrieval
            metrics_out["siglip_duplicate_fraction"] = duplicate_fraction_total / batch_size_retrieval
            metrics_out["siglip_positive_prob"] = positive_prob_total / batch_size_retrieval
            metrics_out["siglip_random_r1"] = random_r1_total / batch_size_retrieval
            recall1_hits = float(retrieval_stats.get("recall_at_1_hits", 0.0))
            recall5_hits = float(retrieval_stats.get("recall_at_5_hits", 0.0))
            metrics_out["siglip_recall_at_1_hits"] = recall1_hits
            metrics_out["siglip_recall_at_5_hits"] = recall5_hits
            if batch_size_retrieval > 0:
                metrics_out["siglip_recall_at_1"] = recall1_hits / batch_size_retrieval
                metrics_out["siglip_recall_at_5"] = recall5_hits / batch_size_retrieval
            pos_prob_max_total = retrieval_stats.get("positive_prob_max_total")
            if pos_prob_max_total is not None:
                metrics_out["siglip_positive_prob_max"] = float(pos_prob_max_total) / batch_size_retrieval
            etm_prob_total = retrieval_stats.get("etm_prob_total")
            etm_prob_count = retrieval_stats.get("etm_prob_count", 0.0)
            if etm_prob_count:
                metrics_out["etm_positive_prob"] = float(etm_prob_total) / max(1.0, float(etm_prob_count))
            etg_next_total = retrieval_stats.get("etg_next_total")
            etg_next_count = retrieval_stats.get("etg_next_count", 0.0)
            if etg_next_count:
                metrics_out["etg_next_acc"] = float(etg_next_total) / max(1.0, float(etg_next_count))
            etg_copy_total = retrieval_stats.get("etg_copy_total")
            etg_copy_count = retrieval_stats.get("etg_copy_count", 0.0)
            if etg_copy_count:
                metrics_out["etg_copy_rate"] = float(etg_copy_total) / max(1.0, float(etg_copy_count))

        if self.tail_class_ids:
            tail_labels = set(self.tail_class_ids)
            tail_labels_seen: Set[str] = set()
            hits_k = {1: 0, 5: 0}
            totals_k = {1: 0, 5: 0}
            per_label_hits: Dict[int, Dict[str, Tuple[int, int]]] = {1: {}, 5: {}}
            availability: Dict[str, Tuple[int, int]] = {}
            for idx, pos_id in enumerate(positive_ids):
                if pos_id not in tail_labels:
                    continue
                tail_labels_seen.add(pos_id)
                details = candidate_details[idx] if idx < len(candidate_details) else {}
                top_entries = details.get("top") or []
                if not top_entries and "ids" in details:
                    ids_seq = details.get("ids", [])
                    top_entries = [(tid, "", 0.0) for tid in ids_seq[:5]]
                candidate_count = int(details.get("candidate_count", 0) or len(details.get("ids", [])))
                avail_pos, avail_total = availability.get(pos_id, (0, 0))
                if candidate_count > 0:
                    availability[pos_id] = (avail_pos + 1, avail_total + 1)
                else:
                    availability[pos_id] = (avail_pos, avail_total + 1)
                for k in (1, 5):
                    slice_entries = top_entries[:k] if top_entries else []
                    hit = 1 if any(entry[0] == pos_id for entry in slice_entries) else 0
                    hits_k[k] += hit
                    totals_k[k] += 1
                    per_label = per_label_hits.setdefault(k, {})
                    label_hits, label_total = per_label.get(pos_id, (0, 0))
                    per_label[pos_id] = (label_hits + hit, label_total + 1)
            tail_info = {
                "labels_with_pos": tail_labels_seen,
                "total_pos_examples": totals_k[1],
                "tail_hits@1": hits_k[1],
                "tail_total@1": totals_k[1],
                "tail_hits@5": hits_k[5],
                "tail_total@5": totals_k[5],
                "tail_hits_per_label@1": per_label_hits.get(1, {}),
                "tail_hits_per_label@5": per_label_hits.get(5, {}),
                "tail_availability": availability,
            }

        return metrics_out, samples, tail_info

    # ------------------------------------------------------------------
    def _maybe_save_checkpoint(
        self,
        epoch: int,
        loss: float,
        *,
        samples: Optional[Sequence[Union[Tuple[str, str], Dict[str, Any]]]] = None,
        metrics: Optional[Dict[str, float]] = None,
        best_only: bool = False,
    ) -> Optional[str]:
        if not self.config.is_ref_device:
            return None

        checkpoint_root = self.config.checkpoint_dir or os.path.join(
            self.config.output_dir or ".",
            "checkpoints",
        )
        os.makedirs(checkpoint_root, exist_ok=True)

        if best_only:
            # Delete previous best checkpoint if it exists
            if hasattr(self, 'best_checkpoint_path') and self.best_checkpoint_path and os.path.exists(self.best_checkpoint_path):
                try:
                    os.remove(self.best_checkpoint_path)
                    print(f"[ECGTextStage1Runner] Deleted previous best checkpoint: {self.best_checkpoint_path}")
                except Exception as e:
                    print(f"[ECGTextStage1Runner] Warning: Could not delete previous best checkpoint: {e}")
            
            checkpoint_path = os.path.join(checkpoint_root, f"stage1_best_epoch_{epoch + 1:03d}.pt")
            self.best_checkpoint_path = checkpoint_path
            csv_path = os.path.join(
                checkpoint_root, f"stage1_best_epoch_{epoch + 1:03d}_samples.csv"
            )
        else:
            # Delete previous last checkpoint if it exists
            if hasattr(self, 'last_checkpoint_path') and self.last_checkpoint_path and os.path.exists(self.last_checkpoint_path):
                try:
                    os.remove(self.last_checkpoint_path)
                    print(f"[ECGTextStage1Runner] Deleted previous last checkpoint: {self.last_checkpoint_path}")
                except Exception as e:
                    print(f"[ECGTextStage1Runner] Warning: Could not delete previous last checkpoint: {e}")
            
            checkpoint_path = os.path.join(checkpoint_root, f"stage1_last_epoch_{epoch + 1:03d}.pt")
            self.last_checkpoint_path = checkpoint_path
            csv_path = os.path.join(
                checkpoint_root, f"stage1_last_epoch_{epoch + 1:03d}_samples.csv"
            )

        self._save_checkpoint(
            model=self.bridge,
            optimizer=self.optimizer,
            epoch=epoch,
            loss=loss,
            checkpoint_path=checkpoint_path,
        )

        if samples or metrics:
            self._write_samples_csv(csv_path, samples or [], metrics=metrics)

        return checkpoint_path

    def _write_samples_csv(
        self,
        csv_path: str,
        samples: Sequence[Union[Tuple[str, str], Dict[str, Any]]],
        metrics: Optional[Dict[str, float]] = None,
    ) -> None:
        if not self.config.is_ref_device:
            return

        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        base_fields = [
            "ecg_id",
            "alignment",
            "row_pos_mass",
            "row_neg_mass",
            "ground_truth_pos_ids",
            "ground_truth_pos_texts",
            "ground_truth_pos_probs",
            "ground_truth_neg_ids",
            "ground_truth_neg_texts",
            "ground_truth_neg_probs",
            "top_pred_ids",
            "top_pred_texts",
            "top_pred_probs",
            "siglip_candidate_count",
            "siglip_duplicate_fraction",
            "siglip_positive_prob",
            "siglip_positive_prob_max",
            "siglip_random_r1",
            "siglip_recall_at_1",
            "siglip_recall_at_5",
            "siglip_tail_recall_at_1",
            "siglip_tail_recall_at_5",
            "siglip_top5",
            "etm_positive_prob",
            "etm_positive_prob_max",
            "etg_next_acc",
            "etg_copy_rate",
            "report_text",
            "generated_report",
        ]

        metric_exclude = {
            "loss",
            "loss_etc",
            "loss_etm",
            "loss_etg",
            "etm_acc",
            "etm_auroc",
            "siglip_pos_mass",
            "siglip_neg_mass",
            "siglip_candidate_vocab",
            "siglip_recall_at_1",
            "siglip_recall_at_5",
            "siglip_tail_recall_at_1",
            "siglip_tail_recall_at_5",
            "siglip_loss",
            "siglip_loss_raw",
            "siglip_loss_mismatch",
            "logit_scale",
        }

        metric_fields: List[str] = []
        if metrics:
            for key in metrics.keys():
                if key in metric_exclude:
                    continue
                if key not in base_fields and key not in metric_fields:
                    metric_fields.append(key)

        fieldnames = base_fields + metric_fields

        def _format_text(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, str):
                return value
            if isinstance(value, (list, tuple)):
                return "\n".join(str(item) for item in value)
            return str(value)

        def _format_probs(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, (list, tuple)):
                formatted: List[str] = []
                for item in value:
                    try:
                        num = float(item)
                        if math.isnan(num):
                            continue
                        formatted.append(f"{num:.4f}")
                    except (TypeError, ValueError):
                        continue
                return ", ".join(formatted)
            if isinstance(value, (int, float)):
                num = float(value)
                if math.isnan(num):
                    return ""
                return f"{num:.4f}"
            return str(value)

        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()

            for sample in samples:
                row_data = {field: "" for field in fieldnames}
                if not isinstance(sample, dict):
                    reference, prediction = sample
                    row_data["reference"] = reference
                    row_data["prediction"] = prediction
                    writer.writerow(row_data)
                    continue

                row_data["ecg_id"] = sample.get("ecg_id", "")

                for key in (
                    "alignment",
                    "row_pos_mass",
                    "row_neg_mass",
                    "siglip_candidate_count",
                    "siglip_duplicate_fraction",
                    "siglip_positive_prob",
                    "siglip_positive_prob_max",
                    "siglip_random_r1",
                    "etm_positive_prob",
                    "etm_positive_prob_max",
                    "etg_next_acc",
                    "etg_copy_rate",
                ):
                    if key in fieldnames and key in sample and sample[key] is not None:
                        value = sample[key]
                        if isinstance(value, (int, float)) and not math.isnan(float(value)):
                            row_data[key] = float(value)
                        elif isinstance(value, str):
                            row_data[key] = value

                if "ground_truth_pos_ids" in fieldnames:
                    row_data["ground_truth_pos_ids"] = _format_text(sample.get("ground_truth_pos_ids"))
                if "ground_truth_pos_texts" in fieldnames:
                    row_data["ground_truth_pos_texts"] = _format_text(sample.get("ground_truth_pos_texts"))
                if "ground_truth_pos_probs" in fieldnames:
                    row_data["ground_truth_pos_probs"] = _format_probs(sample.get("ground_truth_pos_probs"))
                if "ground_truth_neg_ids" in fieldnames:
                    row_data["ground_truth_neg_ids"] = _format_text(sample.get("ground_truth_neg_ids"))
                if "ground_truth_neg_texts" in fieldnames:
                    row_data["ground_truth_neg_texts"] = _format_text(sample.get("ground_truth_neg_texts"))
                if "ground_truth_neg_probs" in fieldnames:
                    row_data["ground_truth_neg_probs"] = _format_probs(sample.get("ground_truth_neg_probs"))
                if "top_pred_ids" in fieldnames:
                    row_data["top_pred_ids"] = _format_text(sample.get("top_pred_ids"))
                if "top_pred_texts" in fieldnames:
                    row_data["top_pred_texts"] = _format_text(sample.get("top_pred_texts"))
                if "top_pred_probs" in fieldnames:
                    row_data["top_pred_probs"] = _format_probs(sample.get("top_pred_probs"))

                if "siglip_top5" in fieldnames:
                    row_data["siglip_top5"] = _format_text(sample.get("siglip_top5"))

                if "report_text" in fieldnames:
                    row_data["report_text"] = sample.get("report_text", "")
                if "generated_report" in fieldnames:
                    row_data["generated_report"] = sample.get("generated_report", "")

                for key in metric_fields:
                    if key in sample and sample[key] is not None:
                        value = sample[key]
                        if isinstance(value, (int, float)):
                            if math.isnan(float(value)):
                                continue
                            row_data[key] = float(value)
                        else:
                            row_data[key] = value
                for key in metric_fields:
                    if key.startswith("tail/") and not row_data.get(key):
                        if getattr(self.config, "tail_enable", False):
                            row_data[key] = 0.0

                writer.writerow(row_data)

            if metrics:
                summary = {key: "" for key in fieldnames}
                summary["ecg_id"] = "__summary__"
                for key in fieldnames:
                    if key in metrics and metrics[key] is not None:
                        value = metrics[key]
                        if isinstance(value, (int, float)):
                            if math.isnan(float(value)):
                                continue
                            summary[key] = float(value)
                        else:
                            summary[key] = value
                writer.writerow(summary)

    def _collect_text_samples(
        self,
        *,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits: torch.Tensor,
        references: Optional[Sequence[str]],
        ecg_ids: Sequence[Any],
        positive_ids: Sequence[str],
        positive_texts: Sequence[str],
        positive_embeddings: torch.Tensor,
        candidate_info: Sequence[Dict[str, Any]],
        etm_pos_probs: torch.Tensor,
        negative_id_pool: Sequence[Sequence[str]],
        negative_text_pool: Sequence[Sequence[str]],
        ecg_embeddings: torch.Tensor,
        logit_scale: float,
        temperature: Optional[float],
        num_samples: int,
    ) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
        batch_size = input_ids.size(0)
        if batch_size == 0:
            return [], {}

        preds = logits.argmax(dim=-1)
        pad_id = self.tokenizer.pad_token_id

        reference_list: Optional[List[str]] = None
        if references is not None and isinstance(references, (list, tuple)):
            reference_list = [str(r) for r in references]

        logit_scale_value = 1.0
        if isinstance(temperature, (int, float)) and temperature is not None:
            temp_val = float(temperature)
            if temp_val > 0.0 and math.isfinite(temp_val):
                logit_scale_value = 1.0 / temp_val
        if (
            isinstance(logit_scale, (int, float))
            and math.isfinite(float(logit_scale))
            and float(logit_scale) > 0.0
        ):
            logit_scale_value = float(logit_scale)

        positive_embeddings_map: Dict[str, torch.Tensor] = {}
        positive_text_lookup: Dict[str, str] = {}
        for idx, pos_id in enumerate(positive_ids):
            if not pos_id:
                continue
            positive_embeddings_map.setdefault(pos_id, positive_embeddings[idx])
            if idx < len(positive_texts):
                positive_text_lookup.setdefault(pos_id, str(positive_texts[idx]))

        grouped: Dict[str, Dict[str, Any]] = {}

        for idx in range(batch_size):
            mask = attention_mask[idx].bool()
            if mask.numel() > 0:
                mask[0] = False

            pred_ids_full = preds[idx]
            pred_ids = pred_ids_full[mask].tolist()

            gold_ids_full = input_ids[idx]
            gold_ids = gold_ids_full[mask].tolist()

            if reference_list is not None and idx < len(reference_list):
                gold_text = reference_list[idx]
            else:
                gold_text = self.tokenizer.decode(gold_ids, skip_special_tokens=True)

            pred_text = self.tokenizer.decode(pred_ids, skip_special_tokens=True)

            next_targets = gold_ids_full.roll(-1, dims=0)
            if pad_id is not None:
                next_targets[-1] = pad_id
            next_mask = mask.clone()
            if next_mask.numel() > 0:
                next_mask[-1] = False
            if pad_id is not None:
                next_mask = next_mask & (next_targets != pad_id)
            if next_mask.any():
                next_hits = (pred_ids_full == next_targets) & next_mask
                next_acc = float(next_hits.sum().item() / next_mask.sum().item())
            else:
                next_acc = float("nan")

            copy_hits = (pred_ids_full == gold_ids_full) & mask
            copy_rate = float(copy_hits.sum().item() / mask.sum().item()) if mask.any() else float("nan")

            ecg_id = str(ecg_ids[idx]) if idx < len(ecg_ids) else str(idx)
            group = grouped.get(ecg_id)
            if group is None:
                group = {
                    "ecg_vecs": [ecg_embeddings[idx]],
                    "reference": gold_text,
                    "prediction": pred_text,
                    "pos_ids": set(),
                    "pos_texts": {},
                    "candidate_sequence": [],
                    "candidate_texts": {},
                    "explicit_neg_texts": {},
                    "etm_probs": [],
                    "etg_next": [],
                    "etg_copy": [],
                }
                grouped[ecg_id] = group
            else:
                group["ecg_vecs"].append(ecg_embeddings[idx])
                if not group["reference"]:
                    group["reference"] = gold_text
                if not group["prediction"]:
                    group["prediction"] = pred_text

            pos_id = str(positive_ids[idx]) if idx < len(positive_ids) else ""
            pos_text = str(positive_texts[idx]) if idx < len(positive_texts) else ""
            if pos_id:
                group["pos_ids"].add(pos_id)
                if pos_text:
                    group["pos_texts"][pos_id] = pos_text
                elif pos_id in positive_text_lookup:
                    group["pos_texts"][pos_id] = positive_text_lookup[pos_id]

            info = candidate_info[idx] if idx < len(candidate_info) else {}
            candidate_ids_list = [str(tid) for tid in (info.get("ids") or []) if tid is not None]
            candidate_texts_list = [str(text) for text in (info.get("texts") or [])]
            if len(candidate_texts_list) < len(candidate_ids_list):
                candidate_texts_list.extend([""] * (len(candidate_ids_list) - len(candidate_texts_list)))
            pos_indices_info = info.get("positive_indices") or []
            pos_ids_info = [str(pid) for pid in (info.get("pos_ids") or []) if pid]

            group["candidate_sequence"].extend(candidate_ids_list)
            for local_idx, tid in enumerate(candidate_ids_list):
                if not tid:
                    continue
                text_value = candidate_texts_list[local_idx] if local_idx < len(candidate_texts_list) else ""
                if not text_value:
                    text_value = self.text_lookup.get(tid, "")
                if tid not in group["candidate_texts"] or not group["candidate_texts"][tid]:
                    group["candidate_texts"][tid] = text_value

            for local_idx, pid in enumerate(pos_ids_info):
                if not pid:
                    continue
                group["pos_ids"].add(pid)
                if pid not in group["pos_texts"] or not group["pos_texts"][pid]:
                    text_value = ""
                    if local_idx < len(pos_indices_info):
                        cand_idx = pos_indices_info[local_idx]
                        if 0 <= cand_idx < len(candidate_texts_list):
                            text_value = candidate_texts_list[cand_idx]
                    if not text_value:
                        text_value = self.text_lookup.get(pid, positive_text_lookup.get(pid, ""))
                    if not text_value and pid == pos_id:
                        text_value = pos_text
                    if text_value:
                        group["pos_texts"][pid] = text_value

            neg_ids_row = negative_id_pool[idx] if idx < len(negative_id_pool) else []
            neg_texts_row = negative_text_pool[idx] if idx < len(negative_text_pool) else []
            for neg_idx, neg_id in enumerate(neg_ids_row):
                neg_id_str = str(neg_id)
                if not neg_id_str:
                    continue
                text_value = ""
                if neg_idx < len(neg_texts_row):
                    text_value = str(neg_texts_row[neg_idx])
                if not text_value:
                    text_value = self.text_lookup.get(neg_id_str, "")
                group["explicit_neg_texts"][neg_id_str] = text_value
                group["candidate_sequence"].append(neg_id_str)
                if neg_id_str not in group["candidate_texts"] or not group["candidate_texts"][neg_id_str]:
                    group["candidate_texts"][neg_id_str] = text_value

            etm_prob_value = float(etm_pos_probs[idx].item()) if idx < etm_pos_probs.size(0) else float("nan")
            group["etm_probs"].append(etm_prob_value)
            group["etg_next"].append(next_acc)
            group["etg_copy"].append(copy_rate)

        def _embedding_for_id(text_id: str) -> Optional[torch.Tensor]:
            if text_id in positive_embeddings_map:
                return positive_embeddings_map[text_id]
            if self.text_bank_embeddings is not None and text_id in self.text_bank_index:
                idx_bank = self.text_bank_index[text_id]
                return self.text_bank_embeddings[idx_bank]
            return None

        all_rows: List[Dict[str, Any]] = []
        stats = {
            "batch_size": 0.0,
            "pos_mass_total": 0.0,
            "neg_mass_total": 0.0,
            "alignment_total": 0.0,
            "candidate_count_total": 0.0,
            "duplicate_fraction_total": 0.0,
            "positive_prob_total": 0.0,
            "positive_prob_max_total": 0.0,
            "random_r1_total": 0.0,
            "recall_at_1_hits": 0.0,
            "recall_at_5_hits": 0.0,
            "etm_prob_total": 0.0,
            "etm_prob_count": 0.0,
            "etg_next_total": 0.0,
            "etg_next_count": 0.0,
            "etg_copy_total": 0.0,
            "etg_copy_count": 0.0,
        }

        max_neg_setting = getattr(self.config, "siglip_retrieval_negatives", None)
        if max_neg_setting is None:
            max_neg_setting = getattr(self.config, "siglip_bank_negatives", None)
        if max_neg_setting is None:
            max_neg_setting = 5
        try:
            max_neg_report = int(max_neg_setting)
        except (TypeError, ValueError):
            max_neg_report = 5
        if max_neg_report < 0:
            max_neg_report = 0

        for ecg_id, group in grouped.items():
            vecs = torch.stack(group["ecg_vecs"], dim=0)
            ecg_vec = F.normalize(vecs.mean(dim=0), p=2, dim=0, eps=1e-6)

            candidate_sequence = [tid for tid in group["candidate_sequence"] if tid]
            seen_candidates: Set[str] = set()
            candidate_ids_ordered: List[str] = []
            for tid in candidate_sequence:
                if tid not in seen_candidates:
                    seen_candidates.add(tid)
                    candidate_ids_ordered.append(tid)
            for pos_id in group["pos_ids"]:
                if pos_id and pos_id not in seen_candidates:
                    seen_candidates.add(pos_id)
                    candidate_ids_ordered.append(pos_id)

            candidate_embeddings: List[torch.Tensor] = []
            final_ids: List[str] = []
            for tid in candidate_ids_ordered:
                embedding = _embedding_for_id(tid)
                if embedding is None:
                    continue
                candidate_embeddings.append(embedding)
                final_ids.append(tid)

            if not final_ids:
                continue

            candidate_tensor = torch.stack(candidate_embeddings, dim=0)
            logits_row = torch.matmul(candidate_tensor, ecg_vec)
            logits_row = logits_row * logit_scale_value
            probs = torch.softmax(logits_row, dim=0)
            prob_map = {tid: float(probs[idx].item()) for idx, tid in enumerate(final_ids)}

            pos_ids_set: Set[str] = set(group["pos_ids"])
            pos_entries: List[Tuple[str, str, float]] = []
            for tid in pos_ids_set:
                if tid not in prob_map:
                    continue
                text_value = group["pos_texts"].get(tid) or self.text_lookup.get(tid, "")
                pos_entries.append((tid, text_value, prob_map[tid]))
            pos_entries.sort(key=lambda item: item[2], reverse=True)

            explicit_neg_set = set(group["explicit_neg_texts"].keys())
            neg_entries: List[Tuple[str, str, float]] = []
            for tid, text_value in group["explicit_neg_texts"].items():
                prob_val = prob_map.get(tid, float("nan"))
                neg_entries.append((tid, text_value, prob_val))
            neg_entries.sort(
                key=lambda item: (
                    math.isnan(item[2]),
                    -(item[2] if not math.isnan(item[2]) else 0.0),
                )
            )

            other_neg_entries: List[Tuple[str, str, float]] = []
            for tid in final_ids:
                if tid in pos_ids_set or tid in explicit_neg_set:
                    continue
                prob_val = prob_map.get(tid, float("nan"))
                text_value = group["candidate_texts"].get(tid) or self.text_lookup.get(tid, "")
                other_neg_entries.append((tid, text_value, prob_val))
            other_neg_entries.sort(
                key=lambda item: (
                    math.isnan(item[2]),
                    -(item[2] if not math.isnan(item[2]) else 0.0),
                )
            )

            if max_neg_report > 0:
                limited_neg_entries = neg_entries[:max_neg_report]
                remaining = max(0, max_neg_report - len(limited_neg_entries))
                if remaining > 0:
                    limited_neg_entries += other_neg_entries[:remaining]
            else:
                limited_neg_entries = neg_entries + other_neg_entries

            top_indices = torch.argsort(probs, descending=True)
            top_entries: List[Tuple[str, str, float]] = []
            for offset in top_indices[:5].tolist():
                tid = final_ids[offset]
                text_value = group["candidate_texts"].get(tid) or self.text_lookup.get(tid, "")
                top_entries.append((tid, text_value, float(probs[offset].item())))

            pos_mass = sum(prob_map.get(tid, 0.0) for tid in pos_ids_set)
            pos_mass = float(pos_mass)
            neg_mass = float(max(0.0, 1.0 - pos_mass))
            candidate_count = float(len(final_ids))
            total_candidate_with_dups = len(candidate_sequence)
            unique_candidate_with_dups = len(set(candidate_sequence))
            duplicate_fraction = 0.0
            if total_candidate_with_dups > 0:
                duplicate_fraction = 1.0 - (unique_candidate_with_dups / total_candidate_with_dups)

            random_r1 = 1.0 / candidate_count if candidate_count > 0 else 0.0

            pos_prob_mean = float(
                sum(entry[2] for entry in pos_entries) / len(pos_entries)
            ) if pos_entries else float("nan")
            pos_prob_max = float(pos_entries[0][2]) if pos_entries else float("nan")

            etm_valid = [val for val in group["etm_probs"] if not math.isnan(val)]
            etm_prob_mean = float(sum(etm_valid) / len(etm_valid)) if etm_valid else float("nan")
            etm_prob_max = float(max(etm_valid)) if etm_valid else float("nan")

            etg_next_valid = [val for val in group["etg_next"] if not math.isnan(val)]
            etg_next_mean = float(sum(etg_next_valid) / len(etg_next_valid)) if etg_next_valid else float("nan")

            etg_copy_valid = [val for val in group["etg_copy"] if not math.isnan(val)]
            etg_copy_mean = float(sum(etg_copy_valid) / len(etg_copy_valid)) if etg_copy_valid else float("nan")

            recall_at_1_hit = 1.0 if top_entries and top_entries[0][0] in pos_ids_set else 0.0
            recall_at_5_hit = 1.0 if any(entry[0] in pos_ids_set for entry in top_entries) else 0.0

            # Calculate tail-specific recall metrics
            tail_pos_ids = pos_ids_set & self.tail_class_ids if self.tail_class_ids else set()
            tail_recall_at_1_hit = 0.0
            tail_recall_at_5_hit = 0.0
            if tail_pos_ids:
                tail_recall_at_1_hit = 1.0 if top_entries and top_entries[0][0] in tail_pos_ids else 0.0
                tail_recall_at_5_hit = 1.0 if any(entry[0] in tail_pos_ids for entry in top_entries) else 0.0

            siglip_top5_str = "; ".join(
                f"{tid}:{prob:.4f}:{(text or '')[:80]}"
                for tid, text, prob in top_entries
            )

            row = {
                "ecg_id": ecg_id,
                "alignment": pos_mass - neg_mass,
                "row_pos_mass": pos_mass,
                "row_neg_mass": neg_mass,
                "ground_truth_pos_ids": [entry[0] for entry in pos_entries],
                "ground_truth_pos_texts": [entry[1] for entry in pos_entries],
                "ground_truth_pos_probs": [entry[2] for entry in pos_entries],
                "ground_truth_neg_ids": [entry[0] for entry in limited_neg_entries],
                "ground_truth_neg_texts": [entry[1] for entry in limited_neg_entries],
                "ground_truth_neg_probs": [entry[2] for entry in limited_neg_entries],
                "top_pred_ids": [entry[0] for entry in top_entries],
                "top_pred_texts": [entry[1] for entry in top_entries],
                "top_pred_probs": [entry[2] for entry in top_entries],
                "siglip_candidate_count": candidate_count,
                "siglip_duplicate_fraction": duplicate_fraction,
                "siglip_positive_prob": pos_prob_mean,
                "siglip_positive_prob_max": pos_prob_max,
                "siglip_random_r1": random_r1,
                "siglip_recall_at_1": recall_at_1_hit,
                "siglip_recall_at_5": recall_at_5_hit,
                "siglip_tail_recall_at_1": tail_recall_at_1_hit,
                "siglip_tail_recall_at_5": tail_recall_at_5_hit,
                "siglip_top5": siglip_top5_str,
                "etm_positive_prob": etm_prob_mean,
                "etm_positive_prob_max": etm_prob_max,
                "etg_next_acc": etg_next_mean,
                "etg_copy_rate": etg_copy_mean,
                "report_text": group.get("reference", ""),
                "generated_report": group.get("prediction", ""),
                "reference": group.get("reference", ""),
                "prediction": group.get("prediction", ""),
            }
            all_rows.append(row)

            stats["batch_size"] += 1.0
            stats["pos_mass_total"] += pos_mass
            stats["neg_mass_total"] += neg_mass
            stats["alignment_total"] += (pos_mass - neg_mass)
            stats["candidate_count_total"] += candidate_count
            stats["duplicate_fraction_total"] += duplicate_fraction
            if not math.isnan(pos_prob_mean):
                stats["positive_prob_total"] += pos_prob_mean
            if not math.isnan(pos_prob_max):
                stats["positive_prob_max_total"] += pos_prob_max
            stats["random_r1_total"] += random_r1
            stats["recall_at_1_hits"] += recall_at_1_hit
            stats["recall_at_5_hits"] += recall_at_5_hit
            if not math.isnan(etm_prob_mean):
                stats["etm_prob_total"] += etm_prob_mean
                stats["etm_prob_count"] += 1.0
            if not math.isnan(etg_next_mean):
                stats["etg_next_total"] += etg_next_mean
                stats["etg_next_count"] += 1.0
            if not math.isnan(etg_copy_mean):
                stats["etg_copy_total"] += etg_copy_mean
                stats["etg_copy_count"] += 1.0

        all_rows.sort(key=lambda row: row.get("ecg_id", ""))
        if num_samples > 0 and len(all_rows) > num_samples:
            rows_to_log = all_rows[:num_samples]
        else:
            rows_to_log = all_rows

        return rows_to_log, stats

    def _log_text_table(self, samples: List[Tuple[str, str]], key: str = "train/sample_texts") -> None:
        if not samples:
            return
        if not self.wandb_wrapper or not self.wandb_wrapper.is_initialized():
            return

        try:
            import wandb  # type: ignore
        except ImportError:
            return

        table = wandb.Table(columns=["reference", "prediction"], data=samples)
        self.wandb_wrapper.log({key: table})
