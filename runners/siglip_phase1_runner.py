from __future__ import annotations

import csv
import glob
import math
import os
import random
import time
from typing import Any, Dict, Optional, List, Tuple, Iterable, Sequence

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
from utils.lm_injection import build_inputs_with_ecg_prefix, pad_labels_for_ecg_prefix


@RunnerRegistry.register(RunnerName.SIGLIP_PHASE1)
class SiglipPhase1Runner(BaseRunner):
    """Runner implementing SigLIP-style BCE alignment for ECG embeddings."""

    def __init__(
        self,
        encoder: nn.Module,
        bridge: nn.Module,
        decoder: Optional[nn.Module],
        train_dataloader: DataLoader,
        optimizer: Optimizer,
        config: SiglipPhase1Config,
        text_embeddings: torch.Tensor,
        text_id_to_idx: Dict[str, int],
        text_lookup: Dict[str, str],
        quantizer: Optional[nn.Module] = None,
        use_quantized_inputs: bool = True,
        tokenizer_config: Optional[Dict[str, Any]] = None,
        report_text_embedder: Optional[Any] = None,
        wandb_wrapper=None,
        validation_dataloader: Optional[DataLoader] = None,
        scaler: Optional[GradScaler] = None,
        scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    ) -> None:
        super().__init__(config=config, wandb_wrapper=wandb_wrapper)
        self.encoder = encoder
        self.quantizer = quantizer
        self.bridge = bridge
        self.raw_bridge = getattr(bridge, "bridge", bridge)
        self.decoder: Optional[nn.Module] = decoder
        if self.decoder is not None and hasattr(self.decoder, "bridge"):
            self.decoder.bridge = self.raw_bridge

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

        # Phase A-v2: per-ECG free-text report contrastive mode.
        self.contrastive_text_mode = str(
            getattr(config, "contrastive_text_mode", "bank") or "bank"
        ).lower()
        if self.contrastive_text_mode not in {"bank", "report"}:
            raise ValueError(
                f"Unsupported contrastive_text_mode '{self.contrastive_text_mode}'."
            )
        self.report_text_embedder = report_text_embedder
        self.recall_log_examples = int(getattr(config, "recall_log_examples", 8) or 8)
        if self.contrastive_text_mode == "report" and self.report_text_embedder is None:
            raise ValueError(
                "contrastive_text_mode='report' requires a report_text_embedder; "
                "none was provided by the project."
            )

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

        if self.decoder is not None:
            self.decoder.to(self.device)
            if hasattr(self.decoder, "freeze_llm_parameters"):
                try:
                    self.decoder.freeze_llm_parameters(exclude=("bridge",))
                except TypeError:
                    self.decoder.freeze_llm_parameters()
            for param in self.raw_bridge.parameters():
                param.requires_grad = True
            self.decoder.eval()
            self.decoder_tokenizer = getattr(self.decoder, "tokenizer", None)
        else:
            self.decoder_tokenizer = None

        dtype_map = {
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp16": torch.float16,
            "float16": torch.float16,
        }
        self.amp_dtype = dtype_map.get(config.dtype.lower(), None)
        self.use_autocast = self.amp_dtype is not None
        if self.amp_dtype == torch.bfloat16:
            self.scaler = None
        elif self.amp_dtype == torch.float16 and self.scaler is None:
            self.scaler = GradScaler()
        self.grad_clip = 1.0

        self.siglip_loss_weight = float(getattr(config, "siglip_loss_weight", 1.0) or 1.0)
        self.lm_loss_weight = float(getattr(config, "lm_loss_weight", 0.0) or 0.0)
        self.ce_max_length = int(getattr(config, "ce_max_length", 512) or 512)
        self.ce_report_probability = float(getattr(config, "ce_report_probability", 0.6) or 0.6)
        self.ce_max_samples_per_batch = int(getattr(config, "ce_max_samples_per_batch", 64) or 64)
        self.ce_enabled = self.decoder is not None and self.lm_loss_weight > 0
        self._ce_rng = random.Random(int(self.config.seed) + int(self.config.device))
        self.ce_backprop_to_bridge = bool(getattr(config, "ce_backprop_to_bridge", True))
        self.ce_scale_by_pairs = bool(getattr(config, "ce_scale_by_pairs", True))
        self.lm_weight_warmup_steps = int(getattr(config, "lm_weight_warmup_steps", 0) or 0)
        self.ce_scale_cap = float(getattr(config, "ce_scale_cap", 0.0) or 0.0)

        self.report_generation_batch_size = max(1, int(getattr(config, "batch_size", 8) or 8))
        self.report_system_prompt = (
            "You are a cardiology ECG expert. Return exactly one line of clinical findings "
            "separated by semicolons. Do not include explanations, caveats, questions, or extra text."
        )
        self.report_user_prompt = (
            "List the ECG findings in the form '<finding>; <finding>; ...'. "
            "If the tracing is normal, respond with 'Normal ECG'."
        )
        self.report_response_prefix = "Findings:"
        self.report_min_new_tokens = int(getattr(config, "report_min_new_tokens", 24) or 24)
        self.qa_system_prompt = (
            "You are a cardiology ECG expert. Answer each question with 'Yes.' or 'No.' only."
        )
        self.qa_response_prefix = "Answer:"
        self.qa_min_new_tokens = int(getattr(config, "qa_min_new_tokens", 6) or 6)
        self._report_prompt_cache: Optional[Tuple[torch.Tensor, torch.Tensor, int, str]] = None

        if self.decoder is not None and hasattr(self.decoder, "bridge"):
            assert self.decoder.bridge is self.raw_bridge, "decoder.bridge must be tied to retrieval bridge"

        self.train_encoder = bool(getattr(config, "train_encoder", False))
        if self.train_encoder:
            self.encoder.train()
            for param in self.encoder.parameters():
                param.requires_grad = True
        else:
            self.encoder.eval()
            for param in self.encoder.parameters():
                param.requires_grad = False

        if self.quantizer is not None:
            self.quantizer.eval()
            for param in self.quantizer.parameters():
                param.requires_grad = False

        txt_std = float(self.text_embeddings.float().std().item()) if self.text_embeddings.numel() else 1.0
        self.bridge.train()
        self.text_embeddings = F.normalize(self.text_embeddings, dim=-1)
        self.text_embeddings = self.text_embeddings.to(self.device)
        if hasattr(self.raw_bridge, "output_scale"):
            scale_tensor = torch.tensor(txt_std, dtype=self.raw_bridge.output_scale.dtype, device=self.raw_bridge.output_scale.device)
            with torch.no_grad():
                self.raw_bridge.output_scale.copy_(scale_tensor)

        self.best_checkpoint_path: Optional[str] = None
        self.best_metric_value: float = float("inf")
        self.best_metric_epoch: Optional[int] = None
        self.best_metric_name: Optional[str] = None
        self.last_checkpoint_path: Optional[str] = None
        self.last_checkpoint_epoch: Optional[int] = None

        self._initialize_checkpoint_state()
        self._warn_if_decoder_params_not_in_optimizer()

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

    def _bridge_param_groups(self) -> Dict[str, List[torch.nn.Parameter]]:
        """Partition bridge parameters by head for gradient diagnostics."""
        param_groups: Dict[str, List[torch.nn.Parameter]] = {
            "lm_facing": [],
            "retrieval_facing": [],
            "shared": [],
        }
        if self.raw_bridge is None:
            return param_groups

        all_params = [p for p in self.bridge.parameters() if p.requires_grad]
        id_to_param = {id(p): p for p in all_params}

        lm_modules = []
        for name in ("to_llm", "norm_out"):
            module = getattr(self.raw_bridge, name, None)
            if isinstance(module, nn.Module):
                lm_modules.append(module)
        output_scale = getattr(self.raw_bridge, "output_scale", None)
        lm_param_ids: set[int] = set()
        if lm_modules:
            for module in lm_modules:
                for param in module.parameters():
                    if param.requires_grad:
                        lm_param_ids.add(id(param))
        if isinstance(output_scale, torch.nn.Parameter) and output_scale.requires_grad:
            lm_param_ids.add(id(output_scale))

        retrieval_modules = []
        for name in ("pool_norm", "pool_gate", "to_txt"):
            module = getattr(self.raw_bridge, name, None)
            if isinstance(module, nn.Module):
                retrieval_modules.append(module)
        retrieval_param_ids: set[int] = set()
        if retrieval_modules:
            for module in retrieval_modules:
                for param in module.parameters():
                    if param.requires_grad:
                        retrieval_param_ids.add(id(param))

        for pid, param in id_to_param.items():
            if pid in lm_param_ids:
                param_groups["lm_facing"].append(param)
            elif pid in retrieval_param_ids:
                param_groups["retrieval_facing"].append(param)
            else:
                param_groups["shared"].append(param)
        return param_groups

    def _warn_if_decoder_params_not_in_optimizer(self) -> None:
        if self.decoder is None or self.optimizer is None:
            return
        opt_param_ids = {
            id(param)
            for group in self.optimizer.param_groups
            for param in group.get("params", [])
            if param is not None
        }
        missing: List[str] = []
        for name, param in self.decoder.named_parameters():
            if param.requires_grad and id(param) not in opt_param_ids:
                missing.append(name)
        if missing and getattr(self.config, "is_ref_device", True):
            preview = ", ".join(missing[:8])
            print(
                f"[{self.__class__.__name__}] WARNING: {len(missing)} trainable decoder parameters "
                f"are not in the optimizer. Examples: {preview}"
            )

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

    @staticmethod
    def _report_false_negative_mask(reports: List[str], device: torch.device) -> torch.Tensor:
        """Build a [B, B] keep-mask for in-batch report InfoNCE.

        Entry (i, j) is False (i.e. EXCLUDED from negatives) when i != j AND
        report_i == report_j (exact string match). The diagonal is always True
        (it is the positive). Empty/whitespace-only reports are treated as
        distinct so they are not collapsed together as false negatives.
        """
        n = len(reports)
        normalized = [str(r).strip() for r in reports]
        same = torch.zeros((n, n), dtype=torch.bool, device=device)
        for i in range(n):
            ri = normalized[i]
            if not ri:
                continue
            for j in range(i + 1, n):
                if normalized[j] == ri:
                    same[i, j] = True
                    same[j, i] = True
        eye = torch.eye(n, dtype=torch.bool, device=device)
        # Keep diagonal (positives) + everything that is NOT a duplicate off-diagonal.
        keep = (~same) | eye
        return keep

    def _report_infonce_loss(
        self,
        ecg_feats: torch.Tensor,
        report_embs: torch.Tensor,
        reports: List[str],
        scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """ESI/MERL-style symmetric in-batch InfoNCE over per-ECG reports.

        Args:
            ecg_feats:   [B, H] L2-normalized ECG bridge embeddings.
            report_embs: [B, H] L2-normalized report embeddings.
            reports:     list of B report strings (for false-negative masking).
            scale:       scalar temperature divisor (bridge.temperature()).

        Returns (loss, logits[B, B]). Duplicate-report off-diagonal entries are
        masked to -inf so they are not penalized as negatives.
        """
        ecg_feats = F.normalize(ecg_feats, dim=-1)
        report_embs = F.normalize(report_embs, dim=-1)
        logits = (ecg_feats @ report_embs.T) / scale  # [B, B]
        b = logits.size(0)
        keep = self._report_false_negative_mask(reports, logits.device)
        neg_inf = torch.finfo(logits.dtype).min
        logits_masked = logits.masked_fill(~keep, neg_inf)
        target = torch.arange(b, device=logits.device)
        # Symmetric: ECG->report and report->ECG (mask is symmetric).
        loss_i = F.cross_entropy(logits_masked, target)
        loss_t = F.cross_entropy(logits_masked.T, target)
        loss = 0.5 * (loss_i + loss_t)
        return loss.to(ecg_feats.dtype), logits

    @torch.no_grad()
    def _log_recall_examples(
        self,
        ecg_feats: torch.Tensor,
        report_embs: torch.Tensor,
        reports: List[str],
        ecg_ids: List[str],
        scale: torch.Tensor,
        epoch: Optional[int] = None,
    ) -> "wandb.Table":
        """Build (and optionally log) a wandb.Table of best/worst ECG->report recall.

        For each ECG, retrieve over the in-batch report gallery, compute the rank
        of its own (true) report, and select the `recall_log_examples` BEST
        (gt_rank == 1) and WORST (largest gt_rank) cases. Returns the table so it
        is unit-testable without a live wandb run.
        """
        ecg_feats = F.normalize(ecg_feats, dim=-1)
        report_embs = F.normalize(report_embs, dim=-1)
        sims = (ecg_feats @ report_embs.T) / scale  # [B, B]
        b = sims.size(0)
        # Rank of the diagonal (true report) per row: 1 = top-1.
        order = torch.argsort(sims, dim=1, descending=True)  # [B, B]
        diag = torch.arange(b, device=sims.device)
        # position of own index within each row's ranking
        ranks = torch.empty(b, dtype=torch.long, device=sims.device)
        for row in range(b):
            pos = (order[row] == diag[row]).nonzero(as_tuple=False)
            ranks[row] = int(pos[0, 0].item()) + 1 if pos.numel() else b
        top1_idx = order[:, 0].tolist()
        ranks_list = ranks.tolist()

        rows = list(range(b))
        best = sorted(rows, key=lambda r: ranks_list[r])[: self.recall_log_examples]
        worst = sorted(rows, key=lambda r: -ranks_list[r])[: self.recall_log_examples]

        table = wandb.Table(
            columns=[
                "ecg_id",
                "gt_report",
                "top1_retrieved_report",
                "gt_rank",
                "correct",
            ]
        )
        seen: set[int] = set()
        for r in list(best) + list(worst):
            if r in seen:
                continue
            seen.add(r)
            ecg_id = ecg_ids[r] if r < len(ecg_ids) else ""
            gt_report = reports[r] if r < len(reports) else ""
            t1 = top1_idx[r]
            top1_report = reports[t1] if t1 < len(reports) else ""
            gt_rank = int(ranks_list[r])
            table.add_data(
                str(ecg_id),
                str(gt_report),
                str(top1_report),
                gt_rank,
                bool(gt_rank == 1),
            )

        if (
            self.wandb_wrapper
            and self.wandb_wrapper.is_initialized()
            and getattr(self.config, "is_ref_device", True)
        ):
            key = "val/recall_examples"
            if epoch is not None:
                key = f"val/recall_examples_epoch_{epoch}"
            self.wandb_wrapper.log({key: table})
        return table

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

    @staticmethod
    def _sum_grad_abs(parameters: Iterable[torch.nn.Parameter]) -> float:
        total = 0.0
        for param in parameters:
            if param.grad is None:
                continue
            total += float(param.grad.detach().abs().sum().item())
        return total

    def _collect_lr_metrics(self, prefix: str = "train") -> Dict[str, float]:
        if self.optimizer is None:
            return {}
        metrics: Dict[str, float] = {}
        lr_values: List[float] = []
        for index, group in enumerate(self.optimizer.param_groups):
            lr = group.get("lr", None)
            if lr is None:
                continue
            lr_value = float(lr)
            lr_values.append(lr_value)
            name = str(group.get("name") or f"group_{index}")
            metrics[f"{prefix}/lr_{name}"] = lr_value
        if lr_values:
            metrics[f"{prefix}/lr_min"] = float(min(lr_values))
            metrics[f"{prefix}/lr_max"] = float(max(lr_values))
            metrics[f"{prefix}/lr_mean"] = float(sum(lr_values) / len(lr_values))
        return metrics

    def _log_grad_norms(
        self,
        bridge_norm: Optional[float],
        adapter_norm: Optional[float],
        optimizer_step: int,
        bridge_norm_pre_clip: Optional[float] = None,
        adapter_norm_pre_clip: Optional[float] = None,
    ) -> None:
        if not self.config.is_ref_device:
            return
        if not (self.wandb_wrapper and self.wandb_wrapper.is_initialized()):
            return

        payload: Dict[str, float] = {"trainer/optimizer_step": float(optimizer_step)}
        if bridge_norm is not None:
            payload["train/grad_norm_bridge"] = float(bridge_norm)
        if bridge_norm_pre_clip is not None:
            payload["train/grad_norm_bridge_pre_clip"] = float(bridge_norm_pre_clip)
        if adapter_norm is not None:
            payload["train/grad_norm_medgemma_adapter"] = float(adapter_norm)
        if adapter_norm_pre_clip is not None:
            payload["train/grad_norm_medgemma_adapter_pre_clip"] = float(adapter_norm_pre_clip)
        if len(payload) > 1:
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
        # When the encoder is contrastively trained, persist its weights too so the
        # Phase A encoder can be reloaded for linear probing. Backward-compatible:
        # the key is only added when train_encoder is True.
        if bool(getattr(self, "train_encoder", False)):
            try:
                encoder_module = (
                    self.encoder.module if hasattr(self.encoder, "module") else self.encoder
                )
                save_kwargs["encoder_state_dict"] = {
                    k: v.detach().cpu() for k, v in encoder_module.state_dict().items()
                }
            except Exception as exc:  # pragma: no cover - defensive
                print(
                    f"[{self.__class__.__name__}] Failed to capture encoder_state_dict: {exc}"
                )
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
        analysis_rows: Optional[Sequence[Dict[str, Any]]] = None,
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

        rolling_checkpoint_path = os.path.join(checkpoint_dir, "checkpoint.pt")

        payload = {
            "metric_value": metric_value,
            "metric_name": metric_name,
            "is_best": False,
            "is_last": True,
            "train_metrics": train_metrics_payload,
        }
        if val_metrics_payload is not None:
            payload["val_metrics"] = val_metrics_payload

        summary_metrics = self._build_checkpoint_metrics_summary(train_metrics, val_metrics)
        rows_for_csv: Sequence[Dict[str, Any]] = analysis_rows or []

        last_saved = self._save_checkpoint_file(
            checkpoint_path=last_path,
            epoch=epoch,
            loss=float(train_metrics.get("loss", metric_value)),
            payload=payload,
        )

        if last_saved:
            self._save_checkpoint_file(
                checkpoint_path=rolling_checkpoint_path,
                epoch=epoch,
                loss=float(train_metrics.get("loss", metric_value)),
                payload=payload,
            )
            legacy_checkpoint = os.path.join(checkpoint_dir, "Checkpoint.pt")
            if legacy_checkpoint != rolling_checkpoint_path:
                self._cleanup_checkpoint(legacy_checkpoint)
            last_csv = os.path.join(checkpoint_dir, f"siglip_phase1_last_epoch_{epoch}.csv")
            self._write_checkpoint_analysis_csv(
                csv_path=last_csv,
                rows=rows_for_csv,
                metrics=summary_metrics,
            )
            checkpoint_csv = os.path.join(checkpoint_dir, "checkpoint.csv")
            self._write_checkpoint_analysis_csv(
                csv_path=checkpoint_csv,
                rows=rows_for_csv,
                metrics=summary_metrics,
            )
            legacy_csv = os.path.join(checkpoint_dir, "Checkpoint.csv")
            if legacy_csv != checkpoint_csv:
                self._cleanup_checkpoint(legacy_csv)

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
            best_csv = os.path.join(checkpoint_dir, f"siglip_phase1_best_epoch_{epoch}.csv")
            self._write_checkpoint_analysis_csv(
                csv_path=best_csv,
                rows=rows_for_csv,
                metrics=summary_metrics,
            )
            if previous_best and previous_best != best_path:
                self._cleanup_checkpoint(previous_best)

    def _build_checkpoint_metrics_summary(
        self,
        train_metrics: Dict[str, float],
        val_metrics: Optional[Dict[str, float]],
    ) -> Dict[str, float]:
        summary: Dict[str, float] = {}
        for key, value in train_metrics.items():
            if isinstance(value, (int, float)):
                summary[f"train_{key}"] = float(value)
        if val_metrics:
            for key, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    summary[f"val_{key}"] = float(value)
        return summary

    def _write_checkpoint_analysis_csv(
        self,
        csv_path: str,
        rows: Sequence[Dict[str, Any]],
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
            "ground_truth_texts",
            "ground_truth_ids",
            "ground_truth_probs",
            "explicit_neg_texts",
            "explicit_neg_ids",
            "explicit_neg_probs",
            "top_pred_texts",
            "top_pred_ids",
            "top_pred_probs",
            "report_text",
            "generated_report",
            "ce_prompt",
            "ce_target_text",
            "ce_target_type",
        ]

        metric_fields: List[str] = []
        if metrics:
            for key in metrics.keys():
                if key not in base_fields and key not in metric_fields:
                    metric_fields.append(key)

        fieldnames = base_fields + metric_fields

        def _format_text(values: Any) -> str:
            if values is None:
                return ""
            if isinstance(values, str):
                return values
            if isinstance(values, (list, tuple)):
                return "\n".join(str(v) for v in values)
            return str(values)

        def _format_probs(values: Any) -> str:
            if values is None:
                return ""
            if isinstance(values, (list, tuple)):
                return ", ".join(f"{float(v):.4f}" for v in values)
            if isinstance(values, (int, float)):
                return f"{float(values):.4f}"
            return str(values)

        with open(csv_path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()

            for row in rows:
                row_data: Dict[str, Any] = {field: "" for field in fieldnames}
                row_data["ecg_id"] = row.get("ecg_id", "")

                alignment = row.get("alignment")
                if isinstance(alignment, (int, float)):
                    row_data["alignment"] = float(alignment)
                elif alignment is not None:
                    row_data["alignment"] = alignment

                row_pos_mass = row.get("row_pos_mass")
                if isinstance(row_pos_mass, (int, float)):
                    row_data["row_pos_mass"] = float(row_pos_mass)
                elif row_pos_mass is not None:
                    row_data["row_pos_mass"] = row_pos_mass

                row_neg_mass = row.get("row_neg_mass")
                if isinstance(row_neg_mass, (int, float)):
                    row_data["row_neg_mass"] = float(row_neg_mass)
                elif row_neg_mass is not None:
                    row_data["row_neg_mass"] = row_neg_mass

                row_data["ground_truth_texts"] = _format_text(row.get("ground_truth_texts"))
                row_data["ground_truth_ids"] = _format_text(row.get("ground_truth_ids"))
                row_data["ground_truth_probs"] = _format_probs(row.get("ground_truth_probs"))

                row_data["explicit_neg_texts"] = _format_text(row.get("explicit_neg_texts"))
                row_data["explicit_neg_ids"] = _format_text(row.get("explicit_neg_ids"))
                row_data["explicit_neg_probs"] = _format_probs(row.get("explicit_neg_probs"))

                row_data["top_pred_texts"] = _format_text(row.get("top_pred_texts"))
                row_data["top_pred_ids"] = _format_text(row.get("top_pred_ids"))
                row_data["top_pred_probs"] = _format_probs(row.get("top_pred_probs"))

                row_data["report_text"] = row.get("report_text", "")
                generated_report = row.get("generated_report", "")
                if isinstance(generated_report, str):
                    row_data["generated_report"] = self._clean_generated_text(
                        generated_report,
                        [self.report_system_prompt, self.report_user_prompt, self.report_response_prefix],
                    )
                else:
                    row_data["generated_report"] = _format_text(generated_report)

                row_data["ce_prompt"] = row.get("ce_prompt", "")
                row_data["ce_target_text"] = row.get("ce_target_text", "")
                row_data["ce_target_type"] = row.get("ce_target_type", "")

                writer.writerow(row_data)

            if metrics:
                summary = {field: "" for field in fieldnames}
                summary["ecg_id"] = "__summary__"
                for key, value in metrics.items():
                    if key in summary and isinstance(value, (int, float)):
                        summary[key] = float(value)
                    elif key in summary:
                        summary[key] = value
                writer.writerow(summary)

        print(f"[{self.__class__.__name__}] Saved checkpoint CSV: {csv_path}")

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

            lr_metrics_epoch = self._collect_lr_metrics(prefix="train")

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
                base_lr = 0.0
                if self.optimizer and self.optimizer.param_groups:
                    base_lr = float(self.optimizer.param_groups[0].get("lr", 0.0))
                print(
                    f"Epoch {epoch}/{num_epochs} - total_loss: {epoch_metrics['loss']:.4f} "
                    f"siglip_loss: {epoch_metrics.get('siglip_loss', 0.0):.4f} "
                    f"lm_loss: {epoch_metrics.get('lm_loss', 0.0):.4f} "
                    f"pos_prob: {epoch_metrics['pos_prob']:.4f} neg_prob: {epoch_metrics['neg_prob']:.4f} "
                    f"alignment: {alignment:.4f} U: {epoch_metrics.get('avg_candidate_vocab', 0.0):.1f} "
                    f"lr: {base_lr:.6f}"
                )
                train_payload = {f"train/{k}": v for k, v in epoch_metrics.items()}
                train_payload["trainer/epoch"] = float(epoch)
                train_payload["trainer/step"] = float(self.global_step)
                self._log_metrics(train_payload)
                if lr_metrics_epoch:
                    lr_payload = dict(lr_metrics_epoch)
                    lr_payload["trainer/epoch"] = float(epoch)
                    lr_payload["trainer/step"] = float(self.global_step)
                    self._log_metrics(lr_payload)
                self._log_alpha_statistics()

                if val_metrics is not None:
                    print(
                        f"    Validation - total_loss: {val_metrics['loss']:.4f} "
                        f"siglip_loss: {val_metrics.get('siglip_loss', 0.0):.4f} "
                        f"lm_loss: {val_metrics.get('lm_loss', 0.0):.4f} "
                        f"pos_prob: {val_metrics['pos_prob']:.4f} neg_prob: {val_metrics['neg_prob']:.4f} "
                        f"recall@5: {val_metrics['recall_at_5']:.4f} "
                        f"U: {val_metrics.get('avg_candidate_vocab', 0.0):.1f}"
                    )
                    val_payload = {f"val/{k}": v for k, v in val_metrics.items()}
                    val_payload["trainer/epoch"] = float(epoch)
                    self._log_metrics(val_payload)

                # Save the checkpoint BEFORE any wandb artifact/table logging.
                # The table logging can CPU-spin / block for hours when the wandb
                # backend is flaky (observed a 3h+ hang on repeated HTTP 502
                # retries at epoch end), which previously prevented the epoch
                # checkpoint from ever landing and deadlocked the other DDP rank.
                self._maybe_save_epoch_checkpoints(
                    epoch=epoch,
                    train_metrics=epoch_metrics,
                    val_metrics=val_metrics,
                    analysis_rows=analysis_rows,
                )

                if analysis_rows:
                    self._log_retrieval_artifacts(
                        epoch=epoch,
                        rows=analysis_rows,
                        best_indices=best_indices or [],
                        worst_indices=worst_indices or [],
                        random_index=random_index,
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
        total_siglip_loss = 0.0
        total_group_loss = 0.0
        total_lm_loss = 0.0
        total_lm_count = 0

        self.bridge.train(mode == RunMode.TRAIN)
        if self.decoder is not None:
            self.decoder.train(mode == RunMode.TRAIN)
        if getattr(self, "train_encoder", False):
            self.encoder.train(mode == RunMode.TRAIN)

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
            total_siglip_loss += metrics.get("siglip_loss_sum", 0.0)
            total_group_loss += metrics.get("group_loss_sum", 0.0)
            total_lm_loss += metrics.get("lm_loss_sum", 0.0)
            total_lm_count += metrics.get("lm_sample_count", 0)

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
                        siglip=f"{(total_siglip_loss / max(1, total_pairs)):.3f}",
                        lm=f"{(total_lm_loss / max(1, total_lm_count)):.3f}" if total_lm_count else "0.000",
                        align=f"{alignment_avg:.3f}",
                        U=int(current_vocab),
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
                            "train/siglip_loss": float(total_siglip_loss / max(1, total_pairs)),
                            "train/lm_loss": float(total_lm_loss / max(1, total_lm_count)) if total_lm_count else 0.0,
                            "trainer/step": float(self.global_step),
                        }
                    )

        epoch_loss = total_loss / max(1, total_pairs)
        epoch_pos = pos_mass_sum / max(1, row_counter)
        epoch_neg = neg_mass_sum / max(1, row_counter)
        avg_masked_candidates = total_masked_candidates / max(1, total_rows)
        avg_candidate_vocab = candidate_vocab_sum / max(1, candidate_vocab_count)
        avg_siglip_loss = total_siglip_loss / max(1, total_pairs)
        avg_group_loss = total_group_loss / max(1, total_pairs)
        avg_lm_loss = (total_lm_loss / max(1, total_lm_count)) if total_lm_count else 0.0

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
            "siglip_loss": float(avg_siglip_loss),
            "group_loss": float(avg_group_loss),
            "lm_loss": float(avg_lm_loss),
            "lm_samples": float(total_lm_count),
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

        ce_loss_tensor: Optional[torch.Tensor] = None
        ce_texts: List[str] = []
        ce_types: List[str] = []
        siglip_contrib_value = 0.0
        ce_contrib_value = 0.0
        valid_rows = 0

        report_mode = self.contrastive_text_mode == "report"
        report_labels: Optional[torch.Tensor] = None
        report_mask: Optional[torch.Tensor] = None
        report_text_ids_local: Optional[List[int]] = None
        with autocast_ctx:
            pooled, bridge_tokens = self.bridge(**bridge_inputs)
            pooled = F.normalize(pooled, dim=-1)

            if report_mode:
                # ESI/MERL-style in-batch InfoNCE over per-ECG free-text reports.
                reports_batch = list(batch.get("reports", [""] * pooled.size(0)))
                report_embs = self.report_text_embedder.embed(reports_batch)
                report_embs = report_embs.to(device=pooled.device, dtype=pooled.dtype)
                siglip_primary, logits = self._report_infonce_loss(
                    ecg_feats=pooled,
                    report_embs=report_embs,
                    reports=reports_batch,
                    scale=self.bridge.temperature(),
                )
                group_loss = logits.new_zeros(())
                siglip_loss = siglip_primary + group_loss
                total_loss = siglip_loss * self.siglip_loss_weight
                siglip_contrib_value = float(
                    (siglip_loss * self.siglip_loss_weight).detach().cpu()
                )
                # Diagonal == positive; reuse logits/labels for pos/neg-mass metrics.
                b = pooled.size(0)
                report_labels = torch.eye(b, device=pooled.device)
                report_mask = torch.isfinite(logits)
                report_text_ids_local = list(range(b))
                ce_prompts, ce_targets, ce_types = [], [], []
                loss = total_loss / self.grad_accum

            if not report_mode:
                logits = pooled @ text_vecs.T
                logits = logits / self.bridge.temperature()
                mask = weights_mask
                alpha_col = self._alpha_for_text_ids(text_ids) if self.use_focal_infonce else None
                gamma_pos = self.focal_gamma_pos if self.use_focal_infonce else 0.0
                gamma_neg = self.focal_gamma_neg if self.use_focal_infonce else 0.0
                pos_mask_for_count = (labels > 0.5) & mask
                valid_rows = int(pos_mask_for_count.any(dim=1).sum().item())
                if self.loss_type == "siglip_bce":
                    siglip_primary = self._siglip_bce_loss(
                        logits=logits,
                        labels=labels,
                        mask=mask,
                        alpha_col=alpha_col,
                        gamma_pos=gamma_pos,
                        gamma_neg=gamma_neg,
                        detach_weights=self.focal_detach_weights,
                    )
                else:
                    siglip_primary = self._infonce_loss(
                        logits=logits,
                        labels=labels,
                        mask=mask,
                        alpha_col=alpha_col,
                        gamma_pos=gamma_pos,
                        gamma_neg=gamma_neg,
                        detach_weights=self.focal_detach_weights,
                    )
                group_loss = (
                    self._compute_group_auxiliary_loss(logits, labels, text_ids)
                    if self.mutual_exclusive_groups and self.group_loss_weight > 0
                    else logits.new_zeros(())
                )
                siglip_loss = siglip_primary + group_loss
                total_loss = siglip_loss * self.siglip_loss_weight
                siglip_contrib_value = float((siglip_loss * self.siglip_loss_weight).detach().cpu())

                ce_prompts: List[str] = []
                ce_targets: List[str] = []
                ce_types: List[str] = []
                if self.ce_enabled and bridge_inputs.get("codes") is not None:
                    reports = batch.get("reports", [""] * len(signals))
                    ce_indices, ce_inputs_unused, ce_prompts, ce_targets, ce_types = self._sample_ce_targets(reports, text_ids, labels)
                    if ce_prompts and ce_targets:
                        codes_tensor = bridge_inputs.get("codes")
                        use_bridge = self.ce_backprop_to_bridge
                        ce_codes = None
                        ce_bridge_tokens = None
                        if use_bridge:
                            ce_bridge_tokens = bridge_tokens[ce_indices] if bridge_tokens is not None else None
                        else:
                            if isinstance(codes_tensor, torch.Tensor):
                                ce_codes = codes_tensor[ce_indices]
                        ce_loss_tensor = self._run_decoder_ce(
                            ce_prompts,
                            ce_targets,
                            ce_codes,
                            ce_types,
                            ecg_embeddings=ce_bridge_tokens,
                        )
                        if ce_loss_tensor is not None:
                            lm_w = self.lm_loss_weight
                            if self.lm_weight_warmup_steps > 0:
                                lm_w = lm_w * min(1.0, self.global_step / float(self.lm_weight_warmup_steps))
                            if self.ce_scale_by_pairs:
                                ce_batch = max(1, len(ce_prompts))
                                pair_norm = max(1, valid_rows)
                                scale = pair_norm / ce_batch
                                if self.ce_scale_cap > 0.0:
                                    scale = min(scale, self.ce_scale_cap)
                                contrib = ce_loss_tensor * lm_w * scale
                            else:
                                scale = 1.0
                                contrib = ce_loss_tensor * lm_w
                            total_loss = total_loss + contrib
                            ce_contrib_value = float(contrib.detach().cpu())
                    else:
                        ce_prompts = []
                        ce_targets = []

            loss = total_loss / self.grad_accum

        siglip_loss_value = float(siglip_loss.detach().cpu())
        group_loss_value = float(group_loss.detach().cpu())
        ce_loss_value = float(ce_loss_tensor.detach().cpu()) if ce_loss_tensor is not None else 0.0
        ce_sample_count = len(ce_prompts) if ce_loss_tensor is not None else 0

        if self.scaler is not None:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        self.global_step += 1
        perform_step = self.global_step % self.grad_accum == 0
        optimizer_step = self.global_step // self.grad_accum
        should_log_grads = (
            getattr(self.config, "is_ref_device", True)
            and perform_step
            and optimizer_step > 0
            and (optimizer_step % 50 == 0)
        )

        grad_norm_bridge: Optional[float] = None
        grad_norm_bridge_pre_clip: Optional[float] = None
        grad_norm_adapter: Optional[float] = None
        grad_norm_adapter_pre_clip: Optional[float] = None

        if perform_step:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)

            bridge_params = [param for param in self.bridge.parameters() if param.requires_grad]
            pre_clip_sum = self._sum_grad_abs(bridge_params) if should_log_grads else None
            if bridge_params:
                grad_norm_bridge_pre_clip = self._compute_grad_norm(bridge_params)

            encoder_params: List[torch.nn.Parameter] = []
            grad_norm_encoder_pre_clip: Optional[float] = None
            if getattr(self, "train_encoder", False):
                encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
                if encoder_params:
                    grad_norm_encoder_pre_clip = self._compute_grad_norm(encoder_params)
            groups = self._bridge_param_groups()
            pre_lm = self._compute_grad_norm(groups["lm_facing"]) if groups["lm_facing"] else 0.0
            pre_ret = self._compute_grad_norm(groups["retrieval_facing"]) if groups["retrieval_facing"] else 0.0
            pre_shared = self._compute_grad_norm(groups["shared"]) if groups["shared"] else 0.0

            inner_adapter = getattr(self.bridge, "bridge", None)
            adapter_params: List[torch.nn.Parameter] = []
            if inner_adapter is not None:
                adapter_params = [param for param in inner_adapter.parameters() if param.requires_grad]
            if adapter_params:
                grad_norm_adapter_pre_clip = self._compute_grad_norm(adapter_params)

            torch.nn.utils.clip_grad_norm_(bridge_params, self.grad_clip)
            if encoder_params:
                torch.nn.utils.clip_grad_norm_(encoder_params, self.grad_clip)

            if bridge_params:
                grad_norm_bridge = min(self._compute_grad_norm(bridge_params), self.grad_clip)
            if adapter_params:
                grad_norm_adapter = min(self._compute_grad_norm(adapter_params), self.grad_clip)

            if should_log_grads:
                post_clip_sum = self._sum_grad_abs(bridge_params)
                payload = {
                    "train/bridge_grad_sum": float(post_clip_sum),
                    "trainer/step": float(self.global_step),
                }
                if pre_clip_sum is not None:
                    payload["train/bridge_grad_sum_pre_clip"] = float(pre_clip_sum)
                if groups["lm_facing"]:
                    post_lm = min(self._compute_grad_norm(groups["lm_facing"]), self.grad_clip)
                    payload["train/grad_norm_bridge_to_llm_pre_clip"] = float(pre_lm)
                    payload["train/grad_norm_bridge_to_llm"] = float(post_lm)
                if groups["retrieval_facing"]:
                    post_ret = min(self._compute_grad_norm(groups["retrieval_facing"]), self.grad_clip)
                    payload["train/grad_norm_bridge_to_txt_pre_clip"] = float(pre_ret)
                    payload["train/grad_norm_bridge_to_txt"] = float(post_ret)
                if groups["shared"]:
                    post_shared = min(self._compute_grad_norm(groups["shared"]), self.grad_clip)
                    payload["train/grad_norm_bridge_shared_pre_clip"] = float(pre_shared)
                    payload["train/grad_norm_bridge_shared"] = float(post_shared)
                if encoder_params:
                    post_enc = min(self._compute_grad_norm(encoder_params), self.grad_clip)
                    payload["train/grad_norm_encoder_pre_clip"] = float(grad_norm_encoder_pre_clip or 0.0)
                    payload["train/grad_norm_encoder"] = float(post_enc)
                self._log_metrics(payload)

            # Final safety net: the decoder's unfrozen LLM params (e.g. last-N
            # layers + final norm) are in the optimizer but NOT clipped above, so
            # a non-finite grad there would corrupt them on step(). Skip the whole
            # optimizer step if ANY trainable grad is non-finite (standard
            # divergence guard; bf16 has no GradScaler to do this for us).
            step_grads_finite = True
            for group in self.optimizer.param_groups:
                for param in group["params"]:
                    if param.grad is not None and not torch.isfinite(param.grad).all():
                        step_grads_finite = False
                        break
                if not step_grads_finite:
                    break

            if not step_grads_finite:
                self._nonfinite_step_count = getattr(self, "_nonfinite_step_count", 0) + 1
                if getattr(self.config, "is_ref_device", True) and self._nonfinite_step_count <= 20:
                    print(f"[SigLIP] Skipping optimizer step with non-finite grads "
                          f"(count={self._nonfinite_step_count}, opt_step={optimizer_step}).")
            elif self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self._log_grad_norms(
                grad_norm_bridge,
                grad_norm_adapter,
                optimizer_step,
                bridge_norm_pre_clip=grad_norm_bridge_pre_clip,
                adapter_norm_pre_clip=grad_norm_adapter_pre_clip,
            )
            if getattr(self.config, "is_ref_device", True):
                self._log_metrics(
                    {
                        "trainer/step": float(self.global_step),
                        "train/loss_contrib_siglip_step": float(siglip_contrib_value),
                        "train/loss_contrib_lm_step": float(ce_contrib_value),
                    }
                )

        logits_detached = logits.detach()
        if report_mode:
            metrics_labels = report_labels
            metrics_mask = report_mask
            metrics_candidate_vocab = logits_detached.size(1)
        else:
            metrics_labels = labels
            metrics_mask = mask
            metrics_candidate_vocab = len(text_ids)
        with torch.no_grad():
            if report_mode:
                metrics_logits = logits_detached
            else:
                metrics_logits = self._collapse_yes_no_logits(logits_detached, text_ids)
            neg_inf = float("-inf")
            masked_logits = metrics_logits.masked_fill(~metrics_mask, neg_inf).to(torch.float32)
            probs = torch.softmax(masked_logits, dim=1)
            pos_mask = (metrics_labels > 0.5) & metrics_mask
            neg_mask = (~pos_mask) & metrics_mask
            row_pos_mass = (probs * pos_mask).sum(dim=1)
            row_neg_mass = (probs * neg_mask).sum(dim=1)
            pos_prob_sum = row_pos_mass.sum().item()
            neg_prob_sum = row_neg_mass.sum().item()
            row_count = metrics_mask.size(0)
            valid_rows = int(pos_mask.any(dim=1).sum().item())

        loss_numerator = float(loss.detach().cpu()) * self.grad_accum * max(valid_rows, 1)

        return {
            "loss_sum": loss_numerator,
            "num_pairs": max(valid_rows, 1),
            "pos_prob_sum": pos_prob_sum,
            "neg_prob_sum": neg_prob_sum,
            "row_count": row_count,
            "candidate_mask_sum": float(metrics_mask.sum().item()),
            "candidate_vocab": metrics_candidate_vocab,
            "siglip_loss_sum": siglip_loss_value * max(valid_rows, 1),
            "group_loss_sum": group_loss_value * max(valid_rows, 1),
            "lm_loss_sum": ce_loss_value * ce_sample_count,
            "lm_sample_count": ce_sample_count,
        }

    @torch.no_grad()
    def _evaluate(self, dataloader: Optional[DataLoader], epoch: Optional[int] = None) -> Optional[Dict[str, float]]:
        if dataloader is None:
            return None

        self.bridge.eval()
        if self.decoder is not None:
            self.decoder.eval()
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
        total_ce_loss = 0.0
        total_ce_count = 0
        total_siglip_loss = 0.0
        total_group_loss = 0.0
        eval_rng = random.Random(int(self.config.seed) + int(epoch or 0) + int(self.config.device))

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

        report_mode = self.contrastive_text_mode == "report"
        report_recall_logged = False
        batch_count = 0
        for batch in iterator:
            batch_count += 1
            signals = batch["signals"].to(self.device)
            labels = batch["labels"].to(self.device)
            weights = batch["weights"].to(self.device)
            text_ids: List[str] = batch["text_ids"]
            ecg_ids: List[str] = batch.get("ecg_ids", [""] * len(signals))
            reports: List[str] = batch.get("reports", [""] * len(ecg_ids))

            if report_mode:
                bridge_inputs = self._compute_bridge_inputs(signals)
                pooled, _ = self.bridge(**bridge_inputs)
                pooled = F.normalize(pooled, dim=-1)
                report_embs = self.report_text_embedder.embed(reports)
                report_embs = report_embs.to(device=pooled.device, dtype=pooled.dtype)
                scale = self.bridge.temperature()
                r_loss, r_logits = self._report_infonce_loss(
                    ecg_feats=pooled,
                    report_embs=report_embs,
                    reports=list(reports),
                    scale=scale,
                )
                b = pooled.size(0)
                r_labels = torch.eye(b, device=pooled.device)
                r_mask = torch.isfinite(r_logits)
                masked = r_logits.masked_fill(~r_mask, float("-inf")).to(torch.float32)
                probs = torch.softmax(masked, dim=1)
                pos_mask = (r_labels > 0.5) & r_mask
                neg_mask = (~pos_mask) & r_mask
                pos_mass_sum += (probs * pos_mask).sum().item()
                neg_mass_sum += (probs * neg_mask).sum().item()
                row_counter += b
                total_loss += float(r_loss.item()) * b
                total_pairs += b
                total_siglip_loss += float(r_loss.item()) * b
                total_masked_candidates += float(r_mask.sum().item())
                total_rows += b
                candidate_vocab_sum += b
                candidate_vocab_count += 1
                batch_recalls = compute_recall_at_many(masked, r_labels, ks=recall_keys)
                for k, (value_sum, count) in batch_recalls.items():
                    recall_sums[k] += value_sum
                    recall_counts[k] += count
                # Log good/bad recall examples once per validation (first batch).
                if not report_recall_logged:
                    self._log_recall_examples(
                        ecg_feats=pooled,
                        report_embs=report_embs,
                        reports=list(reports),
                        ecg_ids=list(ecg_ids),
                        scale=scale,
                        epoch=epoch,
                    )
                    report_recall_logged = True
                continue

            text_indices = [self.text_id_to_idx[tid] for tid in text_ids]
            text_vecs = self.text_embeddings[text_indices]
            text_vecs = F.normalize(text_vecs, dim=-1)

            bridge_inputs = self._compute_bridge_inputs(signals)
            codes_cpu = None
            if bridge_inputs.get("codes") is not None and isinstance(bridge_inputs["codes"], torch.Tensor):
                codes_cpu = bridge_inputs["codes"].detach().cpu()
            pooled, bridge_tokens = self.bridge(**bridge_inputs)
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
            siglip_loss = infonce_loss + group_loss
            batch_loss = siglip_loss * self.siglip_loss_weight

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
            total_siglip_loss += siglip_loss.item() * max(valid_rows, 1)
            total_group_loss += group_loss.item() * max(valid_rows, 1)

            logits_for_metrics = masked_logits
            batch_recalls = compute_recall_at_many(logits_for_metrics, labels, ks=recall_keys)
            for k, (value_sum, count) in batch_recalls.items():
                recall_sums[k] += value_sum
                recall_counts[k] += count

            ce_prompts_eval: List[str] = []
            ce_targets_eval: List[str] = []
            ce_types: List[str] = []
            ce_target_map: dict[int, tuple[str, str, str]] = {}
            if self.ce_enabled and bridge_inputs.get("codes") is not None:
                reports_seq = batch.get("reports", [""] * len(signals))
                ce_indices, ce_inputs_unused, ce_prompts_eval, ce_targets_eval, ce_types = self._sample_ce_targets(reports_seq, text_ids, labels, rng=eval_rng)
                if ce_prompts_eval and ce_targets_eval:
                    codes_tensor = bridge_inputs.get("codes")
                    use_bridge = self.ce_backprop_to_bridge
                    ce_codes = None
                    ce_bridge_tokens = None
                    if use_bridge:
                        ce_bridge_tokens = bridge_tokens[ce_indices] if bridge_tokens is not None else None
                    else:
                        if isinstance(codes_tensor, torch.Tensor):
                            ce_codes = codes_tensor[ce_indices]
                    ce_loss_tensor = self._run_decoder_ce(
                        ce_prompts_eval,
                        ce_targets_eval,
                        ce_codes,
                        ce_types,
                        ecg_embeddings=ce_bridge_tokens,
                    )
                    if ce_loss_tensor is not None:
                        ce_value = float(ce_loss_tensor.item())
                        total_ce_loss += ce_value * len(ce_prompts_eval)
                        total_ce_count += len(ce_prompts_eval)
                    for local_pos, row_idx in enumerate(ce_indices):
                        ce_target_map[row_idx] = (
                            ce_prompts_eval[local_pos],
                            ce_targets_eval[local_pos],
                            ce_types[local_pos],
                        )

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

                ce_prompt_text, ce_target_text, ce_target_type = ce_target_map.get(row, ("", "", ""))
                codes_entry = None
                if codes_cpu is not None and row < codes_cpu.size(0):
                    codes_entry = codes_cpu[row].clone()
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
                        "report_text": reports[row] if row < len(reports) else "",
                        "ce_prompt": ce_prompt_text,
                        "ce_target_text": ce_target_text,
                        "ce_target_type": ce_target_type,
                        "quantized_codes": codes_entry,
                        "generated_report": None,
                    }
                )
                index = len(analysis_rows) - 1
                best_samples.append((index, alignment_row))
                worst_samples.append((index, alignment_row))

        if self.config.is_ref_device:
            print(f"[SigLIP] Validation epoch {epoch or 'N/A'}: processed {batch_count} batches, collected {len(analysis_rows)} analysis rows")

        avg_loss = total_loss / max(1, total_pairs)
        avg_pos = pos_mass_sum / max(1, row_counter)
        avg_neg = neg_mass_sum / max(1, row_counter)
        avg_masked_candidates = total_masked_candidates / max(1, total_rows)
        avg_candidate_vocab = candidate_vocab_sum / max(1, candidate_vocab_count)
        avg_siglip_loss = total_siglip_loss / max(1, total_pairs)
        avg_group_loss = total_group_loss / max(1, total_pairs)
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
            "siglip_loss": float(avg_siglip_loss),
            "group_loss": float(avg_group_loss),
        }
        if total_ce_count:
            metrics["lm_loss"] = float(total_ce_loss / max(1, total_ce_count))
            metrics["lm_samples"] = float(total_ce_count)
        else:
            metrics["lm_loss"] = 0.0
            metrics["lm_samples"] = 0.0
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

            if (
                getattr(self.config, "output_dir", None)
                and epoch is not None
                and getattr(self.config, "is_ref_device", True)
            ):
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

        # Always add analysis_rows when on ref device (even if empty) to ensure CSV generation
        if self.config.is_ref_device:
            print(f"[SigLIP] Collected {len(analysis_rows)} analysis rows for epoch {epoch or 'N/A'}")
            if analysis_rows:
                best_samples.sort(key=lambda x: x[1], reverse=True)
                worst_samples.sort(key=lambda x: x[1])
                top_best = best_samples[:5]
                top_worst = worst_samples[:5]
                random_idx = random.randrange(len(analysis_rows)) if analysis_rows else None
                metrics["analysis_rows"] = analysis_rows
                metrics["best_indices"] = top_best
                metrics["worst_indices"] = top_worst
                metrics["random_index"] = random_idx
            else:
                # Empty lists to trigger empty CSV generation for debugging
                metrics["analysis_rows"] = []
                metrics["best_indices"] = []
                metrics["worst_indices"] = []
                metrics["random_index"] = None
        return metrics

    def _compute_bridge_inputs(self, signals: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Extract tokenizer features or codes for bridge consumption.

        When ``train_encoder`` is True the encoder forward runs WITH autograd so
        gradients from the contrastive loss flow into the raw signal encoder.
        Otherwise (default) the whole feature extraction is wrapped in no_grad.
        """
        train_encoder = bool(getattr(self, "train_encoder", False))
        encoder_ctx = torch.enable_grad() if train_encoder else torch.no_grad()
        with encoder_ctx:
            encoder_feats = self.encoder(signals)
            features: torch.Tensor
            codes = None

            if self.quantizer is not None and self.use_quantized_inputs:
                keep = getattr(self.config, "num_codebooks_kept", None)
                offset_cfg = getattr(self.config, "codebook_offset", 0)
                # The quantizer is frozen; on the quantized path do not propagate
                # encoder gradients through it (Phase A uses the continuous path).
                quantizer_input = encoder_feats.detach() if train_encoder else encoder_feats
                quantized_outputs = self.quantizer(quantizer_input, return_all_codes=True)
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
        artifact_dir: Optional[str] = None
        base_dirs: List[Optional[str]] = [
            getattr(self.config, "output_dir", None),
            getattr(self.config, "checkpoint_dir", None),
            getattr(self.config, "base_checkpoint_path", None),
        ]
        base_dirs.append(os.path.join(os.getcwd(), "checkpoints"))
        for base_dir in base_dirs:
            if not base_dir:
                continue
            candidate = os.path.join(base_dir, "artifacts")
            try:
                ensure_dir(candidate)
            except OSError:
                continue
            artifact_dir = candidate
            break
        if artifact_dir is None:
            artifact_dir = os.path.join(os.getcwd(), "artifacts")
            ensure_dir(artifact_dir)

        rank_suffix = ""
        rank = int(getattr(self.config, "device", 0))
        if not getattr(self.config, "is_ref_device", True):
            rank_suffix = f"_rank{rank}"

        retrieval_columns = [
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
            "ground_truth_all_ids",
            "ground_truth_all_texts",
            "ground_truth_all_probs",
            "top_pred_ids",
            "top_pred_texts",
            "top_pred_probs",
            "report_text",
            "ce_target_text",
            "ce_target_type",
            "generated_report",
        ]
        generated_columns = [
            "ecg_id",
            "report_text",
            "generated_report",
            "ce_prompt",
            "ce_target_text",
            "ce_target_type",
            "alignment",
            "row_pos_mass",
            "row_neg_mass",
            "top_pred_texts",
            "top_pred_probs",
            "top_pred_ids",
        ]

        if not rows:
            df_empty = pd.DataFrame(columns=retrieval_columns)
            csv_path_empty = os.path.join(artifact_dir, f"val_epoch_{epoch}_retrieval{rank_suffix}.csv")
            df_empty.to_csv(csv_path_empty, index=False)
            print(f"[SigLIP] Saved retrieval CSV: {csv_path_empty}")

            gen_df_empty = pd.DataFrame(columns=generated_columns)
            gen_csv_path_empty = os.path.join(artifact_dir, f"val_epoch_{epoch}_generated_reports{rank_suffix}.csv")
            gen_df_empty.to_csv(gen_csv_path_empty, index=False)
            print(f"[SigLIP] Saved generated reports CSV: {gen_csv_path_empty}")

            if (
                self.wandb_wrapper
                and self.wandb_wrapper.is_initialized()
                and getattr(self.config, "is_ref_device", True)
            ):
                retrieval_table_empty = wandb.Table(columns=retrieval_columns)
                generated_table_empty = wandb.Table(columns=generated_columns)
                self.wandb_wrapper.log(
                    {
                        f"val/retrieval_table_full_epoch_{epoch}": retrieval_table_empty,
                        f"val/generated_reports_full_epoch_{epoch}": generated_table_empty,
                    }
                )
            return

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
                    generated_report = row.get("generated_report")
                    if generated_report:
                        cleaned_report = self._clean_generated_text(
                            generated_report,
                            [self.report_system_prompt, self.report_user_prompt, self.report_response_prefix],
                        )
                        if cleaned_report != generated_report:
                            generated_report = cleaned_report
                            row["generated_report"] = cleaned_report
                    if (generated_report is None or generated_report == "") and self.decoder is not None:
                        codes_tensor = row.get("quantized_codes")
                        if isinstance(codes_tensor, torch.Tensor):
                            generated_report = self._generate_report_from_codes(codes_tensor)
                            generated_report = self._clean_generated_text(
                                generated_report,
                                [self.report_system_prompt, self.report_user_prompt, self.report_response_prefix],
                            )
                            row["generated_report"] = generated_report
                    ce_target = row.get("ce_target_text", "")
                    ce_type = row.get("ce_target_type", "")
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
                            "report_text": row.get("report_text", ""),
                            "ce_target": ce_target,
                            "ce_target_type": ce_type,
                            "generated_report": generated_report or "",
                        }
                    )

        generated_rows: List[dict] = []
        row_retrieval_entries: List[dict] = []
        row_generated_entries: List[dict] = []

        df_rows = []
        generation_start = time.perf_counter()
        progress_bar = tqdm(
            rows,
            desc="Generating validation reports",
            total=len(rows),
            leave=False,
            dynamic_ncols=True,
        )
        csv_path = os.path.join(artifact_dir, f"val_epoch_{epoch}_retrieval{rank_suffix}.csv")
        gen_csv_path = os.path.join(artifact_dir, f"val_epoch_{epoch}_generated_reports{rank_suffix}.csv")
        if os.path.exists(csv_path):
            os.remove(csv_path)
        if os.path.exists(gen_csv_path):
            os.remove(gen_csv_path)

        chunk_size = max(1, int(getattr(self.config, "analysis_partial_save_size", 200)))
        retrieval_chunk: List[dict] = []
        generated_chunk: List[dict] = []
        retrieval_header_written = False
        generated_header_written = False

        ecg_to_codes: Dict[str, torch.Tensor] = {}
        ecg_to_row_indices: Dict[str, List[int]] = {}

        def _flush_chunks() -> None:
            nonlocal retrieval_header_written, generated_header_written
            if retrieval_chunk:
                chunk_df = pd.DataFrame(retrieval_chunk, columns=retrieval_columns)
                chunk_df.to_csv(
                    csv_path,
                    mode="a",
                    header=not retrieval_header_written,
                    index=False,
                )
                retrieval_header_written = True
                retrieval_chunk.clear()
            if generated_chunk:
                chunk_gen_df = pd.DataFrame(generated_chunk, columns=generated_columns)
                chunk_gen_df.to_csv(
                    gen_csv_path,
                    mode="a",
                    header=not generated_header_written,
                    index=False,
                )
                generated_header_written = True
                generated_chunk.clear()

        for row_idx, row in enumerate(progress_bar):
            all_labels_str, all_ids_str, all_probs_str = _format_all_labels(row)
            generated_report = row.get("generated_report", "")
            if generated_report:
                cleaned_report = self._clean_generated_text(
                    generated_report,
                    [self.report_system_prompt, self.report_user_prompt, self.report_response_prefix],
                )
                if cleaned_report != generated_report:
                    generated_report = cleaned_report
                    row["generated_report"] = cleaned_report
            cleaned_report = row.get("generated_report", "") or ""
            codes_tensor = row.get("quantized_codes")
            if (not cleaned_report) and self.decoder is not None and isinstance(codes_tensor, torch.Tensor):
                ecg_id = row["ecg_id"]
                if ecg_id not in ecg_to_codes:
                    ecg_to_codes[ecg_id] = codes_tensor
                ecg_to_row_indices.setdefault(ecg_id, []).append(row_idx)

            generated_entry = {
                "ecg_id": row["ecg_id"],
                "report_text": row.get("report_text", ""),
                "generated_report": cleaned_report,
                "ce_prompt": row.get("ce_prompt", ""),
                "ce_target_text": row.get("ce_target_text", ""),
                "ce_target_type": row.get("ce_target_type", ""),
                "alignment": float(row.get("alignment", 0.0)),
                "row_pos_mass": float(row.get("row_pos_mass", 0.0)),
                "row_neg_mass": float(row.get("row_neg_mass", 0.0)),
                "top_pred_texts": _format_list(row.get("top_pred_texts", [])),
                "top_pred_probs": _format_probs(row.get("top_pred_probs", [])),
                "top_pred_ids": _format_list(row.get("top_pred_ids", [])),
            }
            generated_rows.append(generated_entry)
            row_generated_entries.append(generated_entry)

            retrieval_entry = {
                "ecg_id": row["ecg_id"],
                "alignment": float(row.get("alignment", 0.0)),
                "row_pos_mass": float(row.get("row_pos_mass", 0.0)),
                "row_neg_mass": float(row.get("row_neg_mass", 0.0)),
                "ground_truth_pos_ids": _format_list(row.get("ground_truth_ids", [])),
                "ground_truth_pos_texts": _format_list(row.get("ground_truth_texts", [])),
                "ground_truth_pos_probs": _format_probs(row.get("ground_truth_probs", [])),
                "ground_truth_neg_ids": _format_list(row.get("explicit_neg_ids", [])),
                "ground_truth_neg_texts": _format_list(row.get("explicit_neg_texts", [])),
                "ground_truth_neg_probs": _format_probs(row.get("explicit_neg_probs", [])),
                "ground_truth_all_ids": all_ids_str,
                "ground_truth_all_texts": all_labels_str,
                "ground_truth_all_probs": all_probs_str,
                "top_pred_ids": _format_list(row.get("top_pred_ids", [])),
                "top_pred_texts": _format_list(row.get("top_pred_texts", [])),
                "top_pred_probs": _format_probs(row.get("top_pred_probs", [])),
                "report_text": row.get("report_text", ""),
                "ce_target_text": row.get("ce_target_text", ""),
                "ce_target_type": row.get("ce_target_type", ""),
                "generated_report": cleaned_report,
            }
            df_rows.append(retrieval_entry)
            row_retrieval_entries.append(retrieval_entry)

            retrieval_chunk.append(retrieval_entry)
            generated_chunk.append(generated_entry)
            if (len(df_rows) % chunk_size) == 0:
                _flush_chunks()

        progress_bar.close()
        _flush_chunks()

        if ecg_to_codes and self.decoder is not None:
            ecg_items = list(ecg_to_codes.items())
            # Cap the number of full report generations (inspection-only). On a
            # 27B decoder, generating all ~3.6k val reports costs ~7h/epoch and
            # blocks the next epoch; a small sample is enough for qualitative CSV.
            gen_cap = int(getattr(self.config, "val_report_generation_max_ecgs", 0) or 0)
            if gen_cap > 0 and len(ecg_items) > gen_cap:
                ecg_items = ecg_items[:gen_cap]
            batch_size = self.report_generation_batch_size
            for start in range(0, len(ecg_items), batch_size):
                batch = ecg_items[start : start + batch_size]
                codes_batch = [item[1] for item in batch]
                generated_texts = self._generate_reports_batch(codes_batch)
                for (ecg_id, _), gen_text in zip(batch, generated_texts):
                    cleaned_text = self._clean_generated_text(
                        gen_text,
                        [self.report_system_prompt, self.report_user_prompt, self.report_response_prefix],
                    )
                    for idx in ecg_to_row_indices.get(ecg_id, []):
                        rows[idx]["generated_report"] = cleaned_text
                        row_generated_entries[idx]["generated_report"] = cleaned_text
                        row_retrieval_entries[idx]["generated_report"] = cleaned_text

        # Optionally save generated token IDs (JSONL) for a sample
        if (
            getattr(self.config, "log_generated_token_ids", False)
            and self.decoder is not None
            and getattr(self.config, "is_ref_device", True)
        ):
            token_sample_k = int(getattr(self.config, "generated_token_sample_k", 50) or 50)
            # Prefer the ecg_to_codes pool if available; otherwise sample from rows
            sample_items: List[tuple[str, torch.Tensor]] = []
            if ecg_to_codes:
                items = list(ecg_to_codes.items())
                if len(items) > token_sample_k:
                    rng = random.Random(self.config.seed + epoch)
                    items = rng.sample(items, token_sample_k)
                sample_items = [(ecg_id, codes) for ecg_id, codes in items]
            else:
                candidates: List[tuple[str, torch.Tensor]] = []
                for row in rows:
                    ecg_id = row.get("ecg_id", "")
                    codes_tensor = row.get("quantized_codes")
                    if isinstance(codes_tensor, torch.Tensor):
                        candidates.append((str(ecg_id), codes_tensor))
                if candidates:
                    if len(candidates) > token_sample_k:
                        rng = random.Random(self.config.seed + epoch)
                        candidates = rng.sample(candidates, token_sample_k)
                    sample_items = candidates

            if sample_items:
                codes_batch = [codes for _, codes in sample_items]
                ecg_ids_batch = [ecg_id for ecg_id, _ in sample_items]
                decoded_texts, token_ids = self._generate_reports_batch(codes_batch, return_token_ids=True)  # type: ignore[assignment]
                artifact_dir = os.path.join(self.config.output_dir, "artifacts")
                ensure_dir(artifact_dir)
                rank = int(getattr(self.config, "device", 0))
                rank_suffix = ""
                if not getattr(self.config, "is_ref_device", True):
                    rank_suffix = f"_rank{rank}"
                jsonl_path = os.path.join(
                    artifact_dir,
                    f"val_epoch_{epoch}_generated_tokens{rank_suffix}.jsonl",
                )
                import json
                with open(jsonl_path, "w") as handle:
                    for ecg_id, text, toks in zip(ecg_ids_batch, decoded_texts, token_ids.tolist()):
                        rec = {
                            "ecg_id": ecg_id,
                            "generated_text": text,
                            "token_ids": toks,
                        }
                        handle.write(json.dumps(rec) + "\n")
                print(f"[SigLIP] Saved generated token IDs JSONL: {jsonl_path}")

                if self.wandb_wrapper and self.wandb_wrapper.is_initialized():
                    preview = min(10, len(ecg_ids_batch))
                    table = wandb.Table(columns=["ecg_id", "len", "first_20_tokens", "generated_text"])
                    for i in range(preview):
                        toks = token_ids[i].tolist()
                        table.add_data(ecg_ids_batch[i], int(len(toks)), str(toks[:20]), decoded_texts[i])
                    self.wandb_wrapper.log({f"val/generated_token_ids_epoch_{epoch}": table})

        for row in rows:
            if isinstance(row.get("quantized_codes"), torch.Tensor):
                row["quantized_codes"] = None

        generation_elapsed = time.perf_counter() - generation_start
        print(
            f"[SigLIP] Generated validation reports for {len(rows)} rows in {generation_elapsed:.2f}s"
        )

        df = pd.DataFrame(df_rows, columns=retrieval_columns)
        save_start = time.perf_counter()
        df.to_csv(csv_path, index=False)
        retrieval_write_time = time.perf_counter() - save_start
        print(f"[SigLIP] Saved retrieval CSV in {retrieval_write_time:.2f}s: {csv_path}")

        gen_df = pd.DataFrame(generated_rows, columns=generated_columns)
        save_start = time.perf_counter()
        gen_df.to_csv(gen_csv_path, index=False)
        generated_write_time = time.perf_counter() - save_start
        print(
            f"[SigLIP] Saved generated reports CSV in {generated_write_time:.2f}s: {gen_csv_path}"
        )

        if (
            self.wandb_wrapper
            and self.wandb_wrapper.is_initialized()
            and getattr(self.config, "is_ref_device", True)
        ):
            retrieval_table_full = wandb.Table(columns=list(df.columns))
            for record in df.itertuples(index=False, name=None):
                retrieval_table_full.add_data(*record)
            generated_table_full = wandb.Table(columns=list(gen_df.columns))
            for record in gen_df.itertuples(index=False, name=None):
                generated_table_full.add_data(*record)
            self.wandb_wrapper.log(
                {
                    f"val/retrieval_table_full_epoch_{epoch}": retrieval_table_full,
                    f"val/generated_reports_full_epoch_{epoch}": generated_table_full,
                }
            )

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
                "report_text",
                "ce_target",
                "ce_target_type",
                "generated_report",
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
                    sample["report_text"],
                    sample["ce_target"],
                    sample["ce_target_type"],
                    sample["generated_report"],
                )
            self.wandb_wrapper.log({f"val/retrieval_samples_epoch_{epoch}": table})

            if generated_rows:
                rng = random.Random(self.config.seed + epoch)
                sample_rows = rng.sample(generated_rows, min(5, len(generated_rows)))
                gen_table = wandb.Table(columns=["ecg_id", "report_text", "generated_report", "ce_target_text", "ce_target_type"])
                for row in sample_rows:
                    gen_table.add_data(
                        row["ecg_id"],
                        row["report_text"],
                        row["generated_report"],
                        row["ce_target_text"],
                        row["ce_target_type"],
                    )
                self.wandb_wrapper.log({f"val/generated_reports_epoch_{epoch}": gen_table})

    def _sample_ce_targets(
        self,
        reports: Sequence[str],
        text_ids: Sequence[str],
        labels: torch.Tensor,
        rng: Optional[random.Random] = None,
    ) -> Tuple[List[int], List[str], List[str], List[str], List[str]]:
        if not self.ce_enabled:
            return [], [], [], []
        rng = rng or self._ce_rng
        labels_cpu = labels.detach().cpu()
        indices: List[int] = []
        ce_inputs: List[str] = []
        prompts: List[str] = []
        targets: List[str] = []
        target_types: List[str] = []
        for row_idx in range(labels_cpu.size(0)):
            row = labels_cpu[row_idx]
            pos_cols = torch.nonzero(row > 0.5, as_tuple=False).flatten().tolist()
            qa_ids = [text_ids[col] for col in pos_cols if text_ids[col].startswith("QA_") and text_ids[col].endswith("_yes")]
            fallback_ids = [text_ids[col] for col in pos_cols]
            report_text = str(reports[row_idx]) if row_idx < len(reports) else ""
            if report_text and rng.random() < self.ce_report_probability:
                prompt, target, target_text = self._build_report_prompt_and_target(report_text)
                target_type = "report"
            elif qa_ids:
                tid = rng.choice(qa_ids)
                qa_string = self.text_lookup.get(tid, tid)
                prompt, target, target_text = self._build_qa_prompt_and_target(qa_string)
                target_type = "qa"
            elif fallback_ids:
                tid = fallback_ids[0]
                fallback_text = self.text_lookup.get(tid, tid)
                prompt, target, target_text = self._build_report_prompt_and_target(fallback_text)
                target_type = "positive"
            elif report_text:
                prompt, target, target_text = self._build_report_prompt_and_target(report_text)
                target_type = "report"
            else:
                fallback_text = "No abnormal ECG findings detected."
                prompt, target, target_text = self._build_report_prompt_and_target(fallback_text)
                target_type = "fallback"
            indices.append(row_idx)
            prompts.append(prompt)
            targets.append(target_text)
            ce_inputs.append(f"{prompt}\n{target}")
            target_types.append(target_type)

        if self.ce_max_samples_per_batch > 0 and len(indices) > self.ce_max_samples_per_batch:
            selected_positions = sorted(rng.sample(range(len(indices)), self.ce_max_samples_per_batch))
            indices = [indices[pos] for pos in selected_positions]
            ce_inputs = [ce_inputs[pos] for pos in selected_positions]
            prompts = [prompts[pos] for pos in selected_positions]
            targets = [targets[pos] for pos in selected_positions]
            target_types = [target_types[pos] for pos in selected_positions]

        return indices, ce_inputs, prompts, targets, target_types

    def _build_ce_batch(
        self,
        prompts: Sequence[str],
        targets: Sequence[str],
        target_types: Optional[Sequence[str]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build CE batch with proper prompt/target separation for label masking.

        Args:
            prompts: List of prompts (will be masked in labels with -100)
            targets: List of targets (will be predicted by the model)
            target_types: Optional list describing target categories (e.g., report, qa)

        Returns:
            Tuple of (input_ids, attention_mask, labels)
        """
        if not prompts or not targets:
            raise ValueError("Cannot build CE batch without prompts and targets.")
        if len(prompts) != len(targets):
            raise ValueError(f"Prompts and targets must have same length: {len(prompts)} != {len(targets)}")
        if self.decoder_tokenizer is None:
            raise ValueError("Decoder tokenizer is unavailable for CE batching.")

        tokenizer = self.decoder_tokenizer
        use_chat = hasattr(tokenizer, "apply_chat_template")
        response_prefix_report = (self.report_response_prefix + " ").strip()
        response_prefix_qa = (self.qa_response_prefix + " ").strip()

        def _split_user(prompt_text: str, expected_prefix: str, fallback: str) -> str:
            if prompt_text.startswith(expected_prefix):
                remainder = prompt_text[len(expected_prefix):].lstrip("\n")
                return remainder if remainder else fallback
            if "\n" in prompt_text:
                remainder = prompt_text.split("\n", 1)[1].strip()
                return remainder if remainder else fallback
            stripped = prompt_text.strip()
            return stripped if stripped else fallback

        if use_chat:
            prefix_texts: List[str] = []
            full_texts: List[str] = []
            for idx, (prompt, target) in enumerate(zip(prompts, targets)):
                t_type = target_types[idx] if target_types is not None and idx < len(target_types) else "report"
                if t_type == "qa":
                    system_prompt = self.qa_system_prompt
                    user_prompt = _split_user(prompt, self.qa_system_prompt, prompt.strip() or self.report_user_prompt).strip()
                    assistant_prefix = response_prefix_qa
                else:
                    system_prompt = self.report_system_prompt
                    user_prompt = _split_user(prompt, self.report_system_prompt, self.report_user_prompt).strip()
                    assistant_prefix = response_prefix_report
                prefix_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": assistant_prefix},
                ]
                full_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": (assistant_prefix + target).strip()},
                ]
                prefix_texts.append(
                    tokenizer.apply_chat_template(prefix_messages, tokenize=False, add_generation_prompt=True)
                )
                full_texts.append(
                    tokenizer.apply_chat_template(full_messages, tokenize=False, add_generation_prompt=False)
                )
            enc_full = tokenizer(
                full_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.ce_max_length,
            )
            enc_prefix = tokenizer(
                prefix_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.ce_max_length,
            )
        else:
            prefix_sequences: List[str] = []
            full_sequences: List[str] = []
            for idx, (prompt, target) in enumerate(zip(prompts, targets)):
                t_type = target_types[idx] if target_types is not None and idx < len(target_types) else "report"
                if t_type == "qa":
                    prefix = response_prefix_qa
                else:
                    prefix = response_prefix_report
                prompt_body = prompt.strip()
                prefix_sequences.append(f"{prompt_body}\n{prefix}")
                full_sequences.append(f"{prompt_body}\n{prefix} {target}".strip())
            enc_full = tokenizer(
                full_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.ce_max_length,
            )
            enc_prefix = tokenizer(
                prefix_sequences,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.ce_max_length,
            )

        input_ids = enc_full["input_ids"]
        attention_mask = enc_full["attention_mask"]
        labels = input_ids.clone()

        batch_size = labels.size(0)
        prompt_lengths = enc_prefix["attention_mask"].sum(dim=1)
        for i in range(batch_size):
            prompt_end = min(int(prompt_lengths[i].item()), input_ids.size(1))
            labels[i, :prompt_end] = -100

        return input_ids, attention_mask, labels

    def _run_decoder_ce(
        self,
        prompts: Sequence[str],
        targets: Sequence[str],
        quantized_codes: Optional[torch.Tensor],
        target_types: Optional[Sequence[str]] = None,
        *,
        ecg_embeddings: Optional[torch.Tensor] = None,
    ) -> Optional[torch.Tensor]:
        """Run decoder cross-entropy loss with prompt/target separation.

        Args:
            prompts: List of prompts (masked in labels)
            targets: List of targets (predicted by model)
            quantized_codes: ECG codes tensor

        Returns:
            CE loss tensor or None if invalid inputs
        """
        if not prompts or not targets or self.decoder is None:
            return None
        if quantized_codes is None and ecg_embeddings is None:
            return None
        if len(prompts) != len(targets):
            return None

        text_input_ids, text_attention_mask, labels = self._build_ce_batch(prompts, targets, target_types)
        text_input_ids = text_input_ids.to(self.device)
        text_attention_mask = text_attention_mask.to(self.device)
        labels = labels.to(self.device)

        ecg_tokens = ecg_embeddings
        if ecg_tokens is not None and not isinstance(ecg_tokens, torch.Tensor):
            return None
        if ecg_tokens is None:
            if quantized_codes is None or not isinstance(quantized_codes, torch.Tensor):
                return None
            bridge_module = getattr(self.decoder, "bridge", None)
            if bridge_module is None:
                return None
            bridge_out = bridge_module(quantized_codes.to(self.device))
            if isinstance(bridge_out, tuple):
                ecg_tokens = bridge_out[0]
            elif isinstance(bridge_out, dict):
                ecg_tokens = bridge_out.get("token_embeddings") or bridge_out.get("embeddings")
            else:
                ecg_tokens = bridge_out
        ecg_tokens = ecg_tokens.to(self.device)
        if ecg_tokens.dim() == 2:
            ecg_tokens = ecg_tokens.unsqueeze(1)

        target_std = getattr(self.config, "target_embedding_std", None)
        inputs_embeds, attention_mask = build_inputs_with_ecg_prefix(
            self.decoder.llm_model,
            text_input_ids,
            text_attention_mask,
            ecg_tokens,
            target_embedding_std=target_std,
        )
        labels = pad_labels_for_ecg_prefix(labels, ecg_tokens.size(1))

        outputs = self.decoder.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )
        loss = outputs.loss
        # NaN guard: a microbatch whose labels are ALL ignore_index makes HF's
        # cross-entropy average over zero valid tokens -> NaN (more likely with
        # small ce_max_samples_per_batch). bf16 overflow can also yield non-finite
        # loss. Returning None makes the caller skip the CE contribution for this
        # step rather than poisoning the optimized loss / decoder gradients.
        if loss is None or not torch.isfinite(loss):
            self._ce_nonfinite_count = getattr(self, "_ce_nonfinite_count", 0) + 1
            if getattr(self.config, "is_ref_device", True) and self._ce_nonfinite_count <= 20:
                print(f"[SigLIP] Skipping non-finite CE loss "
                      f"(count={self._ce_nonfinite_count}, n_prompts={len(prompts)}).")
            return None
        return loss

    def _get_report_prompt_tensors(self) -> Tuple[torch.Tensor, torch.Tensor, int, str]:
        if self._report_prompt_cache is not None:
            return self._report_prompt_cache
        if self.decoder_tokenizer is None:
            raise RuntimeError("Decoder tokenizer is required for report generation")

        response_prefix = (self.report_response_prefix + " ").strip()
        tokenizer = self.decoder_tokenizer
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": self.report_system_prompt},
                {"role": "user", "content": self.report_user_prompt},
                {"role": "assistant", "content": response_prefix},
            ]
            prompt_str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_str = f"{self.report_system_prompt}\n{self.report_user_prompt}\n{response_prefix}"

        enc = tokenizer(
            [prompt_str],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.ce_max_length,
        )
        prompt_input_ids = enc["input_ids"].to(self.device)
        prompt_attention_mask = enc["attention_mask"].to(self.device)
        available = max(1, self.ce_max_length - prompt_input_ids.shape[1])
        max_new_tokens = min(self.ce_max_length, max(8, available))

        self._report_prompt_cache = (prompt_input_ids, prompt_attention_mask, max_new_tokens, response_prefix)
        return self._report_prompt_cache

    def _generate_reports_batch(self, codes_list: Sequence[Optional[torch.Tensor]], *, return_token_ids: bool = False) -> List[str] | tuple[List[str], torch.Tensor]:
        if self.decoder is None or self.decoder_tokenizer is None:
            return ["" for _ in codes_list]
        if not codes_list:
            return []

        processed_tensors: List[torch.Tensor] = []
        valid_indices: List[int] = []
        for idx, codes in enumerate(codes_list):
            if not isinstance(codes, torch.Tensor):
                continue
            codes_tensor = codes.to(self.device)
            if codes_tensor.dim() == 1:
                codes_tensor = codes_tensor.unsqueeze(0)
            elif codes_tensor.dim() >= 2 and codes_tensor.size(0) != 1:
                codes_tensor = codes_tensor.unsqueeze(0)
            processed_tensors.append(codes_tensor)
            valid_indices.append(idx)

        results = ["" for _ in codes_list]
        if not processed_tensors:
            return results

        quantized_codes = torch.cat(processed_tensors, dim=0).to(self.device)
        prompt_input_ids, prompt_attention_mask, max_new_tokens, response_prefix = self._get_report_prompt_tensors()
        prompt_input_ids = prompt_input_ids.repeat(quantized_codes.size(0), 1)
        prompt_attention_mask = prompt_attention_mask.repeat(quantized_codes.size(0), 1)

        tokenizer = self.decoder_tokenizer
        bridge_module = getattr(self.decoder, "bridge", None)
        if bridge_module is None:
            return results

        with torch.no_grad():
            bridge_out = bridge_module(quantized_codes)
            if isinstance(bridge_out, tuple):
                ecg_tokens = bridge_out[0]
            elif isinstance(bridge_out, dict):
                ecg_tokens = bridge_out.get("token_embeddings") or bridge_out.get("embeddings")
            else:
                ecg_tokens = bridge_out
            if ecg_tokens.dim() == 2:
                ecg_tokens = ecg_tokens.unsqueeze(1)

            target_std = getattr(self.config, "target_embedding_std", None)
            inputs_embeds, attention_mask = build_inputs_with_ecg_prefix(
                self.decoder.llm_model,
                prompt_input_ids,
                prompt_attention_mask,
                ecg_tokens,
                target_embedding_std=target_std,
            )
            min_new_tokens = min(self.report_min_new_tokens, max_new_tokens)
            generated = self.decoder.llm_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                min_new_tokens=min_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                eos_token_id=tokenizer.eos_token_id,
            )

        sequences = getattr(generated, "sequences", generated)
        if not isinstance(sequences, torch.Tensor):
            sequences = torch.as_tensor(sequences)
        if sequences.dim() == 1:
            sequences = sequences.unsqueeze(0)
        prompt_len = prompt_input_ids.size(1)
        if sequences.size(1) > prompt_len:
            sequences = sequences[:, prompt_len:]
        decoded = tokenizer.batch_decode(sequences, skip_special_tokens=True)

        response_prefix_clean = response_prefix.strip()
        for stored_idx, text in zip(valid_indices, decoded):
            body = text.strip().replace("\n", " ")
            if not body:
                body = "Normal ECG"
            cleaned = f"{response_prefix_clean} {body}".strip()
            results[stored_idx] = cleaned
        if return_token_ids:
            return results, sequences.detach().cpu()
        return results

    def _generate_report_from_codes(self, codes: torch.Tensor) -> str:
        """Generate a full ECG report from quantized codes using the CE prompt."""
        reports = self._generate_reports_batch([codes])
        return reports[0] if reports else ""

    def generate_qa_answer(self, codes: torch.Tensor, question: str, max_new_tokens: int = 10) -> str:
        """Generate a yes/no answer to a QA question from quantized codes.

        Args:
            codes: Quantized ECG codes tensor
            question: Question text (e.g., "Is T wave inversion present?")
            max_new_tokens: Maximum tokens to generate (default 10 for short answers)

        Returns:
            Generated answer (e.g., "Yes." or "No.")
        """
        if self.decoder is None or self.decoder_tokenizer is None:
            return ""

        codes_tensor = codes.to(self.device)
        if codes_tensor.dim() == 1:
            codes_tensor = codes_tensor.unsqueeze(0)
        elif codes_tensor.dim() >= 2 and codes_tensor.size(0) != 1:
            codes_tensor = codes_tensor.unsqueeze(0)

        tokenizer = self.decoder_tokenizer
        response_prefix = (self.qa_response_prefix + " ").strip()
        question_text = question.strip()
        if hasattr(tokenizer, "apply_chat_template"):
            messages = [
                {"role": "system", "content": self.qa_system_prompt},
                {"role": "user", "content": question_text},
                {"role": "assistant", "content": response_prefix},
            ]
            prompt_str = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt_str = f"{self.qa_system_prompt}\n{question_text}\n{response_prefix}"

        enc = tokenizer(
            [prompt_str],
            return_tensors="pt",
            padding=False,
            truncation=True,
            max_length=self.ce_max_length,
        )

        prompt_input_ids = enc["input_ids"].to(self.device)
        prompt_attention_mask = enc["attention_mask"].to(self.device)
        max_new_tokens = max(4, min(max_new_tokens, self.ce_max_length))

        bridge_module = getattr(self.decoder, "bridge", None)
        if bridge_module is None:
            return ""

        with torch.no_grad():
            bridge_out = bridge_module(codes_tensor)
            if isinstance(bridge_out, tuple):
                ecg_tokens = bridge_out[0]
            elif isinstance(bridge_out, dict):
                ecg_tokens = bridge_out.get("token_embeddings") or bridge_out.get("embeddings")
            else:
                ecg_tokens = bridge_out
        if ecg_tokens.dim() == 2:
            ecg_tokens = ecg_tokens.unsqueeze(1)
        target_std = getattr(self.config, "target_embedding_std", None)
        inputs_embeds, attention_mask = build_inputs_with_ecg_prefix(
            self.decoder.llm_model,
            prompt_input_ids,
            prompt_attention_mask,
            ecg_tokens,
            target_embedding_std=target_std,
        )
        min_new_tokens = min(self.qa_min_new_tokens, max_new_tokens)
        generated = self.decoder.llm_model.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            min_new_tokens=min_new_tokens,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            eos_token_id=tokenizer.eos_token_id,
        )

        sequences = getattr(generated, "sequences", generated)
        if not isinstance(sequences, torch.Tensor):
            sequences = torch.as_tensor(sequences)
        if sequences.dim() == 1:
            sequences = sequences.unsqueeze(0)

        prompt_len = prompt_input_ids.size(1)
        if sequences.size(1) > prompt_len:
            sequences = sequences[:, prompt_len:]

        text = tokenizer.batch_decode(sequences, skip_special_tokens=True)[0].strip()
        if text.startswith(self.qa_response_prefix):
            text = text[len(self.qa_response_prefix):].strip()
        if not text:
            return "Yes."
        return text

    @staticmethod
    def _clean_generated_text(text: str, extra_forbidden: Optional[Sequence[str]] = None) -> str:
        if not text:
            return text
        extra = set(s.lower() for s in (extra_forbidden or []))
        lines = []
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            lower = stripped.lower()
            if lower in {"user", "model", "assistant"}:
                continue
            if "you are a medical expert" in lower:
                continue
            if "you are a cardiology ecg expert" in lower:
                continue
            if lower.startswith("analyze this ecg"):
                continue
            if lower in extra:
                continue
            lines.append(stripped)
        return "\n".join(lines) if lines else text.strip()

    def _build_report_prompt_and_target(self, report_text: str) -> tuple[str, str, str]:
        cleaned_report = report_text.strip()
        prompt = f"{self.report_system_prompt}\n{self.report_user_prompt}"
        target = f"{self.report_response_prefix} {cleaned_report}".strip()
        return prompt, target, cleaned_report

    def _build_qa_prompt_and_target(self, qa_text: str) -> tuple[str, str, str]:
        raw = qa_text.strip()
        question = raw
        answer = "Yes."
        if "A:" in raw:
            question_part, answer_part = raw.split("A:", 1)
            question = question_part.strip()
            answer = answer_part.strip()
        prompt = f"{self.qa_system_prompt}\n{question}"
        target = f"{self.qa_response_prefix} {answer}".strip()
        return prompt, target, answer
