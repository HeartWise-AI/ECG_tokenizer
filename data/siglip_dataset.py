"""Dataset and collate utilities for SigLIP Phase-1 alignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from utils.ddp import DistributedUtils
from utils.constants import lead_to_idx
from dataset_generation.siglip_shared import SIGLIP_TARGETED_HARD_NEGATIVES


def _load_waveform(path: str) -> np.ndarray:
    """Load waveform from an .npy file with basic validation."""
    waveform = np.load(path)
    if waveform.ndim == 3 and waveform.shape[-1] == 1:
        waveform = waveform.squeeze(-1)
    if waveform.ndim != 2:
        raise ValueError(f"Expected waveform with shape (time, leads); got {waveform.shape} from {path}")
    return waveform


def _normalize_waveform(
    waveform: np.ndarray,
    normalize: bool,
    lead_stats: dict[str, dict[str, float]] | None,
) -> np.ndarray:
    if not normalize or lead_stats is None:
        return waveform
    normalized = np.zeros_like(waveform)
    for lead_name, lead_idx in lead_to_idx.items():
        stats = lead_stats.get(lead_name)
        if stats is None:
            raise KeyError(f"Missing normalization statistics for lead '{lead_name}'")
        normalized[:, lead_idx] = (waveform[:, lead_idx] - stats["mean"]) / stats["std"]
    return normalized


@dataclass
class SiglipSample:
    ecg_id: str
    targets: List[tuple[str, int, float]]
    split: str
    report: str = ""


class SiglipDataset(Dataset):
    """Dataset yielding ECG signals and explicit SigLIP supervision tuples."""

    def __init__(
        self,
        mapping_csv: str,
        text_bank_csv: str,
        split: str,
        expected_waveform_length: int,
        num_leads: int,
        normalize_waveforms: bool,
        lead_stats: dict[str, dict[str, float]] | None,
        qa_positive_weight_multiplier: float = 1.0,
        signal_path_column: str = "ecg_id",
        shuffle: bool = False,
        shuffle_seed: int | None = None,
    ) -> None:
        mapping_df = pd.read_csv(mapping_csv)
        if split:
            mapping_df = mapping_df[mapping_df["split"].str.lower() == split.lower()]
        if mapping_df.empty:
            raise ValueError(f"No rows found for split='{split}' in {mapping_csv}")

        text_df = pd.read_csv(text_bank_csv)
        if text_df.empty:
            raise ValueError(f"text_bank file {text_bank_csv} is empty")

        self.text_lookup: Dict[str, str] = dict(zip(text_df["text_id"], text_df["text"]))
        self.all_text_ids: List[str] = list(self.text_lookup.keys())

        self.qa_positive_weight_multiplier = float(qa_positive_weight_multiplier) if qa_positive_weight_multiplier else 1.0

        report_lookup: Dict[str, str] = {}
        for candidate in ("diagnosis", "report", "original_report"):
            if candidate in mapping_df.columns:
                subset = mapping_df[["ecg_id", candidate]].dropna(subset=[candidate])
                if subset.empty:
                    continue
                subset = subset.astype({candidate: str})
                subset = subset.drop_duplicates(subset=["ecg_id"], keep="first")
                report_lookup = dict(zip(subset["ecg_id"], subset[candidate]))
                break

        grouped = mapping_df.groupby("ecg_id")
        self.samples: List[SiglipSample] = []
        for ecg_id, group in grouped:
            entries = list(
                zip(
                    group["text_id"].tolist(),
                    group["label"].astype(int).tolist(),
                    group["weight"].astype(float).tolist(),
                )
            )
            ecg_split = group["split"].iloc[0] if "split" in group.columns else split or "train"
            adjusted_entries: list[tuple[str, int, float]] = []
            for text_id, label, weight in entries:
                if label > 0 and text_id.startswith("QA_"):
                    weight *= self.qa_positive_weight_multiplier
                adjusted_entries.append((text_id, label, weight))
            self.samples.append(
                SiglipSample(
                    ecg_id=ecg_id,
                    targets=adjusted_entries,
                    split=str(ecg_split),
                    report=report_lookup.get(ecg_id, ""),
                )
            )

        if shuffle:
            rng = np.random.default_rng(shuffle_seed)
            rng.shuffle(self.samples)
        else:
            self.samples.sort(key=lambda item: item.ecg_id)

        self.expected_waveform_length = expected_waveform_length
        self.num_leads = num_leads
        self.normalize_waveforms = normalize_waveforms
        self.lead_stats = lead_stats
        self.signal_path_column = signal_path_column

    def __len__(self) -> int:
        return len(self.samples)

    def _load_signal(self, ecg_path: str) -> np.ndarray:
        waveform = _load_waveform(ecg_path)
        if waveform.shape[1] != self.num_leads:
            raise ValueError(f"Waveform {ecg_path} has {waveform.shape[1]} leads; expected {self.num_leads}")
        if waveform.shape[0] > self.expected_waveform_length:
            step = waveform.shape[0] // self.expected_waveform_length
            waveform = waveform[::step, :]
        if waveform.shape[0] != self.expected_waveform_length:
            raise ValueError(
                f"Waveform {ecg_path} length {waveform.shape[0]} != {self.expected_waveform_length}"
            )
        waveform = _normalize_waveform(waveform, self.normalize_waveforms, self.lead_stats)
        return waveform.T.astype(np.float32)  # leads x time

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        signal = self._load_signal(sample.ecg_id)
        return {
            "signal": signal,
            "ecg_id": sample.ecg_id,
            "targets": sample.targets,
            "split": sample.split,
            "report": sample.report,
        }


class SiglipBatchCollatorInfoNCE:
    """Collate function that builds masked InfoNCE supervision matrices."""

    def __init__(
        self,
        text_lookup: Dict[str, str],
        all_text_ids: Sequence[str],
        implicit_negatives_per_batch: int,
        implicit_negatives_per_row: int | None = None,
        negatives_mode: str = "per_batch",
        implicit_weight: float = 1.0,
        seed: int = 0,
        max_hardneg_per_group: int = 3,
    ) -> None:
        self.text_lookup = text_lookup
        self.all_text_ids = list(all_text_ids)
        self.id_set = set(self.all_text_ids)
        self.implicit_negatives_per_batch = max(0, int(implicit_negatives_per_batch))
        self.implicit_negatives_per_row = (
            None if implicit_negatives_per_row is None else max(0, int(implicit_negatives_per_row))
        )
        mode = (negatives_mode or "per_batch").lower()
        self.negatives_mode = "per_row" if mode == "per_row" else "per_batch"
        self.max_hardneg_per_group = max(0, int(max_hardneg_per_group))
        self.rng = np.random.default_rng(seed)

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
        rhythm_candidates: List[str] = []
        for tid in self.all_text_ids:
            if any(token in tid for token in rhythm_tokens):
                if tid.startswith("QA_") and tid.endswith("_yes"):
                    rhythm_candidates.append(tid)
                elif tid.startswith("LBL_"):
                    rhythm_candidates.append(tid)
        self.rhythm_candidates = rhythm_candidates
        family_negs: Dict[str, List[str]] = {}
        for key, values in SIGLIP_TARGETED_HARD_NEGATIVES.items():
            if key not in self.id_set:
                continue
            filtered = [val for val in values if val in self.id_set]
            if filtered:
                family_negs[key] = filtered
        self.family_hard_negs = family_negs

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(int(seed))

    @staticmethod
    def _counterpart(text_id: str) -> str | None:
        if text_id.endswith("yes"):
            return text_id[:-3] + "no"
        if text_id.endswith("no"):
            return text_id[:-2] + "yes"
        return None

    def _extract_item_targets(self, item: dict) -> List[tuple[str, int, float]]:
        triples: List[tuple[str, int, float]] = []
        if "targets" in item and item["targets"]:
            for tid, label, weight in item["targets"]:
                triples.append((tid, int(label), float(weight)))
        else:
            pos_ids = item.get("pos_text_ids", []) or []
            pos_weights = item.get("pos_weights", []) or [1.0] * len(pos_ids)
            for tid, w in zip(pos_ids, pos_weights):
                triples.append((tid, 1, float(w)))
        return [(tid, lab, w) for (tid, lab, w) in triples if tid in self.id_set]

    def _sample_rhythm_negatives(self, positives: List[str]) -> List[str]:
        if self.max_hardneg_per_group == 0 or not self.rhythm_candidates:
            return []
        pos_set = set(positives)
        pool = [tid for tid in self.rhythm_candidates if tid not in pos_set]
        if not pool:
            return []
        sample_size = min(self.max_hardneg_per_group, len(pool))
        indices = self.rng.choice(len(pool), size=sample_size, replace=False)
        return [pool[int(idx)] for idx in indices]

    def _gather_family_negatives(self, positives: Sequence[str]) -> List[str]:
        collected: List[str] = []
        for tid in positives:
            negatives = self.family_hard_negs.get(tid)
            if negatives:
                collected.extend(negatives)
        # Preserve order while removing duplicates
        return list(dict.fromkeys(collected))

    def __call__(self, batch: Sequence[dict]) -> dict:
        batch_size = len(batch)
        signals: List[torch.Tensor] = []
        ecg_ids: List[str] = []
        positives: List[List[str]] = []
        positive_weights: List[List[float]] = []
        required_negs: List[List[str]] = []
        reports: List[str] = []

        for item in batch:
            signals.append(torch.from_numpy(item["signal"]))
            ecg_ids.append(item.get("ecg_id", ""))
            reports.append(item.get("report", ""))

            triples = self._extract_item_targets(item)
            pos_ids: List[str] = []
            pos_ws: List[float] = []
            explicit_negs: List[str] = []

            for tid, label, weight in triples:
                if label > 0:
                    pos_ids.append(tid)
                    pos_ws.append(max(1.0, float(weight)))
                elif weight > 0:
                    explicit_negs.append(tid)

            required: List[str] = []
            for tid_list in (pos_ids, explicit_negs):
                for tid in tid_list:
                    cp = self._counterpart(tid)
                    if cp and cp in self.id_set:
                        required.append(cp)

            required.extend(explicit_negs)
            required.extend(self._sample_rhythm_negatives(pos_ids))
            required.extend(self._gather_family_negatives(pos_ids))
            positives.append(pos_ids)
            positive_weights.append(pos_ws)
            required_negs.append(list(dict.fromkeys(required)))

        candidate_ids: List[str] = []
        candidate_set: set[str] = set()

        def _add_candidate(tid: str) -> None:
            if tid in self.id_set and tid not in candidate_set:
                candidate_set.add(tid)
                candidate_ids.append(tid)

        for pos in positives:
            for tid in pos:
                _add_candidate(tid)
        for negs in required_negs:
            for tid in negs:
                _add_candidate(tid)

        remaining = [tid for tid in self.all_text_ids if tid not in candidate_set]
        if remaining:
            if self.negatives_mode == "per_row" and self.implicit_negatives_per_row:
                for _ in range(batch_size):
                    remaining_pool = [tid for tid in self.all_text_ids if tid not in candidate_set]
                    if not remaining_pool:
                        break
                    sample_size = min(self.implicit_negatives_per_row, len(remaining_pool))
                    if sample_size <= 0:
                        break
                    indices = self.rng.choice(len(remaining_pool), size=sample_size, replace=False)
                    for idx in indices:
                        _add_candidate(remaining_pool[int(idx)])
            elif self.implicit_negatives_per_batch > 0:
                sample_size = min(self.implicit_negatives_per_batch, len(remaining))
                if sample_size > 0:
                    indices = self.rng.choice(len(remaining), size=sample_size, replace=False)
                    for idx in indices:
                        _add_candidate(remaining[int(idx)])

        num_candidates = len(candidate_ids)
        labels = torch.zeros((batch_size, num_candidates), dtype=torch.float32)
        weights = torch.zeros_like(labels)
        id_to_col = {tid: col for col, tid in enumerate(candidate_ids)}

        for row in range(batch_size):
            pos_ids = positives[row]
            pos_ws = positive_weights[row]
            for tid, weight in zip(pos_ids, pos_ws):
                col = id_to_col.get(tid)
                if col is None:
                    continue
                labels[row, col] = 1.0
                weights[row, col] = 1.0
                cp = self._counterpart(tid)
                if cp is not None and cp in id_to_col:
                    weights[row, id_to_col[cp]] = 1.0

            for tid in required_negs[row]:
                col = id_to_col.get(tid)
                if col is not None:
                    weights[row, col] = 1.0

            if not weights[row].any():
                weights[row, 0] = 1.0

        signals_tensor = torch.stack(signals, dim=0)

        return {
            "signals": signals_tensor,
            "ecg_ids": ecg_ids,
            "text_ids": candidate_ids,
            "labels": labels,
            "weights": weights,
            "reports": reports,
        }


def get_siglip_dataloader(
    dataset: SiglipDataset,
    batch_size: int,
    num_workers: int,
    num_replicas: int,
    rank: int,
    shuffle: bool,
    collate_fn: SiglipBatchCollatorInfoNCE,
) -> torch.utils.data.DataLoader:
    return DistributedUtils.get_distributed_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        pin_memory=True,
        collate_fn=collate_fn,
    )


__all__ = ["SiglipDataset", "SiglipBatchCollatorInfoNCE", "get_siglip_dataloader"]
