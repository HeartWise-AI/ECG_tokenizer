"""Dataset utilities for Stage-1 ECG ↔ text alignment training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch.utils.data._utils.collate import default_collate

from data.siglip_dataset import _load_waveform, _normalize_waveform
from utils.ddp import DistributedUtils


@dataclass
class Stage1Sample:
    ecg_path: str
    split: str
    positive_ids: List[str]
    negative_ids: List[str]
    report: str


class ECGTextStage1Dataset(Dataset):
    """Dataset yielding ECG signals paired with positive/negative text supervision."""

    def __init__(
        self,
        mapping_csv: str,
        text_bank_csv: str,
        split: str,
        expected_waveform_length: int,
        num_leads: int,
        normalize_waveforms: bool,
        lead_stats: Dict[str, Dict[str, float]] | None,
        seed: int | None = None,
        max_positives_per_ecg: int | None = None,
    ) -> None:
        mapping_df = pd.read_csv(mapping_csv)
        if split:
            mapping_df = mapping_df[mapping_df["split"].str.lower() == split.lower()]
        if mapping_df.empty:
            raise ValueError(f"No rows for split='{split}' found in {mapping_csv}.")

        text_df = pd.read_csv(text_bank_csv)
        if text_df.empty:
            raise ValueError(f"Text bank '{text_bank_csv}' is empty.")

        self.text_lookup: Dict[str, str] = dict(zip(text_df["text_id"], text_df["text"]))
        self.all_text_ids: List[str] = list(self.text_lookup.keys())

        grouped = mapping_df.groupby("ecg_id")
        samples: List[Stage1Sample] = []
        for ecg_path, group in grouped:
            positives = group[group["label"] > 0]["text_id"].astype(str).tolist()
            negatives = group[group["label"] <= 0]["text_id"].astype(str).tolist()
            positives = [tid for tid in positives if tid in self.text_lookup]
            negatives = [tid for tid in negatives if tid in self.text_lookup]
            if not positives:
                continue
            report = ""
            for candidate in ("diagnosis", "report", "original_report"):
                if candidate in group.columns and pd.notna(group[candidate].iloc[0]):
                    report = str(group[candidate].iloc[0])
                    break
            samples.append(
                Stage1Sample(
                    ecg_path=str(ecg_path),
                    split=str(group["split"].iloc[0] if "split" in group.columns else split or "train"),
                    positive_ids=positives,
                    negative_ids=negatives,
                    report=report,
                )
            )

        if not samples:
            raise ValueError(
                "No Stage-1 samples with positive supervision were found. "
                "Check mapping_csv/text_bank contents."
            )

        self.samples = samples
        self.expected_waveform_length = int(expected_waveform_length)
        self.num_leads = int(num_leads)
        self.normalize_waveforms = bool(normalize_waveforms)
        self.lead_stats = lead_stats
        self.rng = np.random.default_rng(seed)
        self.max_positives_per_ecg = max_positives_per_ecg if max_positives_per_ecg and max_positives_per_ecg > 0 else 1

    def __len__(self) -> int:
        return len(self.samples)

    def reseed(self, seed: int) -> None:
        self.rng = np.random.default_rng(int(seed))

    def _choose_positive_ids(self, sample: Stage1Sample) -> List[str]:
        positives = [pid for pid in sample.positive_ids if pid in self.text_lookup]
        if not positives:
            raise ValueError("No valid positive text IDs available for Stage-1 sample.")
        if self.max_positives_per_ecg >= len(positives):
            return positives
        indices = self.rng.choice(len(positives), size=self.max_positives_per_ecg, replace=False)
        return [positives[int(i)] for i in indices]

    def _load_signal(self, path: str) -> torch.Tensor:
        waveform = _load_waveform(path)
        if waveform.shape[1] != self.num_leads:
            raise ValueError(f"Waveform {path} has {waveform.shape[1]} leads; expected {self.num_leads}.")
        if waveform.shape[0] > self.expected_waveform_length:
            step = waveform.shape[0] // self.expected_waveform_length
            waveform = waveform[::step, :]
        if waveform.shape[0] != self.expected_waveform_length:
            raise ValueError(
                f"Waveform {path} length {waveform.shape[0]} != {self.expected_waveform_length}"
            )
        waveform = _normalize_waveform(waveform, self.normalize_waveforms, self.lead_stats)
        tensor = torch.from_numpy(waveform.T.astype(np.float32))
        return tensor

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]
        signal = self._load_signal(sample.ecg_path)

        positive_ids_raw = self._choose_positive_ids(sample)
        positive_ids: List[str] = []
        positive_texts: List[str] = []
        for pos_id in positive_ids_raw:
            positive_text = self.text_lookup.get(pos_id)
            if positive_text is None:
                continue
            positive_ids.append(pos_id)
            positive_texts.append(positive_text)

        # Return ALL explicit negatives - don't subsample here
        # The negatives will be looked up from pre-encoded text bank embeddings
        explicit_negatives = [
            tid for tid in sample.negative_ids if tid not in positive_ids and tid in self.text_lookup
        ]
        if not explicit_negatives:
            pool = [tid for tid in self.all_text_ids if tid not in positive_ids]
            if not pool:
                raise ValueError("Negative pool is empty; cannot sample negatives for Stage-1 dataset.")
            explicit_negatives = pool
        negative_ids = explicit_negatives
        negative_texts = [self.text_lookup.get(tid, "") for tid in negative_ids]

        if not positive_texts:
            raise KeyError("Unable to retrieve positive/negative texts for Stage-1 dataset entry.")

        return {
            "signal": signal,
            "ecg_id": sample.ecg_path,
            "positive_texts": positive_texts,
            "positive_text_ids": positive_ids[: len(positive_texts)],
            "negative_texts": negative_texts,
            "negative_text_ids": negative_ids,
            "report": sample.report,
            "split": sample.split,
        }


def get_stage1_dataloader(
    dataset: ECGTextStage1Dataset,
    batch_size: int,
    num_workers: int,
    num_replicas: int,
    rank: int,
    shuffle: bool,
) -> torch.utils.data.DataLoader:
    """Return a distributed-aware dataloader for Stage-1 training."""
    return DistributedUtils.get_distributed_dataloader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        collate_fn=stage1_collate,
    )


def stage1_collate(batch: List[Dict[str, object]]) -> Dict[str, object]:
    """Collate Stage-1 samples - ONE sample per ECG with ALL positives and negatives."""

    signals: List[torch.Tensor] = []
    positive_texts_lists: List[List[str]] = []
    positive_ids_lists: List[List[str]] = []
    negative_texts_lists: List[List[str]] = []
    negative_ids_lists: List[List[str]] = []
    reports: List[str] = []
    ecg_ids: List[str] = []

    for sample in batch:
        signal = sample["signal"]
        pos_texts = sample.get("positive_texts") or sample.get("positive_text")
        pos_ids = sample.get("positive_text_ids") or sample.get("positive_text_id")
        neg_texts = sample.get("negative_texts") or sample.get("negative_text")
        neg_ids = sample.get("negative_text_ids") or sample.get("negative_text_id")

        if not isinstance(pos_texts, (list, tuple)):
            pos_texts = [pos_texts]
        if not isinstance(pos_ids, (list, tuple)):
            pos_ids = [pos_ids]
        if not isinstance(neg_texts, (list, tuple)):
            neg_texts = [neg_texts]
        if not isinstance(neg_ids, (list, tuple)):
            neg_ids = [neg_ids]

        pos_text_list = [str(text) for text in pos_texts if text is not None]
        pos_id_list = [str(tid) for tid in pos_ids if tid is not None]
        neg_text_list = [str(text) for text in neg_texts if text is not None]
        neg_id_list = [str(tid) for tid in neg_ids if tid is not None]
        
        if not pos_text_list:
            pos_text_list = [""]
        if not pos_id_list:
            pos_id_list = [""]

        # ONE sample per ECG with ALL positives and negatives
        signals.append(signal)
        positive_texts_lists.append(pos_text_list)
        positive_ids_lists.append(pos_id_list)
        negative_texts_lists.append(neg_text_list)
        negative_ids_lists.append(neg_id_list)
        reports.append(str(sample.get("report", "")))
        ecg_ids.append(str(sample.get("ecg_id", "")))

    stacked_signals = torch.stack(signals, dim=0)

    return {
        "signal": stacked_signals,
        "positive_texts_lists": positive_texts_lists,
        "positive_ids_lists": positive_ids_lists,
        "negative_texts_lists": negative_texts_lists,
        "negative_ids_lists": negative_ids_lists,
        "report": reports,
        "ecg_id": ecg_ids,
    }


__all__ = ["ECGTextStage1Dataset", "get_stage1_dataloader"]
