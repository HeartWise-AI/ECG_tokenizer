"""Phase 2 RLVR dataset — yields (signal, prompt, gt_answer_text).

The GT reference is the row's `generated_answer` text. The Phase 2 reward
(`utils.rewards_labelset.LabelsetReward`) extracts canonical diagnoses
from both the model output and this GT text and computes F1.

The dataset can filter to specific prompt_categories so training focuses
on diagnostic prompts where the verifier has signal (interpretation,
classification, category_*, json_interpretation).
"""

from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from data.dpo_pair_dataset import _load_ecg_waveform


_DEFAULT_SYSTEM = (
    "You are an expert cardiologist. You interpret ECGs and answer in a concise, structured way."
)


def _build_prompt_tokens(prompt_text: str, tokenizer, medgemma_prompt_style: bool):
    if medgemma_prompt_style:
        user_content = (
            "<start_of_image>\n\n"
            f"Question: {prompt_text}\n\n"
            "Respond concisely with the key finding or answer."
        )
        prompt_text_built = (
            "<start_of_turn>system\n"
            f"{_DEFAULT_SYSTEM}<end_of_turn>\n"
            "<start_of_turn>user\n"
            f"{user_content}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
    else:
        messages = [
            {"role": "system", "content": _DEFAULT_SYSTEM},
            {"role": "user", "content": prompt_text},
        ]
        prompt_text_built = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    encoding = tokenizer.encode_plus(
        prompt_text_built, add_special_tokens=True, return_tensors="pt"
    )
    return encoding["input_ids"].squeeze(0), encoding["attention_mask"].squeeze(0)


class GRPOLabelsetDataset(Dataset):
    def __init__(
        self,
        dataset_path: str,
        tokenizer,
        config: Any,
        signal_path_column: str = "waveform_path_psa",
        prompt_column: str = "prompt",
        answer_column: str = "generated_answer",
        filter_categories: Optional[List[str]] = None,
        max_rows: Optional[int] = None,
    ) -> None:
        import pyarrow.parquet as pq
        avail_cols = set(pq.ParquetFile(dataset_path).schema_arrow.names)
        want = [signal_path_column, prompt_column, answer_column, "prompt_category"]
        if "sample_weight" in avail_cols:
            want.append("sample_weight")
        want = [c for c in want if c in avail_cols]
        df = pd.read_parquet(dataset_path, columns=want)
        if filter_categories and "prompt_category" in df.columns:
            df = df[df["prompt_category"].isin(filter_categories)].reset_index(drop=True)
            print(f"[GRPOLabelsetDataset] After category filter {filter_categories}: {len(df)} rows")
        # Drop rows with empty answers
        df = df[df[answer_column].notna() & (df[answer_column].str.len() > 0)].reset_index(drop=True)
        if max_rows is not None and max_rows < len(df):
            df = df.sample(n=max_rows, random_state=42).reset_index(drop=True)
            print(f"[GRPOLabelsetDataset] Subsampled to max_rows={max_rows}")
        self.df = df

        self.tokenizer = tokenizer
        self.config = config
        self.signal_path_column = signal_path_column
        self.prompt_column = prompt_column
        self.answer_column = answer_column
        self.medgemma_prompt_style = bool(getattr(config, "medgemma_prompt_style", False))
        self.ecg_waveform_length = int(getattr(config, "ecg_waveform_length", 2500))
        self.ecg_num_leads = int(getattr(config, "ecg_num_leads", 12))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        waveform = _load_ecg_waveform(
            waveform_path=str(row[self.signal_path_column]),
            target_length=self.ecg_waveform_length,
            num_leads=self.ecg_num_leads,
        )
        signal = torch.from_numpy(waveform).T

        prompt_ids, prompt_mask = _build_prompt_tokens(
            prompt_text=str(row[self.prompt_column]),
            tokenizer=self.tokenizer,
            medgemma_prompt_style=self.medgemma_prompt_style,
        )

        return {
            "signal": signal,
            "prompt_input_ids": prompt_ids,
            "prompt_attention_mask": prompt_mask,
            "ground_truth_text": str(row[self.answer_column]),
            "prompt_category": str(row.get("prompt_category", "classification")),
        }


def grpo_labelset_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    signals = torch.stack([item["signal"] for item in batch], dim=0)
    ground_truths = [item["ground_truth_text"] for item in batch]
    categories = [item.get("prompt_category", "classification") for item in batch]
    prompt_ids_list = [item["prompt_input_ids"] for item in batch]
    prompt_mask_list = [item["prompt_attention_mask"] for item in batch]
    max_prompt_len = max(ids.size(0) for ids in prompt_ids_list)

    padded_ids, padded_masks = [], []
    for ids, mask in zip(prompt_ids_list, prompt_mask_list):
        pad_len = max_prompt_len - ids.size(0)
        if pad_len > 0:
            ids = torch.nn.functional.pad(ids, (0, pad_len), value=0)
            mask = torch.nn.functional.pad(mask, (0, pad_len), value=0)
        padded_ids.append(ids)
        padded_masks.append(mask)

    return {
        "signal": signals,
        "prompt_input_ids": torch.stack(padded_ids, dim=0),
        "prompt_attention_mask": torch.stack(padded_masks, dim=0),
        "ground_truth_text": ground_truths,
        "prompt_category": categories,
    }
