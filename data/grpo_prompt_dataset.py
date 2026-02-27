"""Prompt-only dataset for GRPO training.

Loads a CoT SFT parquet file and returns (signal, prompt_input_ids,
prompt_attention_mask, ground_truth_text) per sample. The prompt is the
system + user turns only (no assistant response), formatted for MedGemma.
"""

import json
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from torch.utils.data import Dataset

from data.dpo_pair_dataset import _load_ecg_waveform


def _parse_messages(raw: Any) -> Optional[Dict[str, str]]:
    """Parse a JSON chat messages column into system/user/assistant content."""
    try:
        msgs = json.loads(raw) if isinstance(raw, str) else raw
        if not isinstance(msgs, list):
            return None
        result: Dict[str, str] = {}
        for msg in msgs:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("system", "user", "assistant"):
                result[role] = content
        if "assistant" not in result:
            return None
        return result
    except Exception:
        return None


def _build_prompt_tokens(
    system_message: str,
    user_message: str,
    tokenizer,
    medgemma_prompt_style: bool,
) -> tuple:
    """Build tokenized prompt (system + user turns only, no assistant).

    Returns (prompt_input_ids, prompt_attention_mask) as 1-D tensors.
    """
    if medgemma_prompt_style:
        # Replace <image> placeholder with MedGemma's <start_of_image>
        user_content = user_message.replace("<image>", "<start_of_image>")
        if "<start_of_image>" not in user_content:
            user_content = "<start_of_image>\n\n" + user_content
        prompt_text = (
            "<start_of_turn>system\n"
            f"{system_message}<end_of_turn>\n"
            "<start_of_turn>user\n"
            f"{user_content}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
    else:
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_message},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    encoding = tokenizer.encode_plus(
        prompt_text, add_special_tokens=True, return_tensors="pt"
    )
    prompt_input_ids = encoding["input_ids"].squeeze(0)  # [seq_len]
    prompt_attention_mask = encoding["attention_mask"].squeeze(0)
    return prompt_input_ids, prompt_attention_mask


class GRPOPromptDataset(Dataset):
    """Dataset that yields prompts + ECG signals for GRPO online generation."""

    def __init__(
        self,
        dataset_path: str,
        tokenizer,
        config: Any,
        signal_path_column: str = "signal_path",
        messages_column: str = "messages",
        report_column: str = "report",
    ) -> None:
        self.df = pd.read_parquet(dataset_path)
        self.tokenizer = tokenizer
        self.config = config
        self.signal_path_column = signal_path_column
        self.messages_column = messages_column
        self.report_column = report_column
        self.medgemma_prompt_style = bool(
            getattr(config, "medgemma_prompt_style", False)
        )
        self.ecg_waveform_length = int(
            getattr(config, "ecg_waveform_length", 2500)
        )
        self.ecg_num_leads = int(getattr(config, "ecg_num_leads", 12))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]

        # Load ECG waveform
        waveform_path = str(row[self.signal_path_column])
        waveform = _load_ecg_waveform(
            waveform_path=waveform_path,
            target_length=self.ecg_waveform_length,
            num_leads=self.ecg_num_leads,
        )
        signal = torch.from_numpy(waveform).T  # [num_leads, length]

        # Parse messages to get system/user/assistant
        parsed = _parse_messages(row[self.messages_column])
        if parsed is None:
            raise ValueError(
                f"Failed to parse messages for row {idx}: {row[self.messages_column]}"
            )

        system_message = parsed.get(
            "system",
            "You are an expert cardiologist. You interpret ECGs and answer in a concise, structured way.",
        )
        user_message = parsed.get("user", "")
        ground_truth_text = parsed.get("assistant", "")

        # Build prompt tokens (system + user only)
        prompt_input_ids, prompt_attention_mask = _build_prompt_tokens(
            system_message=system_message,
            user_message=user_message,
            tokenizer=self.tokenizer,
            medgemma_prompt_style=self.medgemma_prompt_style,
        )

        return {
            "signal": signal,
            "prompt_input_ids": prompt_input_ids,
            "prompt_attention_mask": prompt_attention_mask,
            "ground_truth_text": ground_truth_text,
        }


def grpo_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function that pads prompts to the same length within a batch."""
    signals = torch.stack([item["signal"] for item in batch], dim=0)
    ground_truths = [item["ground_truth_text"] for item in batch]

    # Pad prompt_input_ids and prompt_attention_mask to max length in batch
    prompt_ids_list = [item["prompt_input_ids"] for item in batch]
    prompt_mask_list = [item["prompt_attention_mask"] for item in batch]

    max_prompt_len = max(ids.size(0) for ids in prompt_ids_list)

    # Determine pad token id
    pad_id = 0
    if hasattr(batch[0], "tokenizer"):
        pad_id = batch[0].tokenizer.pad_token_id or 0

    padded_ids = []
    padded_masks = []
    for ids, mask in zip(prompt_ids_list, prompt_mask_list):
        pad_len = max_prompt_len - ids.size(0)
        if pad_len > 0:
            ids = torch.nn.functional.pad(ids, (0, pad_len), value=pad_id)
            mask = torch.nn.functional.pad(mask, (0, pad_len), value=0)
        padded_ids.append(ids)
        padded_masks.append(mask)

    prompt_input_ids = torch.stack(padded_ids, dim=0)
    prompt_attention_mask = torch.stack(padded_masks, dim=0)

    return {
        "signal": signals,
        "prompt_input_ids": prompt_input_ids,
        "prompt_attention_mask": prompt_attention_mask,
        "ground_truth_text": ground_truths,
    }
