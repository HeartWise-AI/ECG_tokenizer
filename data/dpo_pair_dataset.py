import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class TokenizedSample:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    prompt_input_ids: torch.Tensor
    prompt_attention_mask: torch.Tensor


def _load_ecg_waveform(
    waveform_path: str,
    target_length: int,
    num_leads: int,
) -> np.ndarray:
    waveform = np.load(waveform_path)
    if waveform.ndim == 3:
        waveform = waveform.squeeze(-1)
    if waveform.shape[-1] == num_leads:
        pass
    elif waveform.shape[0] == num_leads:
        waveform = waveform.T
    current_length = waveform.shape[0]
    if current_length >= target_length:
        start = (current_length - target_length) // 2
        waveform = waveform[start:start + target_length, :]
    else:
        pad_before = (target_length - current_length) // 2
        pad_after = target_length - current_length - pad_before
        waveform = np.pad(waveform, ((pad_before, pad_after), (0, 0)), mode="edge")
    if waveform.shape[1] != num_leads:
        raise ValueError(f"Unexpected waveform shape {waveform.shape} for {waveform_path}")
    return waveform.astype(np.float32)


def _build_tokens(
    prompt_text: str,
    answer_text: str,
    tokenizer,
    max_length: int,
    num_ecg_tokens: int,
    ecg_token_start_id: Optional[int],
    prefix_tuning: bool,
    medgemma_prompt_style: bool,
) -> TokenizedSample:
    system_message = "You are an expert cardiologist. You interpret ECGs and answer in a concise, structured way."
    if not prompt_text:
        prompt_text = "Analyze this ECG and list the clinical findings."

    if medgemma_prompt_style:
        user_content = (
            "<start_of_image>\n\n"
            f"Question: {prompt_text}\n\n"
            "Respond concisely with the key finding or answer."
        )
        prompt_template_text = (
            "<start_of_turn>system\n"
            f"{system_message}<end_of_turn>\n"
            "<start_of_turn>user\n"
            f"{user_content}<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        full_template_text = (
            "<start_of_turn>system\n"
            f"{system_message}<end_of_turn>\n"
            "<start_of_turn>user\n"
            f"{user_content}<end_of_turn>\n"
            f"<start_of_turn>model\n{answer_text}<end_of_turn>"
        )
    else:
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError("Tokenizer does not support chat template.")
        messages_prompt = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"<image_1> {prompt_text}".strip()},
        ]
        messages_full = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"<image_1> {prompt_text}".strip()},
            {"role": "assistant", "content": answer_text},
        ]
        prompt_template_text = tokenizer.apply_chat_template(
            messages_prompt, tokenize=False, add_generation_prompt=True
        )
        full_template_text = tokenizer.apply_chat_template(
            messages_full, tokenize=False, add_generation_prompt=False
        )

    prompt_encoding = tokenizer.encode_plus(prompt_template_text, add_special_tokens=True, return_tensors=None)
    full_encoding = tokenizer.encode_plus(full_template_text, add_special_tokens=True, return_tensors=None)
    prompt_ids = prompt_encoding.input_ids
    full_ids = full_encoding.input_ids

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    if num_ecg_tokens > 0:
        if prefix_tuning:
            ecg_prefix = torch.full((num_ecg_tokens,), fill_value=pad_id, dtype=torch.long)
        else:
            start_id = int(ecg_token_start_id) if ecg_token_start_id is not None else 0
            ecg_prefix = torch.arange(start_id, start_id + num_ecg_tokens, dtype=torch.long)
    else:
        ecg_prefix = torch.zeros(0, dtype=torch.long)
    prefix_len = int(ecg_prefix.numel())

    max_text_len = max(0, max_length - prefix_len)
    full_ids_trunc = full_ids[:max_text_len]

    text_ids = torch.tensor(full_ids_trunc, dtype=torch.long)
    if medgemma_prompt_style and prefix_len > 0:
        insert_after_image = False
        image_pos = None
        try:
            image_token_id = tokenizer.convert_tokens_to_ids("<start_of_image>")
            if isinstance(image_token_id, (list, tuple)):
                image_token_id = image_token_id[0] if image_token_id else None
        except Exception:
            image_token_id = None
        if image_token_id is not None:
            try:
                image_pos = text_ids.tolist().index(int(image_token_id))
                insert_after_image = True
            except ValueError:
                insert_after_image = False
        if insert_after_image and image_pos is not None:
            input_ids = torch.cat([text_ids[:image_pos + 1], ecg_prefix, text_ids[image_pos + 1:]], dim=0)
        else:
            input_ids = torch.cat([ecg_prefix, text_ids], dim=0)
    else:
        input_ids = torch.cat([ecg_prefix, text_ids], dim=0)

    attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    if input_ids.numel() < max_length:
        pad_len = max_length - input_ids.numel()
        input_ids = torch.nn.functional.pad(input_ids, (0, pad_len), value=pad_id)
        attention_mask = torch.nn.functional.pad(attention_mask, (0, pad_len), value=0)

    prompt_input_ids = torch.tensor(prompt_ids[:max_length], dtype=torch.long)
    prompt_attention_mask = torch.ones_like(prompt_input_ids, dtype=torch.long)
    if prompt_input_ids.numel() < max_length:
        pad_len = max_length - prompt_input_ids.numel()
        prompt_input_ids = torch.nn.functional.pad(prompt_input_ids, (0, pad_len), value=pad_id)
        prompt_attention_mask = torch.nn.functional.pad(prompt_attention_mask, (0, pad_len), value=0)

    labels = input_ids.clone()
    full_prompt_len = min(max_length, prefix_len + len(prompt_ids))
    labels[:full_prompt_len] = -100
    labels = labels.masked_fill(attention_mask == 0, -100)

    return TokenizedSample(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
    )


class DPOPairDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        config: Any,
        max_length: int,
        waveform_key: str = "waveform_path",
        prompt_key: str = "prompt",
        chosen_key: str = "chosen",
        rejected_key: str = "rejected",
        weight_key: str = "weight",
    ) -> None:
        self.records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                self.records.append(json.loads(line))

        self.tokenizer = tokenizer
        self.config = config
        self.max_length = max_length
        self.waveform_key = waveform_key
        self.prompt_key = prompt_key
        self.chosen_key = chosen_key
        self.rejected_key = rejected_key
        self.weight_key = weight_key

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records[idx]
        waveform_path = row[self.waveform_key]
        if not os.path.exists(waveform_path):
            raise FileNotFoundError(f"Missing waveform: {waveform_path}")

        prompt = str(row[self.prompt_key])
        chosen = str(row[self.chosen_key])
        rejected = str(row[self.rejected_key])
        weight = float(row.get(self.weight_key, 1.0))

        waveform = _load_ecg_waveform(
            waveform_path=waveform_path,
            target_length=int(getattr(self.config, "ecg_waveform_length", 2500)),
            num_leads=int(getattr(self.config, "ecg_num_leads", 12)),
        )
        signal = torch.from_numpy(waveform).T  # [12, length]

        num_ecg_tokens = int(getattr(self.config, "num_ecg_tokens", 0) or 0)
        ecg_token_start_id = getattr(self.config, "ecg_token_start_id", None)
        prefix_tuning = bool(getattr(self.config, "prefix_tuning", False))
        medgemma_prompt_style = bool(getattr(self.config, "medgemma_prompt_style", False))

        chosen_tokens = _build_tokens(
            prompt_text=prompt,
            answer_text=chosen,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            num_ecg_tokens=num_ecg_tokens,
            ecg_token_start_id=ecg_token_start_id,
            prefix_tuning=prefix_tuning,
            medgemma_prompt_style=medgemma_prompt_style,
        )
        rejected_tokens = _build_tokens(
            prompt_text=prompt,
            answer_text=rejected,
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            num_ecg_tokens=num_ecg_tokens,
            ecg_token_start_id=ecg_token_start_id,
            prefix_tuning=prefix_tuning,
            medgemma_prompt_style=medgemma_prompt_style,
        )

        return {
            "idx": idx,
            "signal": signal,
            "chosen_input_ids": chosen_tokens.input_ids,
            "chosen_attention_mask": chosen_tokens.attention_mask,
            "chosen_labels": chosen_tokens.labels,
            "rejected_input_ids": rejected_tokens.input_ids,
            "rejected_attention_mask": rejected_tokens.attention_mask,
            "rejected_labels": rejected_tokens.labels,
            "prompt_input_ids": chosen_tokens.prompt_input_ids,
            "prompt_attention_mask": chosen_tokens.prompt_attention_mask,
            "weight": torch.tensor(weight, dtype=torch.float32),
        }


def dpo_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    def _stack(key: str, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        tensors = [item[key] for item in batch]
        out = torch.stack(tensors, dim=0)
        if dtype is not None:
            out = out.to(dtype)
        return out

    return {
        "signal": _stack("signal", dtype=torch.float32),
        "chosen_input_ids": _stack("chosen_input_ids"),
        "chosen_attention_mask": _stack("chosen_attention_mask"),
        "chosen_labels": _stack("chosen_labels"),
        "rejected_input_ids": _stack("rejected_input_ids"),
        "rejected_attention_mask": _stack("rejected_attention_mask"),
        "rejected_labels": _stack("rejected_labels"),
        "prompt_input_ids": _stack("prompt_input_ids"),
        "prompt_attention_mask": _stack("prompt_attention_mask"),
        "weight": _stack("weight", dtype=torch.float32),
    }
