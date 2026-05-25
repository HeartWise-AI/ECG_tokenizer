#!/usr/bin/env python3
"""Lightweight rejection-sampling fine-tune on judge-selected ECG QA answers."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_dpo_policy import (  # noqa: E402
    _forward_logits_and_labels,
    _sequence_logp,
    build_tokens,
    load_ecg_waveform,
    load_model,
    set_trainable,
)


def should_step_optimizer(step: int, total_steps: int, grad_accum_steps: int) -> bool:
    grad_accum_steps = max(1, int(grad_accum_steps))
    return step % grad_accum_steps == 0 or step == total_steps


def set_rft_trainable(model: torch.nn.Module, mode: str) -> None:
    if mode == "llm":
        for param in model.parameters():
            param.requires_grad = False
        for name, param in model.named_parameters():
            if name.startswith("decoder.llm_model"):
                param.requires_grad = True
        llm = getattr(getattr(model, "decoder", None), "llm_model", None)
        if llm is not None:
            if hasattr(llm, "gradient_checkpointing_enable"):
                llm.gradient_checkpointing_enable()
                print("[rft] gradient checkpointing enabled on LLM")
            if hasattr(llm, "config"):
                llm.config.use_cache = False
    else:
        set_trainable(model, mode)


class RFTDataset(Dataset):
    def __init__(
        self,
        path: str,
        tokenizer,
        config,
        max_length: int,
        waveform_key: str = "waveform_path",
        prompt_key: str = "prompt",
        answer_key: str = "chosen",
        weight_key: str = "weight",
    ) -> None:
        self.records: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    self.records.append(json.loads(line))
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = max_length
        self.waveform_key = waveform_key
        self.prompt_key = prompt_key
        self.answer_key = answer_key
        self.weight_key = weight_key

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.records[idx]
        waveform = load_ecg_waveform(
            waveform_path=row[self.waveform_key],
            target_length=int(getattr(self.config, "ecg_waveform_length", 2500)),
            num_leads=int(getattr(self.config, "ecg_num_leads", 12)),
        )
        signal = torch.from_numpy(waveform).T
        tokens = build_tokens(
            prompt_text=str(row[self.prompt_key]),
            answer_text=str(row[self.answer_key]),
            tokenizer=self.tokenizer,
            max_length=self.max_length,
            num_ecg_tokens=int(getattr(self.config, "num_ecg_tokens", 0)),
            ecg_token_start_id=getattr(self.config, "ecg_token_start_id", None),
            prefix_tuning=bool(getattr(self.config, "prefix_tuning", False)),
            medgemma_prompt_style=bool(getattr(self.config, "medgemma_prompt_style", False)),
        )
        return {
            "signal": signal,
            "input_ids": tokens.input_ids,
            "attention_mask": tokens.attention_mask,
            "labels": tokens.labels,
            "prompt_input_ids": tokens.prompt_input_ids,
            "prompt_attention_mask": tokens.prompt_attention_mask,
            "weight": torch.tensor(float(row.get(self.weight_key, 1.0)), dtype=torch.float32),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--grad_accum_steps", type=int, default=1)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--trainable", choices=["lora", "llm", "bridge", "projection", "all"], default="lora")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    model, tokenizer, config = load_model(args.checkpoint, device)
    set_rft_trainable(model, args.trainable)
    if args.trainable in {"lora", "llm"}:
        model.eval()
    else:
        model.train()

    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"[rft] trainable params: {sum(p.numel() for p in trainable_params):,}")
    optimizer = torch.optim.AdamW(trainable_params, lr=args.lr)

    max_length = args.max_length or int(getattr(config, "max_token_length", 640))
    dataset = RFTDataset(args.jsonl, tokenizer, config, max_length=max_length)
    if len(dataset) == 0:
        raise SystemExit("RFT dataset is empty")
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)

    use_autocast = args.precision in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if args.precision == "bf16" else torch.float16
    grad_accum_steps = max(1, int(args.grad_accum_steps))
    global_step = 0

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(args.epochs):
        running = 0.0
        total_steps = len(loader)
        for step, batch in enumerate(loader, start=1):
            signal = batch["signal"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            prompt_input_ids = batch["prompt_input_ids"].to(device)
            prompt_attention_mask = batch["prompt_attention_mask"].to(device)
            weight = batch["weight"].to(device)

            with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                logits, labels_expanded = _forward_logits_and_labels(
                    model,
                    signal,
                    input_ids,
                    attention_mask,
                    labels,
                    prompt_input_ids,
                    prompt_attention_mask,
                )
                logp = _sequence_logp(logits, labels_expanded)
                num_tokens = (labels_expanded != -100).sum(dim=-1).clamp_min(1).float()
                loss = ((-logp / num_tokens) * weight).mean()
                loss = loss / grad_accum_steps

            loss.backward()
            running += float(loss.detach().cpu())
            if should_step_optimizer(step, total_steps, grad_accum_steps):
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            if step % 25 == 0:
                print(f"[rft] epoch={epoch} step={step}/{len(loader)} loss={running / step:.4f}")

        avg_loss = running / max(1, len(loader))
        out_path = out_dir / f"rft_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": avg_loss,
            "config": config,
            "use_lora": bool(getattr(config, "use_lora", False)),
        }, out_path)
        print(f"[rft] epoch={epoch} avg_loss={avg_loss:.4f}; saved {out_path}")

    final_path = out_dir / "best_model.pt"
    torch.save({
        "epoch": args.epochs,
        "step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "config": config,
        "use_lora": bool(getattr(config, "use_lora", False)),
    }, final_path)
    print(f"[rft] saved final {final_path}")


if __name__ == "__main__":
    main()
