#!/usr/bin/env python3
"""Offline GRPO over pre-scored candidate groups for ECG QA."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_dpo_policy import (  # noqa: E402
    _forward_logits_and_labels,
    _sequence_logp,
    build_tokens,
    load_ecg_waveform,
    load_model,
    set_trainable,
)


def advantages(scores: List[float], clip: float) -> torch.Tensor:
    vals = torch.tensor(scores, dtype=torch.float32)
    if vals.numel() < 2:
        return vals * 0.0
    std = vals.std()
    if float(std) < 1e-6:
        return vals * 0.0
    return ((vals - vals.mean()) / (std + 1e-6)).clamp(-clip, clip)


def load_groups(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def train_group(model, tokenizer, config, group: Dict[str, Any], device, max_length: int, adv_clip: float,
                precision: str) -> tuple[torch.Tensor, Dict[str, float]]:
    candidates = group["candidates"]
    scores = [float(c["score"]) for c in candidates]
    adv = advantages(scores, adv_clip).to(device)
    if float(adv.abs().max().item()) < 1e-6:
        return None, {"skipped": 1.0}

    waveform = load_ecg_waveform(
        waveform_path=group["waveform_path"],
        target_length=int(getattr(config, "ecg_waveform_length", 2500)),
        num_leads=int(getattr(config, "ecg_num_leads", 12)),
    )
    signal_one = torch.from_numpy(waveform).T
    signal = signal_one.unsqueeze(0).expand(len(candidates), -1, -1).to(device)

    toks = [
        build_tokens(
            prompt_text=str(group["prompt"]),
            answer_text=str(c["text"]),
            tokenizer=tokenizer,
            max_length=max_length,
            num_ecg_tokens=int(getattr(config, "num_ecg_tokens", 0)),
            ecg_token_start_id=getattr(config, "ecg_token_start_id", None),
            prefix_tuning=bool(getattr(config, "prefix_tuning", False)),
            medgemma_prompt_style=bool(getattr(config, "medgemma_prompt_style", False)),
        )
        for c in candidates
    ]
    input_ids = torch.stack([t.input_ids for t in toks]).to(device)
    attention_mask = torch.stack([t.attention_mask for t in toks]).to(device)
    labels = torch.stack([t.labels for t in toks]).to(device)
    prompt_input_ids = torch.stack([t.prompt_input_ids for t in toks]).to(device)
    prompt_attention_mask = torch.stack([t.prompt_attention_mask for t in toks]).to(device)

    use_autocast = precision in {"bf16", "fp16"}
    autocast_dtype = torch.bfloat16 if precision == "bf16" else torch.float16
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
        n_tok = (labels_expanded != -100).sum(dim=-1).clamp_min(1).float()
        avg_logp = logp / n_tok
        loss = -(adv * avg_logp).mean() * float(group.get("reward_span", 1.0))

    return loss, {
        "skipped": 0.0,
        "mean_score": float(np.mean(scores)),
        "span": float(max(scores) - min(scores)),
        "loss": float(loss.detach().cpu()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--groups", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--adv_clip", type=float, default=2.0)
    parser.add_argument("--max_length", type=int, default=None)
    parser.add_argument("--trainable", choices=["lora", "bridge", "projection", "all"], default="lora")
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    rng = np.random.default_rng(args.seed)

    model, tokenizer, config = load_model(args.checkpoint, device)
    set_trainable(model, args.trainable)
    if args.trainable == "lora":
        model.eval()
    else:
        model.train()

    trainable = [p for p in model.parameters() if p.requires_grad]
    print(f"[offline-grpo] trainable params: {sum(p.numel() for p in trainable):,}")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    max_length = args.max_length or int(getattr(config, "max_token_length", 640))
    groups = load_groups(args.groups)
    if not groups:
        raise SystemExit("No offline GRPO groups")

    global_step = 0
    metrics = []
    for epoch in range(args.epochs):
        order = rng.permutation(len(groups))
        running = []
        skipped = 0
        for j, idx in enumerate(order, start=1):
            group = groups[int(idx)]
            optimizer.zero_grad(set_to_none=True)
            loss, info = train_group(
                model, tokenizer, config, group, device, max_length, args.adv_clip, args.precision
            )
            if loss is None:
                skipped += 1
                continue
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
            if not torch.isfinite(grad_norm):
                optimizer.zero_grad(set_to_none=True)
                skipped += 1
                continue
            optimizer.step()
            global_step += 1
            info["grad_norm"] = float(grad_norm.detach().cpu())
            running.append(info)
            if j % 25 == 0:
                mean_loss = float(np.mean([x["loss"] for x in running])) if running else 0.0
                mean_span = float(np.mean([x["span"] for x in running])) if running else 0.0
                print(
                    f"[offline-grpo] epoch={epoch} group={j}/{len(groups)} "
                    f"loss={mean_loss:+.4f} span={mean_span:.3f} skipped={skipped}"
                )

        summary = {
            "epoch": epoch,
            "step": global_step,
            "updates": len(running),
            "skipped": skipped,
            "mean_loss": float(np.mean([x["loss"] for x in running])) if running else 0.0,
            "mean_span": float(np.mean([x["span"] for x in running])) if running else 0.0,
        }
        metrics.append(summary)
        ckpt = out_dir / f"offline_grpo_epoch_{epoch}.pt"
        torch.save({
            "epoch": epoch,
            "step": global_step,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
            "use_lora": bool(getattr(config, "use_lora", False)),
        }, ckpt)
        print(f"[offline-grpo] saved {ckpt}; {summary}")

    final = out_dir / "best_model.pt"
    torch.save({
        "epoch": args.epochs,
        "step": global_step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "config": config,
        "use_lora": bool(getattr(config, "use_lora", False)),
    }, final)
    print(f"[offline-grpo] saved final {final}")


if __name__ == "__main__":
    main()
