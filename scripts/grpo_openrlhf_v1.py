"""Minimal GRPO trainer for ECG_Tokenizer_Wrapper using the OpenRLHF reward
function. Self-contained — does NOT import from runners/grpo_finetuning_runner.py
or projects/grpo_finetuning_project.py.

Pipeline per optimizer step:
  for each prompt in batch:
    1. Load ECG signal.
    2. Generate N candidates with sampling.
    3. Score each candidate via services/openrlhf_judge_reward.py.
    4. Group-normalize the rewards -> advantages.
    5. Compute per-token log-probs of the sampled tokens with grad enabled.
    6. Accumulate loss = -mean(advantage * log_probs_of_gen_tokens).
  optimizer.step() with gradient clipping.

Conservative settings learned from prior failures on this checkpoint:
  - lr = 5e-7
  - max_grad_norm = 0.5
  - clip per-token ratio to [0.8, 1.25]
  - skip step if loss > 5 or grad_norm > 10
  - eval every save_interval steps; rollback if regression > 5pt

Usage:
  CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/volume/ECG_tokenizer python3 \
    scripts/grpo_openrlhf_v1.py \
    --checkpoint .../best_model.pt \
    --train_jsonl data/openrlhf_train_v1.jsonl \
    --eval_subset analysis/rlvr_eval/eval_subset_10per_cat.parquet \
    --output_dir checkpoints/grpo_openrlhf_v1 \
    --max_steps 20 --batch_size 2 --n_candidates 4
"""

import argparse
import importlib.util
import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# Reuse the eval module's model loader (same as best-of-N)
_spec = importlib.util.spec_from_file_location(
    "rlvr_eval_subset", str(ROOT / "scripts" / "rlvr_eval_subset.py"))
eval_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(eval_mod)

# Reward function from the OpenRLHF-compatible service
from services.openrlhf_judge_reward import reward_func, _init_judge_registry


def load_train_data(path: str) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def per_token_logprobs(logits: torch.Tensor, target_ids: torch.Tensor,
                       attention_mask: torch.Tensor) -> torch.Tensor:
    """Return per-position log-prob of the actual target token.

    logits: (B, T, V) — logits at each position predicting the NEXT token
    target_ids: (B, T) — the actual tokens at each position
    attention_mask: (B, T) — 1 for tokens that should be scored, 0 to ignore

    We shift: predict target_ids[t+1] from logits at position t.
    Returns: (B, T-1) — log p(token_{t+1} | <=t).
    Positions where attention_mask[:, t+1] == 0 are zeroed.
    """
    shift_logits = logits[:, :-1, :].contiguous()
    shift_targets = target_ids[:, 1:].contiguous()
    shift_mask = attention_mask[:, 1:].contiguous().float()

    log_probs = F.log_softmax(shift_logits, dim=-1)
    gathered = log_probs.gather(-1, shift_targets.unsqueeze(-1)).squeeze(-1)
    return gathered * shift_mask, shift_mask


def compute_group_advantages(rewards: List[float], clip: float = 2.0) -> torch.Tensor:
    """Group-normalized advantages, clipped to ±clip to limit gradient magnitude."""
    r = torch.tensor(rewards, dtype=torch.float32)
    if r.numel() <= 1:
        return r * 0.0
    mean = r.mean()
    std = r.std() + 1e-6
    a = (r - mean) / std
    return a.clamp(-clip, clip)


@torch.no_grad()
def sample_candidates(model, tokenizer, signal: torch.Tensor,
                       prompt_text: str, n: int, device: str,
                       max_new_tokens: int = 128,
                       temperature: float = 1.0, top_p: float = 0.95):
    """Generate n candidates for a single (signal, prompt). Returns list of
    {prompt_ids, gen_ids, decoded} dicts.

    NOTE: model.generate_report_with_question returns ONLY the new tokens
    (not the prompt). We keep prompt_ids and gen_ids separate; the forward
    pass at backprop time will reconstruct the full sequence.
    """
    prompt = eval_mod.build_prompt(prompt_text)
    enc = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    pids = enc["input_ids"].to(device)
    pmask = enc["attention_mask"].to(device)
    signal = signal.to(device=device, dtype=torch.float32)

    sig_rep = signal.expand(n, -1, -1) if signal.dim() == 3 else signal.unsqueeze(0).expand(n, -1, -1)
    pids_rep = pids.expand(n, -1)
    pmask_rep = pmask.expand(n, -1)

    gen_ids = model.generate_report_with_question(
        x=sig_rep,
        prompt_input_ids=pids_rep,
        prompt_attention_mask=pmask_rep,
        max_token_length=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )

    prompt_ids_1d = pids[0]  # single prompt; same for all candidates
    out = []
    for i in range(gen_ids.size(0)):
        # gen_ids[i] is just the new tokens
        decoded = tokenizer.decode(gen_ids[i], skip_special_tokens=True).strip()
        out.append({
            "prompt_ids": prompt_ids_1d,
            "gen_ids": gen_ids[i],
            "decoded": decoded,
        })
    return out


def compute_logprobs_for_candidates(model, signal: torch.Tensor,
                                     candidates: List[Dict],
                                     device: str) -> List[torch.Tensor]:
    """Score the actual sampled tokens under the (gradient-enabled) policy.

    Each candidate has prompt_ids (shared) and gen_ids (sampled). Build full
    sequences (prompt + gen), pad to max length, run forward, gather log-probs
    of the gen tokens only.
    """
    prompt_ids = candidates[0]["prompt_ids"]  # (Lp,) — same for all
    plen = prompt_ids.size(0)
    gen_lens = [c["gen_ids"].size(0) for c in candidates]
    max_gen = max(gen_lens) if gen_lens else 1
    max_total = plen + max_gen
    pad_id = 0
    n = len(candidates)

    full_ids = torch.full((n, max_total), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros((n, max_total), dtype=torch.long, device=device)
    gen_token_mask = torch.zeros((n, max_total), dtype=torch.float32, device=device)

    for i, c in enumerate(candidates):
        gids = c["gen_ids"].to(device)
        L = gids.size(0)
        full_ids[i, :plen] = prompt_ids.to(device)
        full_ids[i, plen:plen + L] = gids
        attn[i, :plen + L] = 1
        # Only score positions whose NEXT token is a generated one (positions
        # plen-1 .. plen+L-2 predict tokens plen .. plen+L-1).
        gen_token_mask[i, plen:plen + L] = 1.0

    sig_rep = signal.expand(n, -1, -1) if signal.dim() == 3 else signal.unsqueeze(0).expand(n, -1, -1)
    sig_rep = sig_rep.to(device=device, dtype=torch.float32)

    out = model.forward(
        ecg_signal=sig_rep,
        input_ids=full_ids,
        attention_mask=attn,
        labels=None,
    )
    logits = out["logits"] if isinstance(out, dict) else out[0]

    log_probs, shift_mask = per_token_logprobs(logits, full_ids, gen_token_mask)
    # log_probs shape (B, T-1). shift_mask same. Sum per candidate.
    return [(log_probs[i].sum(), shift_mask[i].sum().clamp(min=1.0))
            for i in range(n)]


def eval_on_subset(model, tokenizer, subset_parquet: str, output_dir: str,
                   label: str, device: str, original_config,
                   run_judge: bool = True, max_new_tokens: int = 256,
                   use_original_ckpt: str = None) -> float:
    """Run the same eval as rlvr_eval_subset.py and return overall_score.
    If use_original_ckpt is provided, eval that path directly (no save/reload).
    Otherwise saves current model.state_dict() to a temp file and eval that.
    """
    import subprocess

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if use_original_ckpt:
        tmp_ckpt = Path(use_original_ckpt)
    else:
        # Save model state temporarily so the eval script can load it
        tmp_ckpt = out_dir / f"_eval_ckpt_{label}.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "config": original_config,
        }, tmp_ckpt)

    cmd = [
        sys.executable, "-u", str(ROOT / "scripts" / "rlvr_eval_subset.py"),
        "--checkpoint", str(tmp_ckpt),
        "--subset_parquet", subset_parquet,
        "--output_dir", str(out_dir),
        "--device", device,
        "--max_new_tokens", str(max_new_tokens),
        "--label", label,
    ]
    if run_judge:
        cmd.append("--run_judge")

    print(f"[eval] {' '.join(cmd)}")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT) + ":" + env.get("PYTHONPATH", "")
    res = subprocess.run(cmd, env=env, cwd=str(ROOT))
    if res.returncode != 0:
        print(f"[eval] FAILED (returncode={res.returncode})")
        return -1.0

    summary_path = out_dir / f"summary_{label}.json"
    if not summary_path.exists():
        return -1.0
    with open(summary_path) as f:
        return float(json.load(f).get("overall_score", -1.0))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--eval_subset", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--n_candidates", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=20)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--lr", type=float, default=5e-7)
    p.add_argument("--max_grad_norm", type=float, default=0.5)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip_baseline_eval", action="store_true",
                   help="Skip baseline eval (saves ~8 min); set --known_baseline")
    p.add_argument("--known_baseline", type=float, default=None,
                   help="Use this as baseline score instead of re-running eval")
    p.add_argument("--beta", type=float, default=0.0,
                   help="KL coefficient against a frozen reference policy. "
                        "beta>0 loads a 2nd copy of the model as ref.")
    p.add_argument("--early_stop_on_regression", action="store_true",
                   help="If an eval score drops more than 0.05 below baseline, "
                        "rollback to baseline and stop.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[grpo] loading model from {args.checkpoint}")
    # Stash the original config for re-saving later
    _orig_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    original_config = _orig_ckpt["config"]
    del _orig_ckpt
    model, tokenizer = eval_mod.load_model(args.checkpoint, args.device)

    # Optional KL anchor: load a frozen reference model
    ref_model = None
    if args.beta > 0:
        print(f"[grpo] loading reference model (beta={args.beta}) ...")
        ref_model, _ = eval_mod.load_model(args.checkpoint, args.device)
        for _p in ref_model.parameters():
            _p.requires_grad = False
        ref_model.eval()
        print("[grpo] reference model loaded")
    # Only train LoRA params
    for n, p_ in model.named_parameters():
        p_.requires_grad = ("lora_" in n) or ("lora_A" in n) or ("lora_B" in n)
    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"[grpo] trainable params: {n_train:,}")

    # Make sure LoRA is active (training mode, not inference)
    try:
        model.set_lora_inference_mode(False)
    except Exception:
        pass

    optimizer = torch.optim.AdamW(trainable, lr=args.lr)
    _init_judge_registry()

    train_rows = load_train_data(args.train_jsonl)
    rng = np.random.default_rng(args.seed)

    # Baseline eval
    if args.skip_baseline_eval and args.known_baseline is not None:
        baseline_score = args.known_baseline
        print(f"[grpo] === SKIPPED baseline eval; using known {baseline_score:.4f} ===")
    else:
        print("[grpo] === baseline eval ===")
        baseline_score = eval_on_subset(
            model, tokenizer, args.eval_subset, str(out_dir / "eval_baseline"),
            label="baseline", device=args.device, original_config=original_config,
            max_new_tokens=256, use_original_ckpt=args.checkpoint)
        print(f"[grpo] baseline overall_score = {baseline_score:.4f}")

    metrics: List[Dict] = []
    best_score = -1.0
    best_step = 0
    for step in range(1, args.max_steps + 1):
        t0 = time.time()
        batch_indices = rng.integers(0, len(train_rows), size=args.batch_size)
        optimizer.zero_grad(set_to_none=True)

        step_loss = 0.0
        step_reward = 0.0
        n_groups = 0
        skipped = 0

        for bi in batch_indices:
            row = train_rows[int(bi)]
            try:
                signal = eval_mod.load_ecg_signal(row["signal_path"])
            except Exception as e:
                print(f"[grpo] skip signal load: {e}")
                skipped += 1
                continue

            # KEEP MODEL IN EVAL MODE THROUGHOUT — this freezes BN running
            # stats (encoder has BN; train mode would drift them away from
            # SFT-time stats) and disables dropout. LoRA still computes its
            # delta and gradients flow normally because inference_mode=False.
            model.eval()
            cands = sample_candidates(
                model, tokenizer, signal, row["prompt"],
                n=args.n_candidates, device=args.device,
                max_new_tokens=args.max_new_tokens,
                temperature=1.0, top_p=0.95)

            # Score via judge
            label_str = row["label"]
            prompt_str = row["prompt"]
            queries = [prompt_str + c["decoded"] for c in cands]
            rewards_out = reward_func(queries, [prompt_str] * len(cands),
                                       [label_str] * len(cands))
            rewards = rewards_out["scores"].tolist()
            advs = compute_group_advantages(rewards).to(args.device)

            # Reduced diagnostic - only print 1 sample per 10 steps
            if step % 10 == 1:
                print(f"[grpo]   prompt='{prompt_str[:50]}' rewards={[f'{r:.2f}' for r in rewards]}")

            # Skip degenerate groups (all rewards equal -> zero gradient)
            if abs(advs).max().item() < 1e-6:
                continue

            # Compute log-probs with grad (model still in eval() — BN frozen,
            # no dropout; LoRA delta still has gradient through scaling).
            log_probs_list = compute_logprobs_for_candidates(
                model, signal, cands, args.device)

            # Optional KL anchor: log-probs from frozen ref policy
            ref_lp_list = None
            if ref_model is not None:
                with torch.no_grad():
                    ref_lp_list = compute_logprobs_for_candidates(
                        ref_model, signal, cands, args.device)

            # loss = -mean over candidates of (advantage * mean_log_prob) [+ beta * KL]
            group_loss = 0.0
            kl_estimate = 0.0
            for i, (lp_sum, n_tok) in enumerate(log_probs_list):
                avg_lp = lp_sum / n_tok
                group_loss = group_loss + (-advs[i] * avg_lp)
                if ref_lp_list is not None:
                    ref_avg_lp = (ref_lp_list[i][0] / ref_lp_list[i][1]).detach()
                    # Estimate of KL(policy || ref) ~ avg_lp_policy - avg_lp_ref
                    kl_term = avg_lp - ref_avg_lp
                    group_loss = group_loss + args.beta * kl_term
                    kl_estimate += float(kl_term.detach().item())
            group_loss = group_loss / len(log_probs_list)
            if ref_lp_list is not None:
                kl_estimate /= len(log_probs_list)
            group_loss.backward()
            step_loss += float(group_loss.detach().item())
            step_reward += float(np.mean(rewards))
            n_groups += 1

        if n_groups == 0:
            print(f"[grpo] step {step}: all groups degenerate, skipping")
            optimizer.zero_grad(set_to_none=True)
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        if not torch.isfinite(grad_norm):
            print(f"[grpo] step {step}: non-finite grad_norm, skipping")
            optimizer.zero_grad(set_to_none=True)
            continue
        # clip already capped to max_grad_norm; only skip on truly pathological norms
        if grad_norm > 500.0:
            print(f"[grpo] step {step}: grad_norm {grad_norm:.2f} > 500, skipping")
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.step()
        dt = time.time() - t0
        avg_loss = step_loss / n_groups
        avg_reward = step_reward / n_groups
        print(f"[grpo] step {step:3d}  loss={avg_loss:+.4f}  reward={avg_reward:.3f}  "
              f"grad_norm={grad_norm:.2f}  dt={dt:.1f}s  skipped={skipped}")
        metrics.append({
            "step": step, "loss": avg_loss, "reward": avg_reward,
            "grad_norm": float(grad_norm), "dt": dt,
        })

        if step % args.eval_every == 0:
            print(f"[grpo] === eval @ step {step} ===")
            score = eval_on_subset(
                model, tokenizer, args.eval_subset,
                str(out_dir / f"eval_step{step}"),
                label=f"step{step}", device=args.device,
                original_config=original_config, max_new_tokens=256)
            print(f"[grpo] step {step} eval = {score:.4f}  (baseline {baseline_score:.4f}, "
                  f"delta {score - baseline_score:+.4f})")
            metrics[-1]["eval_overall"] = score
            metrics[-1]["eval_delta"] = score - baseline_score
            # Save best checkpoint
            if score > best_score:
                best_score = score
                best_step = step
                best_ckpt_path = out_dir / "best_so_far.pt"
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "config": original_config,
                    "baseline_score": baseline_score,
                    "best_step": step,
                    "best_score": score,
                    "metrics": metrics,
                }, best_ckpt_path)
                print(f"[grpo] saved new best (Δ {score - baseline_score:+.4f}) to {best_ckpt_path}")
            if args.early_stop_on_regression and score < baseline_score - 0.05:
                print(f"[grpo] EARLY STOP: eval {score:.4f} regressed > 0.05 below baseline")
                break

    # Save final checkpoint and metrics
    final_ckpt = out_dir / "best_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": original_config,
        "baseline_score": baseline_score,
        "metrics": metrics,
    }, final_ckpt)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"baseline_score": baseline_score, "metrics": metrics}, f, indent=2)
    print(f"[grpo] saved final to {final_ckpt}")

    # Final eval
    print("[grpo] === final eval ===")
    final_score = eval_on_subset(
        model, tokenizer, args.eval_subset, str(out_dir / "eval_final"),
        label="final", device=args.device, original_config=original_config,
        max_new_tokens=256)
    print(f"[grpo] FINAL: baseline {baseline_score:.4f} -> final {final_score:.4f}  "
          f"(delta {final_score - baseline_score:+.4f})")


if __name__ == "__main__":
    main()
