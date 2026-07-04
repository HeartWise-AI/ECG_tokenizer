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
# Verifiable per-category rewards (deterministic, no LLM judge required)
from services.verifiable_reward import verify as verifiable_verify


def _try_init_wandb(args):
    if not args.wandb_project:
        return None
    try:
        import wandb

        tags = [t.strip() for t in args.wandb_tags.split(",") if t.strip()]
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity or None,
            name=args.wandb_run_name or None,
            group=args.wandb_group or None,
            tags=tags or None,
            mode=args.wandb_mode or None,
            config=vars(args),
        )
        print(f"[wandb] run initialized: {getattr(run, 'url', None)}")
        return run
    except Exception as e:
        msg = f"[wandb] init failed: {e}"
        if args.wandb_required:
            raise RuntimeError(msg) from e
        print(msg)
        return None


def _wandb_log(run, payload: Dict, step: int) -> None:
    if run is None:
        return
    run.log(payload, step=step)


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


def parse_category_filter(raw: str) -> set:
    return {x.strip() for x in (raw or "").split(",") if x.strip()}


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


def build_text_candidate(tokenizer, prompt_text: str, answer_text: str,
                         device: str, max_new_tokens: int) -> Dict:
    """Build a teacher-forced candidate from a known answer string."""
    prompt = eval_mod.build_prompt(prompt_text)
    enc = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    answer = str(answer_text or "").strip()
    ans = tokenizer(answer, add_special_tokens=False, return_tensors="pt")
    gen_ids = ans["input_ids"][0]
    if max_new_tokens > 0:
        gen_ids = gen_ids[:max_new_tokens]
    if gen_ids.numel() == 0:
        return {}
    return {
        "prompt_ids": enc["input_ids"][0].to(device),
        "gen_ids": gen_ids.to(device),
        "decoded": answer,
    }


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

    # Condition the ECG bridge on the prompt exactly as rollout generation did
    # (InstructionAwareECGQFormerBridge uses prompt_input_ids); omitting it would
    # score candidates under different ECG embeddings than were sampled.
    prompt_input_ids = prompt_ids.to(device).unsqueeze(0).expand(n, -1)
    prompt_attention_mask = torch.ones((n, plen), dtype=torch.long, device=device)

    out = model.forward(
        ecg_signal=sig_rep,
        input_ids=full_ids,
        attention_mask=attn,
        labels=None,
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
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

    created_tmp_ckpt = False
    if use_original_ckpt:
        tmp_ckpt = Path(use_original_ckpt)
    else:
        # Save model state temporarily so the eval script can load it
        tmp_ckpt = out_dir / f"_eval_ckpt_{label}.pt"
        try:
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": original_config,
            }, tmp_ckpt)
            created_tmp_ckpt = True
        except Exception:
            if tmp_ckpt.exists():
                try:
                    tmp_ckpt.unlink()
                except OSError:
                    pass
            raise

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

    try:
        print(f"[eval] {' '.join(cmd)}")
        env = os.environ.copy()
        env["PYTHONPATH"] = str(ROOT) + ":" + env.get("PYTHONPATH", "")
        env["ECG_BERT_DEVICE"] = device
        res = subprocess.run(cmd, env=env, cwd=str(ROOT))
        if res.returncode != 0:
            print(f"[eval] FAILED (returncode={res.returncode})")
            return -1.0

        summary_path = out_dir / f"summary_{label}.json"
        if not summary_path.exists():
            return -1.0
        with open(summary_path) as f:
            return float(json.load(f).get("overall_score", -1.0))
    finally:
        if created_tmp_ckpt and tmp_ckpt.exists():
            try:
                tmp_ckpt.unlink()
                print(f"[eval] removed temporary checkpoint {tmp_ckpt}")
            except OSError as e:
                print(f"[eval] warning: could not remove temporary checkpoint {tmp_ckpt}: {e}")


def load_eval_summary(output_dir: str, label: str) -> Dict:
    summary_path = Path(output_dir) / f"summary_{label}.json"
    if not summary_path.exists():
        return {}
    with open(summary_path) as f:
        return json.load(f)


def flatten_eval_for_wandb(summary: Dict, prefix: str) -> Dict:
    if not summary:
        return {}
    out = {}
    if "overall_score" in summary:
        out[f"{prefix}/overall_score"] = summary["overall_score"]
    for cat, stats in summary.get("category_aggregates", {}).items():
        mean = stats.get("mean_score")
        count = stats.get("count")
        if mean is not None:
            out[f"{prefix}/category/{cat}/mean_score"] = mean
        if count is not None:
            out[f"{prefix}/category/{cat}/count"] = count
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--train_jsonl", required=True)
    p.add_argument("--eval_subset", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--eval_device", default=None,
                   help="Device for validation generation/judge eval subprocesses. "
                        "Defaults to --device; set e.g. cuda:2 to keep eval off "
                        "the training GPU.")
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--n_candidates", type=int, default=4)
    p.add_argument("--max_steps", type=int, default=20)
    p.add_argument("--max_new_tokens", type=int, default=128)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=0.95)
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
    p.add_argument("--reward_kind", choices=["judge", "verifiable"],
                   default="verifiable",
                   help="judge=LLM-as-a-judge (slow, costs API); "
                        "verifiable=per-category deterministic verifier (fast, free)")
    p.add_argument("--early_stop_on_regression", action="store_true",
                   help="If an eval score drops more than 0.05 below baseline, "
                        "rollback to baseline and stop.")
    p.add_argument("--target_delta", type=float, default=None,
                   help="Stop after an eval reaches baseline + target_delta.")
    p.add_argument("--target_score", type=float, default=None,
                   help="Optional absolute eval score required for target. "
                        "When combined with --target_delta, the stop threshold "
                        "is max(baseline + target_delta, target_score).")
    p.add_argument("--full_finetune", action="store_true",
                   help="Unfreeze the whole MedGemma decoder (full FT) instead "
                        "of LoRA-only. Encoder/quantizer/bridge stay frozen.")
    p.add_argument("--train_lora_modules", default="",
                   help="Optional comma-separated substring filter for LoRA-only "
                        "training, e.g. q_proj,v_proj. Empty trains all LoRA "
                        "parameters.")
    p.add_argument("--train_lora_parts", default="",
                   help="Optional comma-separated LoRA parameter part filter. "
                        "Valid values are A and B, matching lora_A/lora_B. "
                        "Empty trains both parts.")
    p.add_argument("--sft_best_weight", type=float, default=0.0,
                   help="Optional auxiliary NLL weight on the highest-reward "
                        "sample in each group. This is judge-selected "
                        "self-imitation to reduce GRPO variance and move greedy "
                        "decoding toward sampled winners.")
    p.add_argument("--sft_min_best_reward", type=float, default=0.8,
                   help="Only apply --sft_best_weight if the best sampled "
                        "reward is at least this value.")
    p.add_argument("--sft_min_reward_gap", type=float, default=0.1,
                   help="Only apply --sft_best_weight if best reward exceeds "
                        "the group mean by at least this amount.")
    p.add_argument("--sft_on_degenerate_high", action="store_true",
                   help="If all sampled rewards are equal but high, still "
                        "apply the auxiliary NLL to reinforce a judge-approved "
                        "sample. This helps when sampling finds good answers "
                        "but GRPO has zero advantage signal.")
    p.add_argument("--sft_gt_weight", type=float, default=0.0,
                   help="Optional teacher-forced NLL weight on the train-row "
                        "ground_truth answer. This is a single-model weight "
                        "update used to anchor greedy behavior; it is not "
                        "best-of-N inference or checkpoint blending.")
    p.add_argument("--sft_gt_categories", default="",
                   help="Comma-separated category allow-list for "
                        "--sft_gt_weight. Empty means all categories.")
    p.add_argument("--sft_ref_weight", type=float, default=0.0,
                   help="Optional preservation loss against the frozen start "
                        "checkpoint on teacher-forced ground_truth text. This "
                        "penalizes squared average-logprob drift and is useful "
                        "with pseudo-label replay rows. It changes training "
                        "only; final inference is still greedy single-output.")
    p.add_argument("--sft_ref_categories", default="",
                   help="Comma-separated category allow-list for "
                        "--sft_ref_weight. Empty means all categories.")
    p.add_argument("--dpo_gt_weight", type=float, default=0.0,
                   help="Optional online DPO-style loss preferring the "
                        "train-row ground_truth over the model's sampled "
                        "answer, anchored to the frozen start checkpoint.")
    p.add_argument("--dpo_gt_categories", default="",
                   help="Comma-separated category allow-list for "
                        "--dpo_gt_weight. Empty means all categories.")
    p.add_argument("--dpo_beta", type=float, default=0.1,
                   help="Inverse-temperature for --dpo_gt_weight.")
    p.add_argument("--dpo_skip_if_rejected_reward_ge", type=float, default=0.999,
                   help="Skip DPO when the sampled/rejected answer already "
                        "scores at or above this reward.")
    p.add_argument("--skip_final_eval", action="store_true",
                   help="Skip the duplicate final eval after saving best_model.pt. "
                        "Useful when max_steps already landed on an eval boundary.")
    p.add_argument("--wandb_project", default=None,
                   help="Enable Weights & Biases logging under this project.")
    p.add_argument("--wandb_entity", default=None)
    p.add_argument("--wandb_run_name", default=None)
    p.add_argument("--wandb_group", default=None)
    p.add_argument("--wandb_tags", default="")
    p.add_argument("--wandb_mode", default=None,
                   help="Optional W&B mode, e.g. online, offline, disabled.")
    p.add_argument("--wandb_required", action="store_true",
                   help="Fail fast if W&B cannot be initialized.")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    eval_device = args.eval_device or args.device
    sft_gt_categories = parse_category_filter(args.sft_gt_categories)
    sft_ref_categories = parse_category_filter(args.sft_ref_categories)
    dpo_gt_categories = parse_category_filter(args.dpo_gt_categories)
    wandb_run = _try_init_wandb(args)

    print(f"[grpo] loading model from {args.checkpoint}")
    # Stash the original config for re-saving later
    _orig_ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    original_config = _orig_ckpt["config"]
    del _orig_ckpt
    model, tokenizer = eval_mod.load_model(args.checkpoint, args.device)

    # Optional KL anchor: load a frozen reference model
    ref_model = None
    if args.beta > 0 or args.sft_ref_weight > 0 or args.dpo_gt_weight > 0:
        print(f"[grpo] loading reference model "
              f"(beta={args.beta}, sft_ref_weight={args.sft_ref_weight}, "
              f"dpo_gt_weight={args.dpo_gt_weight}) ...")
        ref_model, _ = eval_mod.load_model(args.checkpoint, args.device)
        for _p in ref_model.parameters():
            _p.requires_grad = False
        ref_model.eval()
        print("[grpo] reference model loaded")
    if args.full_finetune:
        # Full FT: unfreeze the entire MedGemma decoder (base weights + LoRA).
        # Keep ECG encoder + quantizer + bridge frozen so we don't break the
        # signal-injection path (those were trained in earlier SFT stages).
        for n, p_ in model.named_parameters():
            p_.requires_grad = n.startswith("decoder.llm_model")
        # Enable gradient checkpointing on the LLM to fit 4B FT in memory
        try:
            llm = model.decoder.llm_model
            if hasattr(llm, "gradient_checkpointing_enable"):
                llm.gradient_checkpointing_enable()
                print("[grpo] gradient checkpointing enabled on LLM")
            if hasattr(llm, "config"):
                llm.config.use_cache = False
        except Exception as e:
            print(f"[grpo] gradient checkpointing setup warning: {e}")
        mode_str = "FULL FINETUNE (decoder.llm_model unfrozen)"
    else:
        # LoRA-only
        lora_module_filter = parse_category_filter(args.train_lora_modules)
        lora_part_filter = {part.upper() for part in parse_category_filter(args.train_lora_parts)}
        valid_lora_parts = {"A", "B"}
        invalid_lora_parts = sorted(lora_part_filter - valid_lora_parts)
        if invalid_lora_parts:
            raise ValueError(
                f"Invalid --train_lora_parts values: {invalid_lora_parts}. "
                "Use A,B, A, B, or leave empty."
            )
        for n, p_ in model.named_parameters():
            is_lora = ("lora_" in n) or ("lora_A" in n) or ("lora_B" in n)
            allowed = (
                not lora_module_filter
                or any(module_name in n for module_name in lora_module_filter)
            )
            if lora_part_filter:
                allowed = allowed and (
                    ("A" in lora_part_filter and "lora_A" in n)
                    or ("B" in lora_part_filter and "lora_B" in n)
                )
            p_.requires_grad = is_lora and allowed
        mode_str = "LoRA-only"
        if lora_module_filter:
            mode_str += f" ({','.join(sorted(lora_module_filter))})"
        if lora_part_filter:
            mode_str += f" parts={','.join(sorted(lora_part_filter))}"

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    print(f"[grpo] mode: {mode_str}; trainable params: {n_train:,}")

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
            label="baseline", device=eval_device, original_config=original_config,
            max_new_tokens=256, use_original_ckpt=args.checkpoint)
        print(f"[grpo] baseline overall_score = {baseline_score:.4f}")
    _wandb_log(wandb_run, {"eval/baseline_overall_score": baseline_score}, step=0)

    target_threshold = None
    target_parts = []
    if args.target_delta is not None:
        target_parts.append(baseline_score + args.target_delta)
    if args.target_score is not None:
        target_parts.append(args.target_score)
    if target_parts:
        target_threshold = max(target_parts)
        print(f"[grpo] target threshold = {target_threshold:.4f} "
              f"(baseline={baseline_score:.4f}, target_delta={args.target_delta}, "
              f"target_score={args.target_score})")

    metrics: List[Dict] = []
    best_score = baseline_score
    best_step = 0

    def run_eval_if_due(step: int) -> bool:
        nonlocal best_score, best_step
        if step % args.eval_every != 0:
            return False

        print(f"[grpo] === eval @ step {step} ===")
        score = eval_on_subset(
            model, tokenizer, args.eval_subset,
            str(out_dir / f"eval_step{step}"),
            label=f"step{step}", device=eval_device,
            original_config=original_config, max_new_tokens=256)
        eval_summary = load_eval_summary(
            str(out_dir / f"eval_step{step}"), f"step{step}")
        print(f"[grpo] step {step} eval = {score:.4f}  (baseline {baseline_score:.4f}, "
              f"delta {score - baseline_score:+.4f})")
        metrics[-1]["eval_overall"] = score
        metrics[-1]["eval_delta"] = score - baseline_score
        _wandb_log(wandb_run, {
            "eval/overall_score": score,
            "eval/delta_vs_baseline": score - baseline_score,
            **flatten_eval_for_wandb(eval_summary, "eval"),
        }, step=step)

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
            return True
        metrics[-1]["target_threshold"] = target_threshold
        metrics[-1]["target_reached"] = (
            target_threshold is not None and score >= target_threshold
        )
        if target_threshold is not None and score >= target_threshold:
            print(f"[grpo] TARGET REACHED: eval {score:.4f} >= "
                  f"{target_threshold:.4f}")
            return True
        return False

    for step in range(1, args.max_steps + 1):
        t0 = time.time()
        batch_indices = rng.integers(0, len(train_rows), size=args.batch_size)
        optimizer.zero_grad(set_to_none=True)

        step_loss = 0.0
        step_reward = 0.0
        sampled_groups = 0
        degenerate_groups = 0
        sft_groups = 0
        sft_gt_groups = 0
        sft_ref_groups = 0
        dpo_gt_groups = 0
        sft_gt_loss_sum = 0.0
        sft_ref_loss_sum = 0.0
        dpo_gt_loss_sum = 0.0
        reward_sum = 0.0
        reward_sq_sum = 0.0
        reward_count = 0
        reward_positive = 0
        reward_perfect = 0
        best_reward_sum = 0.0
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
                temperature=args.temperature, top_p=args.top_p)

            # Score candidates — verifiable (default, deterministic) or judge
            label_str = row["label"]
            prompt_str = row["prompt"]
            label_data = json.loads(label_str)
            gt_text = label_data.get("ground_truth", "")
            cat = label_data.get("category", "classification")
            if args.reward_kind == "verifiable":
                rewards = [verifiable_verify(c["decoded"], label_data, cat) for c in cands]
            else:
                queries = [prompt_str + c["decoded"] for c in cands]
                rewards_out = reward_func(queries, [prompt_str] * len(cands),
                                           [label_str] * len(cands))
                rewards = rewards_out["scores"].tolist()
            advs = compute_group_advantages(rewards).to(args.device)
            best_idx = int(np.argmax(rewards)) if rewards else -1
            best_reward = float(rewards[best_idx]) if best_idx >= 0 else 0.0
            mean_reward = float(np.mean(rewards)) if rewards else 0.0
            degenerate_group = abs(advs).max().item() < 1e-6
            if rewards:
                reward_arr = np.asarray(rewards, dtype=np.float64)
                sampled_groups += 1
                reward_sum += float(reward_arr.sum())
                reward_sq_sum += float((reward_arr * reward_arr).sum())
                reward_count += int(reward_arr.size)
                reward_positive += int((reward_arr > 0.0).sum())
                reward_perfect += int((reward_arr >= 0.999).sum())
                best_reward_sum += best_reward
                degenerate_groups += int(degenerate_group)
            sft_eligible = (
                args.sft_best_weight > 0
                and best_idx >= 0
                and best_reward >= args.sft_min_best_reward
                and (
                    best_reward >= mean_reward + args.sft_min_reward_gap
                    or (args.sft_on_degenerate_high and degenerate_group)
                )
            )
            sft_groups += int(sft_eligible)
            gt_eligible = (
                args.sft_gt_weight > 0
                and bool(gt_text)
                and (not sft_gt_categories or cat in sft_gt_categories)
            )
            ref_gt_eligible = (
                args.sft_ref_weight > 0
                and bool(gt_text)
                and (not sft_ref_categories or cat in sft_ref_categories)
            )
            rejected_idx = int(np.argmin(rewards)) if rewards else -1
            rejected_reward = float(rewards[rejected_idx]) if rejected_idx >= 0 else 0.0
            dpo_gt_eligible = (
                args.dpo_gt_weight > 0
                and bool(gt_text)
                and rejected_idx >= 0
                and rejected_reward < args.dpo_skip_if_rejected_reward_ge
                and (not dpo_gt_categories or cat in dpo_gt_categories)
            )
            gt_candidate = {}
            if gt_eligible or ref_gt_eligible or dpo_gt_eligible:
                gt_candidate = build_text_candidate(
                    tokenizer, prompt_str, gt_text, args.device,
                    args.max_new_tokens)
                gt_eligible = gt_eligible and bool(gt_candidate)
                ref_gt_eligible = ref_gt_eligible and bool(gt_candidate)
                dpo_gt_eligible = dpo_gt_eligible and bool(gt_candidate)
            sft_gt_groups += int(gt_eligible)
            sft_ref_groups += int(ref_gt_eligible)
            dpo_gt_groups += int(dpo_gt_eligible)

            # Reduced diagnostic - only print 1 sample per 10 steps
            if step % 10 == 1:
                print(f"[grpo]   prompt='{prompt_str[:50]}' rewards={[f'{r:.2f}' for r in rewards]}")

            # Skip degenerate groups (all rewards equal -> zero gradient)
            need_candidate_logprobs = (not degenerate_group) or sft_eligible
            need_candidate_logprobs = need_candidate_logprobs or dpo_gt_eligible
            if (
                not need_candidate_logprobs
                and not gt_eligible
                and not ref_gt_eligible
                and not dpo_gt_eligible
            ):
                continue

            # Compute log-probs with grad (model still in eval() — BN frozen,
            # no dropout; LoRA delta still has gradient through scaling).
            log_probs_list = []
            if need_candidate_logprobs:
                log_probs_list = compute_logprobs_for_candidates(
                    model, signal, cands, args.device)

            # Optional KL anchor: log-probs from frozen ref policy
            ref_lp_list = None
            if ref_model is not None and need_candidate_logprobs:
                with torch.no_grad():
                    ref_lp_list = compute_logprobs_for_candidates(
                        ref_model, signal, cands, args.device)

            # loss = -mean over candidates of (advantage * mean_log_prob) [+ beta * KL]
            group_loss = None
            kl_estimate = 0.0
            if need_candidate_logprobs:
                candidate_loss = 0.0
                for i, (lp_sum, n_tok) in enumerate(log_probs_list):
                    avg_lp = lp_sum / n_tok
                    candidate_loss = candidate_loss + (-advs[i] * avg_lp)
                    if ref_lp_list is not None:
                        ref_avg_lp = (ref_lp_list[i][0] / ref_lp_list[i][1]).detach()
                        # Estimate of KL(policy || ref) ~ avg_lp_policy - avg_lp_ref
                        kl_term = avg_lp - ref_avg_lp
                        candidate_loss = candidate_loss + args.beta * kl_term
                        kl_estimate += float(kl_term.detach().item())
                group_loss = candidate_loss / len(log_probs_list)
            if sft_eligible:
                best_lp_sum, best_n_tok = log_probs_list[best_idx]
                best_loss = args.sft_best_weight * (
                    -best_lp_sum / best_n_tok
                )
                group_loss = best_loss if group_loss is None else group_loss + best_loss
            gt_lp_sum = None
            gt_n_tok = None
            if gt_eligible or ref_gt_eligible:
                gt_lp_sum, gt_n_tok = compute_logprobs_for_candidates(
                    model, signal, [gt_candidate], args.device)[0]
            if gt_eligible:
                gt_loss = args.sft_gt_weight * (-gt_lp_sum / gt_n_tok)
                group_loss = gt_loss if group_loss is None else group_loss + gt_loss
                sft_gt_loss_sum += float(gt_loss.detach().item())
            if ref_gt_eligible:
                if ref_model is None:
                    raise RuntimeError("--sft_ref_weight requires a reference model")
                with torch.no_grad():
                    ref_gt_lp_sum, ref_gt_n_tok = compute_logprobs_for_candidates(
                        ref_model, signal, [gt_candidate], args.device)[0]
                    ref_gt_avg_lp = (ref_gt_lp_sum / ref_gt_n_tok).detach()
                gt_avg_lp = gt_lp_sum / gt_n_tok
                ref_loss = args.sft_ref_weight * (
                    gt_avg_lp - ref_gt_avg_lp
                ).pow(2)
                group_loss = ref_loss if group_loss is None else group_loss + ref_loss
                sft_ref_loss_sum += float(ref_loss.detach().item())
            if dpo_gt_eligible:
                if ref_model is None:
                    raise RuntimeError("--dpo_gt_weight requires a reference model")
                if gt_lp_sum is None or gt_n_tok is None:
                    gt_lp_sum, gt_n_tok = compute_logprobs_for_candidates(
                        model, signal, [gt_candidate], args.device)[0]
                rejected_lp_sum, rejected_n_tok = log_probs_list[rejected_idx]
                with torch.no_grad():
                    ref_gt_lp_sum, ref_gt_n_tok = compute_logprobs_for_candidates(
                        ref_model, signal, [gt_candidate], args.device)[0]
                    if ref_lp_list is not None:
                        ref_rejected_lp_sum, ref_rejected_n_tok = ref_lp_list[rejected_idx]
                    else:
                        ref_rejected_lp_sum, ref_rejected_n_tok = compute_logprobs_for_candidates(
                            ref_model, signal, [cands[rejected_idx]], args.device)[0]
                    ref_diff = (
                        (ref_gt_lp_sum / ref_gt_n_tok)
                        - (ref_rejected_lp_sum / ref_rejected_n_tok)
                    ).detach()
                pi_diff = (
                    (gt_lp_sum / gt_n_tok)
                    - (rejected_lp_sum / rejected_n_tok)
                )
                dpo_loss = args.dpo_gt_weight * (
                    -F.logsigmoid(args.dpo_beta * (pi_diff - ref_diff))
                )
                group_loss = dpo_loss if group_loss is None else group_loss + dpo_loss
                dpo_gt_loss_sum += float(dpo_loss.detach().item())
            if ref_lp_list is not None:
                kl_estimate /= len(log_probs_list)
            if group_loss is None:
                continue
            group_loss.backward()
            step_loss += float(group_loss.detach().item())
            step_reward += float(np.mean(rewards))
            n_groups += 1

        if n_groups == 0:
            print(f"[grpo] step {step}: all groups degenerate, skipping")
            optimizer.zero_grad(set_to_none=True)
            candidate_reward_mean = reward_sum / max(reward_count, 1)
            candidate_reward_var = (
                reward_sq_sum / max(reward_count, 1) - candidate_reward_mean ** 2
            )
            step_metrics = {
                "step": step,
                "loss": None,
                "reward": None,
                "grad_norm": None,
                "dt": time.time() - t0,
                "sampled_groups": sampled_groups,
                "optimized_groups": 0,
                "optimized_group_rate": 0.0,
                "degenerate_group_rate": degenerate_groups / max(sampled_groups, 1),
                "sft_group_rate": sft_groups / max(sampled_groups, 1),
                "sft_gt_group_rate": sft_gt_groups / max(sampled_groups, 1),
                "sft_ref_group_rate": sft_ref_groups / max(sampled_groups, 1),
                "dpo_gt_group_rate": dpo_gt_groups / max(sampled_groups, 1),
                "sft_gt_loss_mean": sft_gt_loss_sum / max(sft_gt_groups, 1),
                "sft_ref_loss_mean": sft_ref_loss_sum / max(sft_ref_groups, 1),
                "dpo_gt_loss_mean": dpo_gt_loss_sum / max(dpo_gt_groups, 1),
                "candidate_reward_mean": candidate_reward_mean,
                "candidate_reward_std": float(max(candidate_reward_var, 0.0) ** 0.5),
                "candidate_reward_positive_rate": reward_positive / max(reward_count, 1),
                "candidate_reward_perfect_rate": reward_perfect / max(reward_count, 1),
                "best_reward_mean": best_reward_sum / max(sampled_groups, 1),
                "skipped_signal_count": skipped,
                "skipped_update_reason": "all_groups_degenerate",
            }
            metrics.append(step_metrics)
            _wandb_log(wandb_run, {
                "train/update_skipped": 1,
                "train/optimized_groups": 0,
                "train/optimized_group_rate": 0.0,
                "reward/candidate_mean": candidate_reward_mean,
                "reward/candidate_std": step_metrics["candidate_reward_std"],
                "reward/candidate_positive_rate": reward_positive / max(reward_count, 1),
                "reward/candidate_perfect_rate": reward_perfect / max(reward_count, 1),
                "reward/best_reward_mean": best_reward_sum / max(sampled_groups, 1),
                "reward/degenerate_group_rate": degenerate_groups / max(sampled_groups, 1),
                "reward/sft_group_rate": sft_groups / max(sampled_groups, 1),
                "reward/sft_gt_group_rate": sft_gt_groups / max(sampled_groups, 1),
                "reward/sft_ref_group_rate": sft_ref_groups / max(sampled_groups, 1),
                "reward/dpo_gt_group_rate": dpo_gt_groups / max(sampled_groups, 1),
                "loss/sft_gt_mean": sft_gt_loss_sum / max(sft_gt_groups, 1),
                "loss/sft_ref_mean": sft_ref_loss_sum / max(sft_ref_groups, 1),
                "loss/dpo_gt_mean": dpo_gt_loss_sum / max(dpo_gt_groups, 1),
                "data/skipped_signal_count": skipped,
            }, step=step)
            if run_eval_if_due(step):
                break
            continue

        grad_norm = torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
        if not torch.isfinite(grad_norm):
            print(f"[grpo] step {step}: non-finite grad_norm, skipping")
            optimizer.zero_grad(set_to_none=True)
            continue
        # clip_grad_norm_ already scaled the gradient down to max_grad_norm
        # before this point, so the *applied* update is always safe. The raw
        # norm is naturally much larger for full-FT (4.3B params) than LoRA
        # (24M). Only skip on truly pathological / non-finite norms.
        skip_thresh = 50000.0 if args.full_finetune else 500.0
        if grad_norm > skip_thresh:
            print(f"[grpo] step {step}: grad_norm {grad_norm:.2f} > {skip_thresh:.0f}, skipping")
            optimizer.zero_grad(set_to_none=True)
            continue

        optimizer.step()
        dt = time.time() - t0
        avg_loss = step_loss / n_groups
        avg_reward = step_reward / n_groups
        candidate_reward_mean = reward_sum / max(reward_count, 1)
        candidate_reward_var = (
            reward_sq_sum / max(reward_count, 1) - candidate_reward_mean ** 2
        )
        candidate_reward_std = float(max(candidate_reward_var, 0.0) ** 0.5)
        optimized_group_rate = n_groups / max(sampled_groups, 1)
        print(f"[grpo] step {step:3d}  loss={avg_loss:+.4f}  reward={avg_reward:.3f}  "
              f"grad_norm={grad_norm:.2f}  dt={dt:.1f}s  skipped={skipped}")
        step_metrics = {
            "step": step, "loss": avg_loss, "reward": avg_reward,
            "grad_norm": float(grad_norm), "dt": dt,
            "sampled_groups": sampled_groups,
            "optimized_groups": n_groups,
            "optimized_group_rate": optimized_group_rate,
            "degenerate_group_rate": degenerate_groups / max(sampled_groups, 1),
            "sft_group_rate": sft_groups / max(sampled_groups, 1),
            "sft_gt_group_rate": sft_gt_groups / max(sampled_groups, 1),
            "sft_ref_group_rate": sft_ref_groups / max(sampled_groups, 1),
            "dpo_gt_group_rate": dpo_gt_groups / max(sampled_groups, 1),
            "sft_gt_loss_mean": sft_gt_loss_sum / max(sft_gt_groups, 1),
            "sft_ref_loss_mean": sft_ref_loss_sum / max(sft_ref_groups, 1),
            "dpo_gt_loss_mean": dpo_gt_loss_sum / max(dpo_gt_groups, 1),
            "candidate_reward_mean": candidate_reward_mean,
            "candidate_reward_std": candidate_reward_std,
            "candidate_reward_positive_rate": reward_positive / max(reward_count, 1),
            "candidate_reward_perfect_rate": reward_perfect / max(reward_count, 1),
            "best_reward_mean": best_reward_sum / max(sampled_groups, 1),
            "skipped_signal_count": skipped,
        }
        metrics.append(step_metrics)
        _wandb_log(wandb_run, {
            "train/loss": avg_loss,
            "train/reward_mean_by_optimized_group": avg_reward,
            "train/grad_norm": float(grad_norm),
            "train/step_seconds": dt,
            "train/sampled_groups": sampled_groups,
            "train/optimized_groups": n_groups,
            "train/optimized_group_rate": optimized_group_rate,
            "reward/candidate_mean": candidate_reward_mean,
            "reward/candidate_std": candidate_reward_std,
            "reward/candidate_positive_rate": reward_positive / max(reward_count, 1),
            "reward/candidate_perfect_rate": reward_perfect / max(reward_count, 1),
            "reward/best_reward_mean": best_reward_sum / max(sampled_groups, 1),
            "reward/degenerate_group_rate": degenerate_groups / max(sampled_groups, 1),
            "reward/sft_group_rate": sft_groups / max(sampled_groups, 1),
            "reward/sft_gt_group_rate": sft_gt_groups / max(sampled_groups, 1),
            "reward/sft_ref_group_rate": sft_ref_groups / max(sampled_groups, 1),
            "reward/dpo_gt_group_rate": dpo_gt_groups / max(sampled_groups, 1),
            "loss/sft_gt_mean": sft_gt_loss_sum / max(sft_gt_groups, 1),
            "loss/sft_ref_mean": sft_ref_loss_sum / max(sft_ref_groups, 1),
            "loss/dpo_gt_mean": dpo_gt_loss_sum / max(dpo_gt_groups, 1),
            "data/skipped_signal_count": skipped,
        }, step=step)

        if run_eval_if_due(step):
            break

    # Save final checkpoint and metrics
    final_ckpt = out_dir / "best_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": original_config,
        "baseline_score": baseline_score,
        "target_delta": args.target_delta,
        "target_score": args.target_score,
        "target_threshold": target_threshold,
        "metrics": metrics,
    }, final_ckpt)
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({
            "baseline_score": baseline_score,
            "target_delta": args.target_delta,
            "target_score": args.target_score,
            "target_threshold": target_threshold,
            "best_score": best_score,
            "best_step": best_step,
            "wandb_run_url": getattr(wandb_run, "url", None) if wandb_run else None,
            "metrics": metrics,
        }, f, indent=2)
    print(f"[grpo] saved final to {final_ckpt}")

    # Final eval
    if args.skip_final_eval:
        print("[grpo] === final eval skipped (--skip_final_eval) ===")
    else:
        print("[grpo] === final eval ===")
        final_score = eval_on_subset(
            model, tokenizer, args.eval_subset, str(out_dir / "eval_final"),
            label="final", device=eval_device, original_config=original_config,
            max_new_tokens=256)
        print(f"[grpo] FINAL: baseline {baseline_score:.4f} -> final {final_score:.4f}  "
              f"(delta {final_score - baseline_score:+.4f})")
        _wandb_log(wandb_run, {
            "eval/final_overall_score": final_score,
            "eval/final_delta_vs_baseline": final_score - baseline_score,
        }, step=args.max_steps)

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
