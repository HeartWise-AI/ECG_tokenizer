#!/usr/bin/env python3
"""RLVR eval harness — 10-per-category LLM-judge gating.

Subsets the test parquet to 10 rows per prompt_category (seed=42), runs
open-ended generation with the given checkpoint, saves a CSV in the format
LLM_JUDGE/judge_eval.py expects, and (optionally) invokes the judge.

Usage:
    PYTHONPATH=/volume/ECG_tokenizer python scripts/rlvr_eval_subset.py \\
        --checkpoint /volume/ECG_tokenizer/checkpoints/BEST_LLM/2wjwbk0b_20260413-224103_ENHANCED/best_model.pt \\
        --output_dir /volume/ECG_tokenizer/analysis/rlvr_eval/baseline \\
        --device cuda:2 --n_per_cat 10 --run_judge
"""

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# -----------------------------------------------------------------------------
# Tokenizer shims (must run before importing the model)
# -----------------------------------------------------------------------------
_gemma_mod = importlib.import_module("transformers.models.gemma")
if not hasattr(_gemma_mod, "tokenization_gemma_fast"):
    shim = types.ModuleType("transformers.models.gemma.tokenization_gemma_fast")
    from transformers import GemmaTokenizer
    shim.GemmaTokenizerFast = GemmaTokenizer
    sys.modules["transformers.models.gemma.tokenization_gemma_fast"] = shim
    _gemma_mod.tokenization_gemma_fast = shim

import transformers.tokenization_utils as _tok_utils
if not hasattr(_tok_utils, "Trie"):
    class _Trie:
        def __init__(self, *a, **kw): pass
        def __setstate__(self, state): pass
    _tok_utils.Trie = _Trie

_gemma_tok = importlib.import_module("transformers.models.gemma.tokenization_gemma")
if hasattr(_gemma_tok.GemmaTokenizer, "__setstate__"):
    def _patched_setstate(self, state):
        if "sp_model_kwargs" not in state:
            state["sp_model_kwargs"] = {}
        self.__dict__.update(state)
        import sentencepiece as spm
        self.sp_model = spm.SentencePieceProcessor(**state.get("sp_model_kwargs", {}))
        if "sp_model_proto" in state:
            self.sp_model.LoadFromSerializedProto(state["sp_model_proto"])
        elif hasattr(self, "vocab_file") and self.vocab_file:
            self.sp_model.Load(self.vocab_file)
    _gemma_tok.GemmaTokenizer.__setstate__ = _patched_setstate

from transformers import AutoTokenizer
from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper


SYSTEM_MSG = (
    "You are an expert cardiologist. You interpret ECGs and answer "
    "in a concise, structured way."
)


def load_model(checkpoint_path: str, device: str):
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    hf_name = getattr(cfg, "huggingface_model_name", "google/medgemma-4b-it")
    tokenizer = AutoTokenizer.from_pretrained(hf_name)

    use_lora = any(
        "lora_A" in k or "lora_B" in k or "base_layer" in k
        for k in ckpt["model_state_dict"]
    )
    lora_config = None
    if use_lora:
        lora_config = {
            "r": getattr(cfg, "lora_r", 32),
            "lora_alpha": getattr(cfg, "lora_alpha", 64),
            "lora_dropout": getattr(cfg, "lora_dropout", 0.05),
            "target_modules": getattr(cfg, "lora_target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]),
            "bias": getattr(cfg, "lora_bias", "none"),
            "top_k_layers": getattr(cfg, "lora_top_k_layers", None),
        }

    # Normalize decoder_name — GRPO-saved configs use "MedGemmaDecoder", registry has "MedGemma_Decoder"
    raw_decoder_name = getattr(cfg, "decoder_name", "MedGemma_Decoder")
    if raw_decoder_name == "MedGemmaDecoder":
        raw_decoder_name = "MedGemma_Decoder"
    model = ECG_Tokenizer_Wrapper(
        encoder_name=getattr(cfg, "encoder_name", "Residual_Conv_Encoder"),
        quantizer_name=getattr(cfg, "quantizer_name", "ECG_Tokenizer_Quantizer"),
        decoder_name=raw_decoder_name,
        num_quantizers=getattr(cfg, "num_quantizers", 8),
        codebook_size=getattr(cfg, "codebook_size", 512),
        decoder_mode="llm",
        bridge_name=getattr(cfg, "bridge_name", "InstructionAwareECGQFormerBridge"),
        huggingface_model_name=hf_name,
        llm_input_embedding_size=getattr(cfg, "llm_input_embedding_size", 2560),
        tokenizer=tokenizer,
        num_visual_tokens=getattr(cfg, "num_query_tokens", 32),
        bridge_mid_dim=getattr(cfg, "bridge_mid_dim", 768),
        bridge_num_heads=getattr(cfg, "bridge_num_heads", 12),
        bridge_dropout=getattr(cfg, "bridge_dropout", 0.1),
        bridge_num_special_tokens=getattr(cfg, "bridge_num_special_tokens", 4),
        bridge_qformer_layers=getattr(cfg, "bridge_qformer_layers", 10),
        bridge_text_hidden_size=getattr(cfg, "bridge_text_hidden_size", 768),
        bridge_bias_last_codebook=getattr(cfg, "bridge_bias_last_codebook", 0.5),
        bridge_codebook_dropout=getattr(cfg, "bridge_codebook_dropout", 0.0),
        bridge_cross_every=getattr(cfg, "bridge_cross_every", 2),
        instruction_dropout=0.0,
        use_lora=use_lora,
        lora_config=lora_config,
        num_codebooks_kept=getattr(cfg, "num_codebooks_kept", 8),
        codebook_offset=getattr(cfg, "codebook_offset", -1),
        prefix_tuning=False,
        stage1_checkpoint_path=getattr(cfg, "stage1_checkpoint_path", None),
    ).to(device)

    model._load_state_dict(ckpt["model_state_dict"], strict=True)
    if use_lora:
        model.set_lora_inference_mode(True)
    model.eval()
    return model, tokenizer


def build_prompt(prompt_text: str) -> str:
    user_content = (
        "<start_of_image>\n\n"
        f"Question: {prompt_text}\n\n"
        "Respond concisely with the key finding or answer."
    )
    return (
        "<start_of_turn>system\n"
        f"{SYSTEM_MSG}<end_of_turn>\n"
        "<start_of_turn>user\n"
        f"{user_content}<end_of_turn>\n"
        "<start_of_turn>model\n"
    )


def load_ecg_signal(path: str, target: int = 2500, num_leads: int = 12) -> torch.Tensor:
    wav = np.load(path)
    if wav.ndim == 3:
        wav = wav.squeeze(-1)
    if wav.shape[-1] == num_leads:
        pass
    elif wav.shape[0] == num_leads:
        wav = wav.T
    n = wav.shape[0]
    if n >= target:
        s = (n - target) // 2
        wav = wav[s:s + target]
    else:
        pad_b = (target - n) // 2
        pad_a = target - n - pad_b
        wav = np.pad(wav, ((pad_b, pad_a), (0, 0)), mode="edge")
    return torch.from_numpy(wav.T).unsqueeze(0).float()


def build_subset(test_parquet: str, n_per_cat: int, seed: int) -> pd.DataFrame:
    df = pd.read_parquet(test_parquet)
    if "prompt" not in df.columns:
        raise SystemExit("test parquet missing `prompt` column")
    if "generated_answer" not in df.columns:
        if "report" in df.columns:
            df["generated_answer"] = df["report"]
        else:
            raise SystemExit("test parquet missing `generated_answer` and `report`")
    pieces = []
    rng = np.random.default_rng(seed)
    for cat, group in df.groupby("prompt_category", sort=True):
        idx = rng.permutation(len(group))[: min(n_per_cat, len(group))]
        pieces.append(group.iloc[idx])
    sub = pd.concat(pieces, axis=0).reset_index(drop=True)
    return sub


PARTIAL_REQUIRED_COLUMNS = {
    "row_idx",
    "source_row_idx",
    "waveform_name",
    "waveform_path",
    "question",
    "generation",
    "ground_truth",
    "prompt_category",
    "checkpoint_path",
}


def _csv_str(value, default: str = "") -> str:
    if pd.isna(value):
        return default
    return str(value)


def _partial_checkpoint_path(value) -> str:
    raw = _csv_str(value)
    return str(Path(raw).resolve()) if raw else ""


def expected_partial_identity(sub_row: pd.Series, checkpoint: str) -> Dict[str, object]:
    return {
        "source_row_idx": int(sub_row["source_row_idx"]),
        "waveform_path": str(sub_row["waveform_path_psa"]),
        "question": str(sub_row["prompt"]),
        "ground_truth": str(sub_row["generated_answer"]),
        "prompt_category": str(sub_row["prompt_category"]),
        "checkpoint_path": str(Path(checkpoint).resolve()),
    }


def validate_partial_resume_row(partial_row: pd.Series, sub_row: pd.Series, checkpoint: str) -> None:
    expected = expected_partial_identity(sub_row, checkpoint)
    actual = {
        "source_row_idx": None if pd.isna(partial_row["source_row_idx"]) else int(partial_row["source_row_idx"]),
        "waveform_path": _csv_str(partial_row["waveform_path"]),
        "question": _csv_str(partial_row["question"]),
        "ground_truth": _csv_str(partial_row["ground_truth"]),
        "prompt_category": _csv_str(partial_row["prompt_category"], default="unknown"),
        "checkpoint_path": _partial_checkpoint_path(partial_row["checkpoint_path"]),
    }
    mismatches = [key for key, expected_value in expected.items() if actual[key] != expected_value]
    if mismatches:
        details = ", ".join(
            f"{key}: partial={actual[key]!r} current={expected[key]!r}"
            for key in mismatches
        )
        raise ValueError(details)


@torch.no_grad()
def generate_for_row(
    model, tokenizer, signal: torch.Tensor, prompt_text: str,
    device: str, max_new_tokens: int = 256,
) -> str:
    prompt = build_prompt(prompt_text)
    enc = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    pids = enc["input_ids"].to(device)
    pmask = enc["attention_mask"].to(device)
    signal = signal.to(device=device, dtype=torch.float32)
    gen_ids = model.generate_report_with_question(
        x=signal,
        prompt_input_ids=pids,
        prompt_attention_mask=pmask,
        max_token_length=max_new_tokens,
        do_sample=False,
    )
    text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    return text.strip()


@torch.no_grad()
def generate_batched(
    model, tokenizer, signals: List[torch.Tensor], prompts: List[str],
    device: str, max_new_tokens: int = 256,
) -> List[str]:
    """Batched generation — much faster than one-at-a-time."""
    pad_id = tokenizer.pad_token_id or 0
    # Tokenize all prompts, pad to max length
    encs = [tokenizer(build_prompt(p), add_special_tokens=True, return_tensors="pt") for p in prompts]
    max_len = max(e["input_ids"].size(1) for e in encs)
    pids_list, pmask_list = [], []
    for e in encs:
        ids = e["input_ids"][0]
        mask = e["attention_mask"][0]
        pad = max_len - ids.size(0)
        if pad > 0:
            ids = torch.nn.functional.pad(ids, (pad, 0), value=pad_id)   # left-pad
            mask = torch.nn.functional.pad(mask, (pad, 0), value=0)
        pids_list.append(ids)
        pmask_list.append(mask)
    pids = torch.stack(pids_list).to(device)
    pmask = torch.stack(pmask_list).to(device)
    sig_batch = torch.stack([s.squeeze(0) if s.dim() == 3 else s for s in signals]).to(device=device, dtype=torch.float32)
    gen_ids = model.generate_report_with_question(
        x=sig_batch,
        prompt_input_ids=pids,
        prompt_attention_mask=pmask,
        max_token_length=max_new_tokens,
        do_sample=False,
    )
    texts = [tokenizer.decode(gen_ids[i], skip_special_tokens=True).strip() for i in range(gen_ids.size(0))]
    return texts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--test_parquet",
                        default="/volume/ECG_tokenizer/output/combined_test_qa_m25k_h25k.parquet")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:2")
    parser.add_argument("--n_per_cat", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--run_judge", action="store_true")
    parser.add_argument("--llm_judge_dir", default="/volume/LLM_JUDGE")
    parser.add_argument("--subset_parquet", default=None,
                        help="If set, use this prebuilt subset instead of resampling")
    parser.add_argument("--label", default="run",
                        help="Short tag for output files")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Generation batch size. Keep at 1 for this ECG "
                             "decoder: mixed prompt batches can corrupt greedy "
                             "outputs because of padding/position handling.")
    parser.add_argument("--group_by_prompt", action="store_true",
                        help="Batch only rows with identical prompt text. This "
                             "keeps prompt token lengths identical inside each "
                             "batch and is much safer than mixed-prompt batching.")
    parser.add_argument("--generation_microbatch_size", type=int, default=None,
                        help="Override MedGemma decoder microbatch size used "
                             "inside HF generate(). Defaults to checkpoint config.")
    parser.add_argument("--flush_every", type=int, default=0,
                        help="If >0, write a resumable partial CSV every N "
                             "completed rows.")
    parser.add_argument("--resume_partial", action="store_true",
                        help="Resume from generations_<label>.partial.csv if present.")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.subset_parquet:
        sub = pd.read_parquet(args.subset_parquet)
        print(f"[eval] Loaded prebuilt subset: {len(sub)} rows from {args.subset_parquet}")
    else:
        sub = build_subset(args.test_parquet, args.n_per_cat, args.seed)
        out_subset = Path(args.output_dir) / "subset.parquet"
        sub.to_parquet(out_subset, index=False)
        print(f"[eval] Subset: {len(sub)} rows across {sub['prompt_category'].nunique()} categories")
        print(sub["prompt_category"].value_counts().to_string())
        print(f"[eval] Saved subset to {out_subset}")
    if "source_row_idx" not in sub.columns:
        sub["source_row_idx"] = np.arange(len(sub), dtype=np.int64)
    sub = sub.reset_index(drop=True)

    print(f"[eval] Loading model from {args.checkpoint}")
    model, tokenizer = load_model(args.checkpoint, args.device)
    if args.generation_microbatch_size is not None:
        decoder = getattr(model, "decoder", model)
        if not hasattr(decoder, "generation_microbatch_size"):
            raise SystemExit("Loaded model does not expose generation_microbatch_size")
        decoder.generation_microbatch_size = max(1, int(args.generation_microbatch_size))
        print(f"[eval] generation_microbatch_size={decoder.generation_microbatch_size}")
    print("[eval] Model loaded; running generation")

    batch_size = int(getattr(args, "batch_size", 8))
    generations_by_idx: Dict[int, Dict[str, str]] = {}
    csv_path = Path(args.output_dir) / f"generations_{args.label}.csv"
    partial_csv_path = Path(args.output_dir) / f"generations_{args.label}.partial.csv"

    if args.resume_partial and partial_csv_path.exists():
        partial = pd.read_csv(partial_csv_path)
        if "row_idx" not in partial.columns:
            raise SystemExit(f"Partial CSV missing row_idx: {partial_csv_path}")
        missing = PARTIAL_REQUIRED_COLUMNS - set(partial.columns)
        if missing:
            raise SystemExit(f"Partial CSV missing required columns: {missing}")
        for _, row in partial.iterrows():
            idx = int(row["row_idx"])
            if 0 <= idx < len(sub):
                try:
                    validate_partial_resume_row(row, sub.iloc[idx], args.checkpoint)
                except ValueError as exc:
                    raise SystemExit(
                        f"Partial CSV row_idx={idx} does not match current run: {exc}"
                    ) from exc
                generations_by_idx[idx] = {
                    "row_idx": idx,
                    "source_row_idx": int(row["source_row_idx"]),
                    "waveform_name": "" if pd.isna(row["waveform_name"]) else str(row["waveform_name"]),
                    "waveform_path": "" if pd.isna(row["waveform_path"]) else str(row["waveform_path"]),
                    "question": "" if pd.isna(row["question"]) else str(row["question"]),
                    "generation": "" if pd.isna(row["generation"]) else str(row["generation"]),
                    "ground_truth": "" if pd.isna(row["ground_truth"]) else str(row["ground_truth"]),
                    "prompt_category": "unknown" if pd.isna(row["prompt_category"]) else str(row["prompt_category"]),
                    "checkpoint_path": str(Path(args.checkpoint).resolve()),
                }
        print(f"[eval] Resumed {len(generations_by_idx)}/{len(sub)} rows from {partial_csv_path}")

    def flush_partial(force: bool = False):
        if not force and int(args.flush_every) <= 0:
            return
        rows = [generations_by_idx[i] for i in sorted(generations_by_idx)]
        tmp_path = partial_csv_path.with_suffix(partial_csv_path.suffix + ".tmp")
        pd.DataFrame(rows).to_csv(tmp_path, index=False)
        os.replace(tmp_path, partial_csv_path)

    def run_batch(batch):
        try:
            signals = [load_ecg_signal(str(r["waveform_path_psa"])) for _, r in batch]
            prompts = [str(r["prompt"]) for _, r in batch]
            gens = generate_batched(model, tokenizer, signals, prompts,
                                     args.device, args.max_new_tokens)
        except Exception as e:
            gens = [f"[ERROR: {e}]" for _ in batch]
        for (i, row), gen in zip(batch, gens):
            wp = str(row["waveform_path_psa"])
            wn = os.path.basename(wp).replace(".npy", "")
            generations_by_idx[int(i)] = {
                "row_idx": int(i),
                "source_row_idx": int(row["source_row_idx"]),
                "waveform_name": wn,
                "waveform_path": wp,
                "question": str(row["prompt"]),
                "generation": gen,
                "ground_truth": str(row["generated_answer"]),
                "prompt_category": str(row["prompt_category"]),
                "checkpoint_path": str(Path(args.checkpoint).resolve()),
            }

    done = len(generations_by_idx)
    last_flush_done = done
    if args.group_by_prompt:
        for _, group in sub.groupby("prompt", sort=False):
            group_rows = [
                item for item in group.iterrows()
                if int(item[0]) not in generations_by_idx
            ]
            for batch_start in range(0, len(group_rows), batch_size):
                batch = group_rows[batch_start:batch_start + batch_size]
                run_batch(batch)
                done += len(batch)
                print(f"[eval] {done}/{len(sub)}")
                if int(args.flush_every) > 0 and done - last_flush_done >= int(args.flush_every):
                    flush_partial()
                    last_flush_done = done
    else:
        rows_list = [
            item for item in sub.iterrows()
            if int(item[0]) not in generations_by_idx
        ]
        for batch_start in range(0, len(rows_list), batch_size):
            batch = rows_list[batch_start:batch_start + batch_size]
            run_batch(batch)
            done += len(batch)
            print(f"[eval] {done}/{len(sub)}")
            if int(args.flush_every) > 0 and done - last_flush_done >= int(args.flush_every):
                flush_partial()
                last_flush_done = done

    missing = [i for i in range(len(sub)) if i not in generations_by_idx]
    if missing:
        raise SystemExit(f"Missing {len(missing)} generated rows; first missing idx={missing[0]}")
    flush_partial(force=True)
    generations = [generations_by_idx[i] for i in range(len(sub))]

    pd.DataFrame(generations).to_csv(csv_path, index=False)
    print(f"[eval] Saved generations: {csv_path}")

    if not args.run_judge:
        print("[eval] Skipping judge (use --run_judge to invoke)")
        return

    judge_out = Path(args.output_dir) / f"judge_{args.label}.json"
    cmd = [
        sys.executable, "judge_eval.py",
        "--csv", str(csv_path.resolve()),
        "--output", str(judge_out.resolve()),
    ]
    print(f"[eval] Invoking LLM judge: {' '.join(cmd)}")
    env = os.environ.copy()
    res = subprocess.run(cmd, cwd=args.llm_judge_dir, env=env)
    if res.returncode != 0:
        raise SystemExit(f"judge_eval.py failed with code {res.returncode}")

    with open(judge_out) as f:
        results = json.load(f)
    # Judge writes either {overall_score, category_aggregates: ...} (top-level)
    # or {aggregates: {overall_score, category_aggregates: ...}}. Handle both.
    root = results.get("aggregates", results)
    summary = {
        "overall_score": root.get("overall_score"),
        "category_aggregates": {
            cat: {"count": d.get("count"), "mean_score": d.get("mean_score")}
            for cat, d in root.get("category_aggregates", {}).items()
        },
    }
    summary_path = Path(args.output_dir) / f"summary_{args.label}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] Saved summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
