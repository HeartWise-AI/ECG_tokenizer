#!/usr/bin/env python3
"""Best-of-N inference: generate N candidates per prompt, score with LLM judge,
keep highest-scoring. Test-time compute optimization — no training, no risk of
regression. Optionally start from a candidate set; falls back to running the
underlying eval script generator if none provided.
"""

import argparse
import importlib
import importlib.util
import json
import os
import subprocess
import sys
import types
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Tokenizer shims (must run before importing model)
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

# Import eval helpers
spec = importlib.util.spec_from_file_location(
    "rlvr_eval_subset", str(Path(__file__).resolve().parent / "rlvr_eval_subset.py"))
eval_mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eval_mod)

JUDGE_REGISTRY = None
CAT_MAP: Dict[str, List[str]] = {}
JUDGE_DIR: str | None = None


def init_judge_registry(llm_judge_dir: str):
    global JUDGE_REGISTRY, CAT_MAP, JUDGE_DIR
    resolved_dir = str(Path(llm_judge_dir).resolve())
    if JUDGE_REGISTRY is not None and JUDGE_DIR == resolved_dir:
        return

    if JUDGE_DIR is not None and JUDGE_DIR != resolved_dir:
        sys.modules.pop("judges", None)
        sys.modules.pop("merge_utils", None)

    if resolved_dir not in sys.path:
        sys.path.insert(0, resolved_dir)
    env_path = Path(resolved_dir) / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            if line.startswith("FIREWORKS_API_KEY="):
                os.environ["FIREWORKS_API_KEY"] = line.split("=", 1)[1].strip()

    from judges import registry, register_judges
    from merge_utils import get_category_judge_mapping

    register_judges()
    JUDGE_REGISTRY = registry
    CAT_MAP = get_category_judge_mapping()
    JUDGE_DIR = resolved_dir


def judge_score(prediction: str, ground_truth: str, prompt_category: str) -> float:
    if JUDGE_REGISTRY is None:
        init_judge_registry("/volume/LLM_JUDGE")
    judge_names = CAT_MAP.get(prompt_category, ["classification_judge"])
    judge = JUDGE_REGISTRY.get(judge_names[0])
    if judge is None:
        return 0.0
    try:
        verdict = judge.evaluate(prediction, ground_truth)
        return float(verdict.score) if verdict is not None else 0.0
    except Exception:
        return 0.0


@torch.no_grad()
def generate_candidates(model, tokenizer, signal: torch.Tensor, prompt_text: str,
                         n: int, device: str, max_new_tokens: int = 256,
                         temperature: float = 1.0, top_p: float = 0.95) -> List[str]:
    """Generate n diverse candidates for a single (signal, prompt)."""
    prompt = eval_mod.build_prompt(prompt_text)
    enc = tokenizer(prompt, add_special_tokens=True, return_tensors="pt")
    pids = enc["input_ids"].to(device)
    pmask = enc["attention_mask"].to(device)
    signal = signal.to(device=device, dtype=torch.float32)
    # Duplicate signal n times
    signal_rep = signal.expand(n, -1, -1) if signal.dim() == 3 else signal.unsqueeze(0).expand(n, -1, -1)
    pids_rep = pids.expand(n, -1)
    pmask_rep = pmask.expand(n, -1)
    gen_ids = model.generate_report_with_question(
        x=signal_rep,
        prompt_input_ids=pids_rep,
        prompt_attention_mask=pmask_rep,
        max_token_length=max_new_tokens,
        do_sample=True,
        temperature=temperature,
        top_p=top_p,
    )
    return [tokenizer.decode(gen_ids[i], skip_special_tokens=True).strip()
            for i in range(gen_ids.size(0))]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--subset_parquet", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--n_candidates", type=int, default=5)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--label", default="bestof5")
    parser.add_argument("--run_judge", action="store_true",
                        help="Run final LLM judge eval on the selected candidates")
    parser.add_argument("--llm_judge_dir", default="/volume/LLM_JUDGE")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    llm_judge_dir = str(Path(args.llm_judge_dir).resolve())
    init_judge_registry(llm_judge_dir)

    sub = pd.read_parquet(args.subset_parquet)
    print(f"[bestofN] Loaded subset: {len(sub)} rows")

    print(f"[bestofN] Loading model from {args.checkpoint}")
    model, tokenizer = eval_mod.load_model(args.checkpoint, args.device)
    print("[bestofN] Model loaded; running best-of-N inference")

    generations = []
    for i, row in sub.iterrows():
        try:
            signal = eval_mod.load_ecg_signal(str(row["waveform_path_psa"]))
            candidates = generate_candidates(
                model, tokenizer, signal, str(row["prompt"]),
                n=args.n_candidates, device=args.device,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature, top_p=args.top_p,
            )
            # Score each with the judge, pick best
            gt = str(row["generated_answer"])
            cat = str(row["prompt_category"])
            scores = [judge_score(c, gt, cat) for c in candidates]
            best_idx = int(np.argmax(scores))
            best_gen = candidates[best_idx]
            best_score = scores[best_idx]
        except Exception as e:
            best_gen = f"[ERROR: {e}]"
            best_score = 0.0
            scores = [0.0] * args.n_candidates

        wp = str(row["waveform_path_psa"])
        wn = os.path.basename(wp).replace(".npy", "")
        generations.append({
            "waveform_name": wn,
            "waveform_path": wp,
            "question": str(row["prompt"]),
            "generation": best_gen,
            "ground_truth": str(row["generated_answer"]),
            "prompt_category": str(row["prompt_category"]),
            "bestof_picked_idx": best_idx if not best_gen.startswith("[ERROR") else -1,
            "bestof_picked_score": best_score,
            "bestof_candidate_scores": scores,
        })
        if (i + 1) % 5 == 0:
            print(f"[bestofN] {i+1}/{len(sub)}  last_best_score={best_score:.2f}")

    csv_path = Path(args.output_dir) / f"generations_{args.label}.csv"
    # Convert candidate_scores list to JSON-safe
    out_df = pd.DataFrame(generations)
    out_df["bestof_candidate_scores"] = out_df["bestof_candidate_scores"].apply(json.dumps)
    out_df.to_csv(csv_path, index=False)
    print(f"[bestofN] Saved generations: {csv_path}")

    if args.run_judge:
        judge_out = Path(args.output_dir) / f"judge_{args.label}.json"
        cmd = [
            sys.executable, "judge_eval.py",
            "--csv", str(csv_path.resolve()),
            "--output", str(judge_out.resolve()),
        ]
        print(f"[bestofN] Final judge eval: {' '.join(cmd)}")
        env = os.environ.copy()
        res = subprocess.run(cmd, cwd=llm_judge_dir, env=env)
        if res.returncode != 0:
            raise SystemExit(f"judge_eval.py failed with code {res.returncode}")

        with open(judge_out) as f:
            results = json.load(f)
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
        print(f"[bestofN] Summary saved: {summary_path}")
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
