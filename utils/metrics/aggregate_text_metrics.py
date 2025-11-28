"""
Aggregate text generation metrics (ROUGE, BLEU, METEOR) over full sets.

This utility centralizes metric computation so callers (runners, scripts)
can avoid inlined, duplicated logic. It uses:
 - HuggingFace `evaluate` for ROUGE (with aggregator + stemming) and METEOR
 - SacreBLEU for BLEU-1 and BLEU-4 corpus scores

All heavy imports are done inside the function to keep import cost low for
callers that don't need metrics.
"""

from __future__ import annotations

from typing import Dict, List
import os
import sys
import contextlib


@contextlib.contextmanager
def _silence_stdout_stderr():
    saved_out, saved_err = sys.stdout, sys.stderr
    try:
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull  # type: ignore
            sys.stderr = devnull  # type: ignore
            yield
    finally:
        sys.stdout, sys.stderr = saved_out, saved_err


def aggregate_text_metrics(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute aggregated metrics over the entire set.

    Returns a dictionary including available keys among:
      - rouge1, rouge2 (when available), rougeL
      - bleu1, bleu4
      - meteor

    Any metric that fails to compute will be omitted from the result.
    """
    results: Dict[str, float] = {}

    if not predictions or not references:
        return results

    # Import heavy deps lazily
    _hf_load = None
    _sacrebleu = None
    try:
        from evaluate import load as _hf_load  # type: ignore
    except Exception:
        _hf_load = None
    try:
        import sacrebleu as _sacrebleu  # type: ignore
    except Exception:
        _sacrebleu = None

    # ROUGE (HF evaluate)
    if _hf_load is not None:
        try:
            rouge = _hf_load("rouge")
            r = rouge.compute(
                predictions=predictions,
                references=references,
                use_stemmer=True,
                use_aggregator=True,
            )

            def _x(v):
                try:
                    if hasattr(v, "mid") and hasattr(v.mid, "fmeasure"):
                        return float(v.mid.fmeasure)
                    if hasattr(v, "fmeasure"):
                        return float(v.fmeasure)
                    return float(v)
                except Exception:
                    return None

            v1 = _x(r.get("rouge1", 0.0))
            if v1 is not None:
                results["rouge1"] = v1
            vL = _x(r.get("rougeL", 0.0))
            if vL is not None:
                results["rougeL"] = vL
            if "rouge2" in r:
                v2 = _x(r.get("rouge2"))
                if v2 is not None:
                    results["rouge2"] = v2
        except Exception:
            pass

    # BLEU (SacreBLEU)
    if _sacrebleu is not None:
        try:
            refs_nested = [references]
            bleu4 = _sacrebleu.corpus_bleu(predictions, refs_nested).score / 100.0
            bleu1_scorer = _sacrebleu.BLEU(max_ngram_order=1, effective_order=True)
            bleu1 = bleu1_scorer.corpus_score(predictions, refs_nested).score / 100.0
            results["bleu4"] = float(bleu4)
            results["bleu1"] = float(bleu1)
        except Exception:
            pass

    # METEOR (HF evaluate)
    if _hf_load is not None:
        try:
            # Silence noisy NLTK downloader messages that may occur inside METEOR
            with _silence_stdout_stderr():
                meteor = _hf_load("meteor")
                m = meteor.compute(predictions=predictions, references=references)
            mv = m.get("meteor", None)
            if mv is not None:
                results["meteor"] = float(mv)
        except Exception:
            pass

    return results
