"""
Aggregate text generation metrics (ROUGE, BLEU, METEOR) over full sets.

This utility centralizes metric computation so callers (runners, scripts)
can avoid inlined, duplicated logic.

UPDATED: Now uses the same libraries and approach as the inference script
(generate_all_qa_pairs.py) for consistent metrics:
 - rouge_score.RougeScorer (Google's rouge-score, per-sentence then averaged)
 - nltk.translate.bleu_score.sentence_bleu (per-sentence BLEU)
 - nltk.translate.meteor_score.meteor_score (per-sentence METEOR)

All heavy imports are done inside the function to keep import cost low for
callers that don't need metrics.
"""

from __future__ import annotations

from typing import Dict, List
import os
import sys
import contextlib
import warnings


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

    Uses the SAME approach as the inference script (generate_all_qa_pairs.py):
    - Per-sentence scoring with averaging (not corpus-level)
    - rouge_score library (Google's rouge-score)
    - nltk for BLEU and METEOR

    Returns a dictionary including available keys among:
      - rouge1, rouge2, rougeL
      - bleu1, bleu4
      - meteor

    Any metric that fails to compute will be omitted from the result.
    """
    import numpy as np

    results: Dict[str, float] = {}

    if not predictions or not references:
        return results

    # Import rouge_score (Google's library - same as inference script)
    _rouge_scorer = None
    try:
        from rouge_score import rouge_scorer as _rouge_scorer
    except ImportError:
        pass

    # Import nltk (same as inference script)
    _nltk_bleu = None
    _nltk_meteor = None
    _word_tokenize = None
    _smoother = None
    try:
        from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
        from nltk.translate.meteor_score import meteor_score as _nltk_meteor
        from nltk import word_tokenize as _word_tokenize
        import nltk
        # Ensure required NLTK data is available
        try:
            with _silence_stdout_stderr():
                nltk.download('wordnet', quiet=True)
                nltk.download('punkt', quiet=True)
                nltk.download('punkt_tab', quiet=True)
        except Exception:
            pass
        _nltk_bleu = sentence_bleu
        _smoother = SmoothingFunction()
    except ImportError:
        pass

    # Collectors for per-sentence scores
    rouge1_scores: List[float] = []
    rouge2_scores: List[float] = []
    rougeL_scores: List[float] = []
    bleu1_scores: List[float] = []
    bleu4_scores: List[float] = []
    meteor_scores: List[float] = []

    # Setup ROUGE scorer (matching inference script: use_stemmer=True)
    scorer = None
    if _rouge_scorer is not None:
        scorer = _rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    # Process each (prediction, reference) pair (same as inference script)
    for pred, ref in zip(predictions, references):
        if not pred or not ref:
            continue
        pred = str(pred).strip()
        ref = str(ref).strip()
        if not pred or not ref:
            continue

        # ROUGE (per-sentence, matching inference script)
        if scorer is not None:
            try:
                scores = scorer.score(ref, pred)
                rouge1_scores.append(scores['rouge1'].fmeasure)
                rouge2_scores.append(scores['rouge2'].fmeasure)
                rougeL_scores.append(scores['rougeL'].fmeasure)
            except Exception:
                pass

        # BLEU and METEOR (matching inference script exactly)
        if _word_tokenize is not None and _nltk_bleu is not None:
            try:
                ref_tokens = _word_tokenize(ref.lower())
                pred_tokens = _word_tokenize(pred.lower())
                if ref_tokens and pred_tokens:
                    # BLEU-1 with smoothing method1 (same as inference script)
                    bleu1_scores.append(_nltk_bleu(
                        [ref_tokens], pred_tokens,
                        weights=(1, 0, 0, 0),
                        smoothing_function=_smoother.method1
                    ))
                    # BLEU-4 with smoothing method1 (same as inference script)
                    bleu4_scores.append(_nltk_bleu(
                        [ref_tokens], pred_tokens,
                        weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=_smoother.method1
                    ))
                    # METEOR (same as inference script)
                    if _nltk_meteor is not None:
                        meteor_scores.append(_nltk_meteor([ref_tokens], pred_tokens))
            except Exception:
                pass

    # Compute averages (same as inference script: np.mean of per-sentence scores)
    def safe_mean(lst: List[float]) -> float:
        return float(np.mean(lst)) if lst else 0.0

    if rouge1_scores:
        results["rouge1"] = safe_mean(rouge1_scores)
    if rouge2_scores:
        results["rouge2"] = safe_mean(rouge2_scores)
    if rougeL_scores:
        results["rougeL"] = safe_mean(rougeL_scores)
    if bleu1_scores:
        results["bleu1"] = safe_mean(bleu1_scores)
    if bleu4_scores:
        results["bleu4"] = safe_mean(bleu4_scores)
    if meteor_scores:
        results["meteor"] = safe_mean(meteor_scores)

    return results
