"""
This module contains classes to compute evaluation metrics for LLMs.
Each class provides a static function `compute_score` to compute the metric.
Metrics computed:
 - ROUGE: Computes both ROUGE-1 and ROUGE-L F1 scores.
 - BLEU: Computes both BLEU-1 (unigram) and BLEU-4 scores using SacreBLEU.
 - METEOR: Computes the METEOR score.
 - BERTScore: Computes precision, recall and F1 scores.

Each class is registered with MetricRegistry for consistent retrieval.
"""

import warnings
import os
import sys
import contextlib
from typing import Dict, List, Tuple, Union, Optional
import json as _json
import re as _re
import torch
import numpy as np
from transformers import PreTrainedTokenizerBase

from utils.registry import MetricRegistry
from utils.constants import DEEPECG_CATEGORIES, DEEPECG_DIAGNOSIS_TRANSLATION


@contextlib.contextmanager
def _suppress_stdout_stderr():
    """Temporarily silence stdout/stderr (for noisy 3rd-party downloads)."""
    saved_out, saved_err = sys.stdout, sys.stderr
    try:
        with open(os.devnull, 'w') as devnull:
            sys.stdout = devnull  # type: ignore
            sys.stderr = devnull  # type: ignore
            yield
    finally:
        sys.stdout, sys.stderr = saved_out, saved_err


def decode_assistant_only_text(
    tokenizer: PreTrainedTokenizerBase,
    generated: torch.Tensor,
    labels: torch.Tensor,
    input_ids: Optional[torch.Tensor] = None,
    prompt_input_ids: Optional[torch.Tensor] = None,
    num_ecg_tokens: int = 0,
) -> Tuple[str, str]:
    """Decode assistant-only text for predictions and references.

    Shared helper so both runner logging and metric eval stay in sync.
    Drops prompt tokens (where labels are -100) from both the generated output and
    the reference so that metrics focus on the assistant response.
    
    IMPORTANT: When ECG tokens are injected after <start_of_image>, the generated
    sequence is longer than the original prompt by num_ecg_tokens. This function
    accounts for this by using prompt_input_ids length + num_ecg_tokens as the
    offset to find where the generated answer starts.
    
    Args:
        tokenizer: Tokenizer instance
        generated: Generated token ids (includes prompt with ECG + new generated tokens)
        labels: Label token ids with -100 for prompt tokens (aligned with input_ids, NOT generated)
        input_ids: Original input token ids (question + answer, optional)
        prompt_input_ids: Question-only input ids used for generation (optional but recommended)
        num_ecg_tokens: Number of ECG tokens injected into the sequence (default 0).
    
    Returns:
        Tuple of (prediction_text, reference_text) with normalized whitespace
    """
    label_tensor = labels.detach().cpu()
    generated_tensor = generated.detach().cpu()

    # Reference: drop -100s (prompt tokens) - this is correct regardless of ECG injection
    ref_ids = label_tensor[label_tensor != -100].tolist()

    # Prediction: find where the generated answer starts
    gen_ids = generated_tensor.tolist()
    
    # Best case: we know the prompt length used for generation + ECG tokens
    if prompt_input_ids is not None:
        prompt_len = int(prompt_input_ids.shape[-1]) + int(num_ecg_tokens)
        if len(gen_ids) >= prompt_len:
            gen_ids = gen_ids[prompt_len:]
    elif input_ids is not None:
        # Fallback: use input_ids but this may be wrong if it's question+answer
        # We need to find where the answer starts in labels and use that offset
        label_list = label_tensor.tolist()
        first_answer_idx = next((i for i, t in enumerate(label_list) if t != -100), 0)
        # Adjust for ECG tokens that were injected
        adjusted_idx = first_answer_idx + int(num_ecg_tokens)
        if len(gen_ids) >= adjusted_idx:
            gen_ids = gen_ids[adjusted_idx:]
    else:
        # Last resort: use labels to find first non--100
        label_list = label_tensor.tolist()
        first_answer_idx = next((i for i, t in enumerate(label_list) if t != -100), 0)
        # Adjust for ECG tokens
        adjusted_idx = first_answer_idx + int(num_ecg_tokens)
        if len(gen_ids) >= adjusted_idx:
            gen_ids = gen_ids[adjusted_idx:]

    # Decode and normalize text (strip + collapse whitespace)
    pred = " ".join(tokenizer.decode(gen_ids, skip_special_tokens=True).strip().split())
    ref = " ".join(tokenizer.decode(ref_ids, skip_special_tokens=True).strip().split())
    
    return pred, ref


@MetricRegistry.register("rouge")
class RougeMetric:
    _metric = None  # Lazy-loaded to avoid import errors
    
    @staticmethod
    def compute_score(
        generated_ids: torch.Tensor, 
        labels: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase,
        input_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[float, List[str]]]:
        """
        Computes the average ROUGE-1 and ROUGE-L F1 scores over a batch.

        Args:
            generated_ids: Tensor of generated token ids
            labels: Tensor of reference token ids
            tokenizer: Tokenizer instance
            input_ids: Original input token ids (optional)

        Returns:
            Dictionary with keys "rouge1" and "rougeL" representing their respective F1 scores,
            plus "predictions" and "references".
        """
        # Lazy-load metric
        if RougeMetric._metric is None:
            from evaluate import load as load_metric
            RougeMetric._metric = load_metric("rouge")
        
        predictions = []
        references = []
        
        for i, (gen, lab) in enumerate(zip(generated_ids, labels)):
            ii = input_ids[i] if input_ids is not None else None
            pred, ref = decode_assistant_only_text(tokenizer, gen, lab, ii)
            predictions.append(pred)
            references.append(ref)
        
        # Compute with aggregator and stemming for stability
        result = RougeMetric._metric.compute(
            predictions=predictions, 
            references=references,
            use_stemmer=True,
            use_aggregator=True
        )

        # Ensure float outputs (handle Score objects)
        def _extract_score(x):
            if hasattr(x, 'mid') and hasattr(x.mid, 'fmeasure'):
                return float(x.mid.fmeasure)
            elif hasattr(x, 'fmeasure'):
                return float(x.fmeasure)
            else:
                return float(x)
        
        return {
            "rouge1": _extract_score(result["rouge1"]),
            "rougeL": _extract_score(result["rougeL"]),
            "predictions": predictions,
            "references": references
        }


@MetricRegistry.register("bleu")
class SacreBleuMetric:
    @staticmethod
    def compute_score(
        generated_ids: torch.Tensor,
        labels: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase,
        input_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[float, List[str]]]:
        """
        Computes BLEU scores using SacreBLEU for stability and reproducibility.

        Args:
            generated_ids: Tensor of generated token ids
            labels: Tensor of reference token ids
            tokenizer: Tokenizer instance
            input_ids: Original input token ids (optional)

        Returns:
            Dictionary with keys "bleu1" and "bleu4" scores, plus "predictions" and "references".
        """
        try:
            import sacrebleu
        except ImportError:
            warnings.warn("SacreBLEU not installed. Install with: pip install sacrebleu. Returning 0 scores.")
            return {"bleu1": 0.0, "bleu4": 0.0, "predictions": [], "references": []}
        
        predictions = []
        references = []
        
        for i, (gen, lab) in enumerate(zip(generated_ids, labels)):
            ii = input_ids[i] if input_ids is not None else None
            pred, ref = decode_assistant_only_text(tokenizer, gen, lab, ii)
            if pred.strip() and ref.strip():  # Only include non-empty pairs
                predictions.append(pred)
                references.append(ref)
        
        if not predictions:
            warnings.warn("BLEU skipped: empty predictions/references")
            return {"bleu1": 0.0, "bleu4": 0.0, "predictions": [], "references": []}
        
        # SacreBLEU expects references as [num_refs, N] shape
        refs_nested = [references]
        
        try:
            bleu4 = sacrebleu.corpus_bleu(predictions, refs_nested).score / 100.0
            # Use BLEU scorer with max_ngram_order for BLEU-1
            bleu1_scorer = sacrebleu.BLEU(max_ngram_order=1, effective_order=True)
            bleu1 = bleu1_scorer.corpus_score(predictions, refs_nested).score / 100.0
        except Exception as e:
            warnings.warn(f"BLEU computation failed: {e}. Returning 0 scores.")
            bleu1, bleu4 = 0.0, 0.0
        
        return {
            "bleu1": float(bleu1),
            "bleu4": float(bleu4),
            "predictions": predictions,
            "references": references
        }


@MetricRegistry.register("meteor")
class MeteorMetric:
    _metric = None  # Lazy-loaded
    
    @staticmethod
    def compute_score(
        generated_ids: torch.Tensor,
        labels: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase,
        input_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[float, List[str]]]:
        """
        Computes the average METEOR score over a batch.

        Args:
            generated_ids: Tensor of generated token ids
            labels: Tensor of reference token ids
            tokenizer: Tokenizer instance
            input_ids: Original input token ids (optional)

        Returns:
            Dictionary with key "meteor" mapping to the average METEOR score,
            plus "predictions" and "references".
        """
        # Lazy-load metric
        if MeteorMetric._metric is None:
            try:
                from evaluate import load as load_metric
                with _suppress_stdout_stderr():
                    MeteorMetric._metric = load_metric("meteor")
            except Exception as e:
                warnings.warn(f"METEOR unavailable (likely missing nltk data): {e}. Returning 0.")
                return {"meteor": 0.0, "predictions": [], "references": []}
        
        predictions = []
        references = []
        
        for i, (gen, lab) in enumerate(zip(generated_ids, labels)):
            ii = input_ids[i] if input_ids is not None else None
            pred, ref = decode_assistant_only_text(tokenizer, gen, lab, ii)
            predictions.append(pred)
            references.append(ref)
        
        try:
            # Suppress any NLTK downloader logs during compute
            with _suppress_stdout_stderr():
                result = MeteorMetric._metric.compute(predictions=predictions, references=references)
            meteor_score = float(result["meteor"])
        except Exception as e:
            warnings.warn(f"METEOR computation failed: {e}. Returning 0.")
            meteor_score = 0.0
        
        return {
            "meteor": meteor_score,
            "predictions": predictions,
            "references": references
        }


@MetricRegistry.register("bertscore")
class BertScoreMetric:
    _metric = None  # Lazy-loaded
    _default_model_type = "microsoft/deberta-xlarge-mnli"  # Pinned for reproducibility
    
    @staticmethod
    def compute_score(
        generated_ids: torch.Tensor,
        labels: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase,
        input_ids: Optional[torch.Tensor] = None,
        lang: str = "en",
        model_type: Optional[str] = None,
        rescale_with_baseline: bool = True,
        idf: bool = False,
        device: Optional[str] = None
    ) -> Dict[str, Union[float, List[str]]]:
        """
        Computes mean BERTScore Precision/Recall/F1 over a batch.
        
        Args:
            generated_ids: Tensor of generated token ids
            labels: Tensor of reference token ids
            tokenizer: Tokenizer instance
            input_ids: Original input token ids (optional)
            lang: Language code
            model_type: BERTScore model to use (defaults to pinned DeBERTa)
            rescale_with_baseline: Whether to rescale with baseline
            idf: Whether to use IDF weighting
            device: Device to run on
        
        Returns:
            Dictionary with hf-prec, hf-rec, hf-f1 scores, plus predictions and references.
        """
        # Lazy-load metric
        if BertScoreMetric._metric is None:
            try:
                from evaluate import load as load_metric
                BertScoreMetric._metric = load_metric("bertscore")
            except Exception as e:
                warnings.warn(f"BERTScore unavailable. Install with: pip install bert-score. Error: {e}")
                return {
                    "hf-prec": 0.0,
                    "hf-rec": 0.0,
                    "hf-f1": 0.0,
                    "predictions": [],
                    "references": []
                }
        
        predictions = []
        references = []
        
        for i, (gen, lab) in enumerate(zip(generated_ids, labels)):
            ii = input_ids[i] if input_ids is not None else None
            pred, ref = decode_assistant_only_text(tokenizer, gen, lab, ii)
            predictions.append(pred)
            references.append(ref)
        
        # Use pinned model for reproducibility if not specified
        if model_type is None:
            model_type = BertScoreMetric._default_model_type
        
        try:
            results = BertScoreMetric._metric.compute(
                predictions=predictions,
                references=references,
                lang=lang,
                model_type=model_type,
                rescale_with_baseline=rescale_with_baseline,
                idf=idf,
                device=device
            )
            
            # Handle both list and single value returns
            prec = float(np.mean(results["precision"])) if isinstance(results["precision"], list) else float(results["precision"])
            rec = float(np.mean(results["recall"])) if isinstance(results["recall"], list) else float(results["recall"])
            f1 = float(np.mean(results["f1"])) if isinstance(results["f1"], list) else float(results["f1"])
        except Exception as e:
            warnings.warn(f"BERTScore computation failed: {e}. Returning 0 scores.")
            prec, rec, f1 = 0.0, 0.0, 0.0
        
        return {
            "hf-prec": prec,
            "hf-rec": rec,
            "hf-f1": f1,
            "predictions": predictions,
            "references": references
        }


@MetricRegistry.register("json_parse_rate")
class JsonParseRateMetric:
    @staticmethod
    def compute_score(
        generated_ids: torch.Tensor,
        labels: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase,
        input_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[float, List[str]]]:
        parsed = 0
        total = 0
        predictions: List[str] = []
        references: List[str] = []
        for i, (gen, lab) in enumerate(zip(generated_ids, labels)):
            ii = input_ids[i] if input_ids is not None else None
            pred, ref = decode_assistant_only_text(tokenizer, gen, lab, ii)
            predictions.append(pred)
            references.append(ref)
            total += 1
            ok = False
            try:
                _json.loads(pred)
                ok = True
            except Exception:
                # Try trimming to bracketed JSON
                try:
                    start_obj = pred.find('{')
                    end_obj = pred.rfind('}')
                    start_arr = pred.find('[')
                    end_arr = pred.rfind(']')
                    cand = None
                    if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
                        cand = pred[start_obj:end_obj + 1]
                    elif start_arr != -1 and end_arr != -1 and end_arr > start_arr:
                        cand = pred[start_arr:end_arr + 1]
                    if cand is not None:
                        _json.loads(cand)
                        ok = True
                except Exception:
                    ok = False
            if ok:
                parsed += 1
        rate = float(parsed) / float(max(1, total))
        return {
            "json_parse_rate": rate,
            "predictions": predictions,
            "references": references,
        }


@MetricRegistry.register("structured_f1")
class StructuredSlotF1Metric:
    """
    Slot-level (micro) precision/recall/F1 for structured JSON fields.
    Expects categories similar to DEEPECG_CATEGORIES keys. Applies translation
    normalization from DEEPECG_DIAGNOSIS_TRANSLATION.
    """

    @staticmethod
    def _build_translation_map() -> Dict[str, str]:
        tr_map: Dict[str, str] = {}
        try:
            entries = DEEPECG_DIAGNOSIS_TRANSLATION.get('deepecg', [])
            for item in entries:
                name = str(item.get('column_name', '')).strip().lower()
                trans = str(item.get('translation_en', name)).strip().lower()
                if name:
                    tr_map[name] = trans
        except Exception:
            pass
        # Common aliases
        tr_map.setdefault('sinusal', 'sinus rhythm')
        tr_map.setdefault('afib', 'atrial fibrillation')
        tr_map.setdefault('stemi', 'acute mi')
        return tr_map

    @staticmethod
    def _normalize_label(label: str, tr_map: Dict[str, str]) -> str:
        s = (label or '').strip().lower()
        s = _re.sub(r"[\s\-_/]+", " ", s)
        s = s.replace("’", "'").replace("`", "'")
        if s in tr_map:
            return tr_map[s]
        return s

    @staticmethod
    def _collect_items(obj: Union[dict, list, str, int, float], tr_map: Dict[str, str]) -> List[str]:
        items: List[str] = []
        if isinstance(obj, dict):
            for _, v in obj.items():
                items.extend(StructuredSlotF1Metric._collect_items(v, tr_map))
        elif isinstance(obj, list):
            for v in obj:
                items.extend(StructuredSlotF1Metric._collect_items(v, tr_map))
        elif isinstance(obj, (str, int, float)):
            s = str(obj)
            # Split comma-separated lists cautiously
            parts = [p.strip() for p in s.split(',') if p.strip()]
            if not parts:
                parts = [s.strip()]
            for p in parts:
                if p:
                    items.append(StructuredSlotF1Metric._normalize_label(p, tr_map))
        return items

    @staticmethod
    def _unify_keys(k: str) -> str:
        key = (k or '').strip().upper()
        key = key.replace('INFARCT_ISCHEMIA', 'INFARCT, ISCHEMIA')
        key = key.replace('ISCHEMIA', 'INFARCT, ISCHEMIA') if 'INFARCT' in key else key
        return key

    @staticmethod
    def _extract_slots(json_obj: dict, tr_map: Dict[str, str]) -> List[str]:
        all_items: List[str] = []
        try:
            for k, v in json_obj.items():
                key = StructuredSlotF1Metric._unify_keys(k)
                if key in DEEPECG_CATEGORIES:
                    all_items.extend(StructuredSlotF1Metric._collect_items(v, tr_map))
        except Exception:
            pass
        # Deduplicate
        seen: set[str] = set()
        out: List[str] = []
        for it in all_items:
            if it and it not in seen:
                seen.add(it)
                out.append(it)
        return out

    @staticmethod
    def _try_load_json(text: str) -> Optional[dict]:
        if not text:
            return None
        try:
            obj = _json.loads(text)
            return obj if isinstance(obj, dict) else None
        except Exception:
            # Try to trim to bracketed JSON
            start_obj = text.find('{')
            end_obj = text.rfind('}')
            if start_obj != -1 and end_obj != -1 and end_obj > start_obj:
                try:
                    obj = _json.loads(text[start_obj:end_obj + 1])
                    return obj if isinstance(obj, dict) else None
                except Exception:
                    return None
            return None

    @staticmethod
    def compute_score(
        generated_ids: torch.Tensor,
        labels: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase,
        input_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, Union[float, List[str]]]:
        tr_map = StructuredSlotF1Metric._build_translation_map()
        tp = 0
        fp = 0
        fn = 0
        predictions_out: List[str] = []
        references_out: List[str] = []

        for i, (gen, lab) in enumerate(zip(generated_ids, labels)):
            ii = input_ids[i] if input_ids is not None else None
            pred_text, ref_text = decode_assistant_only_text(tokenizer, gen, lab, ii)
            predictions_out.append(pred_text)
            references_out.append(ref_text)
            pred_obj = StructuredSlotF1Metric._try_load_json(pred_text)
            ref_obj = StructuredSlotF1Metric._try_load_json(ref_text)
            if pred_obj is None or ref_obj is None:
                # If either side isn't structured, skip this sample for F1
                continue
            pred_items = set(StructuredSlotF1Metric._extract_slots(pred_obj, tr_map))
            ref_items = set(StructuredSlotF1Metric._extract_slots(ref_obj, tr_map))
            tp += len(pred_items & ref_items)
            fp += len(pred_items - ref_items)
            fn += len(ref_items - pred_items)

        precision = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        recall = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

        return {
            "structured_precision": precision,
            "structured_recall": recall,
            "structured_f1": f1,
            "predictions": predictions_out,
            "references": references_out,
        }


def update_best_metric(
    metric_name: str,
    llm_metrics: Dict[str, Union[float, List[str]]],
    best_metrics: Dict[str, List[Dict[str, Union[float, List[str]]]]],
    K: int = 5
) -> None:
    """
    Update the best (highest-scoring) entries for a given metric.
    The entry (score, predictions, references) is added or replaces an existing entry
    in the provided best_metrics dictionary.
    """
    score = llm_metrics[metric_name]
    predictions = llm_metrics["predictions"]
    references = llm_metrics["references"]
    prompts = llm_metrics.get("prompts", [])
    
    assert isinstance(score, float), f"Expected float for {metric_name}, got {type(score)}"
    assert isinstance(predictions, list), f"Expected list for predictions, got {type(predictions)}"
    assert isinstance(references, list), f"Expected list for references, got {type(references)}"

    entry: Dict[str, Union[float, List[str]]] = {
        "score": score, 
        "predictions": predictions, 
        "references": references,
        "prompts": prompts
    }
    # Get or create the list for the metric
    best_list: List[Dict[str, Union[float, List[str]]]] = best_metrics.setdefault(metric_name, [])
    
    if len(best_list) < K:
        best_list.append(entry)
    else:
        best_list.sort(key=lambda x: x["score"], reverse=True)
        # Replace if the new score is higher than the smallest among the best entries
        last_score = best_list[-1]["score"]
        assert isinstance(last_score, float)
        if score > last_score:
            best_list[-1] = entry
    
    best_list.sort(key=lambda x: x["score"], reverse=True)


def update_worst_metric(
    metric_name: str,
    llm_metrics: Dict[str, Union[float, List[str]]],
    worst_metrics: Dict[str, List[Dict[str, Union[float, List[str]]]]],
    K: int = 5
) -> None:
    """
    Update the worst (lowest-scoring) entries for a given metric.
    The entry (score, predictions, references) is added or replaces an existing entry
    in the provided worst_metrics dictionary.
    """
    score = llm_metrics[metric_name]
    predictions = llm_metrics["predictions"]
    references = llm_metrics["references"]
    prompts = llm_metrics.get("prompts", [])
    
    assert isinstance(score, float), f"Expected float for {metric_name}, got {type(score)}"
    assert isinstance(predictions, list), f"Expected list for predictions, got {type(predictions)}"
    assert isinstance(references, list), f"Expected list for references, got {type(references)}"

    entry: Dict[str, Union[float, List[str]]] = {
        "score": score, 
        "predictions": predictions, 
        "references": references,
        "prompts": prompts
    }
    worst_list = worst_metrics.setdefault(metric_name, [])
    
    if len(worst_list) < K:
        worst_list.append(entry)
    else:
        worst_list.sort(key=lambda x: x["score"])
        # Replace if the new score is lower than the highest in the worst list
        last_score = worst_list[-1]["score"]
        assert isinstance(last_score, float)
        if score < last_score:
            worst_list[-1] = entry
    
    worst_list.sort(key=lambda x: x["score"])


def update_random_batch_metric(
    metric_name: str,
    llm_metrics: Dict[str, Union[float, List[str]]],
    random_metrics: Dict[str, List[Dict[str, Union[float, List[str]]]]],
    k: int = 5
) -> None:
    """
    Update the random batch metric.
    """
    score = llm_metrics[metric_name]
    predictions = llm_metrics["predictions"]
    references = llm_metrics["references"]
    prompts = llm_metrics.get("prompts", [])
    
    assert isinstance(score, float), f"Expected float for {metric_name}, got {type(score)}"
    assert isinstance(predictions, list), f"Expected list for predictions, got {type(predictions)}"
    assert isinstance(references, list), f"Expected list for references, got {type(references)}"
    
    entry: Dict[str, Union[float, List[str]]] = {
        "score": score, 
        "predictions": predictions, 
        "references": references,
        "prompts": prompts
    }
    
    # Get or create the list for the metric
    random_list: List[Dict[str, Union[float, List[str]]]] = random_metrics.setdefault(metric_name, [])
    
    if len(random_list) < k:
        random_list.append(entry)


# Enhanced offline evaluation functions
def compute_bertscore_offline(
    predictions: List[str],
    references: List[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    batch_size: int = 32,
    device: Optional[str] = None
) -> Dict[str, float]:
    """
    Compute BERTScore directly from text strings using bert_score library.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        model_type: BERTScore model to use
        batch_size: Batch size for computation
        device: Device to run on ('cuda' or 'cpu')
    
    Returns:
        Dictionary with precision, recall, and F1 scores
    """
    try:
        from bert_score import score
    except ImportError:
        warnings.warn("bert-score not installed. Install with: pip install bert-score")
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Filter empty strings
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
    if not valid_pairs:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}
    
    valid_preds, valid_refs = zip(*valid_pairs)
    
    P, R, F1 = score(
        list(valid_preds),
        list(valid_refs),
        lang='en',
        model_type=model_type,
        verbose=False,
        device=device,
        batch_size=batch_size
    )
    
    return {
        "precision": float(torch.mean(P).item()),
        "recall": float(torch.mean(R).item()),
        "f1": float(torch.mean(F1).item()),
        "n_samples": len(valid_preds)
    }


def compute_bertscore_by_category(
    predictions: List[str],
    references: List[str],
    categories: List[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    batch_size: int = 32,
    min_samples: int = 5
) -> Dict[str, Dict[str, float]]:
    """
    Compute BERTScore grouped by category.
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        categories: List of category labels (same length as predictions)
        model_type: BERTScore model to use
        batch_size: Batch size for computation
        min_samples: Minimum samples per category to compute score
    
    Returns:
        Dictionary mapping category to scores
    """
    from collections import defaultdict
    
    # Group by category
    category_data = defaultdict(lambda: {'predictions': [], 'references': []})
    for pred, ref, cat in zip(predictions, references, categories):
        if pred.strip() and ref.strip():
            category_data[cat]['predictions'].append(pred)
            category_data[cat]['references'].append(ref)
    
    results = {}
    for category, data in category_data.items():
        if len(data['predictions']) >= min_samples:
            scores = compute_bertscore_offline(
                data['predictions'],
                data['references'],
                model_type=model_type,
                batch_size=batch_size
            )
            results[category] = scores
    
    return results


def compute_bertscore_by_dataset(
    predictions: List[str],
    references: List[str],
    filenames: List[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    batch_size: int = 32
) -> Dict[str, Dict[str, float]]:
    """
    Compute BERTScore grouped by dataset (MIMIC vs MHI based on filename pattern).
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts  
        filenames: List of filenames to identify dataset
        model_type: BERTScore model to use
        batch_size: Batch size for computation
    
    Returns:
        Dictionary with MIMIC and MHI scores
    """
    mimic_data = {'predictions': [], 'references': []}
    mhi_data = {'predictions': [], 'references': []}
    
    for pred, ref, fname in zip(predictions, references, filenames):
        if pred.strip() and ref.strip():
            # Identify dataset based on filename pattern
            if fname.startswith('4'):
                mimic_data['predictions'].append(pred)
                mimic_data['references'].append(ref)
            elif fname.startswith('0'):
                mhi_data['predictions'].append(pred)
                mhi_data['references'].append(ref)
    
    results = {}
    
    if mimic_data['predictions']:
        results['MIMIC'] = compute_bertscore_offline(
            mimic_data['predictions'],
            mimic_data['references'],
            model_type=model_type,
            batch_size=batch_size
        )
    
    if mhi_data['predictions']:
        results['MHI'] = compute_bertscore_offline(
            mhi_data['predictions'],
            mhi_data['references'],
            model_type=model_type,
            batch_size=batch_size
        )
    
    return results


def compute_metrics_for_texts(
    predictions: List[str],
    references: List[str],
    metrics: List[str] = ["rouge", "bleu", "meteor", "bertscore"],
    categories: Optional[List[str]] = None,
    filenames: Optional[List[str]] = None,
    compute_category_scores: bool = False,
    compute_dataset_scores: bool = False
) -> Dict[str, Union[float, Dict]]:
    """
    Compute metrics directly from text strings (for offline evaluation).
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        metrics: List of metric names to compute
        categories: Optional list of category labels
        filenames: Optional list of filenames for dataset identification
        compute_category_scores: Whether to compute per-category scores
        compute_dataset_scores: Whether to compute per-dataset scores
    
    Returns:
        Dictionary of metric scores
    """
    results = {"overall": {}}
    
    # Compute overall metrics
    for metric in metrics:
        if metric == "rouge":
            try:
                from evaluate import load as load_metric
                rouge_metric = load_metric("rouge")
                rouge_scores = rouge_metric.compute(
                    predictions=predictions,
                    references=references,
                    use_stemmer=True,
                    use_aggregator=True
                )
                results["overall"]["rouge1"] = float(rouge_scores["rouge1"])
                results["overall"]["rougeL"] = float(rouge_scores["rougeL"])
            except Exception as e:
                warnings.warn(f"ROUGE computation failed: {e}")
                results["overall"]["rouge1"] = 0.0
                results["overall"]["rougeL"] = 0.0
                
        elif metric == "bleu":
            try:
                import sacrebleu
                refs_nested = [references]
                bleu4 = sacrebleu.corpus_bleu(predictions, refs_nested).score / 100.0
                bleu1_scorer = sacrebleu.BLEU(max_ngram_order=1, effective_order=True)
                bleu1 = bleu1_scorer.corpus_score(predictions, refs_nested).score / 100.0
                results["overall"]["bleu1"] = float(bleu1)
                results["overall"]["bleu4"] = float(bleu4)
            except Exception as e:
                warnings.warn(f"BLEU computation failed: {e}")
                results["overall"]["bleu1"] = 0.0
                results["overall"]["bleu4"] = 0.0
                
        elif metric == "meteor":
            try:
                from evaluate import load as load_metric
                meteor_metric = load_metric("meteor")
                meteor_scores = meteor_metric.compute(
                    predictions=predictions,
                    references=references
                )
                results["overall"]["meteor"] = float(meteor_scores["meteor"])
            except Exception as e:
                warnings.warn(f"METEOR computation failed: {e}")
                results["overall"]["meteor"] = 0.0
                
        elif metric == "bertscore":
            bert_scores = compute_bertscore_offline(predictions, references)
            results["overall"]["bertscore_precision"] = bert_scores["precision"]
            results["overall"]["bertscore_recall"] = bert_scores["recall"]
            results["overall"]["bertscore_f1"] = bert_scores["f1"]
    
    # Compute category-specific scores if requested
    if compute_category_scores and categories is not None:
        results["per_category"] = compute_bertscore_by_category(
            predictions, references, categories
        )
    
    # Compute dataset-specific scores if requested
    if compute_dataset_scores and filenames is not None:
        results["per_dataset"] = compute_bertscore_by_dataset(
            predictions, references, filenames
        )
    
    return results


def analyze_generation_results(
    json_path: str,
    csv_path: Optional[str] = None,
    output_path: str = "metric_analysis.json",
    metrics: List[str] = ["bertscore"],
    verbose: bool = True
) -> Dict:
    """
    Analyze generation results from JSON file with optional CSV metadata.
    
    Args:
        json_path: Path to generation JSON file
        csv_path: Optional path to CSV with metadata
        output_path: Path to save analysis results
        metrics: List of metrics to compute
        verbose: Whether to print analysis
    
    Returns:
        Dictionary with analysis results
    """
    import json
    import pandas as pd
    from pathlib import Path
    
    # Load JSON data
    with open(json_path, 'r') as f:
        json_data = json.load(f)
    
    # Extract predictions and references
    predictions = []
    references = []
    filenames = []
    
    for filename, entry in json_data.items():
        pred = entry.get('Generation', '')
        ref = entry.get('Ground truth', '')
        if pred and ref:
            predictions.append(pred)
            references.append(ref)
            filenames.append(filename)
    
    # Load CSV metadata if provided
    categories = None
    if csv_path and Path(csv_path).exists():
        csv_data = pd.read_csv(csv_path, low_memory=False)
        csv_mapping = {}
        for _, row in csv_data.iterrows():
            if pd.notna(row.get('waveform_name')):
                fname = str(row['waveform_name'])
                csv_mapping[fname] = row.get('prompt_category', 'unknown') or 'unknown'
        
        categories = [csv_mapping.get(f, 'unknown') for f in filenames]
    
    # Compute metrics
    results = compute_metrics_for_texts(
        predictions=predictions,
        references=references,
        metrics=metrics,
        categories=categories,
        filenames=filenames,
        compute_category_scores=(categories is not None),
        compute_dataset_scores=True
    )
    
    # Add metadata
    results['metadata'] = {
        'total_samples': len(predictions),
        'json_path': json_path,
        'csv_path': csv_path
    }
    
    # Save results
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"\n{'='*60}")
        print("GENERATION ANALYSIS RESULTS")
        print(f"{'='*60}")
        print(f"Total samples: {len(predictions)}")
        
        if 'overall' in results:
            print("\nOverall Metrics:")
            for metric, score in results['overall'].items():
                print(f"  {metric}: {score:.4f}")
        
        if 'per_dataset' in results:
            print("\nPer-Dataset BERTScore:")
            for dataset, scores in results['per_dataset'].items():
                print(f"  {dataset}: F1={scores['f1']:.4f} (n={scores.get('n_samples', 0)})")
        
        print(f"\nResults saved to {output_path}")
    
    return results
