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
    generated_only_new_tokens: bool = True,
) -> Tuple[str, str]:
    """Decode assistant-only text for predictions and references.

    Shared helper so both runner logging and metric eval stay in sync.
    Drops prompt tokens (where labels are -100) from both the generated output and
    the reference so that metrics focus on the assistant response.
    
    IMPORTANT: When generation uses `inputs_embeds` (as MedGemma does), HuggingFace's
    generate() returns ONLY the newly generated tokens, NOT the prompt. Set
    `generated_only_new_tokens=True` (default) in this case to skip prompt trimming.
    
    Args:
        tokenizer: Tokenizer instance
        generated: Generated token ids (may be new tokens only OR full sequence with prompt)
        labels: Label token ids with -100 for prompt tokens (aligned with input_ids, NOT generated)
        input_ids: Original input token ids (question + answer, optional)
        prompt_input_ids: Question-only input ids used for generation (optional but recommended)
        num_ecg_tokens: Number of ECG tokens injected into the sequence (default 0).
        generated_only_new_tokens: If True (default), assume generated contains only new tokens
            (as when using inputs_embeds). If False, assume generated contains prompt + new tokens
            (as when using input_ids for generation).
    
    Returns:
        Tuple of (prediction_text, reference_text) with normalized whitespace
    """
    label_tensor = labels.detach().cpu()
    generated_tensor = generated.detach().cpu()

    # Reference: drop -100s (prompt tokens) - this is correct regardless of ECG injection
    ref_ids = label_tensor[label_tensor != -100].tolist()

    # Prediction: get the generated token ids
    gen_ids = generated_tensor.tolist()
    
    # When using inputs_embeds for generation (MedGemma, most modern VLMs),
    # HF generate() returns ONLY the newly generated tokens, NOT the prompt.
    # In this case, we should NOT trim anything - just decode as-is.
    #
    # When using input_ids for generation (legacy path),
    # HF generate() returns prompt + new tokens, so we need to trim the prompt.
    
    if not generated_only_new_tokens:
        # Legacy path: generated contains prompt + new tokens, need to trim
        expected_prompt_len = 0
        if prompt_input_ids is not None:
            expected_prompt_len = int(prompt_input_ids.shape[-1]) + int(num_ecg_tokens)
        elif input_ids is not None:
            label_list = label_tensor.tolist()
            first_answer_idx = next((i for i, t in enumerate(label_list) if t != -100), 0)
            expected_prompt_len = first_answer_idx + int(num_ecg_tokens)
        
        if expected_prompt_len > 0 and len(gen_ids) > expected_prompt_len:
            gen_ids = gen_ids[expected_prompt_len:]
    # else: generated_only_new_tokens=True, use gen_ids as-is (no trimming needed)

    # Decode and normalize text (strip + collapse whitespace)
    pred = " ".join(tokenizer.decode(gen_ids, skip_special_tokens=True).strip().split())
    ref = " ".join(tokenizer.decode(ref_ids, skip_special_tokens=True).strip().split())
    
    return pred, ref


@MetricRegistry.register("rouge")
class RougeMetric:
    @staticmethod
    def compute_score(
        generated_ids: torch.Tensor,
        labels: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase,
        input_ids: Optional[torch.Tensor] = None,
        prompt_input_ids: Optional[torch.Tensor] = None,
        num_ecg_tokens: int = 0
    ) -> Dict[str, Union[float, List[str]]]:
        from utils.metrics.aggregate_text_metrics import aggregate_text_metrics

        predictions = []
        references = []
        for i, (gen, lab) in enumerate(zip(generated_ids, labels)):
            ii = input_ids[i] if input_ids is not None else None
            pi = prompt_input_ids[i] if prompt_input_ids is not None else None
            pred, ref = decode_assistant_only_text(tokenizer, gen, lab, ii, pi, num_ecg_tokens, generated_only_new_tokens=True)
            predictions.append(pred)
            references.append(ref)

        agg = aggregate_text_metrics(predictions, references)
        return {
            "rouge1": agg.get("rouge1", 0.0),
            "rouge2": agg.get("rouge2", 0.0),
            "rougeL": agg.get("rougeL", 0.0),
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
        input_ids: Optional[torch.Tensor] = None,
        prompt_input_ids: Optional[torch.Tensor] = None,
        num_ecg_tokens: int = 0
    ) -> Dict[str, Union[float, List[str]]]:
        from utils.metrics.aggregate_text_metrics import aggregate_text_metrics

        predictions = []
        references = []
        for i, (gen, lab) in enumerate(zip(generated_ids, labels)):
            ii = input_ids[i] if input_ids is not None else None
            pi = prompt_input_ids[i] if prompt_input_ids is not None else None
            pred, ref = decode_assistant_only_text(tokenizer, gen, lab, ii, pi, num_ecg_tokens, generated_only_new_tokens=True)
            if pred.strip() and ref.strip():
                predictions.append(pred)
                references.append(ref)

        if not predictions:
            return {"bleu1": 0.0, "bleu4": 0.0, "predictions": [], "references": []}

        agg = aggregate_text_metrics(predictions, references)
        return {
            "bleu1": agg.get("bleu1", 0.0),
            "bleu4": agg.get("bleu4", 0.0),
            "predictions": predictions,
            "references": references
        }


@MetricRegistry.register("meteor")
class MeteorMetric:
    @staticmethod
    def compute_score(
        generated_ids: torch.Tensor,
        labels: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase,
        input_ids: Optional[torch.Tensor] = None,
        prompt_input_ids: Optional[torch.Tensor] = None,
        num_ecg_tokens: int = 0
    ) -> Dict[str, Union[float, List[str]]]:
        from utils.metrics.aggregate_text_metrics import aggregate_text_metrics

        predictions = []
        references = []
        for i, (gen, lab) in enumerate(zip(generated_ids, labels)):
            ii = input_ids[i] if input_ids is not None else None
            pi = prompt_input_ids[i] if prompt_input_ids is not None else None
            pred, ref = decode_assistant_only_text(tokenizer, gen, lab, ii, pi, num_ecg_tokens, generated_only_new_tokens=True)
            predictions.append(pred)
            references.append(ref)

        agg = aggregate_text_metrics(predictions, references)
        return {
            "meteor": agg.get("meteor", 0.0),
            "predictions": predictions,
            "references": references
        }


@MetricRegistry.register("bertscore")
class BertScoreMetric:
    _metric = None
    _default_model_type = "microsoft/deberta-xlarge-mnli"

    @staticmethod
    def compute_score(
        generated_ids: torch.Tensor,
        labels: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase,
        input_ids: Optional[torch.Tensor] = None,
        prompt_input_ids: Optional[torch.Tensor] = None,
        num_ecg_tokens: int = 0,
        lang: str = "en",
        model_type: Optional[str] = None,
        rescale_with_baseline: bool = True,
        idf: bool = False,
        device: Optional[str] = None
    ) -> Dict[str, Union[float, List[str]]]:
        if BertScoreMetric._metric is None:
            try:
                from evaluate import load as load_metric
                BertScoreMetric._metric = load_metric("bertscore")
            except Exception:
                return {"hf-prec": 0.0, "hf-rec": 0.0, "hf-f1": 0.0, "predictions": [], "references": []}

        predictions = []
        references = []
        for i, (gen, lab) in enumerate(zip(generated_ids, labels)):
            ii = input_ids[i] if input_ids is not None else None
            pi = prompt_input_ids[i] if prompt_input_ids is not None else None
            pred, ref = decode_assistant_only_text(tokenizer, gen, lab, ii, pi, num_ecg_tokens, generated_only_new_tokens=True)
            predictions.append(pred)
            references.append(ref)

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
            prec = float(np.mean(results["precision"]))
            rec = float(np.mean(results["recall"]))
            f1 = float(np.mean(results["f1"]))
        except Exception:
            prec, rec, f1 = 0.0, 0.0, 0.0

        return {"hf-prec": prec, "hf-rec": rec, "hf-f1": f1, "predictions": predictions, "references": references}


@MetricRegistry.register("json_parse_rate")
class JsonParseRateMetric:
    @staticmethod
    def compute_score(
        generated_ids: torch.Tensor,
        labels: torch.Tensor,
        tokenizer: PreTrainedTokenizerBase,
        input_ids: Optional[torch.Tensor] = None,
        prompt_input_ids: Optional[torch.Tensor] = None,
        num_ecg_tokens: int = 0
    ) -> Dict[str, Union[float, List[str]]]:
        parsed = 0
        total = 0
        predictions: List[str] = []
        references: List[str] = []
        for i, (gen, lab) in enumerate(zip(generated_ids, labels)):
            ii = input_ids[i] if input_ids is not None else None
            pi = prompt_input_ids[i] if prompt_input_ids is not None else None
            pred, ref = decode_assistant_only_text(tokenizer, gen, lab, ii, pi, num_ecg_tokens, generated_only_new_tokens=True)
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
        input_ids: Optional[torch.Tensor] = None,
        prompt_input_ids: Optional[torch.Tensor] = None,
        num_ecg_tokens: int = 0
    ) -> Dict[str, Union[float, List[str]]]:
        tr_map = StructuredSlotF1Metric._build_translation_map()
        tp = 0
        fp = 0
        fn = 0
        predictions_out: List[str] = []
        references_out: List[str] = []

        for i, (gen, lab) in enumerate(zip(generated_ids, labels)):
            ii = input_ids[i] if input_ids is not None else None
            pi = prompt_input_ids[i] if prompt_input_ids is not None else None
            pred_text, ref_text = decode_assistant_only_text(tokenizer, gen, lab, ii, pi, num_ecg_tokens, generated_only_new_tokens=True)
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


def compute_bertscore_offline(
    predictions: List[str],
    references: List[str],
    model_type: str = "microsoft/deberta-xlarge-mnli",
    batch_size: int = 32,
    device: Optional[str] = None
) -> Dict[str, float]:
    """Compute BERTScore from text strings using bert_score library."""
    try:
        from bert_score import score
    except ImportError:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
