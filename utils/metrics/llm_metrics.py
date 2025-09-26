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
from typing import Dict, List, Tuple, Union, Optional
import torch
import numpy as np
from transformers import PreTrainedTokenizerBase

from utils.registry import MetricRegistry


def decode_assistant_only_text(
    tokenizer: PreTrainedTokenizerBase,
    generated: torch.Tensor,
    labels: torch.Tensor,
    input_ids: Optional[torch.Tensor] = None
) -> Tuple[str, str]:
    """Decode assistant-only text for predictions and references.

    Shared helper so both runner logging and metric eval stay in sync.
    Drops prompt tokens (where labels are -100) from both the generated output and
    the reference so that metrics focus on the assistant response.
    
    Args:
        tokenizer: Tokenizer instance
        generated: Generated token ids
        labels: Label token ids with -100 for prompt tokens
        input_ids: Original input token ids (optional, for better prompt trimming)
    
    Returns:
        Tuple of (prediction_text, reference_text) with normalized whitespace
    """
    label_tensor = labels.detach().cpu()
    generated_tensor = generated.detach().cpu()

    # Reference: drop -100s (prompt tokens)
    ref_ids = label_tensor[label_tensor != -100].tolist()

    # Prediction: trim prompt if we know its exact length
    gen_ids = generated_tensor.tolist()
    
    if input_ids is not None:
        # Use exact prompt length when available
        prompt_len = int(input_ids.shape[-1])
        if len(gen_ids) >= prompt_len:
            gen_ids = gen_ids[prompt_len:]
    else:
        # Fallback: find first non--100 index in labels
        first_idx = next((i for i, t in enumerate(label_tensor.tolist()) if t != -100), 0)
        if len(gen_ids) >= first_idx:
            gen_ids = gen_ids[first_idx:]

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


# Compatibility functions for offline evaluation
def compute_metrics_for_texts(
    predictions: List[str],
    references: List[str],
    metrics: List[str] = ["rouge", "bleu", "meteor", "bertscore"]
) -> Dict[str, float]:
    """
    Compute metrics directly from text strings (for offline evaluation).
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        metrics: List of metric names to compute
    
    Returns:
        Dictionary of metric scores
    """
    results = {}
    
    # Create dummy tokenizer for compatibility
    from transformers import AutoTokenizer
    dummy_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    
    # Tokenize texts to create dummy tensors
    pred_ids = [dummy_tokenizer.encode(p, add_special_tokens=False) for p in predictions]
    ref_ids = [dummy_tokenizer.encode(r, add_special_tokens=False) for r in references]
    
    # Pad to same length
    max_len = max(max(len(p) for p in pred_ids), max(len(r) for r in ref_ids))
    pred_ids = [p + [dummy_tokenizer.pad_token_id] * (max_len - len(p)) for p in pred_ids]
    ref_ids = [r + [-100] * (max_len - len(r)) for r in ref_ids]
    
    # Convert to tensors
    pred_tensor = torch.tensor(pred_ids)
    ref_tensor = torch.tensor(ref_ids)
    
    for metric in metrics:
        if metric == "rouge":
            metric_obj = RougeMetric()
            scores = metric_obj.compute_score(pred_tensor, ref_tensor, dummy_tokenizer)
            results["rouge1"] = scores["rouge1"]
            results["rougeL"] = scores["rougeL"]
        elif metric == "bleu":
            metric_obj = SacreBleuMetric()
            scores = metric_obj.compute_score(pred_tensor, ref_tensor, dummy_tokenizer)
            results["bleu1"] = scores["bleu1"]
            results["bleu4"] = scores["bleu4"]
        elif metric == "meteor":
            metric_obj = MeteorMetric()
            scores = metric_obj.compute_score(pred_tensor, ref_tensor, dummy_tokenizer)
            results["meteor"] = scores["meteor"]
        elif metric == "bertscore":
            metric_obj = BertScoreMetric()
            scores = metric_obj.compute_score(pred_tensor, ref_tensor, dummy_tokenizer)
            results["bertscore_precision"] = scores["hf-prec"]
            results["bertscore_recall"] = scores["hf-rec"]
            results["bertscore_f1"] = scores["hf-f1"]
    
    return results