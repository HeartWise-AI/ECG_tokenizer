"""Reward functions for GRPO training.

Each reward function takes (generated_text, ground_truth) and returns a float
in [0.0, 1.0]. The ``compute_rewards`` function computes a weighted combination.
"""

import re
from typing import Dict, Optional

import torch


class BertDiagnosisReward:
    """Hybrid diagnosis reward: max(Jaccard, BERT per-class-threshold F1).

    Uses ``heartwise/Bert_diagnosis2classification_En_Fr`` (77-class multi-label
    BERT) with **per-class optimized thresholds** from ``utils.constants``.
    Each of the 77 classes has its own calibrated sigmoid threshold, avoiding
    the problem of a single global threshold being too lenient for some classes
    and too strict for others.

    Final score = max(Jaccard on raw label strings, BERT F1 on thresholded preds).
    """

    # Classes that fire on nearly every ECG and create spurious F1 overlap.
    # 0=Sinusal, 1=Regular, 2=Monomorph — these are noise, not diagnoses.
    IGNORED_CLASSES = {0, 1, 2}

    def __init__(self, device: str = "cuda"):
        from transformers import BertTokenizer, BertForSequenceClassification
        from utils.constants import BERT_CLASS_THRESHOLDS

        model_name = "heartwise/Bert_diagnosis2classification_En_Fr"
        self.device = torch.device(device)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=77
        ).to(self.device)
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        # Per-class threshold tensor [77]
        self.thresholds = torch.tensor(
            BERT_CLASS_THRESHOLDS, dtype=torch.float32, device=self.device
        )
        # Mask: 1.0 for classes we care about, 0.0 for noisy ones
        self.class_mask = torch.ones(77, dtype=torch.float32, device=self.device)
        for idx in self.IGNORED_CLASSES:
            self.class_mask[idx] = 0.0

    @torch.no_grad()
    def _predict(self, text: str) -> torch.Tensor:
        """Return sigmoid probabilities [77] for a text."""
        inputs = self.tokenizer(
            text, padding="max_length", max_length=512,
            truncation=True, return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        logits = self.model(**inputs)["logits"]
        return torch.sigmoid(logits[0])

    def _cosine_sim(self, a: torch.Tensor, b: torch.Tensor) -> float:
        """Cosine similarity on masked probability vectors."""
        a_m = a * self.class_mask
        b_m = b * self.class_mask
        dot = (a_m * b_m).sum()
        norm = torch.norm(a_m) * torch.norm(b_m)
        if norm < 1e-8:
            return 0.0
        return float((dot / norm).cpu())

    def __call__(self, generated_text: str, ground_truth: str) -> float:
        """BERT-based diagnosis reward with per-class thresholds.

        Logic:
        - If neither text activates any BERT class → Jaccard only
        - If TP=0 and FN>0 (missed everything) → soft cosine on BERT probs
          (gives continuous signal for bootstrapping GRPO)
        - Otherwise → average of Jaccard and BERT F1
        """
        gen_block = _extract_answer_block(generated_text)
        gt_block = _extract_answer_block(ground_truth)
        if gen_block is None or gt_block is None:
            return 0.0

        jaccard = _jaccard_on_blocks(gen_block, gt_block)

        gen_probs = self._predict(gen_block)
        gt_probs = self._predict(gt_block)

        gen_preds = (gen_probs > self.thresholds).float() * self.class_mask
        gt_preds = (gt_probs > self.thresholds).float() * self.class_mask

        tp = (gen_preds * gt_preds).sum()
        fp = (gen_preds * (1 - gt_preds)).sum()
        fn = ((1 - gen_preds) * gt_preds).sum()

        if (tp + fp + fn) < 1e-8:
            return jaccard

        if tp < 1e-8 and fn > 0:
            # Soft fallback: cosine similarity on raw BERT probs (continuous signal)
            # Scale by 0.5 so it stays below a real TP match but above hard zero
            cos = self._cosine_sim(gen_probs, gt_probs)
            return cos * 0.5

        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = float((2 * precision * recall / (precision + recall + 1e-8)).cpu())

        return (jaccard + f1) / 2.0


def format_reward(generated_text: str, ground_truth: str) -> float:
    """Return 1.0 if the generated text contains <think>, </think>, and <answer> tags."""
    has_think_open = "<think>" in generated_text
    has_think_close = "</think>" in generated_text
    has_answer = "<answer>" in generated_text
    return 1.0 if (has_think_open and has_think_close and has_answer) else 0.0


def _extract_answer_block(text: str) -> Optional[str]:
    """Extract content between <answer> and </answer> tags."""
    match = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    # Fallback: if <answer> present but no closing tag, take everything after
    match = re.search(r"<answer>(.*)", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return None


def _jaccard_on_blocks(gen_block: str, gt_block: str) -> float:
    """Jaccard similarity between semicolon-separated labels in two text blocks."""
    gen_labels = _parse_diagnosis_labels(gen_block)
    gt_labels = _parse_diagnosis_labels(gt_block)
    if not gen_labels and not gt_labels:
        return 1.0
    if not gen_labels or not gt_labels:
        return 0.0
    return len(gen_labels & gt_labels) / len(gen_labels | gt_labels)


def _parse_diagnosis_labels(answer_block: str) -> set:
    """Parse semicolon-separated diagnosis labels, normalizing whitespace and case."""
    labels = set()
    for label in answer_block.split(";"):
        label = label.strip().lower()
        if label:
            labels.add(label)
    return labels


def diagnosis_accuracy_reward(generated_text: str, ground_truth: str) -> float:
    """Compute Jaccard similarity between diagnosis labels in <answer> blocks."""
    gen_block = _extract_answer_block(generated_text)
    gt_block = _extract_answer_block(ground_truth)
    if gen_block is None or gt_block is None:
        return 0.0
    gen_labels = _parse_diagnosis_labels(gen_block)
    gt_labels = _parse_diagnosis_labels(gt_block)
    if not gen_labels and not gt_labels:
        return 1.0
    if not gen_labels or not gt_labels:
        return 0.0
    intersection = gen_labels & gt_labels
    union = gen_labels | gt_labels
    return len(intersection) / len(union)


# CoT step headers expected in the reasoning trace
_COT_STEP_HEADERS = [
    "Step 1:",
    "Step 2:",
    "Step 3:",
    "Step 4:",
    "Step 5:",
    "Step 6:",
]


def key_evidence_reward(generated_text: str, ground_truth: str) -> float:
    """Check what fraction of CoT step headers have non-empty content after them."""
    if not generated_text:
        return 0.0
    steps_with_content = 0
    for i, header in enumerate(_COT_STEP_HEADERS):
        pos = generated_text.find(header)
        if pos < 0:
            continue
        # Find content after this header up to next header or end
        content_start = pos + len(header)
        content_end = len(generated_text)
        for next_header in _COT_STEP_HEADERS[i + 1:]:
            next_pos = generated_text.find(next_header, content_start)
            if next_pos >= 0:
                content_end = next_pos
                break
        # Also stop at </think> if present
        think_end = generated_text.find("</think>", content_start)
        if think_end >= 0 and think_end < content_end:
            content_end = think_end
        content = generated_text[content_start:content_end].strip()
        if content:
            steps_with_content += 1
    total_headers = len(_COT_STEP_HEADERS)
    return steps_with_content / total_headers


def compute_rewards(
    generated_text: str,
    ground_truth: str,
    weights: Optional[Dict[str, float]] = None,
    bert_reward: Optional["BertDiagnosisReward"] = None,
) -> Dict[str, float]:
    """Compute weighted combination of all reward signals.

    Args:
        generated_text: Model-generated text.
        ground_truth: Ground truth text with expected format.
        weights: Dict mapping reward name to weight.
            Keys: "format", "diagnosis", "evidence".
        bert_reward: Optional BertDiagnosisReward instance. When provided,
            uses BERT F1 for the diagnosis component instead of Jaccard.

    Returns:
        Dict with keys "total", "format", "diagnosis", "evidence".
    """
    if weights is None:
        weights = {"format": 0.2, "diagnosis": 0.5, "evidence": 0.3}

    r_format = format_reward(generated_text, ground_truth)
    if bert_reward is not None:
        r_diagnosis = bert_reward(generated_text, ground_truth)
    else:
        r_diagnosis = diagnosis_accuracy_reward(generated_text, ground_truth)
    r_evidence = key_evidence_reward(generated_text, ground_truth)

    total_weight = sum(weights.values())
    if total_weight <= 0:
        total = 0.0
    else:
        total = (
            weights.get("format", 0.0) * r_format
            + weights.get("diagnosis", 0.0) * r_diagnosis
            + weights.get("evidence", 0.0) * r_evidence
        ) / total_weight

    return {
        "total": total,
        "format": r_format,
        "diagnosis": r_diagnosis,
        "evidence": r_evidence,
    }
