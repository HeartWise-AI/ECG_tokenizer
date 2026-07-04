from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
from tqdm import tqdm


CF_TEMPLATES: Dict[str, str] = {
    "RHYTHM": "Question: {question}\nAnswer: {answer}.",
    "CONDUCTION": "Question: {question}\nAnswer: {answer}.",
    "CHAMBER ENLARGEMENT": "Question: {question}\nAnswer: {answer}.",
    "INFARCT, ISCHEMIA": "Question: {question}\nAnswer: {answer}.",
    "PERICARDITIS": "Question: {question}\nAnswer: {answer}.",
    "OTHER": "Question: {question}\nAnswer: {answer}.",
}


class CFEvaluator:
    """
    Choice-Free evaluator for ECG models.
    Loads a CF dataset and evaluates candidate answers by log-likelihood
    without exposing multiple-choice options in the prompt (per candidate
    scoring). Intended as an early-signal metric during training.
    """

    def __init__(
        self,
        cf_dataset_path: str,
        device: str | torch.device | None = None,
        use_letter_space: bool = False,
    ):
        if not os.path.exists(cf_dataset_path):
            raise FileNotFoundError(f"CF dataset not found: {cf_dataset_path}")
        self.cf_data: List[Dict[str, Any]] = self.load_cf_dataset(cf_dataset_path)
        if device is None:
            dev = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
            self.device = torch.device(dev)
        else:
            self.device = torch.device(device)
        # When true, score candidates in a canonical letter space (A, B, C, ...)
        # instead of full answer strings. This keeps CF scoring aligned with
        # MedGemma prompts that train the model to emit letters.
        self.use_letter_space: bool = bool(use_letter_space)

    @staticmethod
    def load_cf_dataset(path: str) -> List[Dict[str, Any]]:
        if path.endswith(".parquet"):
            import pandas as pd
            df = pd.read_parquet(path)
            return df.to_dict(orient="records")
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("CF dataset must be a list of records")
        return data

    def evaluate(self, model, tokenizer, max_samples: int = 1000) -> Dict[str, float]:
        """
        Run CF evaluation on a model checkpoint.

        Args:
            model: ECG_Tokenizer_Wrapper in LLM mode
            tokenizer: Associated HF tokenizer
            max_samples: Subsample for speed

        Returns:
            metrics dict with overall and per-category accuracy
        """
        model_device = next(model.parameters()).device
        results_by_category: dict[str, list[bool]] = defaultdict(list)

        # Lightweight sampling (full eval can be slow)
        eval_subset = self.cf_data
        if max_samples is not None and max_samples > 0 and len(eval_subset) > max_samples:
            eval_subset = random.sample(eval_subset, max_samples)

        for record in tqdm(eval_subset, desc="CF Evaluation", leave=False):
            try:
                signal = self._load_ecg_signal(record.get("signal_path", ""))
            except Exception:
                # Skip missing/corrupt signals silently to avoid interrupting training
                continue

            question = str(record.get("question", ""))
            candidates = list(record["candidate_answers"])  # copy in case
            # Support multiple correct answers
            gt_indices = record.get("ground_truth_indices")
            if isinstance(gt_indices, list) and len(gt_indices) > 0:
                gt_set = {int(i) for i in gt_indices}
            else:
                gt_set = {int(record.get("ground_truth_index", -1))}
            cat = str(record.get("category", "unknown"))
            use_letters = self.use_letter_space and len(candidates) <= 26
            if use_letters:
                letters = [chr(ord("A") + i) for i in range(len(candidates))]
                base_prompt = self.build_letter_prompt(record, letters, candidates)
                scores = self.score_letter_candidates_batched(
                    model=model,
                    tokenizer=tokenizer,
                    ecg_signal=signal,
                    base_prompt=base_prompt,
                    letters=letters,
                    model_device=model_device,
                )
                # Map predicted letter back to candidate index
                # (same ordering as letters/candidates).
            else:
                base_prompt = self.build_cf_prompt(record)
                # Score candidates via batched log-likelihood
                scores = self.score_all_candidates_batched(
                    model=model,
                    tokenizer=tokenizer,
                    ecg_signal=signal,
                    base_prompt=base_prompt,
                    candidates=candidates,
                    category=cat,
                    model_device=model_device,
                )

            pred_idx = int(torch.tensor(scores).argmax().item()) if scores else -1
            correct = (pred_idx in gt_set) and (min(gt_set) >= 0)
            results_by_category[cat].append(bool(correct))

        # Aggregate accuracies
        metrics: Dict[str, float] = {}
        all_results: list[bool] = []
        for category, results in results_by_category.items():
            if not results:
                continue
            acc = float(sum(bool(x) for x in results)) / float(len(results))
            metrics[f"cf_{category.lower().replace(' ', '_').replace(',', '').replace('/', '_')}_acc"] = acc
            all_results.extend(results)
        if all_results:
            metrics["cf_overall_acc"] = float(sum(all_results)) / float(len(all_results))
        else:
            metrics["cf_overall_acc"] = 0.0

        return metrics

    def evaluate_with_details(
        self,
        model,
        tokenizer,
        max_samples: int = 1000,
        max_details: int = 20,
    ) -> tuple[Dict[str, float], List[Dict[str, Any]]]:
        """
        Run CF evaluation and also return a small set of detailed examples.

        Returns:
            (metrics, details) where details is a list of dicts with keys:
            ['ecg_id','category','question','candidate_answers','ground_truth_indices',
             'pred_index','pred_answer','scores']
        """
        model_device = next(model.parameters()).device
        results_by_category: dict[str, list[bool]] = defaultdict(list)
        details: list[Dict[str, Any]] = []

        eval_subset = self.cf_data
        if max_samples is not None and max_samples > 0 and len(eval_subset) > max_samples:
            eval_subset = random.sample(eval_subset, max_samples)

        detail_quota = int(max(0, max_details or 0))

        for record in tqdm(eval_subset, desc="CF Evaluation", leave=False):
            try:
                signal = self._load_ecg_signal(record.get("signal_path", ""))
            except Exception:
                continue

            question = str(record.get("question", ""))
            candidates = list(record["candidate_answers"])  # copy
            gt_indices = record.get("ground_truth_indices")
            if isinstance(gt_indices, list) and len(gt_indices) > 0:
                gt_set = {int(i) for i in gt_indices}
            else:
                gt_set = {int(record.get("ground_truth_index", -1))}
            cat = str(record.get("category", "unknown"))
            use_letters = self.use_letter_space and len(candidates) <= 26
            if use_letters:
                letters = [chr(ord("A") + i) for i in range(len(candidates))]
                base_prompt = self.build_letter_prompt(record, letters, candidates)
                scores = self.score_letter_candidates_batched(
                    model=model,
                    tokenizer=tokenizer,
                    ecg_signal=signal,
                    base_prompt=base_prompt,
                    letters=letters,
                    model_device=model_device,
                )
            else:
                base_prompt = self.build_cf_prompt(record)
                scores = self.score_all_candidates_batched(
                    model=model,
                    tokenizer=tokenizer,
                    ecg_signal=signal,
                    base_prompt=base_prompt,
                    candidates=candidates,
                    category=cat,
                    model_device=model_device,
                )

            pred_idx = int(torch.tensor(scores).argmax().item()) if scores else -1
            correct = (pred_idx in gt_set) and (min(gt_set) >= 0)
            results_by_category[cat].append(bool(correct))

            if detail_quota > 0:
                details.append({
                    "ecg_id": str(record.get("ecg_id", "")),
                    "category": cat,
                    "question": question,
                    "candidate_answers": candidates,
                    "ground_truth_indices": sorted(list(gt_set)),
                    "pred_index": pred_idx,
                    "pred_answer": (candidates[pred_idx] if 0 <= pred_idx < len(candidates) else "<out_of_range>"),
                    "pred_letter": (letters[pred_idx] if use_letters and 0 <= pred_idx < len(candidates) else None),
                    "scores": [float(s) for s in scores],
                })
                detail_quota -= 1

        metrics: Dict[str, float] = {}
        all_results: list[bool] = []
        for category, results in results_by_category.items():
            if not results:
                continue
            acc = float(sum(bool(x) for x in results)) / float(len(results))
            metrics[f"cf_{category.lower().replace(' ', '_').replace(',', '').replace('/', '_')}_acc"] = acc
            all_results.extend(results)
        metrics["cf_overall_acc"] = (float(sum(all_results)) / float(len(all_results))) if all_results else 0.0

        return metrics, details

    @staticmethod
    def _ensure_bchw(signal: np.ndarray) -> torch.Tensor:
        """Coerce loaded numpy arrays into (B, C, L) tensor with B=1, C=12."""
        if signal.ndim == 3 and signal.shape[-1] == 1:
            signal = signal.squeeze(-1)
        if signal.ndim != 2:
            raise ValueError(f"ECG signal must be 2D (T, C). Got {signal.shape}")
        # Expect (T, C); transpose to (C, T)
        if signal.shape[1] == 12:
            ct = signal.T  # (12, T)
        elif signal.shape[0] == 12:
            ct = signal  # (12, T)
        else:
            # Fallback heuristic: if one dim is 12, use it as channels
            if 12 in signal.shape:
                axis = int(np.where(np.array(signal.shape) == 12)[0][0])
                ct = np.moveaxis(signal, axis, 0)
            else:
                raise ValueError(f"Cannot infer 12-lead axis for ECG with shape {signal.shape}")
        tensor = torch.from_numpy(ct).float().unsqueeze(0)  # (1, 12, T)
        return tensor

    @staticmethod
    def _load_ecg_signal(path: str) -> torch.Tensor:
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Missing ECG file: {path}")
        arr = np.load(path)
        if np.isnan(arr).any():
            raise ValueError("ECG signal contains NaNs")
        return CFEvaluator._ensure_bchw(arr)

    def build_cf_prompt(self, record: Dict[str, Any]) -> str:
        """Build a CF cloze-style base prompt (no candidate list)."""
        ecg_id = str(record.get("ecg_id", "")).strip()
        category = str(record.get("category", "")).strip()
        question = str(record.get("question", "")).strip()
        parts = [
            "ECG interpretation task.",
            (f"ECG ID: {ecg_id}" if ecg_id else None),
            (f"Category: {category}" if category else None),
            (f"Question: {question}" if question else None),
            "Answer:",
        ]
        return "\n".join([p for p in parts if p])

    def build_letter_prompt(
        self,
        record: Dict[str, Any],
        letters: Sequence[str],
        options: Sequence[str],
    ) -> str:
        """
        Build a MedGemma-style prompt that lists lettered options and instructs the
        model to answer with letters only. This aligns CF scoring with letter-based
        training targets.
        """
        question = str(record.get("question", "")).strip()
        multi_label = False
        gt_indices = record.get("ground_truth_indices")
        if isinstance(gt_indices, list) and len(gt_indices) > 1:
            multi_label = True
        selection_text = (
            "Select ALL applicable options from the list below."
            if multi_label else
            "Select the single best option from the list below."
        )
        options_block = "\n".join(
            f"{ltr}. {opt}" for ltr, opt in zip(letters, options)
        )
        user_content = (
            "<start_of_image> [ECG image 1 is attached.]\n\n"
            f"Question: {question}\n\n"
            f"{selection_text}\n"
            "Respond ONLY with the letters of the correct options in ascending order, separated by commas, and nothing else.\n\n"
            f"{options_block}\n\nAnswer:"
        )
        system_message = (
            "You are an expert cardiologist. You interpret ECGs and answer in a concise, structured way."
        )
        return f"{system_message}\n\n{user_content}"

    def score_all_candidates_batched(
        self,
        model,
        tokenizer,
        ecg_signal: torch.Tensor,
        base_prompt: str,
        candidates: Sequence[str],
        category: str | None = None,
        model_device: torch.device | None = None,
    ) -> List[float]:
        """
        Batch score all candidates. Returns list of scores (higher = more likely).
        Scores are the mean log-prob over answer tokens only (prompt masked).
        """
        if not candidates:
            return []

        # Prepare prompt and full candidate texts
        prompt = str(base_prompt)
        cat = (category or "").strip()

        def _fmt(ans: str) -> str:
            # Use category template only to stabilize punctuation/spacing in the appended part
            tpl = CF_TEMPLATES.get(cat) or CF_TEMPLATES["OTHER"]
            # Extract the portion after "Answer:" if present
            text = tpl.format(question="", answer=str(ans).strip())
            if "Answer:" in text:
                text = text.split("Answer:")[-1].strip()
            return text

        full_texts = [prompt + " " + _fmt(c) for c in candidates]

        # Tokenize with padding
        enc = tokenizer(full_texts, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")

        # Inspect expected prefix length from the decoder/bridge
        prefix_len = 0
        try:
            base_model = model.module if hasattr(model, 'module') else model
            decoder = getattr(base_model, 'decoder', None)
            bridge = getattr(decoder, 'bridge', None) if decoder is not None else None
            prefix_len = int(getattr(bridge, 'num_query_tokens', 0) or getattr(decoder, 'num_query_tokens', 0) or 0)
        except Exception:
            prefix_len = 0

        # Compute how many tokens belong to the prompt portion
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        n_prompt_tokens = int(prompt_ids.shape[1])

        # Build labels that ignore the prompt tokens
        labels = input_ids.clone()
        labels[:, :n_prompt_tokens] = -100

        # Prepend soft-prompt placeholders if the decoder expects ECG soft prompts
        if prefix_len > 0:
            pad_id = getattr(tokenizer, 'pad_token_id', 0)
            pad_col = torch.full((input_ids.size(0), prefix_len), int(pad_id), dtype=input_ids.dtype)
            input_ids = torch.cat([pad_col, input_ids], dim=1)
            # Expand attention mask similarly if provided
            if attention_mask is not None:
                pre = torch.ones((attention_mask.size(0), prefix_len), dtype=attention_mask.dtype)
                attention_mask = torch.cat([pre, attention_mask], dim=1)
            # And ensure labels ignore the prefixed soft prompts
            pre_ignore = torch.full((labels.size(0), prefix_len), -100, dtype=labels.dtype)
            labels = torch.cat([pre_ignore, labels], dim=1)

        dev = model_device or self.device
        input_ids = input_ids.to(dev)
        labels = labels.to(dev)
        ecg_signal = ecg_signal.to(dev)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dev)

        # Repeat ECG across candidate batch
        if ecg_signal.dim() == 3:
            ecg_signal = ecg_signal.repeat(input_ids.size(0), 1, 1)

        with torch.no_grad():
            outputs = model(
                ecg_signal=ecg_signal,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            logits = outputs.get("logits")
            if logits is None:
                raise ValueError("Model did not return logits for CF evaluation")

            # Standard LM log-likelihood over labels != -100
            # Shifted to align logits with next-token labels
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]

            log_probs = torch.log_softmax(shift_logits, dim=-1)
            # Replace ignore indices with 0 for safe gather, then mask out
            safe_labels = shift_labels.clone()
            safe_labels[safe_labels == -100] = 0
            token_logprobs = torch.gather(log_probs, 2, safe_labels.unsqueeze(-1)).squeeze(-1)
            mask = (shift_labels != -100).float()
            sum_logprob = (token_logprobs * mask).sum(dim=1)
            len_tokens = mask.sum(dim=1).clamp(min=1)
            scores = (sum_logprob / len_tokens).detach().cpu().tolist()

        return [float(s) for s in scores]

    def score_letter_candidates_batched(
        self,
        model,
        tokenizer,
        ecg_signal: torch.Tensor,
        base_prompt: str,
        letters: Sequence[str],
        model_device: torch.device | None = None,
    ) -> List[float]:
        """
        Score canonical letter candidates (A, B, C, ...) by appending the letter
        to a prompt that already lists the options. Masks out the prompt tokens so
        only the letter contributes to the score.
        """
        if not letters:
            return []

        prompt = str(base_prompt)
        full_texts = [f"{prompt} {ltr}" for ltr in letters]

        enc = tokenizer(full_texts, return_tensors="pt", padding=True)
        input_ids = enc["input_ids"]
        attention_mask = enc.get("attention_mask")

        # Estimate prefix length for soft prompts
        prefix_len = 0
        try:
            base_model = model.module if hasattr(model, 'module') else model
            decoder = getattr(base_model, 'decoder', None)
            bridge = getattr(decoder, 'bridge', None) if decoder is not None else None
            prefix_len = int(getattr(bridge, 'num_query_tokens', 0) or getattr(decoder, 'num_query_tokens', 0) or 0)
        except Exception:
            prefix_len = 0

        # Prompt token count (no letter)
        prompt_ids = tokenizer(prompt, return_tensors="pt")["input_ids"]
        n_prompt_tokens = int(prompt_ids.shape[1])

        labels = input_ids.clone()
        labels[:, :n_prompt_tokens] = -100

        if prefix_len > 0:
            pad_id = getattr(tokenizer, 'pad_token_id', 0)
            pad_col = torch.full((input_ids.size(0), prefix_len), int(pad_id), dtype=input_ids.dtype)
            input_ids = torch.cat([pad_col, input_ids], dim=1)
            if attention_mask is not None:
                pre = torch.ones((attention_mask.size(0), prefix_len), dtype=attention_mask.dtype)
                attention_mask = torch.cat([pre, attention_mask], dim=1)
            pre_ignore = torch.full((labels.size(0), prefix_len), -100, dtype=labels.dtype)
            labels = torch.cat([pre_ignore, labels], dim=1)

        dev = model_device or self.device
        input_ids = input_ids.to(dev)
        labels = labels.to(dev)
        ecg_signal = ecg_signal.to(dev)
        if attention_mask is not None:
            attention_mask = attention_mask.to(dev)

        if ecg_signal.dim() == 3:
            ecg_signal = ecg_signal.repeat(input_ids.size(0), 1, 1)

        with torch.no_grad():
            outputs = model(
                ecg_signal=ecg_signal,
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
            logits = outputs.get("logits")
            if logits is None:
                raise ValueError("Model did not return logits for CF evaluation")

            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:]

            log_probs = torch.log_softmax(shift_logits, dim=-1)
            safe_labels = shift_labels.clone()
            safe_labels[safe_labels == -100] = 0
            token_logprobs = torch.gather(log_probs, 2, safe_labels.unsqueeze(-1)).squeeze(-1)
            mask = (shift_labels != -100).float()
            sum_logprob = (token_logprobs * mask).sum(dim=1)
            len_tokens = mask.sum(dim=1).clamp(min=1)
            scores = (sum_logprob / len_tokens).detach().cpu().tolist()

        return [float(s) for s in scores]
