from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Tuple, Optional
import json

import numpy as np
import torch
from torch.utils.data._utils.collate import default_collate
from transformers import PreTrainedTokenizerBase


SpanMap = Dict[str, List[Tuple[int, int]]]


OUTPUT_FIELD_ORDER = [
    "ecg_abnormal",
    "shd",
    "rhythm",
    "rate_bpm",
    "lvef",
    "acs",
    "afib_risk",
    "conduction",
    "chamber_enlargement",
    "ischemia",
    "pericarditis",
    "other",
    "findings",
]

LABEL_TASKS = {
    "conduction",
    "chamber_enlargement",
    "ischemia",
    "other",
    "findings",
}


@dataclass
class RenderResult:
    text: str
    spans: SpanMap


class DeterministicJSONRenderer:
    def __init__(self, output_field_order: List[str], strict: bool = True):
        self.output_field_order = list(output_field_order)
        self.strict = bool(strict)
        self.buf: List[str] = []
        self.pos = 0
        self.spans: SpanMap = {}

    def _append(self, s: str) -> None:
        self.buf.append(s)
        self.pos += len(s)

    def _add_span(self, path: str, start: int, end: int) -> None:
        self.spans.setdefault(path, []).append((start, end))

    def _emit_json_string(self, s: str, span_path: Optional[str] = None) -> None:
        token = json.dumps(str(s), ensure_ascii=False, separators=(",", ":"))
        start = self.pos
        self._append(token)
        end = self.pos
        if span_path is not None:
            self._add_span(span_path, start, end)

    def _emit_json_int(self, x: int, span_path: Optional[str] = None) -> None:
        token = str(int(x))
        start = self.pos
        self._append(token)
        end = self.pos
        if span_path is not None:
            self._add_span(span_path, start, end)

    def _emit_json_bool(self, b: bool, span_path: Optional[str] = None) -> None:
        token = "true" if b else "false"
        start = self.pos
        self._append(token)
        end = self.pos
        if span_path is not None:
            self._add_span(span_path, start, end)

    def _emit_json_null(self, span_path: Optional[str] = None) -> None:
        start = self.pos
        self._append("null")
        end = self.pos
        if span_path is not None:
            self._add_span(span_path, start, end)

    def _emit_key(self, k: str) -> None:
        self._emit_json_string(k, span_path=None)
        self._append(":")

    def _emit_list_of_strings(self, items: Iterable[str], span_path: str) -> None:
        start_bracket = self.pos
        self._append("[")
        self._add_span(span_path, start_bracket, start_bracket + 1)

        for i, s in enumerate(items):
            if i > 0:
                self._append(",")
            self._emit_json_string(str(s), span_path=span_path)

        end_bracket_pos = self.pos
        self._append("]")
        self._add_span(span_path, end_bracket_pos, end_bracket_pos + 1)

    def _require_key(self, obj: Dict[str, Any], key: str, path: str) -> Any:
        if key not in obj:
            if self.strict:
                raise KeyError(f"Missing required key '{path}.{key}'")
            return None
        return obj[key]

    def render(self, obj: Dict[str, Any]) -> RenderResult:
        if not isinstance(obj, dict):
            raise TypeError(f"Expected dict for render(), got {type(obj)}")

        self.buf = []
        self.pos = 0
        self.spans = {}

        self._append("{")

        self._emit_key("schema_version")
        if "schema_version" not in obj and self.strict:
            raise KeyError("Missing required key 'schema_version'")
        self._emit_json_string(obj.get("schema_version", ""), span_path=None)
        self._append(",")

        self._emit_key("outputs")
        self._append("{")

        outputs = obj.get("outputs")
        if not isinstance(outputs, dict):
            raise TypeError("Missing or invalid 'outputs' in target_json")

        if self.strict:
            missing = [k for k in self.output_field_order if k not in outputs]
            if missing:
                raise KeyError(f"Missing outputs keys: {missing}")

        first_field = True
        for field in self.output_field_order:
            if field not in outputs:
                if self.strict:
                    raise KeyError(f"Missing outputs.{field}")
                continue
            if not first_field:
                self._append(",")
            first_field = False

            self._emit_key(field)
            self._append("{")

            field_obj = outputs[field]
            if not isinstance(field_obj, dict):
                raise TypeError(f"outputs.{field} must be dict")

            is_label_task = field in LABEL_TASKS or ("labels" in field_obj or "labels_present" in field_obj)

            if is_label_task:
                self._emit_key("labels_present")
                lp_path = f"outputs.{field}.labels_present"
                lp_val = self._require_key(field_obj, "labels_present", f"outputs.{field}")
                self._emit_json_bool(bool(lp_val), span_path=lp_path)
                self._append(",")

                self._emit_key("labels")
                labels_path = f"outputs.{field}.labels"
                labels_val = self._require_key(field_obj, "labels", f"outputs.{field}")
                labels_list = list(labels_val) if labels_val is not None else []
                self._emit_list_of_strings(labels_list, span_path=labels_path)
                self._append(",")

                self._emit_key("status")
                status_path = f"outputs.{field}.status"
                status_val = self._require_key(field_obj, "status", f"outputs.{field}")
                self._emit_json_string(status_val, span_path=status_path)
            else:
                self._emit_key("value")
                value_path = f"outputs.{field}.value"
                value = self._require_key(field_obj, "value", f"outputs.{field}")
                if value is None:
                    self._emit_json_null(span_path=value_path)
                elif isinstance(value, (bool, np.bool_)):
                    self._emit_json_bool(bool(value), span_path=value_path)
                elif isinstance(value, (int, np.integer)):
                    self._emit_json_int(int(value), span_path=value_path)
                else:
                    self._emit_json_string(str(value), span_path=value_path)

                if field == "lvef":
                    self._append(",")
                    self._emit_key("unit")
                    unit_val = self._require_key(field_obj, "unit", f"outputs.{field}")
                    if unit_val is None:
                        self._emit_json_null(span_path=None)
                    else:
                        self._emit_json_string(str(unit_val), span_path=None)

                self._append(",")
                self._emit_key("status")
                status_path = f"outputs.{field}.status"
                status_val = self._require_key(field_obj, "status", f"outputs.{field}")
                self._emit_json_string(status_val, span_path=status_path)

            self._append("}")

        self._append("}")

        if "tasks_requested" in obj:
            self._append(",")
            self._emit_key("tasks_requested")
            self._append("[")
            tasks = obj.get("tasks_requested") or []
            for i, t in enumerate(tasks):
                if i > 0:
                    self._append(",")
                self._emit_json_string(str(t), span_path=None)
            self._append("]")
        elif "task" in obj:
            self._append(",")
            self._emit_key("task")
            self._emit_json_string(str(obj.get("task")), span_path=None)

        self._append("}")

        return RenderResult(text="".join(self.buf), spans=self.spans)


def spans_to_intervals(spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    if not spans:
        return []
    spans = sorted(spans)
    merged = [spans[0]]
    for s, e in spans[1:]:
        ps, pe = merged[-1]
        if s <= pe:
            merged[-1] = (ps, max(pe, e))
        else:
            merged.append((s, e))
    return merged


def overlaps_any(token_span: Tuple[int, int], intervals: List[Tuple[int, int]]) -> bool:
    ts, te = token_span
    if te <= ts:
        return False
    for s, e in intervals:
        if e <= ts:
            continue
        if s >= te:
            return False
        return True
    return False


def _normalize_supervised_paths(raw: Any) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        text = raw.strip()
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
                if isinstance(parsed, list):
                    return [str(p) for p in parsed]
            except json.JSONDecodeError:
                pass
        return [p.strip() for p in text.split(",") if p.strip()]
    if isinstance(raw, (list, tuple, set, np.ndarray)):
        return [str(p) for p in raw]
    return [str(raw)]


def _stack_optional_tensors(items: List[Any]) -> Optional[torch.Tensor]:
    if not items:
        return None
    try:
        return default_collate(items)
    except Exception:
        return None


class SchemaSFTBatchCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        max_length: int,
        num_ecg_tokens: int = 0,
        ecg_token_start_id: Optional[int] = None,
        prefix_tuning: bool = False,
        medgemma_prompt_style: bool = False,
        renderer_strict: bool = True,
        supervise_full_json: bool = True,
    ):
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.num_ecg_tokens = int(num_ecg_tokens)
        self.ecg_token_start_id = ecg_token_start_id
        self.prefix_tuning = bool(prefix_tuning)
        self.medgemma_prompt_style = bool(medgemma_prompt_style)
        self.supervise_full_json = bool(supervise_full_json)
        self.renderer = DeterministicJSONRenderer(OUTPUT_FIELD_ORDER, strict=renderer_strict)

        pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        eos_token_id = getattr(self.tokenizer, "eos_token_id", None)
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0] if eos_token_id else None
        if pad_token_id is None:
            pad_token_id = eos_token_id
        if pad_token_id is None:
            raise ValueError("Tokenizer must define pad_token_id or eos_token_id")
        self.pad_token_id = int(pad_token_id)

        bos_token_id = getattr(self.tokenizer, "bos_token_id", None)
        if isinstance(bos_token_id, (list, tuple)):
            bos_token_id = bos_token_id[0] if bos_token_id else None
        self.bos_token_id = int(bos_token_id) if bos_token_id is not None else None

        image_token_id = None
        try:
            image_token_id = self.tokenizer.convert_tokens_to_ids("<start_of_image>")
        except Exception:
            image_token_id = None
        if isinstance(image_token_id, (list, tuple)):
            image_token_id = image_token_id[0] if image_token_id else None
        self.image_token_id = int(image_token_id) if image_token_id is not None else None

    def _build_ecg_prefix(self) -> List[int]:
        if self.num_ecg_tokens <= 0:
            return []
        if self.prefix_tuning or self.ecg_token_start_id is None:
            return [self.pad_token_id] * self.num_ecg_tokens
        start = int(self.ecg_token_start_id)
        return list(range(start, start + self.num_ecg_tokens))

    def _insert_ecg_prefix(self, prompt_ids: List[int]) -> List[int]:
        if self.num_ecg_tokens <= 0:
            return list(prompt_ids)
        prefix = self._build_ecg_prefix()
        if self.medgemma_prompt_style and self.image_token_id is not None:
            try:
                image_pos = prompt_ids.index(self.image_token_id)
                return prompt_ids[: image_pos + 1] + prefix + prompt_ids[image_pos + 1 :]
            except ValueError:
                pass
        if self.medgemma_prompt_style and self.bos_token_id is not None:
            if prompt_ids and prompt_ids[0] == self.bos_token_id:
                return [prompt_ids[0]] + prefix + prompt_ids[1:]
        return prefix + prompt_ids

    def _parse_target_json(self, raw: Any) -> Dict[str, Any]:
        if isinstance(raw, dict):
            return raw
        if isinstance(raw, str):
            return json.loads(raw)
        if isinstance(raw, bytes):
            return json.loads(raw.decode("utf-8"))
        raise TypeError(f"Unsupported target_json type: {type(raw)}")

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        batch = [item for item in batch if item is not None]
        if not batch:
            raise ValueError("All items in the batch were invalid. Check dataset integrity or file paths.")

        input_ids_batch: List[List[int]] = []
        labels_batch: List[List[int]] = []
        attention_masks: List[List[int]] = []
        prompt_input_ids_batch: List[List[int]] = []
        prompt_attention_masks: List[List[int]] = []

        prompt_texts: List[str] = []
        rendered_prompts: List[str] = []
        answer_texts: List[str] = []
        waveform_names: List[str] = []
        prompt_categories: List[str] = []

        signals = []
        pattern_targets = []
        has_pattern_targets = False

        for sample in batch:
            signals.append(sample.get("signal"))

            waveform_names.append(sample.get("waveform_name", ""))
            # Always append to prompt_categories to maintain batch alignment
            prompt_categories.append(str(sample.get("prompt_category", "")))

            prompt_texts.append(str(sample.get("prompt_text", "")))
            rendered_prompt = str(sample.get("rendered_prompt", sample.get("prompt", "")))
            rendered_prompts.append(rendered_prompt)

            raw_answer = sample.get("answer_text")
            if raw_answer is None:
                raw_answer = sample.get("target_json")
            answer_texts.append(str(raw_answer) if raw_answer is not None else "")

            raw_target_json = sample.get("target_json", raw_answer)
            target_json = self._parse_target_json(raw_target_json)

            rr = self.renderer.render(target_json)
            target_text = rr.text

            supervised_paths = _normalize_supervised_paths(sample.get("supervised_paths"))
            supervised_spans: List[Tuple[int, int]] = []
            for path in supervised_paths:
                supervised_spans.extend(rr.spans.get(path, []))
            supervised_intervals = spans_to_intervals(supervised_spans)

            prompt_enc = self.tokenizer.encode_plus(
                rendered_prompt,
                add_special_tokens=True,
                return_tensors=None,
            )
            prompt_ids_base = list(prompt_enc.get("input_ids", []))
            prompt_ids = self._insert_ecg_prefix(prompt_ids_base)

            if len(prompt_ids) >= self.max_length:
                prompt_ids = prompt_ids[: self.max_length]
                input_ids = prompt_ids
                labels = [-100] * len(input_ids)
                target_ids = []
            else:
                target_enc = self.tokenizer(
                    target_text,
                    add_special_tokens=False,
                    return_offsets_mapping=True,
                    padding=False,
                    truncation=False,
                )
                target_ids = list(target_enc.get("input_ids", []))
                offsets = list(target_enc.get("offset_mapping", []))

                max_target_len = self.max_length - len(prompt_ids)
                if max_target_len < len(target_ids):
                    target_ids = target_ids[:max_target_len]
                    offsets = offsets[:max_target_len]

                # Supervise full JSON structure (keys + values) or just values
                if self.supervise_full_json:
                    # Supervise all tokens in the target JSON for proper structure learning
                    masked = [int(tid) for tid in target_ids]
                else:
                    # Legacy behavior: only supervise tokens overlapping supervised_paths
                    masked = []
                    for tid, (cs, ce) in zip(target_ids, offsets):
                        keep = overlaps_any((int(cs), int(ce)), supervised_intervals)
                        masked.append(int(tid) if keep else -100)

                input_ids = prompt_ids + target_ids
                labels = [-100] * len(prompt_ids) + masked

            attention_mask = [1] * len(input_ids)

            pad_len = self.max_length - len(input_ids)
            if pad_len > 0:
                input_ids += [self.pad_token_id] * pad_len
                attention_mask += [0] * pad_len
                labels += [-100] * pad_len

            prompt_ids_trim = prompt_ids_base[: self.max_length]
            prompt_mask = [1] * len(prompt_ids_trim)
            pad_prompt = self.max_length - len(prompt_ids_trim)
            if pad_prompt > 0:
                prompt_ids_trim += [self.pad_token_id] * pad_prompt
                prompt_mask += [0] * pad_prompt

            input_ids_batch.append(input_ids)
            labels_batch.append(labels)
            attention_masks.append(attention_mask)
            prompt_input_ids_batch.append(prompt_ids_trim)
            prompt_attention_masks.append(prompt_mask)

            if "pattern_targets" in sample:
                has_pattern_targets = True
                pattern_targets.append(sample.get("pattern_targets"))

        batch_out: Dict[str, Any] = {
            "signal": default_collate(signals),
            "input_ids": torch.tensor(input_ids_batch, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "labels": torch.tensor(labels_batch, dtype=torch.long),
            "prompt_input_ids": torch.tensor(prompt_input_ids_batch, dtype=torch.long),
            "prompt_attention_mask": torch.tensor(prompt_attention_masks, dtype=torch.long),
            "prompt_text": prompt_texts,
            "rendered_prompt": rendered_prompts,
            "answer_text": answer_texts,
            "waveform_name": waveform_names,
        }

        if prompt_categories:
            batch_out["prompt_category"] = prompt_categories

        if has_pattern_targets:
            batch_out["pattern_targets"] = _stack_optional_tensors(pattern_targets)

        return batch_out
