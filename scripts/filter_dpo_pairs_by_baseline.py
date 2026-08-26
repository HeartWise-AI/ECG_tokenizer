#!/usr/bin/env python3
"""
Filter DPO pairs by baseline preference margin (logp_chosen - logp_rejected).

Usage:
  python scripts/filter_dpo_pairs_by_baseline.py \
    --base_checkpoint checkpoints/BEST_LLM/.../best_model.pt \
    --pairs output/dpo_inference/dpo_5gen_dpo_pairs.jsonl \
    --output output/dpo_inference/dpo_5gen_dpo_pairs.margin05.jsonl \
    --min_margin 0.5 \
    --max_samples 1000

Advanced usage (category-aware thresholds, prompt buckets, and reweighting):
  python scripts/filter_dpo_pairs_by_baseline.py \
    --base_checkpoint checkpoints/BEST_LLM/.../best_model.pt \
    --pairs output/dpo_inference/dpo_5gen_dpo_pairs.jsonl \
    --output output/dpo_inference/dpo_pairs.filtered.jsonl \
    --min_margin 1.0 \
    --category_min_margins_json configs/dpo_min_margins.json \
    --category_weight_mult_json configs/dpo_weight_mult.json \
    --bucket_config_json configs/dpo_buckets.json

Fast path (no model forward): re-filter/reweight an existing file that already
contains `baseline_margin`:
  python scripts/filter_dpo_pairs_by_baseline.py \
    --pairs output/dpo_inference/dpo_5gen_dpo_pairs.margin10.jsonl \
    --output output/dpo_inference/dpo_5gen_dpo_pairs.margin10.reweighted.jsonl \
    --use_existing_margin \
    --min_margin 1.0
"""

import argparse
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformers import AutoTokenizer
from data.dpo_pair_dataset import DPOPairDataset
from models.ecg_tokenizer_wrapper import ECG_Tokenizer_Wrapper
from utils.enums import DecoderMode
from utils.files_handler import load_yaml
from utils.checkpoint_structure import resolve_checkpoint_bridge_option


def _cfg_get(container: Any, key: str, fallback: Any = None) -> Any:
    if container is None:
        return fallback
    if isinstance(container, dict) and key in container:
        return container[key]
    if hasattr(container, key):
        val = getattr(container, key)
        return val if val is not None else fallback
    return fallback


def _sequence_logp(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :]
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    shift_labels = shift_labels.clamp_min(0)
    log_probs = torch.log_softmax(shift_logits, dim=-1)
    token_logp = log_probs.gather(-1, shift_labels.unsqueeze(-1)).squeeze(-1)
    token_logp = token_logp * mask
    return token_logp.sum(dim=-1)


@dataclass(frozen=True)
class BucketRule:
    name: str
    prompt_re: re.Pattern
    category_in: Optional[set[str]]
    category_re: Optional[re.Pattern]
    min_margin: Optional[float]
    weight_mult: Optional[float]


def _load_json_file(path: Optional[str]) -> Any:
    if path is None:
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_bucket_rules(path: Optional[str]) -> list[BucketRule]:
    raw = _load_json_file(path)
    if not raw:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"--bucket_config_json must be a JSON list, got: {type(raw)}")

    rules: list[BucketRule] = []
    for i, entry in enumerate(raw):
        if not isinstance(entry, dict):
            raise ValueError(f"Bucket entry #{i} must be an object, got: {type(entry)}")
        name = entry.get("name")
        if not name:
            raise ValueError(f"Bucket entry #{i} missing required field: name")
        prompt_regex = entry.get("prompt_regex") or entry.get("regex")
        if not prompt_regex:
            raise ValueError(f"Bucket entry '{name}' missing required field: prompt_regex/regex")

        ignore_case = bool(entry.get("ignore_case", True))
        flags = re.IGNORECASE if ignore_case else 0
        try:
            prompt_re = re.compile(str(prompt_regex), flags=flags)
        except re.error as e:
            raise ValueError(f"Invalid prompt regex for bucket '{name}': {e}") from e

        category_in = entry.get("category_in")
        if category_in is not None:
            if not isinstance(category_in, list) or not all(isinstance(x, str) for x in category_in):
                raise ValueError(f"Bucket '{name}' field category_in must be a list[str] if provided.")
            category_in_set: Optional[set[str]] = set(category_in)
        else:
            category_in_set = None

        category_regex = entry.get("category_regex")
        category_re = None
        if category_regex:
            try:
                category_re = re.compile(str(category_regex), flags=flags)
            except re.error as e:
                raise ValueError(f"Invalid category_regex for bucket '{name}': {e}") from e

        min_margin = entry.get("min_margin")
        if min_margin is not None:
            min_margin = float(min_margin)

        weight_mult = entry.get("weight_mult")
        if weight_mult is not None:
            weight_mult = float(weight_mult)

        rules.append(
            BucketRule(
                name=str(name),
                prompt_re=prompt_re,
                category_in=category_in_set,
                category_re=category_re,
                min_margin=min_margin,
                weight_mult=weight_mult,
            )
        )
    return rules


def _match_bucket(prompt: str, category: str, rules: list[BucketRule]) -> Optional[BucketRule]:
    if not rules:
        return None
    prompt = prompt or ""
    category = category or ""
    for rule in rules:
        if rule.category_in is not None and category not in rule.category_in:
            continue
        if rule.category_re is not None and not rule.category_re.search(category):
            continue
        if rule.prompt_re.search(prompt):
            return rule
    return None


def _expand_labels_with_ecg(
    decoder,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    ecg_embeddings: torch.Tensor,
) -> torch.Tensor:
    merged = decoder._inject_ecg_after_image_token(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        ecg_embeddings=ecg_embeddings,
        embed_layer=decoder.llm_model.get_input_embeddings(),
    )
    if merged is not None:
        labels_out = merged[3]
        if labels_out is None:
            raise ValueError("Failed to expand labels with ECG injection.")
        return labels_out

    prefix_len = int(ecg_embeddings.size(1))
    if prefix_len <= 0:
        return labels
    ignore_pad = torch.full(
        (labels.size(0), prefix_len),
        -100,
        dtype=labels.dtype,
        device=labels.device,
    )
    return torch.cat([ignore_pad, labels], dim=1)


def _forward_logits_and_labels(
    model,
    ecg_signal: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    labels: torch.Tensor,
    prompt_input_ids: Optional[torch.Tensor],
    prompt_attention_mask: Optional[torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    ecg_signal = ecg_signal.to(dtype=torch.float32)
    features = model.encoder(ecg_signal)
    quantized, indices, _ = model.quantizer(features)
    quantized_codes = model._extract_primary_codes(indices, model.num_codebooks_kept, model.codebook_offset)

    if not hasattr(model.decoder, "_compute_ecg_embeddings"):
        raise ValueError("Decoder does not support ECG embeddings for DPO filtering.")

    ecg_embeddings, _ = model.decoder._compute_ecg_embeddings(
        quantized,
        quantized_codes,
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
    )

    outputs = model.decoder(
        quantized_features=None,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        quantized_codes=None,
        ecg_embeddings=ecg_embeddings,
        prompt_input_ids=prompt_input_ids,
        prompt_attention_mask=prompt_attention_mask,
    )
    logits = outputs["logits"]
    labels_expanded = _expand_labels_with_ecg(
        decoder=model.decoder,
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels,
        ecg_embeddings=ecg_embeddings,
    )
    return logits, labels_expanded


def load_model(base_checkpoint: str, device: torch.device):
    checkpoint_data = torch.load(base_checkpoint, map_location="cpu", weights_only=False)
    config = checkpoint_data["config"]

    checkpoint_dir = os.path.dirname(base_checkpoint)
    config_yaml_path = os.path.join(checkpoint_dir, "config.yaml")
    yaml_config = load_yaml(config_yaml_path) if os.path.exists(config_yaml_path) else None

    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    decoder_mode = config.decoder_mode if isinstance(config.decoder_mode, DecoderMode) else DecoderMode(config.decoder_mode)
    num_visual_tokens = getattr(config, "num_query_tokens", getattr(config, "num_visual_tokens", None))

    num_codebooks_kept = _cfg_get(yaml_config, "num_codebooks_kept", None)
    if num_codebooks_kept is None:
        num_codebooks_kept = _cfg_get(config, "num_codebooks_kept", None)

    checkpoint_state_dict = checkpoint_data["model_state_dict"]
    mix_gate_key = "decoder.bridge.mix_gate.0.weight"
    if mix_gate_key in checkpoint_state_dict:
        mix_gate_shape = checkpoint_state_dict[mix_gate_key].shape
        hidden_dim = mix_gate_shape[0]
        inferred_codebooks = mix_gate_shape[1] // hidden_dim
        if num_codebooks_kept is None or num_codebooks_kept != inferred_codebooks:
            num_codebooks_kept = inferred_codebooks

    codebook_offset = _cfg_get(yaml_config, "codebook_offset", _cfg_get(config, "codebook_offset", 0))

    lora_config = getattr(config, "lora_config", None)
    if lora_config is None and bool(getattr(config, "use_lora", False)):
        lora_config = {
            "r": int(getattr(config, "lora_r", 16)),
            "lora_alpha": int(getattr(config, "lora_alpha", getattr(config, "lora_r", 16))),
            "lora_dropout": float(getattr(config, "lora_dropout", 0.0)),
            "target_modules": list(getattr(config, "lora_target_modules", None) or []),
            "bias": str(getattr(config, "lora_bias", "none")),
        }
        try:
            setattr(config, "lora_config", lora_config)
        except Exception:
            pass

    model = ECG_Tokenizer_Wrapper(
        encoder_name=config.encoder_name,
        quantizer_name=config.quantizer_name,
        decoder_name=config.decoder_name,
        num_quantizers=int(getattr(config, "num_quantizers", 8)),
        codebook_size=int(getattr(config, "codebook_size", 512)),
        decoder_mode=decoder_mode,
        huggingface_model_name=config.huggingface_model_name,
        llm_input_embedding_size=int(config.llm_input_embedding_size),
        bridge_name=config.bridge_name,
        num_visual_tokens=num_visual_tokens,
        bridge_mid_dim=int(getattr(config, "bridge_mid_dim", 512)),
        bridge_num_heads=int(getattr(config, "bridge_num_heads", 8)),
        bridge_dropout=float(getattr(config, "bridge_dropout", 0.1)),
        bridge_num_special_tokens=int(getattr(config, "bridge_num_special_tokens", 4)),
        bridge_qformer_layers=getattr(config, "bridge_qformer_layers", None),
        bridge_text_hidden_size=getattr(config, "bridge_text_hidden_size", None),
        bridge_bias_last_codebook=_cfg_get(yaml_config, "bridge_bias_last_codebook", _cfg_get(config, "bridge_bias_last_codebook", None)),
        bridge_codebook_dropout=_cfg_get(yaml_config, "bridge_codebook_dropout", _cfg_get(config, "bridge_codebook_dropout", None)),
        bridge_mix_strategy=resolve_checkpoint_bridge_option(config, yaml_config, "bridge_mix_strategy"),
        bridge_token_axis=resolve_checkpoint_bridge_option(config, yaml_config, "bridge_token_axis"),
        bridge_cross_every=_cfg_get(yaml_config, "bridge_cross_every", _cfg_get(config, "bridge_cross_every", None)),
        instruction_dropout=_cfg_get(yaml_config, "instruction_dropout", _cfg_get(config, "instruction_dropout", 0.0)),
        stage1_checkpoint_path=None,
        use_lora=bool(getattr(config, "use_lora", False)),
        lora_config=lora_config,
        tokenizer=tokenizer,
        ecg_token_start_id=None,
        ecg_waveform_length=int(getattr(config, "ecg_waveform_length", 2500)),
        ecg_num_leads=int(getattr(config, "ecg_num_leads", 12)),
        default_generation_kwargs=getattr(config, "default_generation_kwargs", None),
        num_codebooks_kept=num_codebooks_kept,
        codebook_offset=codebook_offset,
    )

    model._load_state_dict(checkpoint_state_dict, strict=False)

    try:
        if bool(getattr(config, "use_lora", False)):
            model.set_lora_inference_mode(True)
    except Exception:
        pass

    model.eval()
    model.to(device)
    return model, tokenizer, config


def collate_with_idx(batch):
    def _stack(key: str, dtype: Optional[torch.dtype] = None) -> torch.Tensor:
        tensors = [item[key] for item in batch]
        out = torch.stack(tensors, dim=0)
        if dtype is not None:
            out = out.to(dtype)
        return out

    return {
        "idx": torch.tensor([item["idx"] for item in batch], dtype=torch.long),
        "signal": _stack("signal", dtype=torch.float32),
        "chosen_input_ids": _stack("chosen_input_ids"),
        "chosen_attention_mask": _stack("chosen_attention_mask"),
        "chosen_labels": _stack("chosen_labels"),
        "rejected_input_ids": _stack("rejected_input_ids"),
        "rejected_attention_mask": _stack("rejected_attention_mask"),
        "rejected_labels": _stack("rejected_labels"),
        "prompt_input_ids": _stack("prompt_input_ids"),
        "prompt_attention_mask": _stack("prompt_attention_mask"),
        "weight": _stack("weight", dtype=torch.float32),
    }


def _atomic_json_dump(path: str, payload: dict) -> None:
    output_dir = os.path.dirname(path) or "."
    os.makedirs(output_dir, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=output_dir,
            prefix=f".{os.path.basename(path)}.",
            suffix=".tmp",
            delete=False,
        ) as f:
            tmp_path = f.name
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.unlink(tmp_path)


def main():
    parser = argparse.ArgumentParser(description="Filter/reweight DPO pairs using baseline margin (optionally category-aware).")
    parser.add_argument("--base_checkpoint", default=None, help="Base (pre-DPO) checkpoint path (required unless --use_existing_margin).")
    parser.add_argument("--pairs", required=True, help="Input DPO pairs JSONL path.")
    parser.add_argument("--output", required=True, help="Output JSONL path.")
    parser.add_argument("--min_margin", type=float, default=0.5)
    parser.add_argument("--category_min_margins_json", type=str, default=None, help="JSON mapping: effective_category -> min_margin override.")
    parser.add_argument("--category_weight_mult_json", type=str, default=None, help="JSON mapping: effective_category -> weight multiplier.")
    parser.add_argument("--bucket_config_json", type=str, default=None, help="JSON list of prompt buckets; first match wins.")
    parser.add_argument("--category_key", type=str, default="prompt_category")
    parser.add_argument("--prompt_key", type=str, default="prompt")
    parser.add_argument("--weight_key", type=str, default="weight")
    parser.add_argument("--baseline_margin_key", type=str, default="baseline_margin")
    parser.add_argument("--effective_category_key", type=str, default="effective_category")
    parser.add_argument("--max_weight", type=float, default=None, help="Clamp final weight to this max (optional).")
    parser.add_argument("--write_all", action="store_true", help="Write all rows (adds keep_by_margin flag) instead of filtering.")
    parser.add_argument("--use_existing_margin", action="store_true", help="Skip model forward; require baseline margin present in JSONL.")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--stats_json", type=str, default=None)
    args = parser.parse_args()

    category_min_margins = _load_json_file(args.category_min_margins_json) or {}
    if not isinstance(category_min_margins, dict):
        raise ValueError("--category_min_margins_json must be a JSON object mapping category -> float.")
    category_min_margins = {str(k): float(v) for k, v in category_min_margins.items()}

    category_weight_mult = _load_json_file(args.category_weight_mult_json) or {}
    if not isinstance(category_weight_mult, dict):
        raise ValueError("--category_weight_mult_json must be a JSON object mapping category -> float.")
    category_weight_mult = {str(k): float(v) for k, v in category_weight_mult.items()}

    bucket_rules = _load_bucket_rules(args.bucket_config_json)

    def _effective_category_and_overrides(record: dict) -> tuple[str, Optional[BucketRule]]:
        raw_cat = str(record.get(args.category_key, "unknown"))
        prompt = str(record.get(args.prompt_key, ""))
        rule = _match_bucket(prompt=prompt, category=raw_cat, rules=bucket_rules)
        if rule is not None:
            return rule.name, rule
        return raw_cat, None

    def _min_margin_for(category: str, rule: Optional[BucketRule]) -> float:
        if rule is not None and rule.min_margin is not None:
            return float(rule.min_margin)
        return float(category_min_margins.get(category, args.min_margin))

    def _weight_mult_for(category: str, rule: Optional[BucketRule]) -> float:
        if rule is not None and rule.weight_mult is not None:
            return float(rule.weight_mult)
        return float(category_weight_mult.get(category, 1.0))

    # Stats accumulators
    all_margins: list[float] = []
    kept_margins: list[float] = []
    total = 0
    kept = 0

    per_cat_total = Counter()
    per_cat_kept = Counter()
    per_cat_all_margins: dict[str, list[float]] = defaultdict(list)
    per_cat_kept_margins: dict[str, list[float]] = defaultdict(list)
    per_cat_weight_base: dict[str, list[float]] = defaultdict(list)
    per_cat_weight_final: dict[str, list[float]] = defaultdict(list)
    per_cat_min_margin_used: dict[str, list[float]] = defaultdict(list)
    per_cat_weight_mult_used: dict[str, list[float]] = defaultdict(list)

    if not args.use_existing_margin and not args.base_checkpoint:
        raise ValueError("--base_checkpoint is required unless --use_existing_margin is set.")

    output_dir = os.path.dirname(args.output) or "."
    os.makedirs(output_dir, exist_ok=True)
    out_tmp_path = None
    out_f = tempfile.NamedTemporaryFile(
        "w",
        encoding="utf-8",
        dir=output_dir,
        prefix=f".{os.path.basename(args.output)}.",
        suffix=".tmp",
        delete=False,
    )
    out_tmp_path = out_f.name

    def _write_record(record: dict, margin: float, effective_category: str, rule: Optional[BucketRule]) -> None:
        nonlocal kept
        min_margin_used = _min_margin_for(effective_category, rule)
        weight_mult = _weight_mult_for(effective_category, rule)
        weight_base_raw = record.get("weight_orig", None)
        if weight_base_raw is None:
            weight_base_raw = record.get(args.weight_key, 1.0)
        weight_base = float(weight_base_raw)
        weight_final = weight_base * float(weight_mult)
        if args.max_weight is not None:
            weight_final = min(weight_final, float(args.max_weight))

        keep_by_margin = margin >= min_margin_used

        record_out = dict(record)
        record_out[args.baseline_margin_key] = float(margin)
        record_out[args.effective_category_key] = str(effective_category)
        record_out["weight_orig"] = float(weight_base)
        record_out[args.weight_key] = float(weight_final)
        record_out["weight_mult"] = float(weight_mult)
        record_out["min_margin_used"] = float(min_margin_used)
        if args.write_all:
            record_out["keep_by_margin"] = bool(keep_by_margin)

        if args.write_all or keep_by_margin:
            out_f.write(json.dumps(record_out) + "\n")
        if keep_by_margin:
            kept += 1
            kept_margins.append(margin)
            per_cat_kept[effective_category] += 1
            per_cat_kept_margins[effective_category].append(margin)

        per_cat_weight_base[effective_category].append(weight_base)
        per_cat_weight_final[effective_category].append(weight_final)
        per_cat_min_margin_used[effective_category].append(min_margin_used)
        per_cat_weight_mult_used[effective_category].append(weight_mult)

    try:
        if args.use_existing_margin:
            # No model forward - just (re)filter/reweight using precomputed margins.
            with open(args.pairs, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if args.max_samples is not None and i >= int(args.max_samples):
                        break
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    if args.baseline_margin_key not in record:
                        raise ValueError(
                            f"Missing '{args.baseline_margin_key}' in record {i}. "
                            "Remove --use_existing_margin or precompute margins first."
                        )
                    margin = float(record[args.baseline_margin_key])
                    effective_category, rule = _effective_category_and_overrides(record)

                    total += 1
                    all_margins.append(margin)
                    per_cat_total[effective_category] += 1
                    per_cat_all_margins[effective_category].append(margin)
                    _write_record(record, margin, effective_category, rule)
        else:
            device = torch.device(f"cuda:{args.device}" if torch.cuda.is_available() else "cpu")
            model, tokenizer, config = load_model(args.base_checkpoint, device)

            max_length = int(getattr(config, "max_token_length", 640))
            dataset = DPOPairDataset(
                path=args.pairs,
                tokenizer=tokenizer,
                config=config,
                max_length=max_length,
                waveform_key=getattr(config, "waveform_key", "waveform_path"),
                prompt_key=getattr(config, "prompt_key", "prompt"),
                chosen_key=getattr(config, "chosen_key", "chosen"),
                rejected_key=getattr(config, "rejected_key", "rejected"),
                weight_key=getattr(config, "weight_key", "weight"),
            )

            if args.max_samples and args.max_samples < len(dataset):
                dataset = Subset(dataset, list(range(int(args.max_samples))))

            loader = DataLoader(
                dataset,
                batch_size=args.batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True,
                collate_fn=collate_with_idx,
            )

            use_autocast = device.type == "cuda"
            autocast_dtype = torch.bfloat16 if use_autocast else torch.float32

            with torch.no_grad():
                for batch in tqdm(loader, desc="Filtering", total=len(loader)):
                    signal = batch["signal"].to(device)
                    prompt_input_ids = batch["prompt_input_ids"].to(device)
                    prompt_attention_mask = batch["prompt_attention_mask"].to(device)

                    chosen_input_ids = batch["chosen_input_ids"].to(device)
                    chosen_attention_mask = batch["chosen_attention_mask"].to(device)
                    chosen_labels = batch["chosen_labels"].to(device)
                    rejected_input_ids = batch["rejected_input_ids"].to(device)
                    rejected_attention_mask = batch["rejected_attention_mask"].to(device)
                    rejected_labels = batch["rejected_labels"].to(device)

                    with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_autocast):
                        logits_c, labels_c = _forward_logits_and_labels(
                            model,
                            signal,
                            chosen_input_ids,
                            chosen_attention_mask,
                            chosen_labels,
                            prompt_input_ids,
                            prompt_attention_mask,
                        )
                        logits_r, labels_r = _forward_logits_and_labels(
                            model,
                            signal,
                            rejected_input_ids,
                            rejected_attention_mask,
                            rejected_labels,
                            prompt_input_ids,
                            prompt_attention_mask,
                        )
                        logp_c = _sequence_logp(logits_c, labels_c)
                        logp_r = _sequence_logp(logits_r, labels_r)

                    margins = (logp_c - logp_r).detach().cpu().tolist()
                    idxs = batch["idx"].detach().cpu().tolist()

                    for idx, margin in zip(idxs, margins):
                        record = dataset.dataset.records[idx] if isinstance(dataset, Subset) else dataset.records[idx]
                        effective_category, rule = _effective_category_and_overrides(record)

                        total += 1
                        all_margins.append(margin)
                        per_cat_total[effective_category] += 1
                        per_cat_all_margins[effective_category].append(margin)
                        _write_record(record, float(margin), effective_category, rule)

        out_f.close()

        def _summary(vals):
            if not vals:
                return {}
            vals_sorted = sorted(vals)
            n = len(vals_sorted)
            def _pct(p):
                if n == 1:
                    return vals_sorted[0]
                idx = int(round(p * (n - 1)))
                return vals_sorted[idx]
            return {
                "mean": sum(vals_sorted) / n,
                "median": _pct(0.5),
                "p05": _pct(0.05),
                "p25": _pct(0.25),
                "p75": _pct(0.75),
                "p95": _pct(0.95),
            }

        stats = {
            "total": total,
            "kept": kept,
            "kept_ratio": kept / total if total else 0.0,
            "min_margin": args.min_margin,
            "all_margins": _summary(all_margins),
            "kept_margins": _summary(kept_margins),
            "output": args.output,
        }

        # Per-category stats
        per_category = {}
        for cat in sorted(per_cat_total.keys()):
            total_cat = int(per_cat_total[cat])
            kept_cat = int(per_cat_kept.get(cat, 0))
            min_margin_used_mean = (
                sum(per_cat_min_margin_used.get(cat, [])) / len(per_cat_min_margin_used.get(cat, []))
                if per_cat_min_margin_used.get(cat) else None
            )
            weight_mult_used_mean = (
                sum(per_cat_weight_mult_used.get(cat, [])) / len(per_cat_weight_mult_used.get(cat, []))
                if per_cat_weight_mult_used.get(cat) else None
            )
            per_category[cat] = {
                "total": total_cat,
                "kept": kept_cat,
                "kept_ratio": (kept_cat / total_cat) if total_cat else 0.0,
                "min_margin_config": category_min_margins.get(cat, None),
                "weight_mult_config": category_weight_mult.get(cat, None),
                "min_margin_used_mean": min_margin_used_mean,
                "weight_mult_used_mean": weight_mult_used_mean,
                "all_margins": _summary(per_cat_all_margins.get(cat, [])),
                "kept_margins": _summary(per_cat_kept_margins.get(cat, [])),
                "weight_base_mean": (sum(per_cat_weight_base.get(cat, [])) / len(per_cat_weight_base.get(cat, [])))
                if per_cat_weight_base.get(cat) else None,
                "weight_final_mean": (sum(per_cat_weight_final.get(cat, [])) / len(per_cat_weight_final.get(cat, [])))
                if per_cat_weight_final.get(cat) else None,
            }
        stats["per_effective_category"] = per_category

        print(json.dumps(stats, indent=2))

        if args.stats_json:
            _atomic_json_dump(args.stats_json, stats)

        os.replace(out_tmp_path, args.output)
        out_tmp_path = None
    finally:
        if not out_f.closed:
            out_f.close()
        if out_tmp_path and os.path.exists(out_tmp_path):
            os.unlink(out_tmp_path)


if __name__ == "__main__":
    main()
