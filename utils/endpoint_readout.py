"""Per-endpoint readout policy for binary/quantitative endpoint questions.

WHY THIS EXISTS
---------------
Greedy decoding commits to an argmax over the whole vocabulary, so the model's
answer *prior* dominates the decision. On ACS that discards roughly a third of the
recoverable recall: the generated sentence gives sensitivity 0.48 at specificity
0.98, while the SAME frozen model scored by its Yes/No log-prob margin reaches
AUROC 0.85 and a 0.70/0.84 operating point.

The best readout is ENDPOINT-SPECIFIC - measured, not assumed (see the table in
`ENDPOINT_POLICY`). Numeric parsing wins where the model emits a number; the
generated text wins where the margin is uninformative; the margin wins where the
text answer is miscalibrated.

CALIBRATION PROVENANCE
----------------------
Platt (a, b) and the decision threshold were fitted on TRAIN-side ECGs
(patient-disjoint from test, verified 0 overlap) and then LOCKED - no test data
touched the fit, and the full test set stays available for evaluation. Source:
`scripts/calibrate_endpoints_on_train.py` -> analysis/endpoint_pyes_concatmix/
endpoint_readout_train_calibrated.json (concat_linear checkpoint, REGEN parquet,
5,958 stratified train ECGs).

DEPLOYMENT CONTRACT
-------------------
This is a scoring rule, not a generator: it needs no ground truth at inference and
is unrelated to the BANNED best-of-N judge selection. Where policy is "margin", the
reported binary answer comes from the score, which CAN disagree with the sentence
the model writes - that disagreement is where the recovered ACS sensitivity comes
from, so callers must decide what to surface (see `ReadoutResult.disagrees`).
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Literal, Mapping, Optional

from utils.artifact_provenance import (
    exclusive_artifact_lock,
    file_identity,
    validate_artifact_bundle,
)
from utils.endpoint_contract import (
    endpoint_contract_sha256,
    endpoint_implementation_identity,
)

Policy = Literal["numeric", "text", "margin"]

CALIBRATION_JSON = Path(
    "/volume/ECG_tokenizer/analysis/endpoint_pyes_concatmix/"
    "endpoint_readout_train_calibrated.json"
)

# Per-endpoint policy. `measured` records the head-to-head that chose it, so a future
# reader can see the decision was evidence-based and re-check it after a retrain.
ENDPOINT_POLICY: Dict[str, Dict[str, object]] = {
    "acute_coronary_occlusion": {
        "policy": "margin",
        "measured": "margin AUROC 0.85 (95% CI 0.82-0.87), locked op point sens 0.70 / spec 0.84; "
                    "greedy text sens 0.48 / spec 0.98. Margin wins decisively.",
        "caveat": "ECE 0.170 - the RANKING transfers but the probability scale does not "
                  "(train prevalence != test). Use the threshold as a decision rule; do NOT "
                  "report these as calibrated probabilities.",
    },
    "lvef_lte_40": {
        "policy": "numeric",
        "measured": "parsed numeric EF AUROC 0.83 (95% CI 0.82-0.85, full test) vs margin 0.75. "
                    "Asking a binary question throws away the model's own continuous estimate.",
        "caveat": "Falls back to margin only if no EF value can be parsed.",
    },
    "lvef_lt_50": {
        "policy": "numeric",
        "measured": "same parsed-EF path as lvef_lte_40; margin only reaches 0.72.",
        "caveat": None,
    },
    "incident_afib_5y": {
        "policy": "text",
        "measured": "generated text bal-acc 0.73 vs margin AUROC 0.61 (ensemble 0.57, i.e. "
                    "ensembling makes it worse). Consistent with the earlier P1 finding that "
                    "AFib P(Yes) regresses toward chance.",
        "caveat": None,
    },
    "shd": {
        "policy": "text",
        "measured": "margin ranks better (AUROC 0.73 vs text bal-acc 0.68) but the locked "
                    "threshold converts that into bal-acc 0.65 - WORSE than text. Sensitivity "
                    "also swung 0.48 -> 0.72 between two training draws.",
        "caveat": "UNRESOLVED - revisit once the threshold is stable across sampling. Keeping "
                  "text is the conservative choice, not a positive result for text.",
    },
}

_EF_RE = re.compile(r"ejection fraction is\s*(\d+(?:\.\d+)?)\s*%", re.I)
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_LEADING_NEGATIVE_RE = re.compile(
    r"^(?:no\b(?!\s+(?:question|doubt)\b)|low\s+(?:risk|likelihood|probability)\b)",
    re.I,
)
_UNLIKELY_RE = re.compile(
    r"(?:^|\b(?:is|are|was|were|appears?|seems?)\s+)unlikely\b",
    re.I,
)
_CLAUSE_SEPARATOR_RE = re.compile(r"\s*(?:[.;]|\bbut\b|\bhowever\b|\bwhereas\b)\s*", re.I)
_ENDPOINT_TEXT_TERMS = {
    "incident_afib_5y": ("atrial fibrillation", "high risk", "elevated risk"),
    "acute_coronary_occlusion": (
        "acute coronary occlusion",
        "acute coronary artery occlusion",
    ),
    "shd": ("structural heart disease",),
}
_DEFAULT_TEXT_TERMS = (
    "high risk",
    "elevated risk",
    "atrial fibrillation",
    "acute coronary occlusion",
    "acute coronary artery occlusion",
    "structural heart disease",
    "present",
)
_KNOWN_ENDPOINT_FINDING_TERMS = (
    "atrial fibrillation",
    "acute coronary occlusion",
    "acute coronary artery occlusion",
    "structural heart disease",
)


@dataclass
class ReadoutResult:
    endpoint: str
    policy: Policy
    answer: Optional[bool]          # None when the policy could not produce an answer
    score: Optional[float]          # calibrated probability (margin) or parsed value (numeric)
    text_answer: Optional[bool]     # what greedy decoding said, always populated when parseable
    disagrees: bool                 # score-based answer != text answer (callers must handle)


def _require_file_identity(value: object, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"calibration provenance field '{field}' is not a file identity")
    path = value.get("path")
    size = value.get("size")
    digest = value.get("sha256")
    if (
        not isinstance(path, str)
        or not path
        or not isinstance(size, int)
        or isinstance(size, bool)
        or size < 0
        or not isinstance(digest, str)
        or not _SHA256_RE.fullmatch(digest)
    ):
        raise ValueError(f"calibration provenance field '{field}' is invalid")
    return value


def _validate_calibration_entry(endpoint: str, value: object) -> Dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError(f"calibration for '{endpoint}' is not an object")
    try:
        entry = {
            "a": float(value["a"] if "a" in value else value["platt_a"]),
            "b": float(value["b"] if "b" in value else value["platt_b"]),
            "threshold": float(
                value["threshold"]
                if "threshold" in value
                else value["locked_threshold_prob"]
            ),
        }
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"calibration for '{endpoint}' is incomplete or invalid") from exc
    if not all(math.isfinite(entry[key]) for key in ("a", "b", "threshold")):
        raise ValueError(f"calibration for '{endpoint}' contains a non-finite value")
    if not 0.0 <= entry["threshold"] <= 1.0:
        raise ValueError(f"calibration threshold for '{endpoint}' must be between 0 and 1")
    return entry


def _read_committed_calibration_document() -> Mapping[str, object]:
    bundle_manifest = CALIBRATION_JSON.with_name(
        "endpoint_readout_train_calibrated.bundle.json"
    )
    bundle_lock = CALIBRATION_JSON.parent / ".endpoint-readout-train.lock"
    required_files = (
        CALIBRATION_JSON.name,
        "endpoint_readout_train_calibrated.md",
        "train_calib_margins.npz",
    )
    with exclusive_artifact_lock(bundle_lock):
        manifest_identity = file_identity(bundle_manifest)
        manifest = validate_artifact_bundle(
            bundle_manifest,
            required_files=required_files,
        )
        expected_member = manifest["files"].get(CALIBRATION_JSON.name)
        if not isinstance(expected_member, Mapping):
            raise ValueError("locked calibration bundle has an invalid JSON member")
        try:
            payload = CALIBRATION_JSON.read_bytes()
        except OSError as exc:
            raise ValueError("locked calibration changed during committed read") from exc
        payload_identity = {
            "path": str(CALIBRATION_JSON.resolve()),
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        if payload_identity != dict(expected_member):
            raise ValueError("locked calibration changed during committed read")
        try:
            validate_artifact_bundle(
                bundle_manifest,
                required_files=required_files,
            )
            if file_identity(bundle_manifest) != manifest_identity:
                raise ValueError("locked calibration changed during committed read")
        except (OSError, ValueError) as exc:
            raise ValueError("locked calibration changed during committed read") from exc

    try:
        document = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError) as exc:
        raise ValueError("locked calibration artifact is invalid JSON") from exc
    if not isinstance(document, Mapping):
        raise ValueError("locked calibration artifact must be an object")
    return document


def _load_calibration(checkpoint_path: str | Path) -> Dict[str, Dict[str, float]]:
    if not CALIBRATION_JSON.exists():
        raise FileNotFoundError(
            f"locked calibration not found at {CALIBRATION_JSON}; regenerate with "
            "scripts/calibrate_endpoints_on_train.py"
        )
    document = _read_committed_calibration_document()
    try:
        provenance = document["provenance"]
        raw = document["endpoints"]
    except (KeyError, TypeError) as exc:
        raise ValueError("locked calibration artifact is missing required fields") from exc
    if not isinstance(provenance, Mapping):
        raise ValueError("locked calibration artifact has invalid provenance")
    if provenance.get("schema_version") != 2 or provenance.get("kind") != "endpoint_train_calibration":
        raise ValueError("locked calibration artifact has an unsupported provenance contract")

    checkpoint_identity = _require_file_identity(provenance.get("checkpoint"), "checkpoint")
    for field in ("train_parquet", "test_parquet", "test_margins", "script"):
        _require_file_identity(provenance.get(field), field)
    expected_contract = endpoint_contract_sha256(ENDPOINT_POLICY)
    if provenance.get("endpoint_contract_sha256") != expected_contract:
        raise ValueError("locked calibration endpoint prompt contract does not match this code")
    if checkpoint_identity != file_identity(checkpoint_path):
        raise ValueError("locked calibration checkpoint does not match the requested checkpoint")

    cache_provenance = provenance.get("test_margin_cache_provenance")
    semantic = cache_provenance.get("semantic") if isinstance(cache_provenance, Mapping) else None
    if not isinstance(semantic, Mapping):
        raise ValueError("locked calibration is missing test margin cache provenance")
    if (
        semantic.get("checkpoint") != checkpoint_identity
        or semantic.get("parquet") != provenance.get("test_parquet")
        or semantic.get("endpoint_contract_sha256") != expected_contract
        or semantic.get("implementation") != endpoint_implementation_identity()
    ):
        raise ValueError("locked calibration is not bound to its test margin inputs")
    if not isinstance(raw, Mapping):
        raise ValueError("locked calibration endpoints field is not an object")
    return {str(key): _validate_calibration_entry(str(key), value) for key, value in raw.items()}


def has_binary_negation(
    text: str,
    positive_terms: Iterable[str] = (
        "high risk",
        "elevated risk",
        "atrial fibrillation",
        "present",
        "structural heart disease",
        "acute coronary occlusion",
        "acute coronary artery occlusion",
    ),
) -> bool:
    """Return whether a binary positive finding is explicitly negated."""
    normalized = str(text).strip().lower()
    if _LEADING_NEGATIVE_RE.match(normalized) or _UNLIKELY_RE.search(normalized):
        return True
    for term in positive_terms:
        escaped = re.escape(term).replace(r"\ ", r"\s+")
        patterns = (
            rf"\b(?:no|without)\s+(?:any\s+|a\s+|an\s+|the\s+)?{escaped}\b",
            rf"\b(?:no|without)\s+(?:(?:clear|convincing|definite)\s+)?"
            rf"(?:evidence|findings?|signs?|indication)\s+"
            rf"(?:of|for|to\s+(?:suggest|support|indicate))\s+"
            rf"(?:any\s+|a\s+|an\s+|the\s+)?{escaped}\b",
            rf"\b(?:does|do|did)\s+not\s+"
            rf"(?:show|demonstrate|indicate|suggest|support|reveal|identify|detect)\s+"
            rf"(?:any\s+|a\s+|an\s+|the\s+)?{escaped}\b",
            rf"\b{escaped}\b\s+(?:is|are|was|were)\s+not\s+"
            r"(?:present|seen|identified|detected|evident|likely)\b",
            rf"\bnot\s+{escaped}\b",
            rf"\blow\s+(?:risk|likelihood|probability)\s+(?:of|for)\s+{escaped}\b",
            rf"\b(?:risk|likelihood|probability)\s+(?:of|for)\s+{escaped}\b"
            rf"[^.;]*?\b(?:is|was|appears?|seems?|remains?)\s+low\b",
            rf"\b{escaped}\b[^.;]*?\b(?:possible\s+but\s+unlikely|unlikely)\b",
        )
        if any(re.search(pattern, normalized) for pattern in patterns):
            return True
    return False


def parse_text_yes_no(text: str, *, endpoint: str | None = None) -> Optional[bool]:
    normalized = str(text).strip().lower()
    terms = _ENDPOINT_TEXT_TERMS.get(endpoint) if endpoint is not None else None
    positive_phrases = terms or _DEFAULT_TEXT_TERMS
    analysis_text = re.sub(r"\bpossible\s+but\s+unlikely\b", "unlikely", normalized)
    outcomes: set[bool] = set()
    for clause in _CLAUSE_SEPARATOR_RE.split(analysis_text):
        if not clause:
            continue
        if endpoint is not None:
            mentions_target = any(term in clause for term in positive_phrases)
            mentions_other_endpoint = any(
                term in clause
                for term in _KNOWN_ENDPOINT_FINDING_TERMS
                if term not in positive_phrases
            )
            if mentions_other_endpoint and not mentions_target:
                continue
        if has_binary_negation(clause, positive_phrases):
            outcomes.add(False)
        elif re.match(r"^yes\b", clause) or any(
            phrase in clause for phrase in positive_phrases
        ):
            outcomes.add(True)
    if len(outcomes) > 1:
        return None
    return next(iter(outcomes)) if outcomes else None


def parse_ef(text: str) -> Optional[float]:
    m = _EF_RE.search(str(text))
    if not m:
        return None
    value = float(m.group(1))
    return value if math.isfinite(value) and 0.0 <= value <= 100.0 else None


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def read_endpoint(
    endpoint: str,
    *,
    generation: str,
    margin: Optional[float] = None,
    calibration: Optional[Dict[str, Dict[str, float]]] = None,
    checkpoint_path: Optional[str | Path] = None,
) -> ReadoutResult:
    """Apply the measured policy for `endpoint`.

    `margin` is logit(Yes) - logit(No) at the first answer position, as produced by
    scripts/evaluate_endpoint_pyes_calibrated.get_margins. It is required only for
    endpoints whose policy is "margin" (and as the numeric fallback).
    """
    if endpoint not in ENDPOINT_POLICY:
        raise KeyError(f"no measured policy for endpoint '{endpoint}'; "
                       f"known: {sorted(ENDPOINT_POLICY)}")
    policy: Policy = ENDPOINT_POLICY[endpoint]["policy"]  # type: ignore[assignment]
    text_answer = parse_text_yes_no(generation, endpoint=endpoint)

    if policy == "numeric":
        ef = parse_ef(generation)
        if ef is not None:
            threshold = 40.0 if endpoint == "lvef_lte_40" else 50.0
            answer = ef <= threshold if endpoint == "lvef_lte_40" else ef < threshold
            return ReadoutResult(endpoint, "numeric", answer, ef, text_answer,
                                 disagrees=text_answer is not None and answer != text_answer)
        policy = "margin"  # documented fallback when no EF value is present

    if policy == "margin":
        if margin is None:
            raise ValueError(f"margin is required for endpoint '{endpoint}'")
        if not math.isfinite(float(margin)):
            raise ValueError("margin must be finite")
        if calibration is None:
            if checkpoint_path is None:
                raise ValueError(
                    "checkpoint_path is required when loading the locked calibration"
                )
            calibration = _load_calibration(checkpoint_path)
        try:
            cal = _validate_calibration_entry(endpoint, calibration[endpoint])
        except KeyError as exc:
            raise ValueError(f"calibration is missing endpoint '{endpoint}'") from exc
        prob = _sigmoid(cal["a"] * margin + cal["b"])
        answer = prob >= cal["threshold"]
        return ReadoutResult(endpoint, "margin", answer, prob, text_answer,
                             disagrees=text_answer is not None and answer != text_answer)

    return ReadoutResult(endpoint, "text", text_answer, None, text_answer, False)
