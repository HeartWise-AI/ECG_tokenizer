"""Per-endpoint readout policy for binary/quantitative endpoint questions.

WHY THIS EXISTS
---------------
Greedy decoding commits to an argmax over the whole vocabulary, so the model's
answer *prior* dominates the decision. On ACS that discards roughly a third of the
recoverable recall: the generated sentence gives sensitivity 0.48 at specificity
0.98, while the SAME frozen model scored by its Yes/No log-prob margin reaches
AUROC 0.85 and a 0.70/0.84 operating point.

The best readout is ENDPOINT-SPECIFIC — measured, not assumed (see the table in
`ENDPOINT_POLICY`). Numeric parsing wins where the model emits a number; the
generated text wins where the margin is uninformative; the margin wins where the
text answer is miscalibrated.

CALIBRATION PROVENANCE
----------------------
Platt (a, b) and the decision threshold were fitted on TRAIN-side ECGs
(patient-disjoint from test, verified 0 overlap) and then LOCKED — no test data
touched the fit, and the full test set stays available for evaluation. Source:
`scripts/calibrate_endpoints_on_train.py` -> analysis/endpoint_pyes_concatmix/
endpoint_readout_train_calibrated.json (concat_linear checkpoint, REGEN parquet,
5,958 stratified train ECGs).

DEPLOYMENT CONTRACT
-------------------
This is a scoring rule, not a generator: it needs no ground truth at inference and
is unrelated to the BANNED best-of-N judge selection. Where policy is "margin", the
reported binary answer comes from the score, which CAN disagree with the sentence
the model writes — that disagreement is where the recovered ACS sensitivity comes
from, so callers must decide what to surface (see `ReadoutResult.disagrees`).
"""
from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional

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
        "caveat": "ECE 0.170 — the RANKING transfers but the probability scale does not "
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
                    "threshold converts that into bal-acc 0.65 — WORSE than text. Sensitivity "
                    "also swung 0.48 -> 0.72 between two training draws.",
        "caveat": "UNRESOLVED — revisit once the threshold is stable across sampling. Keeping "
                  "text is the conservative choice, not a positive result for text.",
    },
}

_EF_RE = re.compile(r"ejection fraction is\s*(\d+(?:\.\d+)?)\s*%", re.I)


@dataclass
class ReadoutResult:
    endpoint: str
    policy: Policy
    answer: Optional[bool]          # None when the policy could not produce an answer
    score: Optional[float]          # calibrated probability (margin) or parsed value (numeric)
    text_answer: Optional[bool]     # what greedy decoding said, always populated when parseable
    disagrees: bool                 # score-based answer != text answer (callers must handle)


def _load_calibration() -> Dict[str, Dict[str, float]]:
    if not CALIBRATION_JSON.exists():
        raise FileNotFoundError(
            f"locked calibration not found at {CALIBRATION_JSON}; regenerate with "
            "scripts/calibrate_endpoints_on_train.py"
        )
    raw = json.loads(CALIBRATION_JSON.read_text())["endpoints"]
    return {
        k: {"a": float(v["platt_a"]), "b": float(v["platt_b"]),
            "threshold": float(v["locked_threshold_prob"])}
        for k, v in raw.items()
    }


def parse_text_yes_no(text: str) -> Optional[bool]:
    t = str(text).strip().lower()
    if t.startswith("yes") or "high risk" in t or "present" in t:
        return True
    if t.startswith("no") or t.startswith("low") or "unlikely" in t or "not present" in t:
        return False
    return None


def parse_ef(text: str) -> Optional[float]:
    m = _EF_RE.search(str(text))
    return float(m.group(1)) if m else None


def read_endpoint(
    endpoint: str,
    *,
    generation: str,
    margin: Optional[float] = None,
    calibration: Optional[Dict[str, Dict[str, float]]] = None,
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
    text_answer = parse_text_yes_no(generation)

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
            return ReadoutResult(endpoint, "margin", text_answer, None, text_answer, False)
        cal = (calibration or _load_calibration())[endpoint]
        prob = 1.0 / (1.0 + math.exp(-(cal["a"] * margin + cal["b"])))
        answer = prob >= cal["threshold"]
        return ReadoutResult(endpoint, "margin", answer, prob, text_answer,
                             disagrees=text_answer is not None and answer != text_answer)

    return ReadoutResult(endpoint, "text", text_answer, None, text_answer, False)
