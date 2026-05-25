"""Per-category verifiable rewards for ECG QA.

WHY: The LLM-as-a-judge is slow, expensive, and noisy. For each
prompt_category in our dataset, the ground truth has a STRUCTURED format
(Yes/No prefix, numeric range, key=value, etc.) that can be checked
deterministically — that's a verifiable reward.

Each verifier returns a float in [0, 1]:
  - 1.0 = output matches GT
  - 0.5 = partial credit
  - 0.0 = miss / format violation

Categories and their verifiers (see route() below):
  binary_yesno: random_finding_question, structural_heart_disease,
    category_pericarditis, category_chamber_enlargement,
    category_rhythm, category_conduction, category_infarct_ischemia
  risk_level: afib_risk (Low/High/Uncertain)
  numeric_lvef: lvef (within ±5 percentage points)
  numeric_interval: ecg_interval (extract first number)
  axis: localization_qrs_axis (Left/Right/Normal)
  culprit: culprit_artery (artery name + completeness)
  json_match: json_interpretation (F1 over key=value pairs)
  ontology_f1: interpretation, classification, category_other
  lead_list: localization_t_wave
  urgency: urgency_assessment
  acs: acs_severity (Yes/No + culprit + completeness)
"""

from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

try:
    from utils.rewards_labelset import _load_alias_index
except Exception:
    _load_alias_index = None

try:
    from utils.constants import (
        BERT_CLASS_THRESHOLDS,
        DEEPECG_CATEGORIES,
        DEEPECG_PATHOLOGICAL_LIMIT,
        ECG_PATTERNS,
    )
except Exception:
    BERT_CLASS_THRESHOLDS = []
    DEEPECG_CATEGORIES = {}
    DEEPECG_PATHOLOGICAL_LIMIT = {}
    ECG_PATTERNS = []


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _payload(obj: Any) -> Optional[Dict[str, Any]]:
    """Return structured label payloads passed through OpenRLHF JSON labels."""
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, str):
        s = obj.strip()
        if s.startswith("{") and s.endswith("}"):
            try:
                parsed = json.loads(s)
            except Exception:
                return None
            if isinstance(parsed, dict):
                return parsed
    return None


def _obj_text(obj: Any) -> str:
    payload = _payload(obj)
    if payload is not None:
        for key in ("ground_truth", "text", "answer", "reference", "prediction"):
            value = payload.get(key)
            if value is not None:
                return str(value)
    return "" if obj is None else str(obj)

_PREFIX_JUNK_RE = re.compile(
    r"^\s*(?:"
    r"einfo\s*:|info\s*:|note\s*:|answer\s*:|user|assistant|model|"
    r"(?:ei|en|ef)\s+(?=(?:yes|no)\b)|"
    r"-{2,}|\*{2,}|={2,}|"
    r"here (?:is|are)[^\n:]*[:\n]|"
    r"[:;,.\-*\s]+"
    r")\s*",
    re.IGNORECASE)

_BAD_PREAMBLE_RE = re.compile(
    r"^\s*(?:"
    r"questions?\s+about|"
    r"to\s+answer|"
    r"i\s+(?:need|would\s+need|cannot|can't)|"
    r"the\s+value\s+if\s+known|"
    r"tell\s+me\s+what|"
    r"display\s+ecg|"
    r"efective\b|"
    r"if\s+no\s+data|"
    r"findings\s+and|"
    r"item\s+list\s+of\s+findings|"
    r"ebook\b|"
    r"a\s+[\"']?(?:low|high|uncertain|moderate)\s+risk|"
    r"\?\s*yes/no"
    r")",
    re.IGNORECASE,
)


def _strip_prefix(s: str) -> tuple:
    """Strip leading junk the model sometimes emits (': ', 'info:', '---',
    'user\\n', 'Here is an ECG:', bare punctuation). Returns
    (clean_string, had_prefix). The LLM-judge reads past these with only a
    mild penalty (~0.7-0.8 vs 1.0); our parsers must do the same instead of
    failing to 0.0. (Root cause of biggest verifier↔judge gap, found
    2026-05.)"""
    if not s:
        return "", False
    orig = s
    # apply up to 3 times to peel stacked junk ("---\nuser\n: ")
    for _ in range(3):
        m = _PREFIX_JUNK_RE.match(s)
        if not m or m.end() == 0:
            break
        s = s[m.end():]
    return s.strip(), (s.strip() != orig.strip())


def _has_bad_preamble(s: str) -> bool:
    """Prompt echoes / instruction fragments are format failures to the judge."""
    return bool(_BAD_PREAMBLE_RE.match(s or ""))


def _first_word(s: str) -> str:
    """Return the first alphabetic word, lowercased. Strips colons/dashes."""
    s = _strip_prefix(s)[0]
    s = s.strip().lstrip(":-*. ").strip()
    m = re.match(r"([A-Za-z]+)", s)
    return m.group(1).lower() if m else ""


def _extract_first_number(s: str) -> Optional[float]:
    m = re.search(r"-?\d+(?:\.\d+)?", s or "")
    return float(m.group(0)) if m else None


def _is_ms_interval_text(s: str) -> bool:
    s = (s or "").lower()
    return bool(re.search(r"\b(?:qtc|qt|pr|qrs|interval)\b", s)) and "ms" in s


def _canonical_yesno(s: str) -> Optional[str]:
    """Normalize: 'Yes', 'No', or 'Borderline' → first word. Handles the
    *** STEMI *** prefix by checking if 'Yes' appears anywhere in first 50 chars."""
    if not s:
        return None
    s_clean = _strip_prefix(s)[0]
    s_lower = s_clean.strip().lower()
    fw = _first_word(s)
    if fw in ("yes", "y", "positive", "abnormal"):
        return "yes"
    if fw in ("no", "n", "negative", "normal"):
        return "no"
    if fw in ("borderline",):
        return "borderline"
    # Special: "*** ... *** - Yes ..." or "*** ... ***" → treat as Yes
    if s_lower.startswith("***") or "stemi" in s_lower[:80] or "acute mi" in s_lower[:80]:
        return "yes"
    return None


# ─────────────────────────────────────────────────────────────────────────
# Per-category verifiers
# ─────────────────────────────────────────────────────────────────────────

def verify_binary_yesno(pred: str, gt: str) -> float:
    """1.0 if both yes/no canonicalize to the same value, else 0."""
    pyn = _canonical_yesno(pred)
    gyn = _canonical_yesno(gt)
    if pyn is None or gyn is None:
        return 0.0
    return 1.0 if pyn == gyn else 0.0


def _extract_clear_yesno(s: str) -> Optional[str]:
    """Find the answer yes/no even after harmless prefix text."""
    s = _strip_prefix(s or "")[0]
    low = s.lower()
    if "yes/no" in low or "yes or no" in low:
        # Remove prompt wording before searching for the actual answer.
        low = re.sub(r"\byes\s*/\s*no\b", " ", low)
        low = re.sub(r"\byes\s+or\s+no\b", " ", low)
    if re.search(r"\bno\s+(?:yes/no\s+)?answer\b", low):
        return None
    matches = []
    for m in re.finditer(r"(?:^|[\n|.;:,-]\s*)(yes|no)\b", low):
        matches.append(m.group(1))
    if matches:
        return matches[-1]
    return _canonical_yesno(s)


def verify_extracted_yesno(pred: str, gt: str) -> float:
    pyn = _extract_clear_yesno(pred)
    gyn = _extract_clear_yesno(gt)
    if pyn is None or gyn is None:
        return 0.0
    return 1.0 if pyn == gyn else 0.0


def verify_yesno_with_findings(pred: str, gt: str) -> float:
    """For the diagnostic 'category_*' prompts. GT is one of:
      - 'No - no evidence of X'              (negative)
      - 'Yes - finding1; finding2; ...'      (positive + findings)
      - free-text findings ('Sinus rhythm (HR: 61); Regular rhythm')

    Scoring: the Yes/No gate is a NECESSARY gate but earns only a small floor;
    the findings F1 carries the rest (so the model can't farm the gate word):
      * gate disagrees -> 0.0
      * No  -> 1.0 if findings clean, else 0.2 + 0.8 * F1 (spurious findings on a
               negative are penalized, as the judge does)
      * Yes -> 0.1 + 0.9 * F1(findings)
      * GT with no gate (pure findings) -> F1 directly
    (2026-05 de-game: lowered the gate floors Yes 0.2->0.1 / No 0.4->0.2 — the old
    floors over-rewarded a correct gate regardless of findings; gap vs judge
    +0.087 -> +0.075 with no corr/saturation cost. The judge heavily penalizes
    wrong findings even when Yes/No is right.)
    """
    gyn = _canonical_yesno(gt)
    pyn = _canonical_yesno(pred)

    # GT has an explicit Yes/No gate
    if gyn in ("yes", "no"):
        if pyn != gyn:
            if gyn == "yes" and pyn is None:
                # Judge gives partial credit when the answer omits the Yes
                # gate but otherwise names the requested positive finding.
                return 0.7 * verify_ontology_f1(pred, gt)
            return 0.0
        f1 = verify_ontology_f1(pred, gt)
        if gyn == "no":
            # Negative examples are often "No - no evidence of X". If the
            # model also mentions positive findings, the judge penalizes it.
            return 1.0 if f1 == 1.0 else 0.2 + 0.8 * f1
        # positive: gate right, now grade the findings
        return 0.1 + 0.9 * f1
    # No gate in GT -> pure findings comparison (e.g. category_rhythm)
    return verify_ontology_f1(pred, gt)


def verify_pericarditis(pred: str, gt: str) -> float:
    """Pericarditis is deterministic: the label match is pericarditis itself.

    The generated QA variants disagree about whether the answer mentions ST
    changes. Treat "acute pericarditis", "pericarditis", and ontology aliases
    as the same positive finding, regardless of the ST-change suffix.
    """
    pyn = _canonical_yesno(pred)
    gyn = _canonical_yesno(gt)
    if gyn in ("yes", "no") and pyn in ("yes", "no") and pyn != gyn:
        return 0.0
    if gyn == "no":
        return 1.0 if pyn == "no" else 0.0

    p_terms = _ontology_terms(pred)
    g_terms = _ontology_terms(gt)
    pericarditis_terms = {"pericarditis", "acute pericarditis"}
    if g_terms & pericarditis_terms:
        return 1.0 if (p_terms & pericarditis_terms) else 0.0
    return verify_yesno_with_findings(pred, gt)


def _hr_penalty(pred: str, gt: str) -> float:
    """If GT states 'HR: X bpm' and pred states a heart rate, return a
    multiplier that penalizes large HR errors (the judge does this for
    rhythm/interpretation). No HR in GT → 1.0 (no penalty)."""
    g = re.search(r"hr[:\s]*([0-9]+(?:\.[0-9]+)?)\s*bpm", gt or "", re.I) \
        or re.search(r"([0-9]+(?:\.[0-9]+)?)\s*bpm", gt or "", re.I)
    if not g:
        return 1.0
    p = re.search(r"hr[:\s]*([0-9]+(?:\.[0-9]+)?)\s*bpm", pred or "", re.I) \
        or re.search(r"([0-9]+(?:\.[0-9]+)?)\s*bpm", pred or "", re.I)
    if not p:
        return 0.8  # GT has HR, pred omitted it — mild penalty
    diff = abs(float(p.group(1)) - float(g.group(1)))
    if diff < 10:
        return 1.0
    return 0.0


def _find_hr(s: str) -> Optional[float]:
    m = re.search(r"hr[:\s]*([0-9]+(?:\.[0-9]+)?)\s*bpm", s or "", re.I) \
        or re.search(r"([0-9]+(?:\.[0-9]+)?)\s*bpm", s or "", re.I)
    return float(m.group(1)) if m else None


def _hr_accuracy(pred: str, gt: str) -> Optional[float]:
    """Graded heart-rate accuracy in [0, 1] (1.0 within 5 bpm, linear to 0 at
    30 bpm). Returns None when the GT states no HR (no HR component). Pred that
    omits HR while GT has one scores 0.0. Used as an additive component for the
    interpretation reward, which BERT diagnosis-labels can't see on their own."""
    g = _find_hr(gt)
    if g is None:
        return None
    p = _find_hr(pred)
    if p is None:
        return 0.0
    diff = abs(p - g)
    if diff <= 5.0:
        return 1.0
    if diff >= 30.0:
        return 0.0
    return 1.0 - (diff - 5.0) / 25.0


def verify_rhythm(pred: str, gt: str) -> float:
    """category_rhythm: Yes/No gate + findings F1, then scaled by an
    HR-accuracy penalty. The judge penalizes wrong heart rate even when
    the rhythm label is right (e.g. GT 'Sinus rhythm (HR: 54)' vs pred
    'Sinus rhythm (HR: 80)' → judge 0.70 not 1.0)."""
    base = verify_yesno_with_findings(pred, gt)
    return base * _hr_penalty(pred, gt)


def _ecg_class(s: str) -> Optional[str]:
    """3-way ECG class from a classification answer: normal / borderline /
    abnormal. The class is the primary verifiable signal."""
    s = _strip_prefix(s)[0].lower()
    if not s:
        return None
    # borderline first (most specific)
    if "borderline" in s or "minor finding" in s:
        return "borderline"
    # Normal phrasings that contain 'abnormal' as a substring ("No significant
    # abnormalities") must be matched before the abnormal token.
    if ("no significant abnormalit" in s or re.search(r"\bnormal\s+ecg\b", s)
            or "is normal" in s or "within normal limits" in s or "routine follow" in s):
        return "normal"
    # Explicit abnormal / pathology token takes priority over the leading Yes/No,
    # which is question-dependent and ambiguous: "Is this normal?" -> "No - Abnormal
    # ECG ..." is ABNORMAL, not normal. (Old code's `startswith("no")->normal`
    # misclassed ~23% of rows -> auto-1.0 ignoring findings.)
    if (re.search(r"\babnormal\s+ecg\b", s) or "pathological finding" in s
            or "significant abnormalit" in s):
        return "abnormal"
    # Last resort: bare leading Yes/No when no explicit class token is present.
    if s.startswith("yes"):
        return "abnormal"
    if s.startswith("no"):
        return "normal"
    return None


def verify_classification(pred: str, gt: str) -> float:
    """classification: the answer states an ECG class (normal / borderline /
    abnormal) plus a findings list. Scoring:
      * class mismatch -> 0.0
      * class match, GT has no findings (or normal) -> 1.0
      * class match, GT has findings -> 0.2 + 0.8 * F1(findings ontology)
    The class is a NECESSARY gate but not sufficient: it earns only a small 0.2
    credit, the findings F1 carries the rest. (2026-05 de-game: the old 0.5
    floor let the model farm the class word — verifier paid 0.78 where the judge
    paid 0.52, gap +0.25; floor 0.2 cuts the gap to +0.20, raises corr 0.59->0.65,
    and widens within-group spread for GRPO. The 0.2 floor still keeps a correct
    class above a wrong one when findings overlap is zero, preserving the
    class-correctness gradient.)"""
    gc = _ecg_class(gt)
    pc = _ecg_class(pred)
    gyn = _canonical_yesno(gt)
    pyn = _canonical_yesno(pred)
    if gyn in ("yes", "no") and pyn in ("yes", "no") and pyn != gyn:
        return 0.0
    if gc is None:
        # GT has no clear class -> fall back to findings F1
        return verify_ontology_f1(pred, gt)
    if pc != gc:
        return 0.0
    # class matches — does the GT enumerate findings?
    gl = gt.lower()
    has_findings = ("finding" in gl) and (";" in gt or ":" in gt)
    if not has_findings or gc == "normal":
        return 1.0
    return 0.2 + 0.8 * verify_ontology_f1(pred, gt)


def verify_risk_level(pred: str, gt: str) -> float:
    """afib_risk: STRICT binary — 1.0 iff the risk level (low/high/uncertain)
    matches the GT, else 0.0. Nothing else counts. Prefix junk is stripped
    first so 'info: Low risk ...' still parses as 'low'.
    (Per spec 2026-05: afib is 1 only if the risk matches, else 0.)"""
    def level(s):
        s = _strip_prefix(s)[0].strip().lower()
        head = s[:120]
        if re.search(r"\b(?:has|with|current(?:ly)?)\s+(?:afib|atrial fibrillation)\b", head):
            return None
        levels = {
            "low": bool(re.search(r"\blow\s+risk\b", head)),
            "high": bool(re.search(r"\bhigh\s+risk\b", head)),
            "uncertain": bool(re.search(r"\b(?:uncertain|unknown|moderate|intermediate)\s+risk\b", head)),
        }
        if sum(levels.values()) > 1:
            return None
        if head.startswith("low risk"):
            return "low"
        if head.startswith("high risk"):
            return "high"
        if head.startswith(("uncertain risk", "unknown risk", "moderate risk", "intermediate risk")):
            return "uncertain"
        if head.startswith("yes"):  # 'Yes - high/low risk ...'
            if re.search(r"\blow\s+risk\b", head):
                return "low"
            if re.search(r"\bhigh\s+risk\b", head):
                return "high"
        if head.startswith("no"):   # 'No - low risk ...'
            if re.search(r"\blow\s+risk\b", head):
                return "low"
            if re.search(r"\bhigh\s+risk\b", head):
                return "high"
        return None
    p = level(pred)
    g = level(gt)
    # Pure binary per spec: 1.0 iff risk level matches, else 0.0. Nothing
    # else counts. (This intentionally diverges from the LLM-judge, which
    # does nuanced scoring — by design we want a clean verifiable signal.)
    return 1.0 if (p is not None and g is not None and p == g) else 0.0


def verify_numeric_lvef(pred: str, gt: str, tol: float = 5.0) -> float:
    """LVEF: GT 'left ventricular ejection fraction is X%'. Compare X within ±tol."""
    # extract first percentage in each string
    p_match = re.search(r"(\d+(?:\.\d+)?)\s*%", pred or "")
    g_match = re.search(r"(\d+(?:\.\d+)?)\s*%", gt or "")
    if not p_match or not g_match:
        return 0.0
    p = float(p_match.group(1))
    g = float(g_match.group(1))
    d = abs(p - g)
    if d <= tol:
        return 1.0
    if d <= 2 * tol:
        return 0.7
    return max(0.0, 1.0 - (d - 2 * tol) / 20.0)


def verify_numeric_interval(pred: str, gt: str) -> float:
    """ecg_interval: compare numeric values.

    ECG intervals measured in ms (QT/QTc/PR/QRS) get full credit when the
    absolute error is <=40 ms. Heart-rate-only prompts get full credit within
    10 bpm.
    """
    p = _extract_first_number(pred)
    g = _extract_first_number(gt)
    if p is None or g is None or g == 0:
        return 0.0
    if p < 0 or g < 0:
        return 0.0
    if _is_ms_interval_text(pred) or _is_ms_interval_text(gt):
        d = abs(p - g)
        if d <= 40.0:
            return 1.0
        return max(0.0, 1.0 - (d - 40.0) / 40.0)
    tol = 10.0
    d = abs(p - g)
    if d <= tol:
        return 1.0
    # steep falloff: zero credit by ~2.5x tolerance
    return max(0.0, 1.0 - d / (1.5 * tol))


def verify_axis(pred: str, gt: str) -> float:
    """localization_qrs_axis: Left/Right/Normal axis deviation. The GT is
    'No - Left axis deviation (...)' style. The model usually rambles
    ('questions about them. Here is an ECG: ...') and the judge scores ~0.
    We require the prediction to actually mention an axis direction with
    the word 'axis' nearby — a stray 'left' in a rambling answer is NOT
    credit. (Bug 2026-05: verifier gave 1.0 on rambling gens judge gave 0.)"""
    def axis(s):
        if _has_bad_preamble(s):
            return None
        s = _strip_prefix(s)[0].lower()
        if "axis" not in s:
            return None
        # direction must be within 25 chars of the word 'axis'
        for direction in ("left", "right", "normal"):
            for m in re.finditer(direction, s):
                ai = s.find("axis")
                while ai != -1:
                    if abs(ai - m.start()) <= 25:
                        return direction
                    ai = s.find("axis", ai + 1)
        return None
    gyn = _canonical_yesno(gt)
    pyn = _canonical_yesno(pred)
    if gyn in ("yes", "no") and pyn in ("yes", "no") and pyn != gyn:
        return 0.0
    if gyn in ("yes", "no") and pyn is None:
        return 0.0
    p = axis(pred)
    g = axis(gt)
    return 1.0 if (p is not None and p == g) else 0.0


def verify_culprit(pred: str, gt: str) -> float:
    """culprit_artery: score = 0.7 * (vessel match) + 0.3 * (segment match).
    Vessel ∈ {LAD, RCA, LCx, Left Main}. Segment ∈ {proximal, mid, distal}.
    Completeness (complete/incomplete) is IGNORED — nothing else counts.
    (Per spec 2026-05: vessel 0.7, segment +0.3, nothing else.)"""
    def parse(s):
        s = (s or "").lower()
        # Vessel — order matters: check 'left main' before 'lad'/'lcx'
        vessels = []
        if "left main" in s or "lmca" in s:
            vessels.append("left_main")
        if "lad" in s or "left anterior descending" in s:
            vessels.append("lad")
        if "rca" in s or "right coronary" in s or "pda" in s or "posterior descending" in s:
            vessels.append("rca")
        if "lcx" in s or "circumflex" in s or "ramus" in s:
            vessels.append("lcx")
        vessels = list(dict.fromkeys(vessels))
        vessel = vessels[0] if len(vessels) == 1 else None
        # Segment
        seg = None
        if "proximal" in s:
            seg = "proximal"
        elif "distal" in s:
            seg = "distal"
        elif "mid" in s:
            seg = "mid"
        return vessel, seg

    p_v, p_s = parse(pred)
    g_v, g_s = parse(gt)
    if g_v is None:
        return 0.0
    score = 0.0
    if p_v == g_v:
        score += 0.7
    if g_s is not None and p_s == g_s:
        score += 0.3
    elif g_s is None:
        # GT has no segment → vessel match alone is full credit
        if p_v == g_v:
            score = 1.0
    return score


def verify_lead_list(pred: str, gt: str) -> float:
    """localization_t_wave: 't wave in [region1 leads X,Y; region2 leads ...]'.
    F1 over the set of mentioned regions."""
    regions = ["inferior", "lateral", "anterior", "septal", "posterior", "high lateral"]
    def regions_in(s):
        s = (s or "").lower()
        return {r for r in regions if r in s}
    P = regions_in(pred)
    G = regions_in(gt)
    if not G:
        return 1.0 if not P else 0.5
    if not P:
        return 0.0
    inter = len(P & G)
    if inter == 0:
        return 0.0
    prec = inter / len(P)
    rec = inter / len(G)
    return 2 * prec * rec / (prec + rec)


def verify_urgency(pred: str, gt: str) -> float:
    """urgency_assessment: 'URGENT: ...' or 'NON-URGENT: ...'."""
    def level(s):
        s = (s or "").upper()
        if "URGENT" in s and "NON" not in s[:20] and "NOT URGENT" not in s[:20]:
            return "urgent"
        if "NON-URGENT" in s or "NOT URGENT" in s or "NON URGENT" in s:
            return "non-urgent"
        if "ROUTINE" in s or "FOLLOW-UP" in s.replace("FOLLOWUP", "FOLLOW-UP"):
            return "routine"
        return None
    p = level(pred)
    g = level(gt)
    return 1.0 if (p is not None and p == g) else 0.0


def _acs_occlusion_type(s: str) -> Optional[str]:
    """'complete' / 'incomplete' / None from an ACS answer."""
    s = _strip_prefix(s or "")[0].lower()
    if "incomplete" in s:
        return "incomplete"
    if "complete" in s:
        return "complete"
    return None


# No-subtype keys, ordered: 0=no disease, 1=non-obstructive, 2=obstructive,
# 3=chronic. 'non-obstructive' must be checked before 'obstructive' (substring).
_ACS_NO_SUBTYPES = (
    ("no coronary disease", "no evidence of coronary"),
    ("non-obstructive", "non obstructive"),
    ("obstructive",),
    ("chronic",),
)


def _acs_no_subtype(s: str) -> Optional[int]:
    s = _strip_prefix(s or "")[0].lower()
    for i, keys in enumerate(_ACS_NO_SUBTYPES):
        if any(k in s for k in keys):
            return i
    return None


def _acs_has_vessel(s: str) -> bool:
    s = (s or "").lower()
    if "not documented" in s or "not available" in s or "not determined" in s:
        return False
    return any(v in s for v in (
        "lad", "left anterior descending", "rca", "right coronary", "pda",
        "posterior descending", "lcx", "circumflex", "ramus", "left main", "lmca"))


def verify_acs(pred: str, gt: str) -> float:
    """acs_severity: graded localization + severity reward.

    The acute Yes/No gate is necessary — a wrong gate scores 0. Then:
      * No  -> 0.4 + 0.6 * (No-subtype match: no-disease / obstructive / chronic)
      * Yes -> graded over whatever the GT documents: occlusion type
               (complete vs incomplete) and culprit vessel (verify_culprit).
               0.5/0.5 when both present, else the single available component;
               a bare 'Yes' with no detail -> 1.0.

    No flat 0.7 floor: the old version paid any correct 'Yes' >= 0.7 regardless
    of type/vessel (gameable) and every correct 'No' exactly 1.0 (binary, ~89%
    saturated). Grading type + vessel + No-subtype spreads the reward so GRPO
    gets a real within-group gradient. (2026-05 redesign.)
    """
    p_yn = _canonical_yesno(pred)
    g_yn = _canonical_yesno(gt)
    if g_yn is None:
        return 0.0
    if p_yn != g_yn:
        return 0.0
    if g_yn == "no":
        g_sub = _acs_no_subtype(gt)
        if g_sub is None:
            return 1.0  # GT states no subtype to grade
        return 0.4 + 0.6 * (1.0 if _acs_no_subtype(pred) == g_sub else 0.0)
    # Yes — grade the components the GT actually documents.
    parts, weights = [], []
    g_type = _acs_occlusion_type(gt)
    if g_type is not None:
        parts.append(1.0 if _acs_occlusion_type(pred) == g_type else 0.0)
        weights.append(0.5)
    if _acs_has_vessel(gt):
        parts.append(verify_culprit(pred, gt))  # returns 0.0 if GT names no vessel
        weights.append(0.5)
    if not parts:
        return 1.0  # correct gate, nothing further to grade
    return sum(p * w for p, w in zip(parts, weights)) / sum(weights)


# Structural heart disease conditions (EchoNext 7-label SHD) -> detection keywords.
# Generation emits the canonical phrase; the verifier matches any keyword.
_SHD_CONDITIONS = {
    "mitral regurgitation": ("mitral regurg", "mitral valve regurg"),
    "aortic stenosis": ("aortic stenosis",),
    "aortic regurgitation": ("aortic regurg", "aortic valve regurg"),
    "tricuspid regurgitation": ("tricuspid regurg",),
    "reduced lvef": ("ejection fraction", "lvef", "reduced ef", "low ef",
                     "lv systolic dysfunction", "reduced systolic"),
    "rv dysfunction": ("right ventricular systolic", "rv systolic",
                       "right ventricular dysfunction", "rv dysfunction"),
    "increased lv wall thickness": ("wall thickness", "hypertrophy", "lvh"),
}


def _shd_conditions(s: str) -> Set[str]:
    s = _strip_prefix(s or "")[0].lower()
    out = set()
    for canon, keys in _SHD_CONDITIONS.items():
        if any(k in s for k in keys):
            out.add(canon)
    return out


def verify_structural(pred: str, gt: str) -> float:
    """structural_heart_disease: graded gate + which SHD conditions are present.

    The GT now enumerates the echo-confirmed conditions (moderate-or-severe
    MR/AS/AR/TR, reduced LVEF<=45%, RV systolic dysfunction, increased LV wall
    thickness) so the answer is no longer a bare Yes/No. Scoring:
      * gate (Yes/No) necessary -> wrong gate = 0.0
      * No -> 1.0
      * Yes, GT names conditions -> 0.3 + 0.7 * F1(conditions); the 0.3 keeps a
        correct 'present' gate above a wrong gate, the F1 carries the rest.
      * Yes, GT names no specific condition -> 1.0 (gate is all there is to grade)
    Replaces the pure binary verify_extracted_yesno that was 100% saturated
    (no within-group GRPO gradient -> contributed to the -0.086 regression).
    (2026-05 de-binarize, EchoNext 7-label GT joined from v1.6.)
    """
    p_yn = _extract_clear_yesno(pred)
    g_yn = _extract_clear_yesno(gt)
    if g_yn is None:
        return verify_extracted_yesno(pred, gt)
    if p_yn != g_yn:
        return 0.0
    if g_yn == "no":
        return 1.0
    gc = _shd_conditions(gt)
    if not gc:
        return 1.0
    return 0.3 + 0.7 * _set_f1(_shd_conditions(pred), gc)


def verify_json_match(pred: str, gt: str) -> float:
    """json_interpretation: parse both as JSON, F1 over flattened (key, value) tuples."""
    def flatten(d):
        if not isinstance(d, dict):
            return set()
        out = set()
        for k, v in d.items():
            if isinstance(v, list):
                for x in v:
                    out.add((k, str(x).strip()))
            else:
                out.add((k, str(v).strip()))
        return out

    def try_parse(s):
        s = (s or "").strip()
        # Find a {...} block
        m = re.search(r"\{.*\}", s, re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

    P = flatten(try_parse(pred) or {})
    G = flatten(try_parse(gt) or {})
    if not G:
        return 0.0 if P else 1.0
    if not P:
        return 0.0
    inter = len(P & G)
    if inter == 0:
        return 0.0
    prec = inter / len(P)
    rec = inter / len(G)
    return 2 * prec * rec / (prec + rec)


# Small built-in ontology of common ECG diagnostic findings
_DEFAULT_ONTOLOGY = [
    "sinus rhythm", "sinus bradycardia", "sinus tachycardia",
    "sinus arrhythmia",
    "bradycardia", "tachycardia", "regular rhythm", "irregular rhythm",
    "atrial fibrillation", "afib", "atrial flutter",
    "atrial tachycardia", "ventricular rhythm",
    "premature atrial complex", "premature ventricular complex",
    "supraventricular tachycardia", "ventricular tachycardia",
    "1st degree av block", "first degree av block",
    "2nd degree av block", "second degree av block",
    "third degree av block", "complete heart block",
    "right bundle branch block", "rbbb",
    "left bundle branch block", "lbbb",
    "left anterior fascicular block",
    "atrial paced", "ventricular paced",
    "sequential pacemaker", "dual chamber pacemaker", "ventricular pacemaker",
    "pacemaker",
    "left axis deviation", "right axis deviation",
    "left ventricular hypertrophy", "lvh",
    "right ventricular hypertrophy", "rvh",
    "left atrial enlargement", "right atrial enlargement",
    "anterior infarction", "inferior infarction", "lateral infarction",
    "septal infarction", "posterior infarction",
    "anteroseptal", "anterolateral",
    "st elevation", "st depression", "st changes", "st change",
    "st downslopping", "st upslopping",
    "t wave inversion", "t wave abnormality", "u wave",
    "prolonged qt", "prolonged pr",
    "low voltage", "early repolarization",
    "wolff-parkinson-white", "wpw", "delta wave",
    "pericarditis", "acute mi", "acute myocardial infarction",
    "stemi", "borderline ecg", "abnormal ecg", "normal ecg",
    "q wave",
]

_ONTOLOGY_SET = sorted(_DEFAULT_ONTOLOGY, key=len, reverse=True)
_ONTOLOGY_ALIAS_PATH = "/volume/LLM_JUDGE/ontology/ecg_ontology.json"
_ALIAS_TO_CANON: Optional[Dict[str, str]] = None
_ALIAS_PATTERN: Optional[re.Pattern] = None

_RHYTHM_TERMS = {
    "sinus rhythm", "sinus bradycardia", "sinus tachycardia",
    "sinus arrhythmia", "atrial fibrillation", "atrial flutter",
    "atrial tachycardia", "ventricular rhythm", "ventricular tachycardia",
    "ventricular pacemaker", "sequential pacemaker", "pacemaker",
    "atrial paced", "ventricular paced",
}
_RHYTHM_PRISTINE_LABELS = {
    "Afib",
    "Atrial flutter",
    "Atrial tachycardia (>= 100 BPM)",
    "Ectopic atrial rhythm (< 100 BPM)",
    "Ventricular Rhythm",
    "Ventricular tachycardia",
    "Supraventricular tachycardia",
    "Junctional rhythm",
    "Bradycardia",
    "Brugada",
    "Wolff-Parkinson-White (Pre-excitation syndrome)",
    "Irregularly irregular",
    "Regularly irregular",
    "Premature ventricular complex",
    "Premature atrial complex",
}

_BERT_LABEL_BY_LOWER = {str(label).lower(): str(label) for label in ECG_PATTERNS}
_BERT_THRESHOLD_BY_LABEL = {
    label: float(BERT_CLASS_THRESHOLDS[i])
    for i, label in enumerate(ECG_PATTERNS)
    if i < len(BERT_CLASS_THRESHOLDS)
}
_CLASSIFICATION_BERT_NOISE = {"Sinusal", "Regular", "Monomorph", "no_qrs"}
_BERT_LABELER = None
_BERT_LABELER_FAILED = False
_BERT_TEXT_CACHE: Dict[str, Set[str]] = {}


def _labels_containing(*needles: str) -> Set[str]:
    out = set()
    for label in ECG_PATTERNS:
        low = label.lower()
        if all(n in low for n in needles):
            out.add(label)
    return out


_BERT_CATEGORY_LABELS: Dict[str, Set[str]] = {
    "classification": set(ECG_PATTERNS) - _CLASSIFICATION_BERT_NOISE,
    "interpretation": set(ECG_PATTERNS) - _CLASSIFICATION_BERT_NOISE,
    "category_rhythm": set(DEEPECG_CATEGORIES.get("RHYTHM", [])),
    "category_conduction": set(DEEPECG_CATEGORIES.get("CONDUCTION", [])),
    "category_chamber_enlargement": set(DEEPECG_CATEGORIES.get("CHAMBER ENLARGEMENT", [])),
    "category_pericarditis": set(DEEPECG_CATEGORIES.get("PERICARDITIS", [])),
    "category_infarct_ischemia": set(DEEPECG_CATEGORIES.get("INFARCT, ISCHEMIA", [])),
    "category_other": set(DEEPECG_CATEGORIES.get("OTHER", [])),
    "localization_t_wave": _labels_containing("t wave inversion"),
    "localization_st_elevation": _labels_containing("st elevation"),
    "localization_st_depression": _labels_containing("st depression"),
    "localization_q_wave": _labels_containing("q wave"),
    "localization_qrs_axis": {"Left axis deviation", "Right axis deviation", "Right superior axis"},
}


def _canonical_bert_label(label: Any) -> Optional[str]:
    if label is None:
        return None
    raw = str(label).strip()
    if not raw:
        return None
    direct = _BERT_LABEL_BY_LOWER.get(raw.lower())
    if direct:
        return direct
    # A few common report aliases that the model/ontology use for the same
    # DeepECG labels.
    alias = {
        "lvh": "Left ventricular hypertrophy",
        "left ventricular hypertrophy": "Left ventricular hypertrophy",
        "rvh": "Right ventricular hypertrophy",
        "right ventricular hypertrophy": "Right ventricular hypertrophy",
        "lae": "Left atrial enlargement",
        "left atrial enlargement": "Left atrial enlargement",
        "rae": "Right atrial enlargement",
        "right atrial enlargement": "Right atrial enlargement",
        "afib": "Afib",
        "atrial fibrillation": "Afib",
        "sinus rhythm": "Sinusal",
        "sinusal": "Sinusal",
        "1st degree av block": "1st degree AV block",
        "first degree av block": "1st degree AV block",
        "right bundle branch block": "Right bundle branch block",
        "rbbb": "Right bundle branch block",
        "left bundle branch block": "Left bundle branch block",
        "lbbb": "Left bundle branch block",
        "wpw": "Wolff-Parkinson-White (Pre-excitation syndrome)",
        "wolff-parkinson-white": "Wolff-Parkinson-White (Pre-excitation syndrome)",
        "acute pericarditis": "Acute pericarditis",
        "pericarditis": "Acute pericarditis",
        "acute mi": "Acute MI",
        "acute myocardial infarction": "Acute MI",
        "rv1 + sv6 11 mm": "RV1 + SV6 > 11 mm",
        "rsr in v1-v2": "rSR' in V1-V2",
    }.get(raw.lower())
    return alias


def _configured_classification_labels(kind: str) -> Set[str]:
    raw = DEEPECG_PATHOLOGICAL_LIMIT.get("deepecg", {}).get(kind, [])
    out = set()
    for label in raw:
        canon = _canonical_bert_label(label)
        if canon is not None:
            out.add(canon)
    return out


def _classification_pathological_labels() -> Set[str]:
    return _configured_classification_labels("pathological")


def _classification_borderline_labels() -> Set[str]:
    return _configured_classification_labels("limit")


def _verify_classification_labelset(pred_labels: Set[str], gt_labels: Set[str]) -> float:
    """Severity-aware classification reward using constants.py.

    Pathological misses are near-zero because they are the core ECG
    interpretation signal. Borderline misses are capped, and normal ECGs are
    judged by diagnosis-label F1 after normal/noise labels are removed.
    """
    pred = set(pred_labels) - _CLASSIFICATION_BERT_NOISE
    gt = set(gt_labels) - _CLASSIFICATION_BERT_NOISE
    pathological = _classification_pathological_labels()
    borderline = _classification_borderline_labels()

    gt_pathological = gt & pathological
    pred_pathological = pred & pathological
    if gt_pathological:
        if not gt_pathological <= pred_pathological:
            return 0.1 if pred else 0.0
        return _set_f1(pred, gt)

    gt_borderline = gt & borderline
    pred_borderline = pred & borderline
    if gt_borderline:
        score = _set_f1(pred, gt)
        if not gt_borderline <= pred_borderline:
            return min(score, 0.5)
        return score

    return _set_f1(pred, gt)


_BERT_TEXT_ALIASES = {
    "lvh": "Left ventricular hypertrophy",
    "left ventricular hypertrophy": "Left ventricular hypertrophy",
    "rvh": "Right ventricular hypertrophy",
    "right ventricular hypertrophy": "Right ventricular hypertrophy",
    "lae": "Left atrial enlargement",
    "left atrial enlargement": "Left atrial enlargement",
    "rae": "Right atrial enlargement",
    "right atrial enlargement": "Right atrial enlargement",
    "atrial fibrillation": "Afib",
    "afib": "Afib",
    "sinus rhythm": "Sinusal",
    "sinus bradycardia": "Bradycardia",
    "sinus tachycardia": "Atrial tachycardia (>= 100 BPM)",
    "premature ventricular complex": "Premature ventricular complex",
    "pvc": "Premature ventricular complex",
    "premature atrial complex": "Premature atrial complex",
    "pac": "Premature atrial complex",
    "right bundle branch block": "Right bundle branch block",
    "rbbb": "Right bundle branch block",
    "left bundle branch block": "Left bundle branch block",
    "lbbb": "Left bundle branch block",
    "left anterior fascicular block": "Left anterior fascicular block",
    "left posterior fascicular block": "Left posterior fascicular block",
    "left axis deviation": "Left axis deviation",
    "right axis deviation": "Right axis deviation",
    "right superior axis": "Right superior axis",
    "1st degree av block": "1st degree AV block",
    "first degree av block": "1st degree AV block",
    "prolonged qt": "Prolonged QT",
    "low voltage": "Low voltage",
    "early repolarization": "Early repolarization",
    "acute pericarditis": "Acute pericarditis",
    "pericarditis": "Acute pericarditis",
    "acute mi": "Acute MI",
    "acute myocardial infarction": "Acute MI",
    "st downslopping": "ST downslopping",
    "st downsloping": "ST downslopping",
    "st segment downslopping": "ST downslopping",
    "st segment downsloping": "ST downslopping",
    "st upslopping": "ST upslopping",
    "st upsloping": "ST upslopping",
    "u wave": "U wave",
    "delta wave": "Delta wave",
    "brugada": "Brugada",
}


def _bert_labels_from_text(text: str) -> Set[str]:
    low = (text or "").lower()
    found: Set[str] = set()
    for label in ECG_PATTERNS:
        if label.lower() in low:
            found.add(label)
    for alias, label in _BERT_TEXT_ALIASES.items():
        if re.search(r"\b" + re.escape(alias) + r"\b", low):
            found.add(label)
    if "st depression" in low:
        if re.search(r"\b(?:lateral|i,\s*avl|v5|v6)\b", low):
            found.add("ST depression (lateral - I, avL, V5-V6)")
        if re.search(r"\b(?:inferior|ii|iii|avf)\b", low):
            found.add("ST depression (inferior - II, III, aVF)")
        if re.search(r"\b(?:anterior|v3|v4)\b", low):
            found.add("ST depression (anterior - V3-V4)")
        if re.search(r"\b(?:septal|v1|v2)\b", low):
            found.add("ST depression (septal- V1-V2)")
    if "st elevation" in low:
        if re.search(r"\b(?:lateral|i,\s*avl|v5|v6)\b", low):
            found.add("ST elevation (lateral - I, aVL, V5-V6)")
        if re.search(r"\b(?:inferior|ii|iii|avf)\b", low):
            found.add("ST elevation (inferior - II, III, aVF)")
        if re.search(r"\b(?:anterior|v3|v4)\b", low):
            found.add("ST elevation (anterior - V3-V4)")
        if re.search(r"\b(?:septal|v1|v2)\b", low):
            found.add("ST elevation (septal - V1-V2)")
        if re.search(r"\b(?:posterior|v7|v8|v9)\b", low):
            found.add("ST elevation (posterior - V7-V8-V9)")
    if "q wave" in low or "q waves" in low:
        if re.search(r"\b(?:lateral|i,\s*avl|v5|v6)\b", low):
            found.add("Q wave (lateral- I, aVL, V5-V6)")
        if re.search(r"\b(?:inferior|ii|iii|avf)\b", low):
            found.add("Q wave (inferior - II, III, aVF)")
        if re.search(r"\b(?:anterior|v3|v4)\b", low):
            found.add("Q wave (anterior - V3-V4)")
        if re.search(r"\b(?:septal|v1|v2)\b", low):
            found.add("Q wave (septal- V1-V2)")
        if re.search(r"\b(?:posterior|v7|v8|v9)\b", low):
            found.add("Q wave (posterior - V7-V9)")
    if "t wave inversion" in low or "t wave inversions" in low:
        if re.search(r"\b(?:lateral|i,\s*avl|v5|v6)\b", low):
            found.add("T wave inversion (lateral -I, aVL, V5-V6)")
        if re.search(r"\b(?:inferior|ii|iii|avf)\b", low):
            found.add("T wave inversion (inferior - II, III, aVF)")
        if re.search(r"\b(?:anterior|v3|v4)\b", low):
            found.add("T wave inversion (anterior - V3-V4)")
        if re.search(r"\b(?:septal|v1|v2)\b", low):
            found.add("T wave inversion (septal- V1-V2)")
    return {label for label in found if label in _BERT_LABEL_BY_LOWER.values()}


def _coerce_bert_values(values: Any, *, is_logit: bool = False) -> Optional[Set[str]]:
    if not isinstance(values, dict):
        return None
    out: Set[str] = set()
    for key, value in values.items():
        label = _canonical_bert_label(key)
        if label is None:
            continue
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        # Some upstream files call these logits, others store probabilities or
        # hard 0/1 classes. Values outside [0, 1] are treated as logits.
        prob = 1.0 / (1.0 + math.exp(-v)) if (is_logit or v < 0.0 or v > 1.0) else v
        threshold = _BERT_THRESHOLD_BY_LABEL.get(label, 0.5)
        if prob >= threshold:
            out.add(label)
    return out


def _bert_labels_from_payload(obj: Any) -> Optional[Set[str]]:
    payload = _payload(obj)
    if payload is None:
        return None
    for key in ("bert_labels", "active_bert_labels", "labels"):
        values = payload.get(key)
        if isinstance(values, dict):
            return {label for label, active in (
                (_canonical_bert_label(k), bool(v)) for k, v in values.items()
            ) if label is not None and active}
        if isinstance(values, (list, tuple, set)):
            labels = {_canonical_bert_label(v) for v in values}
            return {v for v in labels if v is not None}
    for key in ("bert_values", "bert_probs", "bert_probabilities"):
        labels = _coerce_bert_values(payload.get(key), is_logit=False)
        if labels is not None:
            return labels
    for key in ("bert_logits", "logits"):
        labels = _coerce_bert_values(payload.get(key), is_logit=True)
        if labels is not None:
            return labels
    return None


def _bert_inference_requested(gt: Any) -> bool:
    mode = os.environ.get("ECG_VERIFIER_USE_BERT", "auto").strip().lower()
    if mode in {"0", "false", "no", "off"}:
        return False
    if mode in {"1", "true", "yes", "on"}:
        return True
    return _bert_labels_from_payload(gt) is not None


def _load_bert_labeler():
    global _BERT_LABELER, _BERT_LABELER_FAILED
    if _BERT_LABELER is not None:
        return _BERT_LABELER
    if _BERT_LABELER_FAILED:
        return None
    try:
        import torch
        from huggingface_hub import snapshot_download
        from transformers import BertForSequenceClassification, BertTokenizer

        model_path = os.environ.get("ECG_BERT_MODEL_PATH", "checkpoints/bert")
        model_dir = Path(model_path)
        if not (model_dir / "model.safetensors").exists() and not (model_dir / "pytorch_model.bin").exists():
            model_path = snapshot_download(
                "heartwise/Bert_diagnosis2classification_En_Fr",
                local_dir=str(model_dir),
            )
        device = os.environ.get("ECG_BERT_DEVICE")
        if not device:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = BertTokenizer.from_pretrained(model_path)
        model = BertForSequenceClassification.from_pretrained(
            model_path,
            num_labels=len(ECG_PATTERNS),
        )
        model.to(device)
        model.eval()
        _BERT_LABELER = (torch, tokenizer, model, device)
        return _BERT_LABELER
    except Exception as exc:
        print(f"[verifiable_reward] BERT labeler unavailable: {exc}")
        _BERT_LABELER_FAILED = True
        return None


def _bert_predict_labels(text: str) -> Optional[Set[str]]:
    text = _obj_text(text).strip()
    if not text:
        return set()
    cached = _BERT_TEXT_CACHE.get(text)
    if cached is not None:
        return set(cached)
    labeler = _load_bert_labeler()
    if labeler is None:
        return None
    torch, tokenizer, model, device = labeler
    enc = tokenizer(
        text,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    enc = {k: v.to(device) for k, v in enc.items()}
    with torch.no_grad():
        probs = torch.sigmoid(model(**enc).logits)[0].detach().cpu().float().tolist()
    labels = {
        label
        for label, prob in zip(ECG_PATTERNS, probs)
        if prob >= _BERT_THRESHOLD_BY_LABEL.get(label, 0.5)
    }
    if len(_BERT_TEXT_CACHE) > 4096:
        _BERT_TEXT_CACHE.clear()
    _BERT_TEXT_CACHE[text] = set(labels)
    return labels


def _bert_or_payload_labels(obj: Any, *, use_model: bool) -> Optional[Set[str]]:
    labels = _bert_labels_from_payload(obj)
    if labels is not None:
        return labels
    if not use_model:
        return None
    return _bert_predict_labels(_obj_text(obj))


def _set_f1(pred_labels: Set[str], gt_labels: Set[str]) -> float:
    if not gt_labels:
        return 1.0 if not pred_labels else 0.0
    if not pred_labels:
        return 0.0
    inter = len(pred_labels & gt_labels)
    if inter == 0:
        return 0.0
    prec = inter / len(pred_labels)
    rec = inter / len(gt_labels)
    return 2 * prec * rec / (prec + rec)


def _rhythm_has_contradiction(labels: Set[str]) -> bool:
    if "Regular" in labels and "Irregularly irregular" in labels:
        return True
    if "Sinusal" in labels and (
        "Afib" in labels
        or "Atrial flutter" in labels
        or "Ventricular Rhythm" in labels
        or "Junctional rhythm" in labels
    ):
        return True
    return False


def _pristine_rhythm_score(
    pred_labels: Set[str],
    gt_labels: Set[str],
    pred_text_labels: Set[str],
    score: float,
) -> float:
    """Make ECG rhythm interpretation a hard clinical signal.

    HR agreement is necessary but not sufficient: hard rhythm diagnoses and
    ectopy labels must match exactly, and self-contradictory rhythm text gets
    no reward. We intentionally do not make "Regular" or "Sinusal" hard-extra
    labels because some prompts ask only regularity or only rhythm type.
    """
    contradiction_labels = pred_text_labels or pred_labels
    if _rhythm_has_contradiction(contradiction_labels):
        return 0.0
    hard_pred = pred_labels & _RHYTHM_PRISTINE_LABELS
    hard_gt = gt_labels & _RHYTHM_PRISTINE_LABELS
    if hard_pred != hard_gt:
        return 0.0
    return score


_INTERP_HR_WEIGHT = float(os.environ.get("ECG_INTERP_HR_WEIGHT", "0.3"))


def _verify_interpretation(pred: Any, gt: Any, label_scope: Set[str]) -> Optional[float]:
    """interpretation reward = BERT diagnosis-label F1 (+ HR accuracy when the
    GT states a heart rate): score = (1-w)*F1(BERT_labels) + w*HR_accuracy.

    BERT classifies the generation and the GT into the 77 ECG diagnosis labels;
    it is synonym-robust ('pacing'=='pacemaker') and catches missed/extra
    findings that string ontology-F1 (and a lenient judge) forgive. The additive
    HR term covers BERT's numeric blind spot (it can't see 'HR: 92' vs 'HR: 136').
    GT labels come from the row payload when present (true GT columns), else from
    running BERT on the GT text. Falls back (returns None -> ontology-F1) if the
    labeler is unavailable.

    Validated May 2026 (n=3000 x2): vs the LLM judge this scores ~0.60 corr /
    ~0.19 MAE — below ontology-F1's ~0.69 corr, but deliberately chosen for
    clinical-label fidelity over judge-mimicry (the judge over-credits 'Normal'
    generations whose GT is 'Borderline' with a real finding; BERT does not).
    """
    p_all = _bert_or_payload_labels(pred, use_model=True)
    g_all = _bert_or_payload_labels(gt, use_model=True)
    if p_all is None or g_all is None:
        return None
    f1 = _set_f1(p_all & label_scope, g_all & label_scope)
    hr = _hr_accuracy(_obj_text(pred), _obj_text(gt))
    if hr is None:
        return f1
    w = _INTERP_HR_WEIGHT
    return (1.0 - w) * f1 + w * hr


def _verify_bert_category(pred: Any, gt: Any, category: str) -> Optional[float]:
    label_scope = _BERT_CATEGORY_LABELS.get(category)
    if not label_scope:
        return None

    if category == "interpretation":
        return _verify_interpretation(pred, gt, label_scope)

    pred_text = _obj_text(pred)
    gt_text = _obj_text(gt)
    gyn = _canonical_yesno(gt_text)
    pyn = _canonical_yesno(pred_text)
    if gyn in ("yes", "no") and pyn in ("yes", "no") and pyn != gyn:
        return 0.0

    use_model = _bert_inference_requested(gt)
    g_all = _bert_or_payload_labels(gt, use_model=use_model)
    if g_all is None:
        return None
    p_all = _bert_or_payload_labels(pred, use_model=use_model)
    if p_all is None:
        return None

    pred_text_labels = _bert_labels_from_text(pred_text) & label_scope
    pred_labels = (p_all & label_scope) | pred_text_labels
    gt_payload_labels = g_all & label_scope
    gt_text_labels = _bert_labels_from_text(gt_text) & label_scope
    if category == "classification":
        gt_labels = gt_payload_labels or gt_text_labels
    elif gt_text_labels:
        gt_labels = (gt_payload_labels & gt_text_labels) or gt_text_labels
    else:
        gt_labels = gt_payload_labels
    if category == "classification":
        return _verify_classification_labelset(pred_labels, gt_labels)
    if gyn == "no":
        if pyn == "no" or not pred_labels:
            return 1.0
        return 0.0
    if gyn == "yes" and pyn == "no":
        return 0.0
    score = _set_f1(pred_labels, gt_labels)
    if category == "category_rhythm":
        score = _pristine_rhythm_score(pred_labels, gt_labels, pred_text_labels, score)
        score *= _hr_penalty(pred_text, gt_text)
    return score


def _get_alias_index() -> Tuple[Dict[str, str], Optional[re.Pattern]]:
    global _ALIAS_TO_CANON, _ALIAS_PATTERN
    if _ALIAS_TO_CANON is not None:
        return _ALIAS_TO_CANON, _ALIAS_PATTERN

    alias_to_canon: Dict[str, str] = {}
    if _load_alias_index is not None:
        try:
            alias_to_canon, _ = _load_alias_index(_ONTOLOGY_ALIAS_PATH)
        except Exception:
            alias_to_canon = {}
    alias_to_canon.update({
        "lvh": "left ventricular hypertrophy",
        "lv hypertrophy": "left ventricular hypertrophy",
        "left ventricular hypertrophy": "left ventricular hypertrophy",
        "rvh": "right ventricular hypertrophy",
        "rv hypertrophy": "right ventricular hypertrophy",
        "right ventricular hypertrophy": "right ventricular hypertrophy",
        "pericarditis": "acute pericarditis",
        "acute pericarditis": "acute pericarditis",
    })

    aliases = sorted(alias_to_canon.keys(), key=len, reverse=True)
    if aliases:
        _ALIAS_PATTERN = re.compile(
            r"\b(?:" + "|".join(re.escape(a) for a in aliases) + r")\b",
            re.IGNORECASE,
        )
    _ALIAS_TO_CANON = alias_to_canon
    return _ALIAS_TO_CANON, _ALIAS_PATTERN


def _ontology_terms(s: str) -> Set[str]:
    """Find ontology terms present in the string. Greedy longest-match."""
    s = (s or "").lower()
    found = set()
    alias_to_canon, alias_pattern = _get_alias_index()
    if alias_pattern is not None:
        for m in alias_pattern.finditer(s):
            found.add(alias_to_canon[m.group(0).lower()])
    for term in _ONTOLOGY_SET:
        if term in s:
            found.add(term)
    # Collapse synonyms
    synonyms = {
        "afib": "atrial fibrillation",
        "rbbb": "right bundle branch block",
        "lbbb": "left bundle branch block",
        "lvh": "left ventricular hypertrophy",
        "rvh": "right ventricular hypertrophy",
        "wpw": "wolff-parkinson-white",
        "first degree av block": "1st degree av block",
        "second degree av block": "2nd degree av block",
        "stemi": "st elevation",
        "acute mi": "acute myocardial infarction",
        "pericarditis": "acute pericarditis",
        "st change": "st changes",
    }
    return {synonyms.get(t, t) for t in found}


def verify_ontology_f1(pred: str, gt: str) -> float:
    """F1 over the set of ontology terms found in pred vs gt."""
    if _has_bad_preamble(pred):
        return 0.0
    P = _ontology_terms(pred)
    G = _ontology_terms(gt)
    if not G:
        return 1.0 if not P else 0.0  # nothing to match
    if not P:
        return 0.0
    inter = len(P & G)
    if inter == 0:
        return 0.0
    prec = inter / len(P)
    rec = inter / len(G)
    f1 = 2 * prec * rec / (prec + rec)
    p_rhythm = P & _RHYTHM_TERMS
    g_rhythm = G & _RHYTHM_TERMS
    if p_rhythm and g_rhythm and not (p_rhythm & g_rhythm):
        return min(f1, 0.3)
    return f1


# ─────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────

_ROUTING = {
    # Pure binary yes/no (GT is just 'Yes'/'No' or 'No - <neg statement>')
    "random_finding_question": verify_extracted_yesno,
    "structural_heart_disease": verify_structural,
    # Yes/No GATE + findings content (GT = 'Yes - finding1; finding2' or
    # 'No - no evidence ...' or pure findings text). Binary-only scored
    # these 1.0 where the judge gave 0.2 — must grade findings too.
    "category_pericarditis": verify_yesno_with_findings,
    "category_chamber_enlargement": verify_yesno_with_findings,
    "category_rhythm": verify_rhythm,
    "category_conduction": verify_yesno_with_findings,
    "category_infarct_ischemia": verify_yesno_with_findings,
    "category_other": verify_yesno_with_findings,
    "classification": verify_classification,
    # Categorical
    "afib_risk": verify_risk_level,
    # Numeric
    "lvef": verify_numeric_lvef,
    "ecg_interval": verify_numeric_interval,
    # Specialized
    "localization_qrs_axis": verify_axis,
    "localization_t_wave": verify_lead_list,
    "culprit_artery": verify_culprit,
    "urgency_assessment": verify_urgency,
    "acs_severity": verify_acs,
    "json_interpretation": verify_json_match,
    # Free text: ontology F1
    "interpretation": verify_ontology_f1,
}


_PREFIX_ARTIFACT_RE = re.compile(r"^\s*[:;,.\-*]+\s*", re.UNICODE)


def _format_cap(pred: str) -> Optional[float]:
    """Return a max score cap for format artifacts, or None if clean."""
    if not pred:
        return 0.0
    clean, had_prefix = _strip_prefix(pred)
    if _has_bad_preamble(pred):
        return 0.2
    if not clean:
        return 0.0
    if had_prefix:
        return 0.6
    return None


_HARMLESS_PREFIX_EXEMPT_CATEGORIES = {
    "ecg_interval",
    "lvef",
    "random_finding_question",
    "structural_heart_disease",
}


def _apply_format_cap(score: float, pred: str, category: str) -> float:
    if category in _HARMLESS_PREFIX_EXEMPT_CATEGORIES and not _has_bad_preamble(pred):
        return score
    cap = _format_cap(pred)
    if cap is not None:
        score = min(score, cap)
    return score


def verify(pred: str, gt: str, category: str) -> float:
    pred_text = _obj_text(pred)
    gt_text = _obj_text(gt)
    bert_score = _verify_bert_category(pred, gt, category)
    if bert_score is not None:
        score = float(bert_score)
        score = _apply_format_cap(score, pred_text, category)
        return max(0.0, min(1.0, score))

    fn = _ROUTING.get(category, verify_ontology_f1)
    try:
        score = float(fn(pred_text, gt_text))
    except Exception as e:
        print(f"[verifiable_reward] error in {category}: {e}")
        return 0.0
    score = _apply_format_cap(score, pred_text, category)
    return max(0.0, min(1.0, score))


# Smoke test
if __name__ == "__main__":
    cases = [
        ("structural_heart_disease", "Yes - structural heart disease is present", "Yes - structural heart disease is present based on echocardiography", 1.0),
        ("structural_heart_disease", "No - no structural heart disease", "Yes - structural heart disease is present", 0.0),
        # graded: both SHD conditions correct -> 1.0
        ("structural_heart_disease",
         "Yes - structural heart disease; moderate or severe mitral regurgitation; reduced LV ejection fraction (LVEF <= 45%)",
         "Yes - structural heart disease; moderate or severe mitral regurgitation; reduced LV ejection fraction (LVEF <= 45%)", 1.0),
        # graded: correct 'present' gate but wrong condition -> 0.3 (no more flat binary)
        ("structural_heart_disease",
         "Yes - structural heart disease; moderate or severe aortic stenosis",
         "Yes - structural heart disease; moderate or severe mitral regurgitation", 0.3),
        ("afib_risk", "Low risk - this patient is unlikely to develop AFib", "Low risk - this patient is unlikely to develop atrial fibrillation in the next 5 years", 1.0),
        ("afib_risk", "High risk - this patient has", "Low risk - this patient is unlikely to develop AFib", 0.0),
        ("lvef", "The left ventricular ejection fraction is 35% (moderately reduced)",
                 "The left ventricular ejection fraction is 40% (mildly reduced)", 1.0),  # within 5
        ("lvef", "The left ventricular ejection fraction is 60%",
                 "The left ventricular ejection fraction is 35%", 0.0),  # diff 25 - partial
        ("ecg_interval", "QT interval: 460 ms", "QT interval: 456 ms", 1.0),
        ("localization_qrs_axis", "No - Left axis deviation (-30°)", "No - Left axis deviation (-30° to -90°)", 1.0),
        ("culprit_artery", "The culprit artery is the Proximal LAD with complete occlusion",
                            "The culprit artery is the Proximal LAD with complete occlusion", 1.0),
        # vessel 0.7 + segment 0.3; RCA≠LAD and mid≠proximal → 0.0
        ("culprit_artery", "The culprit artery is the Mid RCA with complete occlusion",
                            "The culprit artery is the Proximal LAD with complete occlusion", 0.0),
        # same vessel (RCA), wrong segment (mid vs proximal) → 0.7
        ("culprit_artery", "The culprit artery is the Mid RCA with complete occlusion",
                            "The culprit artery is the Proximal RCA with complete occlusion", 0.7),
        # afib: risk matches (low) despite prefix junk → 1.0 (pure binary)
        ("afib_risk", "info: Low risk - unlikely to develop AF",
                       "Low risk - this patient is unlikely to develop atrial fibrillation", 1.0),
        ("acs_severity", "No - obstructive coronary disease",
                          "No - obstructive coronary disease without acute occlusion", 1.0),
        ("acs_severity", "Yes - acute coronary occlusion; complete occlusion; culprit is the Proximal LAD",
                          "Yes - acute coronary occlusion; complete occlusion; culprit is the Proximal LAD", 1.0),
        # graded: right vessel but wrong occlusion type -> 0.5 (no more 0.7 floor)
        ("acs_severity", "Yes - acute coronary occlusion; incomplete occlusion; culprit is the Proximal LAD",
                          "Yes - acute coronary occlusion; complete occlusion; culprit is the Proximal LAD", 0.5),
        # graded: correct No gate but wrong subtype -> 0.4 (de-saturates the No majority)
        ("acs_severity", "No - chronic occlusion without acute findings",
                          "No - obstructive coronary disease without acute occlusion", 0.4),
        # classification: right class + perfect findings -> 1.0
        ("classification", "No - Abnormal ECG; Pathological findings: Left atrial enlargement",
                            "No - Abnormal ECG; Pathological findings: Left atrial enlargement", 1.0),
        # class gate: same Yes/No but wrong class (borderline vs abnormal) -> 0.0
        ("classification", "No - Borderline ECG; Minor findings: 1st degree AV block",
                            "No - Abnormal ECG; Pathological findings: Left atrial enlargement", 0.0),
        # _ecg_class fix: 'No - Abnormal ECG' is ABNORMAL (not normal); same class+findings -> 1.0
        ("classification", "No - Abnormal ECG; Pathological findings: Atrial flutter",
                            "No - Abnormal ECG; Pathological findings: Atrial flutter", 1.0),
        ("json_interpretation", '{"RHYTHM":["Sinusal","Regular"],"heart_rate_bpm":75}',
                                 '{"RHYTHM":["Sinusal","Regular"],"heart_rate_bpm":75}', 1.0),
        # findings (afib + ST depression) AND heart rate both match -> 1.0
        ("interpretation", "Atrial fibrillation. (HR: 84.0 bpm); ST depression",
                            "Atrial fibrillation. (HR: 84.0 bpm); ST depression nonspecific", 1.0),
        # same findings but HR wildly off -> additive HR term docks it (BERT path)
        ("interpretation", "Atrial fibrillation. (HR: 140.0 bpm); ST depression",
                            "Atrial fibrillation. (HR: 84.0 bpm); ST depression nonspecific", 0.7),
    ]
    for cat, pred, gt, expected in cases:
        got = verify(pred, gt, cat)
        ok = "✓" if abs(got - expected) < 0.5 else "✗"
        print(f"{ok}  {cat:35s} expected≈{expected:.2f}  got={got:.3f}")
