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
import re
from typing import Dict, List, Optional, Set, Tuple


# ─────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────

def _first_word(s: str) -> str:
    """Return the first alphabetic word, lowercased. Strips colons/dashes."""
    s = (s or "").strip().lstrip(":-*. ").strip()
    m = re.match(r"([A-Za-z]+)", s)
    return m.group(1).lower() if m else ""


def _extract_first_number(s: str) -> Optional[float]:
    m = re.search(r"-?\d+(?:\.\d+)?", s or "")
    return float(m.group(0)) if m else None


def _canonical_yesno(s: str) -> Optional[str]:
    """Normalize: 'Yes', 'No', or 'Borderline' → first word. Handles the
    *** STEMI *** prefix by checking if 'Yes' appears anywhere in first 50 chars."""
    if not s:
        return None
    s_lower = s.strip().lower()
    # Special: "*** ... *** - Yes ..." or "*** ... ***" → treat as Yes
    if s_lower.startswith("***") or "stemi" in s_lower[:80] or "acute mi" in s_lower[:80]:
        return "yes"
    fw = _first_word(s)
    if fw in ("yes", "y", "positive", "abnormal"):
        return "yes"
    if fw in ("no", "n", "negative", "normal"):
        return "no"
    if fw in ("borderline",):
        return "borderline"
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


def verify_risk_level(pred: str, gt: str) -> float:
    """afib_risk: 'Low risk - ...', 'High risk - ...', 'Uncertain risk - ...'."""
    def level(s):
        s = (s or "").strip().lower()
        if s.startswith("low"):
            return "low"
        if s.startswith("high"):
            return "high"
        if s.startswith("uncertain") or s.startswith("unknown"):
            return "uncertain"
        if s.startswith("yes"):  # some GTs are 'Yes - high risk'
            if "low" in s[:40]:
                return "low"
            if "high" in s[:40]:
                return "high"
        return None
    p = level(pred)
    g = level(gt)
    return 1.0 if (p is not None and p == g) else 0.0


def verify_numeric_lvef(pred: str, gt: str, tol: float = 5.0) -> float:
    """LVEF: GT 'left ventricular ejection fraction is X%'. Compare X within ±tol."""
    # extract first percentage in each string
    p_match = re.search(r"(\d+(?:\.\d+)?)\s*%", pred or "")
    g_match = re.search(r"(\d+(?:\.\d+)?)\s*%", gt or "")
    if not p_match or not g_match:
        return 0.0
    p = float(p_match.group(1))
    g = float(g_match.group(1))
    return 1.0 if abs(p - g) <= tol else max(0.0, 1.0 - abs(p - g) / 30.0)


def verify_numeric_interval(pred: str, gt: str) -> float:
    """ecg_interval: extract first number from each, compare within ±10%."""
    p = _extract_first_number(pred)
    g = _extract_first_number(gt)
    if p is None or g is None or g == 0:
        return 0.0
    tol = max(10.0, abs(g) * 0.1)  # ±10% or ±10 (whichever larger)
    return 1.0 if abs(p - g) <= tol else max(0.0, 1.0 - abs(p - g) / (3 * tol))


def verify_axis(pred: str, gt: str) -> float:
    """localization_qrs_axis: Left/Right/Normal axis deviation."""
    def axis(s):
        s = (s or "").lower()
        if "left" in s and "axis" in s:
            return "left"
        if "right" in s and "axis" in s:
            return "right"
        if "normal" in s and "axis" in s:
            return "normal"
        return None
    p = axis(pred)
    g = axis(gt)
    return 1.0 if (p is not None and p == g) else 0.0


def verify_culprit(pred: str, gt: str) -> float:
    """culprit_artery: 'The culprit artery is the X with Y occlusion'.
    Score = 0.5 * (artery_match) + 0.5 * (completeness_match)."""
    arteries = [
        "proximal lad", "mid lad", "distal lad", "lad",
        "proximal lcx", "mid lcx", "distal lcx", "lcx", "circumflex",
        "proximal rca", "mid rca", "distal rca", "rca",
        "left main", "diagonal", "marginal",
    ]
    def parse(s):
        s = (s or "").lower()
        best_a = None
        for a in arteries:
            if a in s:
                # prefer longest match
                if best_a is None or len(a) > len(best_a):
                    best_a = a
        # completeness
        comp = None
        if "incomplete" in s:
            comp = "incomplete"
        elif "complete" in s:
            comp = "complete"
        return best_a, comp
    p_a, p_c = parse(pred)
    g_a, g_c = parse(gt)
    if g_a is None and g_c is None:
        return 0.0
    score = 0.0
    if g_a is not None and p_a == g_a:
        score += 0.5
    if g_c is not None and p_c == g_c:
        score += 0.5
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


def verify_acs(pred: str, gt: str) -> float:
    """acs_severity: 'Yes - acute coronary occlusion; culprit X with Y' or
    'No - obstructive coronary disease ...'."""
    p_yn = _canonical_yesno(pred)
    g_yn = _canonical_yesno(gt)
    if g_yn is None:
        return 0.0
    if p_yn != g_yn:
        return 0.0
    if g_yn == "no":
        return 1.0  # No is full credit when matched
    # Yes: also verify culprit + completeness
    return 0.5 + 0.5 * verify_culprit(pred, gt)


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
    "atrial fibrillation", "afib", "atrial flutter",
    "premature atrial complex", "premature ventricular complex",
    "supraventricular tachycardia", "ventricular tachycardia",
    "1st degree av block", "first degree av block",
    "2nd degree av block", "second degree av block",
    "third degree av block", "complete heart block",
    "right bundle branch block", "rbbb",
    "left bundle branch block", "lbbb",
    "left anterior fascicular block",
    "left axis deviation", "right axis deviation",
    "left ventricular hypertrophy", "lvh",
    "right ventricular hypertrophy", "rvh",
    "left atrial enlargement", "right atrial enlargement",
    "anterior infarction", "inferior infarction", "lateral infarction",
    "septal infarction", "posterior infarction",
    "anteroseptal", "anterolateral",
    "st elevation", "st depression", "st downslopping", "st upslopping",
    "t wave inversion", "t wave abnormality", "u wave",
    "prolonged qt", "prolonged pr",
    "low voltage", "early repolarization",
    "wolff-parkinson-white", "wpw", "delta wave",
    "pericarditis", "acute mi", "acute myocardial infarction",
    "stemi", "borderline ecg", "abnormal ecg", "normal ecg",
    "q wave",
]

_ONTOLOGY_SET = sorted(_DEFAULT_ONTOLOGY, key=len, reverse=True)


def _ontology_terms(s: str) -> Set[str]:
    """Find ontology terms present in the string. Greedy longest-match."""
    s = (s or "").lower()
    found = set()
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
    }
    return {synonyms.get(t, t) for t in found}


def verify_ontology_f1(pred: str, gt: str) -> float:
    """F1 over the set of ontology terms found in pred vs gt."""
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
    return 2 * prec * rec / (prec + rec)


# ─────────────────────────────────────────────────────────────────────────
# Dispatch
# ─────────────────────────────────────────────────────────────────────────

_ROUTING = {
    # Binary yes/no
    "random_finding_question": verify_binary_yesno,
    "structural_heart_disease": verify_binary_yesno,
    "category_pericarditis": verify_binary_yesno,
    "category_chamber_enlargement": verify_binary_yesno,
    "category_rhythm": verify_binary_yesno,
    "category_conduction": verify_binary_yesno,
    "category_infarct_ischemia": verify_binary_yesno,
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
    "classification": verify_ontology_f1,
    "category_other": verify_ontology_f1,
}


_PREFIX_ARTIFACT_RE = re.compile(r"^\s*[:;,.\-*]+\s*", re.UNICODE)


def _prefix_penalty(pred: str) -> float:
    """0.5x penalty if output starts with junk like ': ', '.', etc.
    The LLM-as-a-judge penalizes these too. v10 had a hard 0 penalty which
    broke training; v11 uses 0.5 — strong enough to discourage but doesn't
    zero out the gradient signal."""
    if not pred:
        return 1.0
    p = pred.lstrip()
    bad_leaders = [": ", ":\n", ". ", ".\n", ", ", "info:", "answer:", "note:"]
    for b in bad_leaders:
        if p.lower().startswith(b):
            return 0.5
    if _PREFIX_ARTIFACT_RE.match(pred):
        return 0.5
    return 1.0


def verify(pred: str, gt: str, category: str) -> float:
    fn = _ROUTING.get(category, verify_ontology_f1)
    try:
        score = float(fn(pred, gt))
    except Exception as e:
        print(f"[verifiable_reward] error in {category}: {e}")
        return 0.0
    # Apply prefix penalty — judge will see (and penalize) bad prefixes too.
    score *= _prefix_penalty(pred)
    return score


# Smoke test
if __name__ == "__main__":
    cases = [
        ("structural_heart_disease", "Yes - structural heart disease is present", "Yes - structural heart disease is present based on echocardiography", 1.0),
        ("structural_heart_disease", "No - no structural heart disease", "Yes - structural heart disease is present", 0.0),
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
        ("culprit_artery", "The culprit artery is the Mid RCA with complete occlusion",
                            "The culprit artery is the Proximal LAD with complete occlusion", 0.5),
        ("acs_severity", "No - obstructive coronary disease",
                          "No - obstructive coronary disease without acute occlusion", 1.0),
        ("acs_severity", "Yes - acute coronary occlusion; culprit Proximal LAD complete",
                          "Yes - acute coronary occlusion; culprit Proximal LAD with complete occlusion", 1.0),
        ("json_interpretation", '{"RHYTHM":["Sinusal","Regular"],"heart_rate_bpm":75}',
                                 '{"RHYTHM":["Sinusal","Regular"],"heart_rate_bpm":75}', 1.0),
        ("interpretation", "Atrial fibrillation. ST depression",
                            "Atrial fibrillation. (HR: 84.0 bpm); ST depression nonspecific", 1.0),  # both have afib, st depression
    ]
    for cat, pred, gt, expected in cases:
        got = verify(pred, gt, cat)
        ok = "✓" if abs(got - expected) < 0.5 else "✗"
        print(f"{ok}  {cat:35s} expected≈{expected:.2f}  got={got:.3f}")
