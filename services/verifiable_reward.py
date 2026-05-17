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

_PREFIX_JUNK_RE = re.compile(
    r"^\s*(?:"
    r"info\s*:|note\s*:|answer\s*:|user|assistant|model|"
    r"-{2,}|\*{2,}|={2,}|"
    r"here (?:is|are)[^\n:]*[:\n]|"
    r"[:;,.\-*\s]+"
    r")\s*",
    re.IGNORECASE)


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


def _first_word(s: str) -> str:
    """Return the first alphabetic word, lowercased. Strips colons/dashes."""
    s = _strip_prefix(s)[0]
    s = s.strip().lstrip(":-*. ").strip()
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
    s_clean = _strip_prefix(s)[0]
    s_lower = s_clean.strip().lower()
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


def verify_yesno_with_findings(pred: str, gt: str) -> float:
    """For the diagnostic 'category_*' prompts. GT is one of:
      - 'No - no evidence of X'              (negative)
      - 'Yes - finding1; finding2; ...'      (positive + findings)
      - free-text findings ('Sinus rhythm (HR: 61); Regular rhythm')

    Scoring (matches what the judge does):
      * If GT has a Yes/No gate and pred's gate disagrees -> 0.0
      * Else: 0.4 for correct gate + 0.6 * F1(findings ontology terms).
      * If GT has no gate (pure findings), score = F1(findings) directly.
    The judge heavily penalizes wrong findings even when Yes/No is right,
    so we must check the content, not just the gate. (Bug found 2026-05:
    binary-only verifier scored 1.0 where judge gave 0.2.)
    """
    gyn = _canonical_yesno(gt)
    pyn = _canonical_yesno(pred)

    # GT has an explicit Yes/No gate
    if gyn in ("yes", "no"):
        if pyn != gyn:
            return 0.0
        if gyn == "no":
            return 1.0  # negative case: gate match is the whole answer
        # positive: gate right, now grade the findings
        f1 = verify_ontology_f1(pred, gt)
        return 0.4 + 0.6 * f1
    # No gate in GT -> pure findings comparison (e.g. category_rhythm)
    return verify_ontology_f1(pred, gt)


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
    if diff <= 5:
        return 1.0
    if diff <= 15:
        return 0.85
    return 0.6


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
    # abnormal signals
    if ("abnormal" in s or "pathological finding" in s
            or "significant abnormalit" in s
            or s.startswith("yes")):  # 'Yes - there are pathological findings'
        return "abnormal"
    # normal signals
    if ("normal ecg" in s or "is normal" in s or "routine follow" in s
            or s.startswith("no")):  # 'No - ECG is normal'
        return "normal"
    return None


def verify_classification(pred: str, gt: str) -> float:
    """classification: the answer is verifiable — it states an ECG class
    (normal / borderline / abnormal) plus a findings list. Scoring:
      * class mismatch -> 0.0
      * class match, GT has no findings -> 1.0
      * class match, GT has findings -> 0.5 + 0.5 * F1(findings ontology)
    Class is the dominant signal, so a correct 'abnormal/normal/borderline'
    is already half credit + findings overlap. (User 2026-05: classification
    has verifiable rewards, should score near-100% when the class is right.)"""
    gc = _ecg_class(gt)
    pc = _ecg_class(pred)
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
    return 0.5 + 0.5 * verify_ontology_f1(pred, gt)


def verify_risk_level(pred: str, gt: str) -> float:
    """afib_risk: STRICT binary — 1.0 iff the risk level (low/high/uncertain)
    matches the GT, else 0.0. Nothing else counts. Prefix junk is stripped
    first so 'info: Low risk ...' still parses as 'low'.
    (Per spec 2026-05: afib is 1 only if the risk matches, else 0.)"""
    def level(s):
        s = _strip_prefix(s)[0].strip().lower()
        # Look at the first ~60 chars for the risk word
        head = s[:60]
        if head.startswith("low") or "low risk" in head:
            return "low"
        if head.startswith("high") or "high risk" in head:
            return "high"
        if head.startswith(("uncertain", "unknown", "moderate", "intermediate")) \
                or "uncertain risk" in head or "moderate risk" in head:
            return "uncertain"
        if head.startswith("yes"):  # 'Yes - high/low risk ...'
            if "low" in head:
                return "low"
            if "high" in head:
                return "high"
        if head.startswith("no"):   # 'No - low risk ...'
            if "low" in head:
                return "low"
            if "high" in head:
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
    return 1.0 if abs(p - g) <= tol else max(0.0, 1.0 - abs(p - g) / 30.0)


def verify_numeric_interval(pred: str, gt: str) -> float:
    """ecg_interval: extract first number, compare within a TIGHT tolerance.
    The judge treats interval errors clinically — a 42 ms QT error scores 0
    even though it's <10%. Tolerance tightened to ±5% (min ±8 units) and
    the partial-credit falloff is steep. (Bug found 2026-05: ±10% was far
    too loose vs the judge.)"""
    p = _extract_first_number(pred)
    g = _extract_first_number(gt)
    if p is None or g is None or g == 0:
        return 0.0
    tol = max(8.0, abs(g) * 0.05)  # ±5% or ±8 (whichever larger)
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
        vessel = None
        if "left main" in s or "lmca" in s:
            vessel = "left_main"
        elif "lad" in s or "left anterior descending" in s:
            vessel = "lad"
        elif "rca" in s or "right coronary" in s:
            vessel = "rca"
        elif "lcx" in s or "circumflex" in s:
            vessel = "lcx"
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
    # Pure binary yes/no (GT is just 'Yes'/'No' or 'No - <neg statement>')
    "random_finding_question": verify_binary_yesno,
    "structural_heart_disease": verify_binary_yesno,
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


def _prefix_penalty(pred: str) -> float:
    """Mild 0.9x nudge if output starts with junk like 'info:', '---'.
    The per-category parsers already strip prefixes and score the content
    correctly; the LLM-judge only *mildly* penalizes prefixes (~0.7-0.8,
    not 0.5). A small nudge keeps GRPO gently discouraging junk without
    diverging from the judge or breaking the pure-binary categories.
    (2026-05: the old 0.5x double-penalized — parsers strip prefix AND
    this halved the score, badly diverging from judge + afib binary spec.)"""
    if not pred:
        return 1.0
    p = pred.lstrip()
    bad_leaders = ["info:", "answer:", "note:", "user", "assistant", "---", "***"]
    pl = p.lower()
    for b in bad_leaders:
        if pl.startswith(b):
            return 0.9
    return 1.0


def verify(pred: str, gt: str, category: str) -> float:
    fn = _ROUTING.get(category, verify_ontology_f1)
    try:
        score = float(fn(pred, gt))
    except Exception as e:
        print(f"[verifiable_reward] error in {category}: {e}")
        return 0.0
    # afib_risk is a STRICT binary per spec — no prefix penalty at all.
    if category == "afib_risk":
        return score
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
