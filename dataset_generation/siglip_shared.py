from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Set

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slug(text: str) -> str:
    return _SLUG_RE.sub("_", text.lower()).strip("_")


def text_ids_for_label(label: str, include_qa: bool = True, include_no: bool = False) -> List[str]:
    base = slug(label)
    ids = [f"LBL_{base}"]
    if include_qa:
        ids.append(f"QA_{base}_yes")
        if include_no:
            ids.append(f"QA_{base}_no")
    return ids


EXCLUSIVE_GROUPS: Dict[str, Dict[str, Any]] = {
    "RHYTHM_PRIMARY": {
        "members": {
            "Sinusal",
            "Afib",
            "Atrial flutter",
            "Atrial tachycardia (>= 100 BPM)",
            "Ectopic atrial rhythm (< 100 BPM)",
            "Junctional rhythm",
            "Supraventricular tachycardia",
            "Ventricular Rhythm",
            "Ventricular tachycardia",
        },
        "exclusive": True,
    },
    "RHYTHM_REGULARITY": {
        "members": {
            "Regular",
            "Regularly irregular",
            "Irregularly irregular",
        },
        "exclusive": True,
    },
    "AXIS": {
        "members": {
            "Left axis deviation",
            "Right axis deviation",
            "Right superior axis",
        },
        "exclusive": True,
    },
    "BUNDLE": {
        "members": {
            "Left bundle branch block",
            "Right bundle branch block",
        },
        "exclusive": True,
    },
    "AV_BLOCK": {
        "members": {
            "1st degree AV block",
            "2nd degree AV block - mobitz 1",
            "2nd degree AV block - mobitz 2",
            "Third Degree AV Block",
        },
        "exclusive": True,
    },
    "SIGLIP_VENTRICULAR_COMPLEX": {
        "members": {
            "LV pacing",
            "Ventricular paced",
            "Ventricular Rhythm",
            "Ventricular tachycardia",
            "Left bundle branch block",
            "Supraventricular tachycardia",
            "Afib",
            "Junctional rhythm",
        },
        "exclusive": False,
        "targeted_negatives": {
            "LV pacing": [
                "Ventricular paced",
                "Ventricular Rhythm",
                "Ventricular tachycardia",
                "Left bundle branch block",
            ],
            "Ventricular paced": [
                "LV pacing",
                "Ventricular Rhythm",
                "Ventricular tachycardia",
                "Left bundle branch block",
            ],
            "Ventricular Rhythm": [
                "Afib",
                "Junctional rhythm",
                "Ventricular paced",
                "LV pacing",
                "Ventricular tachycardia",
            ],
            "Ventricular tachycardia": [
                "Ventricular Rhythm",
                "Afib",
                "Supraventricular tachycardia",
                "Ventricular paced",
            ],
        },
    },
    "SIGLIP_ST_SEGMENT": {
        "members": {
            "ST elevation (septal - V1-V2)",
            "ST depression (septal- V1-V2)",
            "ST elevation (anterior - V3-V4)",
            "ST depression (anterior - V3-V4)",
            "ST elevation (inferior - II, III, aVF)",
            "ST depression (inferior - II, III, aVF)",
            "ST elevation (lateral - I, aVL, V5-V6)",
            "ST depression (lateral - I, avL, V5-V6)",
            "ST elevation (posterior - V7-V8-V9)",
        },
        "exclusive": False,
        "targeted_negatives": {
            "ST elevation (septal - V1-V2)": [
                "ST depression (septal- V1-V2)",
            ],
            "ST elevation (anterior - V3-V4)": [
                "ST depression (anterior - V3-V4)",
            ],
            "ST elevation (inferior - II, III, aVF)": [
                "ST depression (inferior - II, III, aVF)",
            ],
            "ST elevation (lateral - I, aVL, V5-V6)": [
                "ST depression (lateral - I, avL, V5-V6)",
            ],
            "ST elevation (posterior - V7-V8-V9)": [
                "ST depression (septal- V1-V2)",
                "ST depression (anterior - V3-V4)",
            ],
        },
    },
    "SIGLIP_AV_BLOCK": {
        "members": {
            "Third Degree AV Block",
            "1st degree AV block",
            "2nd degree AV block - mobitz 1",
            "2nd degree AV block - mobitz 2",
            "Junctional rhythm",
        },
        "exclusive": False,
        "targeted_negatives": {
            "Third Degree AV Block": [
                "1st degree AV block",
                "2nd degree AV block - mobitz 1",
                "2nd degree AV block - mobitz 2",
                "Junctional rhythm",
            ],
        },
    },
    "SIGLIP_U_WAVE": {
        "members": {
            "U wave",
            "Prolonged QT",
            "Low voltage",
            "T wave inversion (lateral -I, aVL, V5-V6)",
            "T wave inversion (anterior - V3-V4)",
        },
        "exclusive": False,
        "targeted_negatives": {
            "U wave": [
                "Prolonged QT",
                "Low voltage",
                "T wave inversion (lateral -I, aVL, V5-V6)",
                "T wave inversion (anterior - V3-V4)",
            ],
        },
    },
    "SIGLIP_LEAD_MISPLACEMENT": {
        "members": {
            "Lead misplacement",
            "LV pacing",
            "Ventricular paced",
            "Ventricular Rhythm",
        },
        "exclusive": False,
        "targeted_negatives": {
            "Lead misplacement": [
                "LV pacing",
                "Ventricular paced",
                "Ventricular Rhythm",
            ],
        },
    },
}


def group_members(config: Dict[str, Any]) -> Set[str]:
    members = config.get("members", []) if isinstance(config, dict) else config
    return set(members)


def group_is_exclusive(config: Dict[str, Any]) -> bool:
    return bool(config.get("exclusive", True))


def targeted_negatives_for_label(label: str) -> List[str]:
    for config in EXCLUSIVE_GROUPS.values():
        targeted = config.get("targeted_negatives") or {}
        if label in targeted:
            return targeted[label]
    return []


def build_siglip_hard_negative_text_ids(include_qa: bool = True) -> Dict[str, List[str]]:
    mapping: Dict[str, List[str]] = {}
    for config in EXCLUSIVE_GROUPS.values():
        targeted = config.get("targeted_negatives")
        if not targeted:
            continue
        for anchor_label, negative_labels in targeted.items():
            anchor_ids = text_ids_for_label(anchor_label, include_qa=include_qa, include_no=False)
            negative_ids: List[str] = []
            for negative_label in negative_labels:
                negative_ids.extend(text_ids_for_label(negative_label, include_qa=include_qa, include_no=False))
            negative_ids = sorted(set(negative_ids) - set(anchor_ids))
            for anchor_id in anchor_ids:
                mapping[anchor_id] = negative_ids
    return mapping


SIGLIP_TARGETED_HARD_NEGATIVES: Dict[str, List[str]] = build_siglip_hard_negative_text_ids(include_qa=True)

