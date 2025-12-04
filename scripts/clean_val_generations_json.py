#!/usr/bin/env python3
"""
Clean validation generations JSON by stripping leaked EOT tokens and extra semicolons.

Usage:
  python scripts/clean_val_generations_json.py /path/to/val_generations_epoch_001_step_70000.json

Writes changes in-place after making a timestamped backup next to the file.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime
from typing import Any, Dict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from runners.llm_finetuning_runner import LLMFinetuningRunner  # reuse sanitizer


def sanitize_text(text: Any) -> str:
    """Normalize a value to string and sanitize chat artifacts."""
    if text is None:
        return ""
    return LLMFinetuningRunner._sanitize_chat_text(str(text))


def clean_val_generations_json(path: str) -> int:
    """Clean a JSON mapping of ECG entries and rewrite in-place.

    Returns the number of entries modified.
    """
    with open(path, "r", encoding="utf-8") as f:
        data: Dict[str, Dict[str, Any]] = json.load(f)

    changed = 0
    for key, entry in list(data.items()):
        if not isinstance(entry, dict):
            continue
        before_gen = entry.get("Generation", "")
        before_gt = entry.get("Ground truth", "")
        before_q = entry.get("Question", "")

        after_gen = sanitize_text(before_gen)
        after_gt = sanitize_text(before_gt)
        after_q = sanitize_text(before_q)

        if after_gen != before_gen or after_gt != before_gt or after_q != before_q:
            entry["Generation"] = after_gen
            entry["Ground truth"] = after_gt
            entry["Question"] = after_q
            changed += 1

    # Backup existing file
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_path = f"{path}.bak.{ts}"
    shutil.copy2(path, backup_path)

    # Write cleaned file
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return changed


def main():
    parser = argparse.ArgumentParser(description="Strip leaked <|eot|> tokens and tidy punctuation in val_generations JSON.")
    parser.add_argument("json_path", help="Path to val_generations_epoch_*.json")
    args = parser.parse_args()

    if not os.path.exists(args.json_path):
        print(f"Error: not found: {args.json_path}")
        sys.exit(1)

    changed = clean_val_generations_json(args.json_path)
    print(f"✓ Cleaned {changed} entr{'y' if changed == 1 else 'ies'} in {args.json_path}")


if __name__ == "__main__":
    main()

