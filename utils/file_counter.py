"""Utility to count .pt/.bin checkpoints under a directory."""

from __future__ import annotations

import os
from typing import Iterable

__all__ = ["count_checkpoints", "list_checkpoint_files"]


EXTENSIONS = {".pt", ".bin", ".ckpt"}


def list_checkpoint_files(root: str) -> list[str]:
    if not root or not os.path.exists(root):
        return []
    matches: list[str] = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            _, ext = os.path.splitext(name)
            if ext.lower() in EXTENSIONS:
                matches.append(os.path.join(dirpath, name))
    return sorted(matches)


def count_checkpoints(root: str) -> int:
    return len(list_checkpoint_files(root))
