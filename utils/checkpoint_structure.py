"""Fail-closed reconstruction of checkpoint-defined bridge structure."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch


BRIDGE_STRUCTURE_OPTIONS = frozenset(
    {"bridge_mix_strategy", "bridge_token_axis"}
)
_BRIDGE_STRUCTURE_PARAMETER_MARKERS = (
    "bridge.blocks.",
    "bridge.embed_tables.",
    "bridge.input_norm.",
    "bridge.mix_gate.",
    "bridge.mix_proj.",
    "bridge.queries",
    "bridge.stage1_blocks.",
    "bridge.time_proj.",
)


def _config_value(container: Any, key: str) -> Any:
    if container is None:
        return None
    if isinstance(container, Mapping):
        return container.get(key)
    return getattr(container, key, None)


def resolve_checkpoint_bridge_option(
    checkpoint_config: Any,
    sidecar_config: Any,
    key: str,
    *,
    default: Any = None,
) -> Any:
    """Prefer embedded checkpoint structure and reject a stale sidecar mismatch."""
    if key not in BRIDGE_STRUCTURE_OPTIONS:
        raise KeyError(f"unsupported bridge structure option: {key}")
    embedded = _config_value(checkpoint_config, key)
    sidecar = _config_value(sidecar_config, key)
    if embedded is not None:
        if sidecar is not None and str(sidecar) != str(embedded):
            raise ValueError(
                f"sidecar {key}={sidecar!r} conflicts with embedded checkpoint "
                f"value {embedded!r}"
            )
        return embedded
    return sidecar if sidecar is not None else default


def _structural_parameters(
    state_dict: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    normalized: dict[str, torch.Tensor] = {}
    for raw_key, tensor in state_dict.items():
        key = raw_key.removeprefix("module.")
        if any(marker in key for marker in _BRIDGE_STRUCTURE_PARAMETER_MARKERS):
            if key in normalized:
                raise RuntimeError(f"duplicate bridge structural tensor: {key}")
            normalized[key] = tensor
    return normalized


def validate_bridge_structural_state_dict(
    checkpoint_state: Mapping[str, torch.Tensor],
    model_state: Mapping[str, torch.Tensor],
) -> None:
    """Reject option-dependent bridge tensor omissions and shape mismatches."""
    checkpoint = _structural_parameters(checkpoint_state)
    model = _structural_parameters(model_state)
    missing = sorted(set(model).difference(checkpoint))
    unexpected = sorted(set(checkpoint).difference(model))
    mismatched = sorted(
        key
        for key in set(checkpoint).intersection(model)
        if tuple(checkpoint[key].shape) != tuple(model[key].shape)
    )
    if missing or unexpected or mismatched:
        details = []
        if missing:
            details.append(f"missing={missing}")
        if unexpected:
            details.append(f"unexpected={unexpected}")
        if mismatched:
            details.append(f"shape_mismatched={mismatched}")
        raise RuntimeError("bridge structural checkpoint mismatch: " + "; ".join(details))
