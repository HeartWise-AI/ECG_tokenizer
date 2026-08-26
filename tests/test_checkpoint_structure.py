"""Contracts for reconstructing structural bridge options from checkpoints."""

from types import SimpleNamespace

import pytest
import torch

from utils.checkpoint_structure import (
    resolve_checkpoint_bridge_option,
    validate_bridge_structural_state_dict,
)


def test_embedded_checkpoint_structure_wins_when_sidecar_agrees():
    embedded = SimpleNamespace(bridge_token_axis="time")
    sidecar = {"bridge_token_axis": "time"}

    assert (
        resolve_checkpoint_bridge_option(
            embedded, sidecar, "bridge_token_axis", default="channel"
        )
        == "time"
    )


def test_stale_sidecar_structure_is_rejected():
    embedded = {"bridge_mix_strategy": "concat_linear"}
    sidecar = {"bridge_mix_strategy": "softmax"}

    with pytest.raises(ValueError, match="conflicts with embedded"):
        resolve_checkpoint_bridge_option(
            embedded,
            sidecar,
            "bridge_mix_strategy",
        )


def test_sidecar_structure_is_legacy_fallback_only():
    assert (
        resolve_checkpoint_bridge_option(
            {},
            {"bridge_token_axis": "channel"},
            "bridge_token_axis",
        )
        == "channel"
    )


def test_full_checkpoint_rejects_wrong_option_dependent_bridge_tensors():
    checkpoint = {
        "decoder.bridge.mix_proj.weight": torch.zeros(2, 2),
        "decoder.bridge.mix_proj.bias": torch.zeros(2),
    }
    model = {
        "decoder.bridge.time_proj.weight": torch.zeros(2, 2),
        "decoder.bridge.time_proj.bias": torch.zeros(2),
    }

    with pytest.raises(RuntimeError, match="bridge structural checkpoint mismatch"):
        validate_bridge_structural_state_dict(checkpoint, model)


def test_full_checkpoint_accepts_exact_structural_tensors():
    checkpoint = {
        "decoder.bridge.time_proj.weight": torch.zeros(2, 3),
        "decoder.bridge.time_proj.bias": torch.zeros(2),
        "decoder.llm.weight": torch.zeros(1),
    }
    model = {
        "decoder.bridge.time_proj.weight": torch.ones(2, 3),
        "decoder.bridge.time_proj.bias": torch.ones(2),
        "unrelated.weight": torch.ones(7),
    }

    validate_bridge_structural_state_dict(checkpoint, model)
