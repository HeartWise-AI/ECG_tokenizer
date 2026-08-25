"""Codebook-fusion strategies of the ECG Q-Former bridge family.

RVQ codebooks are additive, so the bridge offers three fusion modes:
  softmax        — legacy convex-combination gate (back-compat default)
  sum            — plain unweighted sum (ablation floor)
  concat_linear  — concat(8×d_mid) -> Linear, initialised to the identity-sum

The identity init must make concat_linear exactly reproduce sum at step 0, and
old softmax checkpoints must keep loading unchanged.
"""

import pytest
import torch

from models.bridge.bridge import (
    ECGQFormerBridge,
    ECGQFormerBridgeStage1,
    InstructionAwareECGQFormerBridge,
)

VOCAB = 64
NCB = 4
D_MID = 32


def _make_bridge(mix_strategy: str, **overrides) -> ECGQFormerBridge:
    kwargs = dict(
        vocab_size=VOCAB,
        num_codebooks=NCB,
        d_mid=D_MID,
        d_llm=48,
        d_txt=24,
        num_steps=16,
        num_query_tokens=6,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        mix_strategy=mix_strategy,
    )
    kwargs.update(overrides)
    return ECGQFormerBridge(**kwargs)


def _make_ids(batch: int = 2, seq: int = 16) -> torch.Tensor:
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(0, VOCAB, (batch, seq, NCB), generator=g)
    ids[0, -3:, :] = VOCAB  # pad_id positions (fully invalid)
    ids[1, 4, 2:] = -1      # partially masked levels
    return ids


def test_invalid_mix_strategy_rejected():
    with pytest.raises(ValueError, match="mix_strategy"):
        _make_bridge("mean")


def test_softmax_default_keeps_gate_parameters():
    bridge = _make_bridge("softmax")
    keys = bridge.state_dict().keys()
    assert any(k.startswith("mix_gate.") for k in keys)
    assert not any(k.startswith("mix_proj.") for k in keys)


def test_concat_linear_identity_init_equals_sum():
    torch.manual_seed(1)
    cl = _make_bridge("concat_linear").eval()
    summed = _make_bridge("sum").eval()
    # concat_linear's state dict is a superset of sum's (extra mix_proj tensors).
    missing, unexpected = summed.load_state_dict(cl.state_dict(), strict=False)
    assert not missing
    assert all(k.startswith("mix_proj.") for k in unexpected)

    ids = _make_ids()
    with torch.no_grad():
        prefix_cl, vec_cl = cl(ids)
        prefix_sum, vec_sum = summed(ids)
    assert torch.allclose(prefix_cl, prefix_sum, atol=1e-4)
    assert torch.allclose(vec_cl, vec_sum, atol=1e-4)


def test_additive_modes_ignore_invalid_levels():
    torch.manual_seed(2)
    bridge = _make_bridge("sum").eval()
    ids = _make_ids()
    variant = ids.clone()
    # Rewriting a masked level's id must not change the output.
    assert variant[1, 4, 3] == -1
    variant[1, 4, 3] = -7
    with torch.no_grad():
        prefix_a, _ = bridge(ids)
        prefix_b, _ = bridge(variant)
    assert torch.allclose(prefix_a, prefix_b)


def test_concat_linear_gradient_reaches_mix_proj():
    torch.manual_seed(3)
    bridge = _make_bridge("concat_linear").train()
    prefix, _ = bridge(_make_ids())
    prefix.sum().backward()
    assert bridge.mix_proj.weight.grad is not None
    assert bridge.mix_proj.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("mix_strategy", ["softmax", "sum", "concat_linear"])
def test_stage1_to_instruction_bridge_roundtrip(tmp_path, mix_strategy):
    torch.manual_seed(4)
    common = dict(
        vocab_size=VOCAB,
        num_codebooks=NCB,
        d_mid=D_MID,
        d_llm=48,
        d_txt=24,
        num_steps=16,
        num_query_tokens=6,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        mix_strategy=mix_strategy,
    )
    stage1 = ECGQFormerBridgeStage1(**common, txt_vocab_size=100, txt_pad_id=0)
    ckpt_path = tmp_path / "stage1.pt"
    torch.save({"model_state_dict": stage1.state_dict()}, ckpt_path)

    bridge = InstructionAwareECGQFormerBridge(**common)
    bridge.load_stage1_checkpoint(str(ckpt_path))

    fusion_prefix = "mix_gate." if mix_strategy == "softmax" else "mix_proj."
    if mix_strategy == "sum":
        return  # no fusion parameters to compare
    for name, param in stage1.named_parameters():
        if name.startswith(fusion_prefix):
            loaded = dict(bridge.named_parameters())[name]
            assert torch.equal(param, loaded), f"{name} not transferred"
