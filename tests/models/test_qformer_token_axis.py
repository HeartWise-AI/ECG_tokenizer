"""token_axis='time' mode of the ECG Q-Former bridge (Step 3.2).

Codes are assigned per CHANNEL by the RVQ, so the time-axis bridge rebuilds the
continuous post-quant tensor through the FROZEN quantizer itself
(get_output_from_indices — the production x1_split RVQ uses implicit neural
codebooks, so levels >=1 are MLP-conditioned and no static table can reproduce
them), transposes it time-major, and projects each time slice with an identity
init. At step 0 the pre-norm kv features must be exactly zᵀ (+PE) in the first
num_steps dims — the probe-validated view.
"""

import pytest
import torch
from vector_quantize_pytorch import ResidualVQ

from models.bridge.bridge import ECGQFormerBridge

VOCAB = 32
NCB = 4
L = 16      # channel positions
D = 10      # codebook dim (time axis length)
D_MID = 24


def _make_bridge(token_axis="time", **overrides):
    kwargs = dict(
        vocab_size=VOCAB,
        num_codebooks=NCB,
        d_mid=D_MID,
        d_llm=32,
        d_txt=16,
        num_steps=L,
        num_query_tokens=4,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        mix_strategy="concat_linear",
        token_axis=token_axis,
        codebook_dim=D,
    )
    kwargs.update(overrides)
    return ECGQFormerBridge(**kwargs)


def _rvq_and_ids(batch=2, seed=0):
    torch.manual_seed(seed)
    rvq = ResidualVQ(dim=D, num_quantizers=NCB, codebook_size=VOCAB,
                     commitment_weight=0.25, implicit_neural_codebook=False).eval()
    x = torch.randn(batch, L, D)
    with torch.no_grad():
        _, ids, _ = rvq(x)
    return rvq, ids


def test_invalid_token_axis_rejected():
    with pytest.raises(ValueError, match="token_axis"):
        _make_bridge(token_axis="lead")


def test_time_mode_params_and_frozen_rvq_out_of_state_dict():
    bridge = _make_bridge()
    rvq, _ = _rvq_and_ids()
    bridge.attach_quantizer(rvq)
    keys = bridge.state_dict().keys()
    assert any(k.startswith("time_proj.") for k in keys)
    assert not any(k.startswith(("mix_gate.", "mix_proj.", "_frozen_rvq")) for k in keys)


def test_forward_requires_attached_quantizer():
    bridge = _make_bridge()
    ids = torch.randint(0, VOCAB, (2, L, NCB))
    with pytest.raises(RuntimeError, match="attach_quantizer"):
        bridge(ids)


def test_kv_equals_rebuilt_z_transposed_plus_pe():
    """Identity init ⇒ pre-norm kv == get_output_from_indices(ids)ᵀ zero-padded, +PE."""
    bridge = _make_bridge().eval()
    rvq, ids = _rvq_and_ids(seed=1)
    bridge.attach_quantizer(rvq)

    with torch.no_grad():
        z = rvq.get_output_from_indices(ids)                 # (B, L, D)
    expected = z.permute(0, 2, 1)                            # (B, D, L) time-major

    projected = bridge.time_proj(expected)
    assert torch.allclose(projected[..., :L], expected, atol=1e-5)
    assert torch.allclose(projected[..., L:], torch.zeros(z.size(0), D, D_MID - L), atol=1e-6)

    pe = bridge._time_pe(D, D_MID, projected.device, projected.dtype)
    mixed, valid = bridge._embed_and_fuse(ids)
    assert torch.allclose(mixed, bridge.input_norm(projected + pe), atol=1e-5)
    assert valid.shape == (z.size(0), D) and valid.all()


def test_forward_shapes_and_channel_mask_ignored():
    bridge = _make_bridge().eval()
    rvq, ids = _rvq_and_ids(batch=3, seed=2)
    bridge.attach_quantizer(rvq)
    channel_mask = torch.ones(3, L)  # channel-shaped mask must be ignored in time mode
    prefix, vec = bridge(ids, attn_mask=channel_mask)
    assert prefix.shape == (3, 4, 32)
    assert vec.shape == (3, 16)


def test_gradients_flow_to_time_proj_not_rvq():
    bridge = _make_bridge().train()
    rvq, ids = _rvq_and_ids(seed=3)
    bridge.attach_quantizer(rvq)
    prefix, _ = bridge(ids)
    prefix.sum().backward()
    assert bridge.time_proj.weight.grad is not None
    assert bridge.time_proj.weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in rvq.parameters())


@pytest.mark.slow
def test_get_output_from_indices_matches_real_quantizer_z():
    """On the production x1_split checkpoint (implicit neural codebooks):
    get_output_from_indices(ids) must equal the quantizer's forward z."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "probe_tokenizer", "/volume/ECG_tokenizer/scripts/probe_tokenizer.py")
    probe_tokenizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe_tokenizer)
    dev = torch.device("cpu")
    enc, q = probe_tokenizer.load_models(
        "/volume/ECG_tokenizer/checkpoints/x1_split/tokenizer_aux_final.pt", dev)
    x = torch.randn(2, 12, 2500)
    with torch.no_grad():
        z, ids, _ = q(enc(x))
        rebuilt = q.quantizer.get_output_from_indices(ids)
    assert torch.allclose(rebuilt, z, atol=1e-4), (rebuilt - z).abs().max()
