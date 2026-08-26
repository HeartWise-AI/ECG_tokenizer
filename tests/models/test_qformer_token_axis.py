"""token_axis='time' mode of the ECG Q-Former bridge (Step 3.2).

Codes are assigned per CHANNEL by the RVQ, so the time-axis bridge rebuilds the
continuous post-quant tensor through the FROZEN quantizer itself
(get_output_from_indices - the production x1_split RVQ uses implicit neural
codebooks, so levels >=1 are MLP-conditioned and no static table can reproduce
them), transposes it time-major, and projects each time slice with an identity
init. At step 0 the pre-norm kv features must be exactly zᵀ (+PE) in the first
num_steps dims - the probe-validated view.
"""

import importlib.util
import os
from pathlib import Path
import sys
import types
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F


_BRIDGE_PATH = Path(__file__).resolve().parents[2] / "models" / "bridge" / "bridge.py"
_BRIDGE_SPEC = importlib.util.spec_from_file_location("qformer_bridge_under_test", _BRIDGE_PATH)
if _BRIDGE_SPEC is None or _BRIDGE_SPEC.loader is None:
    raise ImportError(f"could not load bridge module from {_BRIDGE_PATH}")
_BRIDGE_MODULE = importlib.util.module_from_spec(_BRIDGE_SPEC)
_BRIDGE_SPEC.loader.exec_module(_BRIDGE_MODULE)
ECGQFormerBridge = _BRIDGE_MODULE.ECGQFormerBridge

VOCAB = 32
NCB = 4
L = 16      # channel positions
D = 10      # codebook dim (time axis length)
D_MID = 24


class _FakeResidualVQ(nn.Module):
    """Small additive RVQ double for bridge contract tests."""

    def __init__(self, *, dim=D, num_quantizers=NCB, codebook_size=VOCAB):
        super().__init__()
        self.dim = int(dim)
        self.num_quantizers = int(num_quantizers)
        self.codebook_size = int(codebook_size)
        generator = torch.Generator().manual_seed(123)
        self.codebooks = nn.Parameter(
            torch.randn(
                self.num_quantizers,
                self.codebook_size,
                self.dim,
                generator=generator,
            )
        )

    def get_output_from_indices(self, ids):
        if ids.dim() != 3 or ids.size(-1) != self.num_quantizers:
            raise ValueError("ids must contain every quantizer level")
        output = torch.zeros(*ids.shape[:-1], self.dim, device=ids.device)
        for level in range(self.num_quantizers):
            output = output + F.embedding(ids[..., level], self.codebooks[level])
        return output


class _TestRMSNorm(nn.Module):
    """RMSNorm fallback for the repository's incomplete local test environment."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = float(eps)

    def forward(self, value):
        scale = value.pow(2).mean(dim=-1, keepdim=True).add(self.eps).rsqrt()
        return value * scale * self.weight


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
    rms_norm = getattr(nn, "RMSNorm", _TestRMSNorm)
    with patch.object(nn, "RMSNorm", rms_norm, create=True):
        return ECGQFormerBridge(**kwargs)


def _rvq_and_ids(batch=2, seed=0):
    generator = torch.Generator().manual_seed(seed)
    rvq = _FakeResidualVQ().eval()
    ids = torch.randint(0, VOCAB, (batch, L, NCB), generator=generator)
    return rvq, ids


def test_invalid_token_axis_rejected():
    with pytest.raises(ValueError, match="token_axis"):
        _make_bridge(token_axis="lead")


def test_time_mode_rejects_softmax_fusion():
    with pytest.raises(ValueError, match="softmax.*token_axis='time'"):
        _make_bridge(mix_strategy="softmax")


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


def test_attach_quantizer_requires_matching_codebook_count():
    bridge = _make_bridge()
    rvq = _FakeResidualVQ(num_quantizers=NCB - 1)
    with pytest.raises(ValueError, match="num_quantizers.*3.*num_codebooks.*4"):
        bridge.attach_quantizer(rvq)


def test_attach_quantizer_requires_matching_codebook_dimension():
    bridge = _make_bridge()
    rvq = _FakeResidualVQ(dim=D + 2)
    with pytest.raises(ValueError, match="codebook_dim 12.*codebook_dim 10"):
        bridge.attach_quantizer(rvq)


@pytest.mark.parametrize("depth", [NCB - 1, NCB + 1])
def test_time_mode_requires_full_codebook_depth(depth):
    bridge = _make_bridge().eval()
    bridge.attach_quantizer(_FakeResidualVQ())
    ids = torch.randint(0, VOCAB, (2, L, depth))
    with pytest.raises(ValueError, match=f"exactly {NCB} codebooks"):
        bridge(ids)


@pytest.mark.parametrize("invalid_id", [-1, VOCAB, VOCAB + 7])
def test_time_mode_rejects_invalid_rvq_ids(invalid_id):
    bridge = _make_bridge().eval()
    bridge.attach_quantizer(_FakeResidualVQ())
    ids = torch.randint(0, VOCAB, (2, L, NCB))
    ids[0, 0, 0] = invalid_id
    with pytest.raises(ValueError, match=f"range \\[0, {VOCAB}\\)"):
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
    channel_mask = torch.zeros(3, L, dtype=torch.bool)
    channel_mask[:, ::2] = True
    with torch.no_grad():
        expected_prefix, expected_vec = bridge(ids)
        prefix, vec = bridge(ids, attn_mask=channel_mask)
    assert prefix.shape == (3, 4, 32)
    assert vec.shape == (3, 16)
    assert torch.equal(prefix, expected_prefix)
    assert torch.equal(vec, expected_vec)


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
@pytest.mark.integration
def test_get_output_from_indices_matches_real_quantizer_z():
    """On the production x1_split checkpoint (implicit neural codebooks):
    get_output_from_indices(ids) must equal the quantizer's forward z."""
    checkpoint_raw = os.environ.get("ECG_TOKENIZER_PROD_CHECKPOINT")
    if not checkpoint_raw:
        pytest.skip("set ECG_TOKENIZER_PROD_CHECKPOINT to run the production RVQ check")

    checkpoint_path = Path(checkpoint_raw).expanduser()
    if not checkpoint_path.is_file():
        pytest.skip(f"production RVQ checkpoint not found: {checkpoint_path}")

    pytest.importorskip("vector_quantize_pytorch")
    probe_path = Path(__file__).resolve().parents[2] / "scripts" / "probe_tokenizer.py"
    spec = importlib.util.spec_from_file_location("probe_tokenizer", probe_path)
    if spec is None or spec.loader is None:
        pytest.fail(f"could not load probe module from {probe_path}")
    probe_tokenizer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe_tokenizer)
    dev = torch.device("cpu")
    enc, q = probe_tokenizer.load_models(str(checkpoint_path), dev)
    x = torch.randn(2, 12, 2500)
    with torch.no_grad():
        z, ids, _ = q(enc(x))
        rebuilt = q.quantizer.get_output_from_indices(ids)
    assert torch.allclose(rebuilt, z, atol=1e-4), (rebuilt - z).abs().max()


def _load_medgemma_decoder(monkeypatch):
    import transformers

    monkeypatch.setattr(
        transformers,
        "AutoModelForImageTextToText",
        type("_UnusedAutoModel", (), {}),
        raising=False,
    )
    models_package = types.ModuleType("models")
    models_package.__path__ = [str(_BRIDGE_PATH.parents[1])]
    bridge_package = types.ModuleType("models.bridge")
    bridge_package.__path__ = [str(_BRIDGE_PATH.parent)]
    for name in (
        "ECGCodeBridge",
        "ECGProjectionBridge",
        "PerceiverProjectionBridge",
        "ECGQFormerBridge",
        "InstructionAwareECGQFormerBridge",
        "SequenceTokenBridge",
        "SimpleTokenBridge",
        "CrossModalSequenceTokenBridge",
    ):
        setattr(bridge_package, name, getattr(_BRIDGE_MODULE, name))
    decoder_package = types.ModuleType("models.decoder")
    decoder_package.__path__ = [str(_BRIDGE_PATH.parents[1] / "decoder")]
    monkeypatch.setitem(sys.modules, "models", models_package)
    monkeypatch.setitem(sys.modules, "models.bridge", bridge_package)
    monkeypatch.setitem(sys.modules, "models.bridge.bridge", _BRIDGE_MODULE)
    monkeypatch.setitem(sys.modules, "models.decoder", decoder_package)

    decoder_path = _BRIDGE_PATH.parents[1] / "decoder" / "medgemma_decoder.py"
    decoder_spec = importlib.util.spec_from_file_location(
        "models.decoder.medgemma_decoder",
        decoder_path,
    )
    assert decoder_spec is not None and decoder_spec.loader is not None
    medgemma_decoder = importlib.util.module_from_spec(decoder_spec)
    monkeypatch.setitem(
        sys.modules,
        "models.decoder.medgemma_decoder",
        medgemma_decoder,
    )
    decoder_spec.loader.exec_module(medgemma_decoder)
    return medgemma_decoder


def test_medgemma_decoder_propagates_configured_codebook_dimension(monkeypatch):
    medgemma_decoder = _load_medgemma_decoder(monkeypatch)

    captured = {}

    class _BridgeConstructed(Exception):
        pass

    class _BridgeSpy:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            raise _BridgeConstructed

    monkeypatch.setattr(medgemma_decoder, "ECGQFormerBridge", _BridgeSpy)
    monkeypatch.setattr(
        medgemma_decoder.AutoConfig,
        "from_pretrained",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(_BridgeConstructed):
        medgemma_decoder.MedGemmaDecoder(
            huggingface_model_name="unused",
            llm_input_embedding_size=32,
            quantized_feature_shape=(L, D),
            bridge_name="ECGQFormerBridge",
            ecg_codebook_size=VOCAB,
            num_visual_tokens=4,
            bridge_mid_dim=D_MID,
            bridge_num_heads=4,
            bridge_mix_strategy="concat_linear",
            bridge_token_axis="time",
            num_quantizers=NCB,
        )

    assert captured["codebook_dim"] == D


def test_medgemma_decoder_preserves_invalid_time_ids_for_bridge_rejection(monkeypatch):
    medgemma_decoder = _load_medgemma_decoder(monkeypatch)
    captured = {}

    class _TimeBridge(nn.Module):
        uses_codes = True
        token_axis = "time"
        pad_id = VOCAB

        def forward(self, ecg_ids, attn_mask=None):
            captured["ids"] = ecg_ids.clone()
            if bool((ecg_ids < 0).any()):
                raise ValueError("invalid RVQ ids reached the bridge")
            return torch.zeros(ecg_ids.size(0), 1, 4), torch.zeros(ecg_ids.size(0), 2)

    class _FakeLLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1, 4)

        def get_input_embeddings(self):
            return self.embedding

    decoder = medgemma_decoder.MedGemmaDecoder.__new__(medgemma_decoder.MedGemmaDecoder)
    nn.Module.__init__(decoder)
    decoder.bridge = _TimeBridge()
    decoder.llm_model = _FakeLLM()
    decoder.continuous_bridge = None
    decoder.continuous_only = False
    decoder._continuous_features = None

    ids = torch.randint(0, VOCAB, (1, L, NCB))
    ids[0, 0, 0] = -1
    with pytest.raises(ValueError, match="invalid RVQ ids reached the bridge"):
        decoder._compute_ecg_embeddings(None, ids)
    assert captured["ids"][0, 0, 0].item() == -1


def test_medgemma_decoder_preserves_single_codebook_time_axis_dimension(monkeypatch):
    medgemma_decoder = _load_medgemma_decoder(monkeypatch)
    captured = {}

    class _TimeBridge(nn.Module):
        uses_codes = True
        token_axis = "time"
        pad_id = VOCAB

        def forward(self, ecg_ids, attn_mask=None):
            captured["ids"] = ecg_ids.clone()
            captured["mask"] = attn_mask.clone()
            if ecg_ids.dim() != 3:
                raise ValueError("time-axis ids lost their codebook dimension")
            return torch.zeros(ecg_ids.size(0), 1, 4)

    class _FakeLLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1, 4)

        def get_input_embeddings(self):
            return self.embedding

    decoder = medgemma_decoder.MedGemmaDecoder.__new__(
        medgemma_decoder.MedGemmaDecoder
    )
    nn.Module.__init__(decoder)
    decoder.bridge = _TimeBridge()
    decoder.llm_model = _FakeLLM()
    decoder.continuous_bridge = None
    decoder.continuous_only = False
    decoder._continuous_features = None

    ids = torch.randint(0, VOCAB, (2, L, 1))
    embeddings, _ = decoder._compute_ecg_embeddings(None, ids)

    assert captured["ids"].shape == (2, L, 1)
    assert captured["mask"].shape == (2, L)
    assert embeddings.shape == (2, 1, 4)


def test_medgemma_decoder_preserves_invalid_channel_levels_for_qformer_masking(
    monkeypatch,
):
    medgemma_decoder = _load_medgemma_decoder(monkeypatch)
    captured = {}

    class _ChannelQFormer(nn.Module):
        uses_codes = True
        token_axis = "channel"
        pad_id = VOCAB

        def forward(self, ecg_ids, attn_mask=None):
            captured["ids"] = ecg_ids.clone()
            captured["mask"] = attn_mask.clone()
            return torch.zeros(ecg_ids.size(0), 1, 4), torch.zeros(ecg_ids.size(0), 2)

    class _FakeLLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.embedding = nn.Embedding(1, 4)

        def get_input_embeddings(self):
            return self.embedding

    monkeypatch.setattr(medgemma_decoder, "ECGQFormerBridge", _ChannelQFormer)
    decoder = medgemma_decoder.MedGemmaDecoder.__new__(
        medgemma_decoder.MedGemmaDecoder
    )
    nn.Module.__init__(decoder)
    decoder.bridge = _ChannelQFormer()
    decoder.llm_model = _FakeLLM()
    decoder.continuous_bridge = None
    decoder.continuous_only = False
    decoder._continuous_features = None

    ids = torch.randint(0, VOCAB, (1, L, NCB))
    ids[0, 0, 0] = -1
    decoder._compute_ecg_embeddings(None, ids)

    assert captured["ids"][0, 0, 0].item() == -1
    assert captured["mask"][0, 0].item() is True


def test_stage1_embedded_structure_rejects_no_sidecar_mismatch(
    monkeypatch, tmp_path
):
    medgemma_decoder = _load_medgemma_decoder(monkeypatch)
    checkpoint = tmp_path / "stage1.pt"
    torch.save(
        {
            "config": {
                "bridge_mix_strategy": "concat_linear",
                "bridge_token_axis": "channel",
            },
            "model_state_dict": {},
        },
        checkpoint,
    )
    decoder = medgemma_decoder.MedGemmaDecoder.__new__(
        medgemma_decoder.MedGemmaDecoder
    )
    nn.Module.__init__(decoder)
    decoder.bridge = nn.Module()
    decoder.bridge.mix_strategy = "concat_linear"
    decoder.bridge.token_axis = "time"
    decoder.stage1_metadata = decoder._inspect_stage1_metadata(str(checkpoint))
    decoder._stage1_config_path = None

    with pytest.raises(ValueError, match="bridge_token_axis"):
        decoder._validate_stage1_bridge_config(str(checkpoint))


def test_stage1_load_rejects_structural_missing_and_unexpected_keys(monkeypatch):
    medgemma_decoder = _load_medgemma_decoder(monkeypatch)

    class _Component(nn.Module):
        def load_stage1_checkpoint(self, *args, **kwargs):
            return {
                "missing_keys": ["time_proj.weight", "time_proj.bias"],
                "unexpected_keys": ["mix_proj.weight", "mix_proj.bias"],
                "shape_mismatched_keys": [],
            }

    decoder = medgemma_decoder.MedGemmaDecoder.__new__(
        medgemma_decoder.MedGemmaDecoder
    )
    nn.Module.__init__(decoder)

    with pytest.raises(RuntimeError, match="structural checkpoint mismatch"):
        decoder._load_stage1_weights(
            stage1_component=_Component(),
            stage1_checkpoint_path="unused.pt",
            component_type="bridge",
            component_identifier="bridge",
        )


def test_stage1_load_rejects_checkpoint_with_all_qformer_blocks_stripped(
    monkeypatch, tmp_path
):
    medgemma_decoder = _load_medgemma_decoder(monkeypatch)
    common = dict(
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
        token_axis="channel",
        codebook_dim=D,
    )
    rms_norm = getattr(nn, "RMSNorm", _TestRMSNorm)
    with patch.object(nn, "RMSNorm", rms_norm, create=True):
        source = _BRIDGE_MODULE.ECGQFormerBridgeStage1(
            **common,
            txt_vocab_size=50,
            txt_pad_id=0,
        )
        target = _BRIDGE_MODULE.InstructionAwareECGQFormerBridge(**common)
    stripped_state = {
        key: value
        for key, value in source.state_dict().items()
        if not key.startswith(("blocks.", "stage1_blocks."))
    }
    checkpoint = tmp_path / "stage1_without_qformer_blocks.pt"
    torch.save(
        {
            "config": {
                "bridge_num_layers": 1,
                "bridge_mix_strategy": "concat_linear",
                "bridge_token_axis": "channel",
            },
            "model_state_dict": stripped_state,
        },
        checkpoint,
    )

    decoder = medgemma_decoder.MedGemmaDecoder.__new__(
        medgemma_decoder.MedGemmaDecoder
    )
    nn.Module.__init__(decoder)
    decoder.stage1_metadata = decoder._inspect_stage1_metadata(str(checkpoint))

    with pytest.raises(RuntimeError, match="declares 1 Q-Former blocks but contains 0"):
        decoder._load_stage1_weights(
            stage1_component=target,
            stage1_checkpoint_path=str(checkpoint),
            component_type="bridge",
            component_identifier="bridge",
        )


@pytest.mark.parametrize(
    ("removed_prefix", "expected_message"),
    (
        ("blocks.1.", "declares 2 Q-Former blocks but contains 1"),
        (
            "stage1_blocks.1.",
            "declares 2 instruction Q-Former blocks but contains 1",
        ),
    ),
)
def test_stage1_load_rejects_sidecar_declared_partial_qformer_blocks(
    monkeypatch,
    tmp_path,
    removed_prefix,
    expected_message,
):
    medgemma_decoder = _load_medgemma_decoder(monkeypatch)
    common = dict(
        vocab_size=VOCAB,
        num_codebooks=NCB,
        d_mid=D_MID,
        d_llm=32,
        d_txt=16,
        num_steps=L,
        num_query_tokens=4,
        num_layers=2,
        num_heads=4,
        dropout=0.0,
        mix_strategy="concat_linear",
        token_axis="channel",
        codebook_dim=D,
    )
    rms_norm = getattr(nn, "RMSNorm", _TestRMSNorm)
    with patch.object(nn, "RMSNorm", rms_norm, create=True):
        source = _BRIDGE_MODULE.ECGQFormerBridgeStage1(
            **common,
            txt_vocab_size=50,
            txt_pad_id=0,
        )
        target = _BRIDGE_MODULE.InstructionAwareECGQFormerBridge(**common)
    stripped_state = {
        key: value
        for key, value in source.state_dict().items()
        if not key.startswith(removed_prefix)
    }
    checkpoint = tmp_path / "stage1_partially_stripped.pt"
    torch.save({"model_state_dict": stripped_state}, checkpoint)
    sidecar = tmp_path / "config.yaml"
    sidecar.write_text(
        "bridge_num_layers: 2\n"
        "bridge_mix_strategy: concat_linear\n"
        "bridge_token_axis: channel\n",
        encoding="utf-8",
    )

    decoder = medgemma_decoder.MedGemmaDecoder.__new__(
        medgemma_decoder.MedGemmaDecoder
    )
    nn.Module.__init__(decoder)
    decoder.bridge = target
    decoder.stage1_metadata = decoder._inspect_stage1_metadata(str(checkpoint))
    decoder._stage1_config_path = sidecar
    decoder._validate_stage1_bridge_config(str(checkpoint))

    with pytest.raises(RuntimeError, match=expected_message):
        decoder._load_stage1_weights(
            stage1_component=target,
            stage1_checkpoint_path=str(checkpoint),
            component_type="bridge",
            component_identifier="bridge",
        )
