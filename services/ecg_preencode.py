"""ECG signal pre-encoder service for OpenRLHF rollouts.

WHY: OpenRLHF + vLLM expects a standard `AutoModelForImageTextToText`-style
interface. Our ECG_Tokenizer_Wrapper has a custom encoder→quantizer→bridge→LM
stack that produces *soft tokens* injected into MedGemma. vLLM has no concept
of our encoder, so we encode signals BEFORE the rollout phase and pass the
resulting soft tokens as part of `mm_inputs`.

This module exposes:
- `load_encoder(checkpoint_path, device)`: load just the encoder + quantizer + bridge
- `encode_one(model, signal_path)`: read .npy → run encoder → return soft tokens
- `batch_encode(model, paths)`: same, batched

Status: SKELETON. The actual hook into OpenRLHF's SingleTurnAgentExecutor
is TODO — see `services/openrlhf_agent.py` once that's built.
"""

from __future__ import annotations

import importlib
import importlib.util
import sys
import types
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _install_tokenizer_shims() -> None:
    """Same shims as scripts/rlvr_eval_subset.py — required to load old checkpoints."""
    _gemma_mod = importlib.import_module("transformers.models.gemma")
    if not hasattr(_gemma_mod, "tokenization_gemma_fast"):
        shim = types.ModuleType("transformers.models.gemma.tokenization_gemma_fast")
        from transformers import GemmaTokenizer
        shim.GemmaTokenizerFast = GemmaTokenizer
        sys.modules["transformers.models.gemma.tokenization_gemma_fast"] = shim
        _gemma_mod.tokenization_gemma_fast = shim
    import transformers.tokenization_utils as _tok_utils
    if not hasattr(_tok_utils, "Trie"):
        class _Trie:
            def __init__(self, *a, **kw): pass
            def __setstate__(self, state): pass
        _tok_utils.Trie = _Trie


def load_encoder(checkpoint_path: str, device: str = "cuda:0"):
    """Load the ECG_Tokenizer_Wrapper checkpoint. We use the full model
    here for simplicity; the LM head can be discarded after extracting the
    bridge output. Returns (model, tokenizer).
    """
    _install_tokenizer_shims()
    spec = importlib.util.spec_from_file_location(
        "rlvr_eval_subset", str(ROOT / "scripts" / "rlvr_eval_subset.py"))
    eval_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(eval_mod)
    return eval_mod.load_model(checkpoint_path, device)


def load_signal(path: str, target_len: int = 2500, num_leads: int = 12) -> torch.Tensor:
    """Read a .npy ECG file and reshape to (1, num_leads, target_len)."""
    arr = np.load(path)
    if arr.ndim == 1:
        arr = arr.reshape(num_leads, -1)
    if arr.shape[0] != num_leads and arr.shape[-1] == num_leads:
        arr = arr.T
    if arr.shape[-1] > target_len:
        arr = arr[..., :target_len]
    elif arr.shape[-1] < target_len:
        pad = target_len - arr.shape[-1]
        arr = np.pad(arr, ((0, 0), (0, pad)))
    return torch.from_numpy(arr).float().unsqueeze(0)


@torch.no_grad()
def batch_encode(model, signal_paths: List[str], device: str = "cuda:0") -> torch.Tensor:
    """TODO: encode N signals to soft tokens (B, T_soft, D_llm).
    Requires exposing `model.encode_to_soft_tokens()` on ECG_Tokenizer_Wrapper —
    currently the encoder is hidden inside `generate_report_with_question()`.
    Until that hook exists, use generate path with prompt_only=True.
    """
    raise NotImplementedError(
        "batch_encode requires exposing the encoder→bridge output on "
        "ECG_Tokenizer_Wrapper as a public method. See TODO in "
        "models/ecg_tokenizer_wrapper.py.")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--signal_path", required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args()

    print(f"[ecg_preencode] loading {args.checkpoint}")
    model, _tok = load_encoder(args.checkpoint, args.device)
    sig = load_signal(args.signal_path)
    print(f"[ecg_preencode] signal shape: {sig.shape}")
    print("[ecg_preencode] encode hook is TODO — see batch_encode docstring")
