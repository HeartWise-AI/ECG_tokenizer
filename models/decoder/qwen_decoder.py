"""Qwen decoder for ECG-conditioned causal language modeling."""

from __future__ import annotations

from typing import Optional, Sequence

import torch
from transformers import AutoModelForCausalLM, PreTrainedModel

from utils.enums import ModelName
from utils.registry import ModelRegistry
from .medgemma_decoder import MedGemmaDecoder


def _load_qwen_model(
    model_name: str,
    torch_dtype: Optional[torch.dtype] = None,
) -> PreTrainedModel:
    loaders: Sequence[type[PreTrainedModel]] = (AutoModelForCausalLM,)
    last_error: Exception | None = None
    for attn_impl in ("flash_attention_2", "sdpa", None):
        for loader in loaders:
            try:
                import transformers as _tf

                dtype_key = "dtype" if int(_tf.__version__.split(".")[0]) >= 5 else "torch_dtype"
                kwargs = {
                    dtype_key: torch_dtype,
                    "trust_remote_code": True,
                }
                if attn_impl:
                    kwargs["attn_implementation"] = attn_impl
                model = loader.from_pretrained(model_name, **kwargs)
                if attn_impl:
                    print(f"   Using attention: {attn_impl}")
                return model
            except Exception as exc:  # pragma: no cover - depends on local HF/cache state
                last_error = exc
                continue

    raise RuntimeError(f"Unable to load Qwen model '{model_name}'. Last error: {last_error}.")


@ModelRegistry.register(ModelName.QWEN_DECODER.value)
class QwenDecoder(MedGemmaDecoder):
    """Qwen causal-LM decoder using ECG soft-prefix conditioning.

    This intentionally reuses the MedGemma decoder's ECG bridge and loss plumbing, but disables
    MedGemma-specific token handling. Qwen receives ECG tokens as continuous prefix embeddings.
    """

    def __init__(self, *args, **kwargs) -> None:
        kwargs.setdefault("add_dec_token", False)
        kwargs.setdefault("pass_token_type_ids", False)
        super().__init__(*args, **kwargs)

    def _load_llm_model(
        self,
        model_name: str,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> PreTrainedModel:
        return _load_qwen_model(model_name, torch_dtype=torch_dtype)

    def _start_image_token_id(self) -> Optional[int]:
        return None
