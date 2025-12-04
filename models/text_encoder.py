import warnings
import torch
import torch.nn as nn
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoConfig

from utils.registry import ModelRegistry


def get_tokenizer(model_name: str = "google/medgemma-4b-it"):
    """Return a tokenizer configured for clinical text (supports MedGemma/Gemma)."""
    trust = True if ("gemma" in model_name.lower() or "medgemma" in model_name.lower()) else False
    return AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        model_max_length=1024,
        padding_side="right",
        truncation_side="right",
        trust_remote_code=trust,
    )


def _resolve_hidden_size(cfg: AutoConfig) -> int:
    hidden = getattr(cfg, "hidden_size", None)
    if hidden is None:
        text_cfg = getattr(cfg, "text_config", None)
        if text_cfg is not None:
            hidden = getattr(text_cfg, "hidden_size", None)
    if hidden is None:
        raise ValueError("Unable to resolve hidden_size from provided model config.")
    return int(hidden)


def _load_text_backbone(model_name: str) -> nn.Module:
    """Load a text backbone for pooling embeddings.

    Tries AutoModel (encoder) first. If unavailable, falls back to AutoModelForCausalLM.
    """
    trust = True if ("gemma" in model_name.lower() or "medgemma" in model_name.lower()) else False
    last_err: Exception | None = None
    try:
        return AutoModel.from_pretrained(model_name, trust_remote_code=trust)
    except Exception as e:
        last_err = e
    try:
        return AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=trust)
    except Exception as e:  # pragma: no cover - depends on HF model availability
        last_err = e
    raise RuntimeError(f"Failed to load text backbone for '{model_name}': {last_err}")


@ModelRegistry.register("text_encoder")
class TextEncoder(nn.Module):
    """Generic text encoder wrapper (BERT, Gemma/MedGemma, etc.) with optional partial freezing.

    Exposes `.tokenizer`, `.bert` (backbone), `.resize_token_embeddings`, and returns
    pooled features projected to `output_dim`.
    """

    def __init__(
        self,
        model_name: str = "google/medgemma-4b-it",
        output_dim: int = 512,
        dropout: float = 0.2,
        freeze_ratio: float = 0.5,
    ) -> None:
        super().__init__()
        self.model_name = model_name
        self.output_dim = int(output_dim)
        self.freeze_ratio = float(freeze_ratio)
        self.dropout = float(dropout)

        self.tokenizer = get_tokenizer(model_name)

        self.bert = _load_text_backbone(model_name)
        try:
            hidden_size = _resolve_hidden_size(self.bert.config)  # type: ignore[arg-type]
        except Exception:
            # Fallback to common attribute
            hidden_size = getattr(self.bert.config, "hidden_size", None) or 768
            hidden_size = int(hidden_size)

        if hasattr(self.bert, "pooler"):
            self.bert.pooler = None

        # Ensure tokenizer has a pad token; prefer EOS when available.
        try:
            if getattr(self.tokenizer, "pad_token_id", None) is None:
                eos_tok = getattr(self.tokenizer, "eos_token", None)
                if eos_tok:
                    self.tokenizer.pad_token = eos_tok
                else:
                    # Add a dedicated [PAD] token
                    self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
                # Resize embeddings to accommodate potential new token
                resize_fn = getattr(self.bert, "resize_token_embeddings", None)
                if callable(resize_fn):
                    resize_fn(len(self.tokenizer))
        except Exception:
            # Non-fatal; runner will check again and error with context if needed
            pass

        self._freeze_partial_bert()

        self.proj = nn.Sequential(
            nn.Dropout(self.dropout),
            nn.Linear(hidden_size, self.output_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
        )

    def _freeze_partial_bert(self) -> None:
        """Freeze a fraction of the backbone starting from the bottom layers."""
        if self.freeze_ratio <= 0.0:
            return

        all_named_params = list(self.bert.named_parameters())
        total_count = len(all_named_params)
        train_count = int(max(0.0, min(1.0, self.freeze_ratio)) * total_count)

        for idx, (_, param) in enumerate(all_named_params):
            if idx < (total_count - train_count):
                param.requires_grad = False

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # Obtain contextualized token representations robustly across model types
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden = None
        if hasattr(outputs, "last_hidden_state") and getattr(outputs, "last_hidden_state") is not None:
            hidden = outputs.last_hidden_state
        elif hasattr(outputs, "hidden_states") and outputs.hidden_states is not None:
            hidden = outputs.hidden_states[-1]
        else:
            # Final fallback: embed tokens only
            try:
                embed_layer = self.bert.get_input_embeddings()
                hidden = embed_layer(input_ids)
            except Exception as e:  # pragma: no cover - defensive fallback
                raise RuntimeError(f"TextEncoder could not obtain hidden states: {e}")

        # Masked mean pooling (robust across tokenizers)
        if attention_mask is None:
            pooled = hidden[:, 0]
        else:
            mask = attention_mask.to(dtype=hidden.dtype).unsqueeze(-1)
            denom = mask.sum(dim=1).clamp_min(1.0)
            pooled = (hidden * mask).sum(dim=1) / denom

        features = self.proj(pooled)
        return features

    def resize_token_embeddings(self, new_size: int) -> None:
        self.bert.resize_token_embeddings(new_size)
