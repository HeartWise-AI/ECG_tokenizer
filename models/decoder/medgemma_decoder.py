"""MedGemma decoder that mirrors the LLaMA ECG token integration path."""

from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForImageTextToText, AutoTokenizer, PreTrainedModel
from transformers.generation.logits_process import LogitsProcessor
import math

from models.bridge.bridge import (
    ECGCodeBridge,
    ECGProjectionBridge,
    PerceiverProjectionBridge,
    ECGQFormerBridge,
    InstructionAwareECGQFormerBridge,
)
from models.bridge import SequenceTokenBridge, SimpleTokenBridge, CrossModalSequenceTokenBridge
from utils.enums import BridgeName, ModelName
from utils.registry import ModelRegistry
from utils.constants import ECG_PATTERNS
from utils.files_handler import load_yaml

logger = logging.getLogger(__name__)


def _coerce_id(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return _coerce_id(value[0])
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _load_medgemma_model(
    model_name: str,
    torch_dtype: Optional[torch.dtype] = None,
) -> PreTrainedModel:
    loaders: Sequence[type[PreTrainedModel]] = (
        AutoModelForImageTextToText,
    )
    last_error: Exception | None = None
    # Try flash_attention_2 first, fall back to default if not available
    for attn_impl in ("flash_attention_2", "sdpa", None):
        for loader in loaders:
            try:
                # transformers >=5.x renamed torch_dtype -> dtype
                import transformers as _tf
                _dtype_key = "dtype" if int(_tf.__version__.split(".")[0]) >= 5 else "torch_dtype"
                kwargs = {
                    _dtype_key: torch_dtype,
                    "trust_remote_code": True,
                }
                if attn_impl:
                    kwargs["attn_implementation"] = attn_impl
                model = loader.from_pretrained(model_name, **kwargs)
                if attn_impl:
                    print(f"   Using attention: {attn_impl}")
                return model
            except Exception as exc:  # pragma: no cover - depends on HF availability
                last_error = exc
                continue
    hint = ""
    if last_error is not None and "does not recognize this architecture" in str(last_error):
        hint = (
            " Transformers >= 4.50.0 is required for Gemma 3 / MedGemma checkpoints. "
            "Please upgrade the transformers package (e.g. pip install -U transformers)."
        )
    raise RuntimeError(
        f"Unable to load MedGemma model '{model_name}'. Last error: {last_error}.{hint}"
    )


@ModelRegistry.register(ModelName.MEDGEMMA_DECODER.value)
class MedGemmaDecoder(nn.Module):
    """Finetuning decoder that injects ECG tokens into the MedGemma text stream."""

    def _load_llm_model(
        self,
        model_name: str,
        torch_dtype: Optional[torch.dtype] = None,
    ) -> PreTrainedModel:
        return _load_medgemma_model(model_name, torch_dtype=torch_dtype)

    def __init__(
        self,
        huggingface_model_name: str = "google/medgemma-4b-it",
        llm_input_embedding_size: int = 4096,
        quantized_feature_shape: Tuple[int, int] = (128, 82),
        bridge_name: Union[BridgeName, str] = BridgeName.LLAMA32_ECG_PROJECTION_BRIDGE,
        adapter_dropout: float = 0.1,
        quantizer: Optional[nn.Module] = None,
        tokenizer: Optional[Any] = None,
        ecg_codebook_size: int = 192,
        num_visual_tokens: Optional[int] = None,
        bridge_mid_dim: int = 512,
        bridge_num_heads: int = 8,
        bridge_dropout: float = 0.1,
        bridge_num_special_tokens: int = 4,
        num_quantizers: int = 1,
        ecg_token_start_id: Optional[int] = None,
        torch_dtype: Optional[torch.dtype] = torch.bfloat16,
        default_generation_kwargs: Optional[Dict[str, Any]] = None,
        prefix_tuning: bool = False,
        label_ignore_index: int = -100,
        stage1_checkpoint_path: Optional[str] = None,
        **unused_kwargs: Any,
    ) -> None:
        super().__init__()

        self.quantizer = quantizer
        self.prefix_tuning = prefix_tuning
        self.label_ignore_index = int(label_ignore_index)
        self.stage1_checkpoint_path = stage1_checkpoint_path
        self._stage1_checkpoint_used = False
        self.stage1_metadata: Dict[str, Any] = {}
        self._stage1_config_path: Optional[Path] = None
        self.debug_ecg_injection = bool(unused_kwargs.pop("debug_ecg_injection", False))
        self.add_dec_token = bool(unused_kwargs.pop("add_dec_token", True))
        self.pass_token_type_ids = bool(unused_kwargs.pop("pass_token_type_ids", True))
        # P1: continuous-feature Perceiver path (built after the main bridge, below).
        self.use_continuous_features = bool(unused_kwargs.pop("use_continuous_features", False))
        self.continuous_num_tokens = int(unused_kwargs.pop("continuous_num_tokens", 32))
        self.continuous_num_heads = int(unused_kwargs.pop("continuous_num_heads", 8))
        # Phase B: when True, skip the discrete code/Q-Former path entirely and emit ONLY the
        # continuous Perceiver tokens (used when the encoder no longer matches the VQ codebook).
        self.continuous_only = bool(unused_kwargs.pop("continuous_only", False))
        self._debug_ecg_injection_logged = False
        if stage1_checkpoint_path:
            self._stage1_config_path = self._infer_stage1_config_path(stage1_checkpoint_path)
            self.stage1_metadata = self._inspect_stage1_metadata(stage1_checkpoint_path)
            if self.stage1_metadata:
                print(
                    f"[Stage1] checkpoint metadata: bridge_tokens={self.stage1_metadata.get('bridge_num_tokens', 'unknown')}, "
                    f"token_dim={self.stage1_metadata.get('bridge_token_dim', 'unknown')}, "
                    f"prefix='{self.stage1_metadata.get('bridge_prefix', 'n/a')}'"
                )

        requested_embedding_size = int(llm_input_embedding_size)
        resolved_embedding_size = requested_embedding_size
        model_hidden_size: Optional[int] = None
        try:
            hf_config = AutoConfig.from_pretrained(
                huggingface_model_name,
                trust_remote_code=True,
            )
        except Exception:
            hf_config = None
        if hf_config is not None:
            model_hidden_size = getattr(hf_config, "hidden_size", None)
            if model_hidden_size is None:
                text_cfg = getattr(hf_config, "text_config", None)
                if text_cfg is not None:
                    model_hidden_size = getattr(text_cfg, "hidden_size", None)
            if model_hidden_size is not None:
                resolved_embedding_size = int(model_hidden_size)
                if resolved_embedding_size != requested_embedding_size:
                    warnings.warn(
                        f"Overriding llm_input_embedding_size={requested_embedding_size} with MedGemma hidden_size={resolved_embedding_size}"
                    )

        self.llm_input_embedding_size = resolved_embedding_size
        llm_input_embedding_size = self.llm_input_embedding_size
        self.qformer_text_projection: Optional[nn.Linear] = None
        self.qformer_text_norm: Optional[nn.LayerNorm] = None
        self.qformer_instruction_dropout: Optional[nn.Dropout] = None
        self.qformer_hidden_size: Optional[int] = None
        self.qformer_text_output_size: Optional[int] = None
        self.bridge_load_info: Optional[Dict[str, list[str]]] = None
        self.pattern_label_count = max(0, int(unused_kwargs.pop("pattern_label_count", len(ECG_PATTERNS))))
        self.pattern_loss_weight = float(unused_kwargs.pop("pattern_loss_weight", 0.3))
        if self.pattern_loss_weight < 0:
            self.pattern_loss_weight = 0.0
        # LVEF soft-decoding loss configuration
        self.lvef_loss_weight = float(unused_kwargs.pop("lvef_loss_weight", 0.0))
        # Auxiliary scalar discrimination heads on the bridge pooled output (0 = disabled)
        self.lvef_head_loss_weight = max(0.0, float(unused_kwargs.pop("lvef_head_loss_weight", 0.0) or 0.0))
        self.shd_head_loss_weight = max(0.0, float(unused_kwargs.pop("shd_head_loss_weight", 0.0) or 0.0))
        self.afib_head_loss_weight = max(0.0, float(unused_kwargs.pop("afib_head_loss_weight", 0.0) or 0.0))
        self.aux_endpoint_specs: Dict[str, Dict[str, Any]] = {}
        self._digit_token_ids: Optional[list[int]] = None
        self._pct_token_id: Optional[int] = None
        # MedGemma's HF generate() misbehaves for batch>1 when using inputs_embeds; default to micro-batch=1
        # for the generation step (encoding stays batched). Can be overridden via config kwarg.
        self.generation_microbatch_size = int(unused_kwargs.pop("generation_microbatch_size", 1))
        self.pattern_classifier: Optional[nn.Module] = None
        self.pattern_loss_fn: Optional[nn.Module] = None
        pattern_pos_weight_cfg = unused_kwargs.pop("pattern_bce_pos_weight", None)
        pos_weight_tensor: Optional[torch.Tensor] = None
        if pattern_pos_weight_cfg is not None:
            if isinstance(pattern_pos_weight_cfg, (int, float)):
                pos_weight_tensor = torch.tensor([float(pattern_pos_weight_cfg)], dtype=torch.float32)
            elif isinstance(pattern_pos_weight_cfg, (list, tuple)):
                pos_weight_tensor = torch.tensor([float(x) for x in pattern_pos_weight_cfg], dtype=torch.float32)
            else:
                raise TypeError(
                    "pattern_bce_pos_weight must be a float or a sequence of floats"
                )
            if self.pattern_label_count > 0:
                if pos_weight_tensor.numel() == 1 and self.pattern_label_count > 1:
                    pos_weight_tensor = pos_weight_tensor.repeat(self.pattern_label_count)
                elif pos_weight_tensor.numel() not in (1, self.pattern_label_count):
                    raise ValueError(
                        f"pattern_bce_pos_weight expects 1 or {self.pattern_label_count} values; "
                        f"received {pos_weight_tensor.numel()}"
                    )
            else:
                warnings.warn(
                    "pattern_bce_pos_weight was provided but pattern_label_count is 0; ignoring pos_weight."
                )
                pos_weight_tensor = None
        if pos_weight_tensor is not None:
            self.register_buffer("_pattern_pos_weight", pos_weight_tensor, persistent=False)
        else:
            self.register_buffer("_pattern_pos_weight", torch.tensor([], dtype=torch.float32), persistent=False)
        self._binary_allowed_token_ids: tuple[int, ...] = tuple()
        self._binary_bad_words_cache: Optional[list[list[int]]] = None

        if isinstance(bridge_name, str):
            try:
                bridge_name = BridgeName(bridge_name)
            except ValueError:
                pass
        self.bridge_name = bridge_name

        bridge_name_str = bridge_name.value if hasattr(bridge_name, "value") else str(bridge_name)

        self.bridge: Optional[Union[
            ECGCodeBridge,
            ECGProjectionBridge,
            PerceiverProjectionBridge,
            SequenceTokenBridge,
            SimpleTokenBridge,
            CrossModalSequenceTokenBridge,
        ]] = None
        self.bridge_config: Optional[Dict[str, Any]] = None

        visual_tokens = num_visual_tokens if num_visual_tokens is not None else quantized_feature_shape[0]
        stage1_bridge_tokens = int(self.stage1_metadata.get("bridge_num_tokens", 0))
        if stage1_bridge_tokens > 0 and stage1_bridge_tokens != visual_tokens:
            print(
                f"[Stage1] overriding bridge_num_visual_tokens: checkpoint expects {stage1_bridge_tokens}, config provided {visual_tokens}"
            )
            visual_tokens = stage1_bridge_tokens

        qformer_layers = int(unused_kwargs.pop("bridge_qformer_layers", 6))
        qformer_text_hidden = int(unused_kwargs.pop("bridge_text_hidden_size", bridge_mid_dim))
        qformer_bias_last = float(unused_kwargs.pop("bridge_bias_last_codebook", 0.5))
        qformer_codebook_dropout = float(unused_kwargs.pop("bridge_codebook_dropout", 0.0))
        qformer_mix_strategy = str(unused_kwargs.pop("bridge_mix_strategy", "softmax") or "softmax")
        qformer_token_axis = str(unused_kwargs.pop("bridge_token_axis", "channel") or "channel")
        qformer_cross_every = int(unused_kwargs.pop("bridge_cross_every", 2))
        instruction_dropout = float(unused_kwargs.pop("instruction_dropout", 0.0))

        code_bridge_aliases = {
            BridgeName.LLAMA32_ECG_CODE_BRIDGE,
            "Llama32_ECGCodeBridge",
            "ECGCodeBridge",
        }
        projection_bridge_aliases = {
            BridgeName.LLAMA32_ECG_PROJECTION_BRIDGE,
            "Llama32_ECGProjectionBridge",
            "ECGProjectionBridge",
        }
        perceiver_bridge_aliases = {
            BridgeName.ECG_PERCEIVER_BRIDGE,
            "ECGPerceiverBridge",
            "PerceiverProjectionBridge",
        }
        instruction_qformer_aliases = {
            BridgeName.ECG_STAGE1_QFORMER_BRIDGE,
            BridgeName.INSTRUCTION_AWARE_ECG_QFORMER_BRIDGE,
            "ECGQFormerBridgeStage1",
            "InstructionAwareECGQFormerBridge",
        }
        qformer_bridge_aliases = {
            BridgeName.LLAMA32_ECG_QFORMER_BRIDGE,
            "Llama32_ECGQFormerBridge",
            "ECGQFormerBridge",
        }
        sequence_token_aliases = {
            BridgeName.LLAMA32_SEQUENCE_TOKEN_BRIDGE,
            "SequenceTokenBridge",
        }
        simple_token_aliases = {
            BridgeName.LLAMA32_SIMPLE_TOKEN_BRIDGE,
            "SimpleTokenBridge",
        }
        cross_modal_aliases = {
            "CrossModalSequenceTokenBridge",
        }

        kept = unused_kwargs.pop('num_codebooks_kept', None)
        offset_raw = unused_kwargs.pop('codebook_offset', 0)

        total_codebooks = max(1, int(num_quantizers))
        requested_keep = int(kept) if kept is not None else total_codebooks
        if requested_keep <= 0 or requested_keep > total_codebooks:
            requested_keep = total_codebooks

        offset = int(offset_raw or 0)
        if requested_keep >= total_codebooks:
            resolved_offset = 0
        else:
            if offset < 0:
                resolved_offset = max(total_codebooks - requested_keep, 0)
            else:
                resolved_offset = max(0, min(offset, total_codebooks - requested_keep))
        self.num_codebooks_kept = requested_keep
        self.codebook_offset = resolved_offset

        candidates = (bridge_name, bridge_name_str)

        def _matches(alias_set: set[Any]) -> bool:
            return any(candidate in alias_set for candidate in candidates)

        if _matches(code_bridge_aliases):
            self.bridge = ECGCodeBridge(
                vocab_size=ecg_codebook_size,
                d_mid=bridge_mid_dim,
                d_model=llm_input_embedding_size,
                num_output_tokens=visual_tokens,
                num_heads=bridge_num_heads,
                num_special_tokens=bridge_num_special_tokens,
                dropout=bridge_dropout,
                num_codebooks=requested_keep,
                codebook_offset=resolved_offset,
                original_num_codebooks=total_codebooks,
            )
            self.bridge_config = {
                "style": "code",
                "bridge_name": bridge_name_str,
                "vocab_size": ecg_codebook_size,
                "mid_dim": bridge_mid_dim,
                "output_tokens": visual_tokens,
                "num_heads": bridge_num_heads,
                "dropout": bridge_dropout,
                "num_special_tokens": bridge_num_special_tokens,
            }
        elif _matches(projection_bridge_aliases):
            feature_dim = quantized_feature_shape[1] if len(quantized_feature_shape) > 1 else llm_input_embedding_size
            # Optional projection-bridge kwargs surfaced from config
            proj_kwargs: Dict[str, Any] = {
                'input_dim': feature_dim,
                'd_model': llm_input_embedding_size,
                'num_tokens': visual_tokens,
                'dropout': bridge_dropout,
            }
            # Positional encoding flexibility
            use_sinusoidal = unused_kwargs.pop('bridge_use_sinusoidal_pos_emb', None)
            max_pos = unused_kwargs.pop('bridge_pos_embedding_max_len', None)
            if use_sinusoidal is not None:
                proj_kwargs['use_sinusoidal_pos_emb'] = bool(use_sinusoidal)
            if max_pos is not None:
                try:
                    proj_kwargs['pos_embedding_max_len'] = int(max_pos)
                except Exception:
                    pass
            # Optional knobs
            softmax_temp = unused_kwargs.pop('bridge_softmax_temp', None)
            mix_residual = unused_kwargs.pop('bridge_mix_residual', None)
            add_modality_embed = unused_kwargs.pop('bridge_add_modality_embed', None)
            add_cls_token = unused_kwargs.pop('bridge_add_cls_token', None)
            if softmax_temp is not None:
                try:
                    proj_kwargs['softmax_temp'] = float(softmax_temp)
                except Exception:
                    pass
            if mix_residual is not None:
                try:
                    proj_kwargs['mix_residual'] = float(mix_residual)
                except Exception:
                    pass
            if add_modality_embed is not None:
                proj_kwargs['add_modality_embed'] = bool(add_modality_embed)
            if add_cls_token is not None:
                proj_kwargs['add_cls_token'] = bool(add_cls_token)

            self.bridge = ECGProjectionBridge(**proj_kwargs)
            self.bridge_config = {
                "style": "projection",
                "bridge_name": bridge_name_str,
                "input_dim": feature_dim,
                "d_model": llm_input_embedding_size,
                "num_tokens": visual_tokens,
                "dropout": bridge_dropout,
            }
        elif _matches(perceiver_bridge_aliases):
            feature_dim = quantized_feature_shape[1] if len(quantized_feature_shape) > 1 else llm_input_embedding_size
            self.bridge = PerceiverProjectionBridge(
                input_dim=feature_dim,
                d_model=llm_input_embedding_size,
                num_output_tokens=visual_tokens,
                num_heads=bridge_num_heads,
                dropout=bridge_dropout,
            )
            self.bridge_config = {
                "style": "perceiver",
                "bridge_name": bridge_name_str,
                "input_dim": feature_dim,
                "d_model": llm_input_embedding_size,
                "output_tokens": visual_tokens,
                "num_heads": bridge_num_heads,
                "dropout": bridge_dropout,
            }
        elif _matches(instruction_qformer_aliases):
            num_steps = quantized_feature_shape[0] if len(quantized_feature_shape) > 0 else visual_tokens
            self.bridge = InstructionAwareECGQFormerBridge(
                vocab_size=ecg_codebook_size,
                num_codebooks=requested_keep,
                d_mid=bridge_mid_dim,
                d_llm=llm_input_embedding_size,
                d_txt=qformer_text_hidden,
                num_steps=num_steps,
                num_query_tokens=visual_tokens,
                num_layers=qformer_layers,
                num_heads=bridge_num_heads,
                dropout=bridge_dropout,
                num_special_tokens=bridge_num_special_tokens,
                bias_last_codebook=qformer_bias_last,
                codebook_dropout=qformer_codebook_dropout,
                mix_strategy=qformer_mix_strategy,
                token_axis=qformer_token_axis,
                cross_every=qformer_cross_every,
            )
            self.bridge_config = {
                "style": "instruction_qformer",
                "bridge_name": bridge_name_str,
                "vocab_size": ecg_codebook_size,
                "mid_dim": bridge_mid_dim,
                "output_tokens": visual_tokens,
                "num_heads": bridge_num_heads,
                "dropout": bridge_dropout,
                "num_layers": qformer_layers,
                "text_hidden_size": qformer_text_hidden,
                "bias_last_codebook": qformer_bias_last,
                "codebook_dropout": qformer_codebook_dropout,
                "mix_strategy": qformer_mix_strategy,
                "token_axis": qformer_token_axis,
                "cross_every": qformer_cross_every,
            }
        elif _matches(qformer_bridge_aliases):
            num_steps = quantized_feature_shape[0] if len(quantized_feature_shape) > 0 else visual_tokens
            self.bridge = ECGQFormerBridge(
                vocab_size=ecg_codebook_size,
                num_codebooks=requested_keep,
                d_mid=bridge_mid_dim,
                d_llm=llm_input_embedding_size,
                d_txt=qformer_text_hidden,
                num_steps=num_steps,
                num_query_tokens=visual_tokens,
                num_layers=qformer_layers,
                num_heads=bridge_num_heads,
                dropout=bridge_dropout,
                num_special_tokens=bridge_num_special_tokens,
                bias_last_codebook=qformer_bias_last,
                codebook_dropout=qformer_codebook_dropout,
                mix_strategy=qformer_mix_strategy,
                token_axis=qformer_token_axis,
            )
            self.bridge_config = {
                "style": "qformer",
                "bridge_name": bridge_name_str,
                "vocab_size": ecg_codebook_size,
                "mid_dim": bridge_mid_dim,
                "output_tokens": visual_tokens,
                "num_heads": bridge_num_heads,
                "dropout": bridge_dropout,
                "num_layers": qformer_layers,
                "text_hidden_size": qformer_text_hidden,
                "bias_last_codebook": qformer_bias_last,
                "codebook_dropout": qformer_codebook_dropout,
                "mix_strategy": qformer_mix_strategy,
                "token_axis": qformer_token_axis,
            }
        elif _matches(sequence_token_aliases):
            self.bridge = SequenceTokenBridge(
                input_shape=quantized_feature_shape,
                output_size=llm_input_embedding_size,
                dropout=bridge_dropout,
            )
            self.bridge_config = {
                "style": "sequence_token",
                "input_shape": quantized_feature_shape,
                "output_size": llm_input_embedding_size,
                "dropout": bridge_dropout,
            }
        elif _matches(simple_token_aliases):
            self.bridge = SimpleTokenBridge(
                input_shape=quantized_feature_shape,
                output_size=llm_input_embedding_size,
                dropout=bridge_dropout,
            )
            self.bridge_config = {
                "style": "simple_sequence_token",
                "input_shape": quantized_feature_shape,
                "output_size": llm_input_embedding_size,
                "dropout": bridge_dropout,
            }
        elif _matches(cross_modal_aliases):
            self.bridge = CrossModalSequenceTokenBridge(
                input_shape=quantized_feature_shape,
                output_size=llm_input_embedding_size,
                dropout=bridge_dropout,
                num_attention_heads=bridge_num_heads,
            )
            self.bridge_config = {
                "style": "cross_modal_sequence_token",
                "input_shape": quantized_feature_shape,
                "output_size": llm_input_embedding_size,
                "dropout": bridge_dropout,
                "num_heads": bridge_num_heads,
            }
        else:
            self.adapter_class: ModelClassT = ModelRegistry.get(bridge_name)
            if self.adapter_class is None:
                raise ValueError(f"Adapter {bridge_name} not found in ModelRegistry")

            adapter_ctor = cast(Any, self.adapter_class)
            adapter_kwargs = {
                'input_shape': quantized_feature_shape,
                'output_size': llm_input_embedding_size,
                'dropout': adapter_dropout
            }

            if 'SequenceToken' in bridge_name_str:
                adapter_kwargs.update({
                    'use_cross_attention': getattr(self, 'use_cross_attention', True),
                    'num_attention_heads': getattr(self, 'num_attention_heads', bridge_num_heads),
                    'intermediate_dim': getattr(self, 'intermediate_dim', None)
                })

            self.adapter = adapter_ctor(**adapter_kwargs)

            num_ecg_tokens_raw = getattr(self.adapter, 'num_tokens', 1)
            try:
                num_ecg_tokens = int(num_ecg_tokens_raw)
            except (TypeError, ValueError):
                num_ecg_tokens = quantized_feature_shape[0]
            self.num_ecg_tokens = num_ecg_tokens
            self.bridge_config = {
                "style": "registry_adapter",
                "adapter_name": bridge_name_str,
                "input_shape": quantized_feature_shape,
                "output_size": llm_input_embedding_size,
                "dropout": adapter_dropout,
            }

        if self.bridge is None:
            if self.adapter is None:
                raise ValueError(
                    "MedGemmaDecoder currently requires an ECG bridge (projection or code)."
                )
            self.num_ecg_tokens = getattr(self.adapter, "num_tokens", visual_tokens)
        else:
            self.num_ecg_tokens = getattr(self.bridge, "num_tokens", visual_tokens)

        # P1: optional parallel continuous-feature (pre-quantization) Perceiver path.
        # Resamples the pre-quant encoder output [B, num_steps, feat_dim] into
        # `continuous_num_tokens` soft tokens that are concatenated with the code-bridge
        # tokens in `_compute_ecg_embeddings`. Trained fresh (random init); kept separate
        # from `self.bridge` so the Stage-1 warmstart does not touch it.
        self.continuous_bridge: Optional[nn.Module] = None
        self._continuous_features: Optional[torch.Tensor] = None
        if self.use_continuous_features:
            cont_feat_dim = (
                quantized_feature_shape[1]
                if len(quantized_feature_shape) > 1
                else llm_input_embedding_size
            )
            self.continuous_bridge = PerceiverProjectionBridge(
                input_dim=cont_feat_dim,
                d_model=llm_input_embedding_size,
                num_output_tokens=self.continuous_num_tokens,
                num_heads=self.continuous_num_heads,
                dropout=bridge_dropout,
            )
            self.num_ecg_tokens = int(self.num_ecg_tokens) + int(self.continuous_num_tokens)
            print(
                f"[P1] continuous Perceiver bridge enabled: input_dim={cont_feat_dim} "
                f"+{self.continuous_num_tokens} tokens -> total ecg tokens={self.num_ecg_tokens}"
            )

        stage1_component_identifier = "decoder"
        stage1_component_type = "decoder"
        stage1_component = None
        if self.bridge is not None:
            stage1_component = self.bridge
            stage1_component_type = "bridge"
            stage1_component_identifier = f"bridge '{bridge_name_str}'"
        elif getattr(self, "adapter", None) is not None:
            stage1_component = self.adapter
            stage1_component_type = "adapter"
            adapter_name = type(self.adapter).__name__
            stage1_component_identifier = f"adapter '{adapter_name}' (config key '{bridge_name_str}')"

        if stage1_checkpoint_path and stage1_component is not None:
            # Fail fast if the bridge config disagrees with the Stage-1 config/metadata.
            self._validate_stage1_bridge_config(stage1_checkpoint_path)
            stage1_loaded = self._load_stage1_weights(
                stage1_component=stage1_component,
                stage1_checkpoint_path=stage1_checkpoint_path,
                component_type=stage1_component_type,
                component_identifier=stage1_component_identifier,
            )
            if stage1_loaded:
                self._stage1_checkpoint_used = True

        if stage1_checkpoint_path:
            if self._stage1_checkpoint_used:
                print(f"[Stage1] checkpoint loaded for {stage1_component_identifier}: {stage1_checkpoint_path}")
            else:
                print(
                    f"[Stage1] checkpoint path provided ({stage1_checkpoint_path}) but {stage1_component_identifier} did not load it; continuing without Stage-1 weights."
                )
        else:
            print("[Stage1] checkpoint path not provided; proceeding without Stage-1 weights.")

        if hasattr(self.bridge, "forward_instruction_hidden"):
            hidden_size = int(self.bridge.queries.size(-1))
            self.qformer_hidden_size = hidden_size
            self.qformer_text_projection = nn.Linear(llm_input_embedding_size, hidden_size)
            self.qformer_text_norm = nn.LayerNorm(hidden_size)
            if instruction_dropout > 0:
                self.qformer_instruction_dropout = nn.Dropout(instruction_dropout)
            self.qformer_text_output_size = int(
                getattr(getattr(self.bridge, "to_txt", None), "out_features", hidden_size)
            )
        else:
            self.qformer_hidden_size = bridge_mid_dim
            self.qformer_text_output_size = self.qformer_hidden_size

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(huggingface_model_name)
        
        # Register [DEC] as special token for MedGemma. Generic causal-LM decoders can disable this.
        if self.add_dec_token and "[DEC]" not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["[DEC]"]})

        candidate_pieces = [
            "Yes",
            "No",
            "yes",
            "no",
            "▁Yes",
            "▁No",
        ]
        candidate_texts = [
            "Yes",
            "No",
            " yes",
            " no",
            "Yes.",
            "No.",
        ]
        allowed_ids_set: set[int] = set()
        for piece in candidate_pieces:
            token_id = self.tokenizer.convert_tokens_to_ids(piece)
            if token_id is None:
                continue
            try:
                token_int = int(token_id)
            except (TypeError, ValueError):
                continue
            if token_int >= 0:
                allowed_ids_set.add(token_int)
        for text in candidate_texts:
            try:
                encoded = self.tokenizer.encode(text, add_special_tokens=False)
            except Exception:
                encoded = []
            for token_id in encoded:
                if isinstance(token_id, int) and token_id >= 0:
                    allowed_ids_set.add(int(token_id))
        self._binary_allowed_token_ids = tuple(sorted(allowed_ids_set))
        self._binary_bad_words_cache = None

        # Avoid expanding LLM vocab with ECG position tokens; rely on soft prompts or cross-attn
        # Q-Former path and projection bridges use continuous ECG embeddings prepended to text embeddings
        self.ecg_token_start_id = None

        self.llm_model: PreTrainedModel = self._load_llm_model(
            huggingface_model_name,
            torch_dtype=torch_dtype,
        )

        hidden_size = getattr(self.llm_model.config, 'hidden_size', None)
        if hidden_size is None:
            text_cfg = getattr(self.llm_model.config, 'text_config', None)
            if text_cfg is not None:
                hidden_size = getattr(text_cfg, 'hidden_size', None)
        if hidden_size is None:
            raise ValueError("MedGemma configuration does not expose hidden_size")
        if llm_input_embedding_size != hidden_size:
            warnings.warn(
                f"llm_input_embedding_size={llm_input_embedding_size} does not match MedGemma hidden_size={hidden_size}; using model hidden_size"
            )
            self.llm_input_embedding_size = int(hidden_size)
            llm_input_embedding_size = self.llm_input_embedding_size

        self.llm = self.llm_model
        self.llm_model.resize_token_embeddings(len(self.tokenizer), mean_resizing=True)  # type: ignore[arg-type]
        
        # Clear any cached token IDs since vocab size changed
        self._start_image_id_cache = None

        self._ecg_embedding_hook_handle = None
        if not self.prefix_tuning and self.ecg_token_start_id is not None:
            self._initialize_ecg_tokens_semantically(self.ecg_token_start_id, self.num_ecg_tokens)
            mask = torch.zeros(len(self.tokenizer), dtype=torch.bool)
            mask[self.ecg_token_start_id:self.ecg_token_start_id + self.num_ecg_tokens] = True
            self.register_buffer("_ecg_embedding_train_mask", mask, persistent=False)
            self._apply_ecg_embedding_mask()

            print(
                f"   ECG token ID range: [{self.ecg_token_start_id}, "
                f"{self.ecg_token_start_id + self.num_ecg_tokens - 1}]"
            )
            print(f"   New vocabulary size: {len(self.llm_model.get_input_embeddings().weight)}")
        else:
            self._ecg_embedding_train_mask = None  # type: ignore[assignment]

        pad_id = _coerce_id(getattr(self.llm_model.config, "pad_token_id", None))
        eos_id = _coerce_id(getattr(self.llm_model.config, "eos_token_id", None))
        if pad_id is None:
            pad_id = eos_id if eos_id is not None else 0
        if eos_id is None:
            eos_id = pad_id
        self.pad_token_id = int(pad_id)
        self.eos_token_id = int(eos_id)
        # Prefer tokenizer-provided eos when available (more robust across configs)
        tok_eos = getattr(self.tokenizer, "eos_token_id", None)
        if tok_eos is not None:
            try:
                self.eos_token_id = int(tok_eos)
            except (TypeError, ValueError):
                pass

        base_defaults: Dict[str, Any] = {
            "do_sample": False,
            "temperature": 0.0,
            "top_p": 1.0,
            "max_new_tokens": 64,
            "min_new_tokens": 0,
            "repetition_penalty": 1.1,
            "no_repeat_ngram_size": 5,
        }
        if default_generation_kwargs:
            base_defaults.update(default_generation_kwargs)
        
        # Determine end-of-turn token ID (MedGemma uses <end_of_turn>, LLaMA uses <|eot_id|>)
        self._eot_token_id = None
        self._eos_token_ids: list[int] = []
        
        # Always include the tokenizer's EOS
        if self.eos_token_id is not None:
            self._eos_token_ids.append(int(self.eos_token_id))
        
        # Try MedGemma's <end_of_turn> first
        end_of_turn_id = self.tokenizer.convert_tokens_to_ids("<end_of_turn>")
        if isinstance(end_of_turn_id, int) and end_of_turn_id > 0:
            self._eot_token_id = end_of_turn_id
            if end_of_turn_id not in self._eos_token_ids:
                self._eos_token_ids.append(end_of_turn_id)
        
        # Fallback to LLaMA's <|eot_id|>
        if self._eot_token_id is None:
            eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if isinstance(eot_id, int) and eot_id > 0:
                self._eot_token_id = eot_id
                if eot_id not in self._eos_token_ids:
                    self._eos_token_ids.append(eot_id)

                    
        if not self.prefix_tuning and self.ecg_token_start_id is not None:
            self.bad_ecg_token_ids: Optional[list[list[int]]] = [[tid] for tid in range(
                self.ecg_token_start_id,
                self.ecg_token_start_id + self.num_ecg_tokens,
            )]
        else:
            self.bad_ecg_token_ids = None
        self.default_generation_params = base_defaults
        # Build bad_words_ids with ECG placeholders (if present) and sentinel markers
        bad_words: list[list[int]] = []
        if self.bad_ecg_token_ids:
            bad_words.extend(self.bad_ecg_token_ids)
        for tok in ("<|start_ecg|>", "<|end_ecg|>"):
            try:
                tid = self.tokenizer.convert_tokens_to_ids(tok)
            except Exception:
                tid = None
            if tid is not None:
                try:
                    tid = int(tid)
                    if tid >= 0:
                        bad_words.append([tid])
                except Exception:
                    pass
        if bad_words:
            self.default_generation_params["bad_words_ids"] = bad_words
        self._initialize_pattern_head()
        self._initialize_aux_endpoint_heads()

    def _sanitize_generate_args(self, args: Dict[str, Any]) -> Dict[str, Any]:
        """Remove sampling-only knobs when sampling is disabled to avoid HF warnings."""
        if not bool(args.get("do_sample", False)):
            for key in ("temperature", "top_k", "top_p"):
                args.pop(key, None)
        return args

    # ------------------------------------------------------------------
    # Logits processors to prevent early EOS and ensure JSON closure
    # ------------------------------------------------------------------
    class PreventEarlyEos(LogitsProcessor):
        def __init__(self, eos_token_id: int, min_tokens: int = 12):
            self.eos_token_id = int(eos_token_id)
            self.min_tokens = int(min_tokens)

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
            if input_ids.size(1) < self.min_tokens:
                scores[:, self.eos_token_id] = -float("inf")
            return scores

    class JsonMustCloseProcessor(LogitsProcessor):
        """
        Disallow EOS while an opened JSON object/array hasn't been closed.
        Uses lightweight decode of the tail to count braces/brackets.
        """
        def __init__(self, tokenizer, eos_token_id: int, lookback_tokens: int = 256):
            self.tok = tokenizer
            self.eos_token_id = int(eos_token_id)
            self.lookback = int(lookback_tokens)

        def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
            B, T = input_ids.size()
            with torch.no_grad():
                for b in range(B):
                    tail = input_ids[b, max(0, T - self.lookback):]
                    try:
                        text = self.tok.decode(tail.tolist(), skip_special_tokens=True)
                    except Exception:
                        text = ""
                    open_obj = text.count("{")
                    close_obj = text.count("}")
                    open_arr = text.count("[")
                    close_arr = text.count("]")
                    unclosed = (open_obj > close_obj) or (open_arr > close_arr)
                    if unclosed:
                        scores[b, self.eos_token_id] = -float("inf")
            return scores

    def _default_logits_processors(self, *, force_json: bool = False, min_tokens: int = 12):
        procs: list[LogitsProcessor] = []
        eos_id = getattr(self, "eos_token_id", None)
        if eos_id is not None:
            procs.append(self.PreventEarlyEos(eos_token_id=eos_id, min_tokens=min_tokens))
            if force_json:
                procs.append(self.JsonMustCloseProcessor(self.tokenizer, eos_token_id=eos_id))
        return procs

    # ------------------------------------------------------------------
    # Task-aware decoding routing
    # ------------------------------------------------------------------
    def _infer_task(self, prompt_text: str) -> str:
        """Infer task type from prompt, extracting only the user question part."""
        t = (prompt_text or "").lower()

        # Extract the actual question, ignoring system prompt
        if "question:" in t:
            question_part = t.split("question:")[-1].strip()
        elif "user" in t:
            parts = t.split("user")
            question_part = parts[-1].strip() if parts else t
        else:
            question_part = t

        # Check task indicators in the question only
        if "json" in question_part or "return json" in question_part or "{\"" in question_part:
            return "json"
        if "yes/no" in question_part or "yes or no" in question_part:
            return "binary"
        if "lvef" in question_part or "ejection fraction" in question_part:
            return "scalar"
        return "free"

    def _decoding_profile(self, task: str) -> dict:
        """Return task-specific decoding settings (excluding temperature/top_p which are user-controlled)."""
        if task == "binary":
            return {
                "max_new_tokens": 3,
                "bad_words_ids": self._binary_bad_words_ids(),
                "force_json": False,
                "min_tokens_guard": 2,
            }
        if task == "json":
            return {
                "max_new_tokens": 196,
                "force_json": True,
                "min_tokens_guard": 16,
            }
        if task == "scalar":
            return {
                "max_new_tokens": 24,
                "min_tokens_guard": 2,
            }
        return {}

    def _inspect_stage1_metadata(self, checkpoint_path: str) -> Dict[str, Any]:
        """Peek into Stage-1 checkpoint for bridge metadata (e.g., token counts)."""
        metadata: Dict[str, Any] = {}
        try:
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        except FileNotFoundError:
            warnings.warn(
                f"Stage-1 checkpoint not found while inspecting metadata: '{checkpoint_path}'"
            )
            return metadata
        except Exception as exc:
            warnings.warn(
                f"Failed to inspect Stage-1 checkpoint '{checkpoint_path}': {exc}"
            )
            return metadata

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        key_candidates = [
            "decoder.bridge.positional_embedding.weight",
            "module.decoder.bridge.positional_embedding.weight",
            "bridge.positional_embedding.weight",
            "module.bridge.positional_embedding.weight",
        ]
        for key in key_candidates:
            tensor = state_dict.get(key)
            if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
                metadata["bridge_num_tokens"] = int(tensor.shape[0])
                metadata["bridge_token_dim"] = int(tensor.shape[1])
                metadata["bridge_prefix"] = key[: -len("positional_embedding.weight")]
                break

        # Also try direct query matrix if positional embeddings are absent
        if "bridge_num_tokens" not in metadata:
            query_keys = [
                "decoder.bridge.queries",
                "module.decoder.bridge.queries",
                "bridge.queries",
                "module.bridge.queries",
            ]
            for key in query_keys:
                tensor = state_dict.get(key)
                if isinstance(tensor, torch.Tensor) and tensor.ndim == 2:
                    metadata["bridge_num_tokens"] = int(tensor.shape[0])
                    metadata["bridge_token_dim"] = int(tensor.shape[1])
                    metadata["bridge_prefix"] = key[: -len("queries")]
                    break

        del checkpoint
        return metadata

    def _infer_stage1_config_path(self, checkpoint_path: str) -> Optional[Path]:
        """Locate a Stage-1 config file relative to the checkpoint path."""
        path = Path(checkpoint_path)
        candidates = [
            path.parent / "config.yaml",
            path.parent / "config.yml",
            path.parent.parent / "config.yaml",
            path.parent.parent / "config.yml",
        ]
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None

    def _validate_stage1_bridge_config(self, stage1_checkpoint_path: str) -> None:
        """Validate bridge hyperparameters against the Stage-1 config/metadata."""
        if self.bridge is None:
            return

        # Prefer explicit config values saved alongside Stage-1 training.
        cfg_path = self._stage1_config_path or self._infer_stage1_config_path(stage1_checkpoint_path)
        cfg: Dict[str, Any] = {}
        if cfg_path is not None and cfg_path.exists():
            try:
                cfg = load_yaml(str(cfg_path))
            except Exception as exc:
                warnings.warn(f"Failed to load Stage-1 config at '{cfg_path}': {exc}")

        mismatches: list[str] = []

        def _require_match(name: str, expected: Optional[Any], actual: Optional[Any]) -> None:
            if expected is None or actual is None:
                return
            try:
                exp_int = int(expected)
                act_int = int(actual)
            except Exception:
                return
            if exp_int != act_int:
                mismatches.append(f"{name} (config {exp_int} vs model {act_int})")

        # Extract expectations from Stage-1 config if present.
        expected_layers = cfg.get("bridge_num_layers")
        expected_heads = cfg.get("bridge_num_heads")
        expected_special = cfg.get("bridge_num_special_tokens")
        expected_queries = cfg.get("num_query_tokens")
        expected_cross_every = cfg.get("cross_every")
        expected_hidden = cfg.get("bridge_hidden_size")

        # Derive actual bridge settings from the instantiated module.
        actual_layers = len(getattr(self.bridge, "blocks", []))
        actual_instruction_layers = len(getattr(self.bridge, "stage1_blocks", [])) if hasattr(self.bridge, "stage1_blocks") else None
        actual_heads = None
        if getattr(self.bridge, "blocks", None):
            first_block = getattr(self.bridge, "blocks")[0]
            if hasattr(first_block, "self_attn"):
                actual_heads = getattr(first_block.self_attn, "num_heads", None)
        actual_special = None
        if getattr(self.bridge, "embed_tables", None):
            vocab_size = getattr(self.bridge, "vocab_size", None)
            first_table = self.bridge.embed_tables[0]
            actual_special = first_table.num_embeddings - int(vocab_size) if vocab_size is not None else None
        actual_queries = getattr(self.bridge, "num_query_tokens", None)
        actual_cross_every = getattr(self.bridge, "stage1_cross_every", getattr(self.bridge, "cross_every", None))
        actual_hidden = None
        if getattr(self.bridge, "embed_tables", None):
            actual_hidden = self.bridge.embed_tables[0].weight.size(1)

        # Validate against Stage-1 config expectations when available.
        _require_match("bridge_num_layers", expected_layers, actual_layers)
        if expected_layers is not None and actual_instruction_layers is not None:
            _require_match("bridge_num_layers (instruction)", expected_layers, actual_instruction_layers)
        _require_match("bridge_num_heads", expected_heads, actual_heads)
        _require_match("bridge_num_special_tokens", expected_special, actual_special)
        _require_match("num_query_tokens", expected_queries, actual_queries)
        _require_match("cross_every", expected_cross_every, actual_cross_every)
        _require_match("bridge_hidden_size", expected_hidden, actual_hidden)

        # Also enforce token count and hidden dim based on checkpoint metadata.
        meta_dim = self.stage1_metadata.get("bridge_token_dim")
        _require_match("bridge_hidden_size (checkpoint)", meta_dim, actual_hidden)

        # mix_strategy / token_axis are structural (different fusion parameters and kv
        # geometry): a mismatch makes strict=False loading silently drop weights, so fail hard.
        for cfg_key, attr in (("bridge_mix_strategy", "mix_strategy"),
                              ("bridge_token_axis", "token_axis")):
            expected = cfg.get(cfg_key)
            actual = getattr(self.bridge, attr, None)
            if expected is not None and actual is not None and str(expected) != str(actual):
                raise ValueError(
                    f"Stage-1 checkpoint '{stage1_checkpoint_path}' was trained with "
                    f"{cfg_key}='{expected}' but the current config instantiated "
                    f"'{actual}'. Set {cfg_key}: {expected} in the Stage-3 config."
                )

        if mismatches:
            details = "; ".join(mismatches)
            warnings.warn(
                f"Stage-1 bridge checkpoint '{stage1_checkpoint_path}' mismatches current config: {details}."
                " Proceeding to load whatever keys match.",
                UserWarning,
            )

    def _initialize_pattern_head(self) -> None:
        """Initialise multilabel projection head for auxiliary ECG targets."""
        # If the loss is disabled (weight <= 0), skip creating the head entirely
        if self.pattern_loss_weight <= 0 or self.pattern_label_count <= 0 or not hasattr(self.bridge, "forward_instruction_hidden"):
            # Ensure no stale classifier remains on the bridge
            if hasattr(self.bridge, "pattern_classifier"):
                delattr(self.bridge, "pattern_classifier")
            self.pattern_classifier = None
            self.pattern_loss_fn = None
            return

        feature_dim = int(self.qformer_text_output_size or self.qformer_hidden_size or self.llm_input_embedding_size)
        hidden_dim = max(128, feature_dim // 2)
        # Attach auxiliary classifier to the bridge so the aux loss comes from the bridge head
        self.bridge.pattern_classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, self.pattern_label_count),
        )
        if hasattr(self, "_pattern_pos_weight") and self._pattern_pos_weight.numel() > 0:
            self.pattern_loss_fn = nn.BCEWithLogitsLoss(pos_weight=self._pattern_pos_weight)
        else:
            self.pattern_loss_fn = nn.BCEWithLogitsLoss()
        # Keep local reference None to avoid ambiguity; classifier lives on bridge
        self.pattern_classifier = None

    def _initialize_aux_endpoint_heads(self) -> None:
        """Attach scalar auxiliary heads (LVEF regression, SHD/AFib binary) to the bridge.

        Each head reads the Q-Former pooled ECG vector (`pooled_queries`) — the same tap
        the pattern head uses — and is trained jointly with the LM loss. Losses are masked
        per-row at forward time so rows without the label contribute nothing.
        """
        self.aux_endpoint_specs = {}
        specs = [
            ("lvef", self.lvef_head_loss_weight, "regression"),
            ("shd", self.shd_head_loss_weight, "binary"),
            ("afib", self.afib_head_loss_weight, "binary"),
        ]
        has_pooled = hasattr(self.bridge, "forward_instruction_hidden")
        feature_dim = int(self.qformer_text_output_size or self.qformer_hidden_size or self.llm_input_embedding_size)
        for name, weight, kind in specs:
            attr = f"{name}_head"
            if has_pooled and weight and weight > 0:
                hidden_dim = max(64, feature_dim // 2)
                setattr(self.bridge, attr, nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim),
                    nn.GELU(),
                    nn.Linear(hidden_dim, 1),
                ))
                self.aux_endpoint_specs[name] = {"weight": float(weight), "kind": kind}
            elif hasattr(self.bridge, attr):
                delattr(self.bridge, attr)

    # ------------------------------------------------------------------
    # Token initialization helpers (borrowed from LLaMA decoder)
    # ------------------------------------------------------------------
    def _load_stage1_weights(
        self,
        *,
        stage1_component: nn.Module,
        stage1_checkpoint_path: str,
        component_type: str,
        component_identifier: str,
    ) -> bool:
        """Load Stage-1 weights into the requested component, with logging."""
        print(f"[Stage1] attempting to load checkpoint for {component_identifier} from '{stage1_checkpoint_path}'")

        if hasattr(stage1_component, "load_stage1_checkpoint"):
            try:
                load_info = stage1_component.load_stage1_checkpoint(stage1_checkpoint_path, strict=False)
                self.bridge_load_info = load_info
                missing = load_info.get("missing_keys") or []
                shape_mismatched = load_info.get("shape_mismatched_keys") or []
                stage1_blocks = load_info.get("stage1_block_count")
                model_blocks = load_info.get("model_block_count")
                if isinstance(stage1_blocks, int) and isinstance(model_blocks, int) and stage1_blocks != model_blocks:
                    warnings.warn(
                        f"{component_identifier} Stage-1 checkpoint expects {stage1_blocks} blocks but config instantiated {model_blocks}. "
                        "Loading available tensors despite mismatch.",
                        UserWarning,
                    )
                stage1_instr_blocks = load_info.get("stage1_instruction_block_count")
                model_instr_blocks = load_info.get("model_instruction_block_count")
                if isinstance(stage1_instr_blocks, int) and isinstance(model_instr_blocks, int) and stage1_instr_blocks != model_instr_blocks:
                    warnings.warn(
                        f"{component_identifier} Stage-1 checkpoint expects {stage1_instr_blocks} instruction blocks but config instantiated {model_instr_blocks}. "
                        "Loading available tensors despite mismatch.",
                        UserWarning,
                    )
                if missing or shape_mismatched:
                    missing_preview = ", ".join(missing[:5])
                    mismatch_preview = ", ".join(
                        f"{name} (ckpt={ckpt_shape}, model={model_shape})"
                        for name, ckpt_shape, model_shape in shape_mismatched[:5]
                    )
                    details = []
                    if missing:
                        details.append(f"missing tensors: {missing_preview}")
                    if shape_mismatched:
                        details.append(f"shape mismatches: {mismatch_preview}")
                    warnings.warn(
                        f"{component_identifier} Stage-1 checkpoint partially loaded: "
                        f"{len(missing)} missing, {len(shape_mismatched)} shape mismatches. "
                        + (" ".join(details) if details else ""),
                        UserWarning,
                    )

                summary: list[str] = []
                stage1_layers = load_info.get("stage1_block_count")
                model_layers = load_info.get("model_block_count")
                if isinstance(stage1_layers, int) and isinstance(model_layers, int):
                    diff = model_layers - stage1_layers
                    if diff > 0:
                        summary.append(f"{diff} new Q-Former layers initialised")
                stage1_instr = load_info.get("stage1_instruction_block_count")
                model_instr = load_info.get("model_instruction_block_count")
                if isinstance(stage1_instr, int) and isinstance(model_instr, int):
                    diff = model_instr - stage1_instr
                    if diff > 0:
                        summary.append(f"{diff} new instruction layers initialised")
                partial = load_info.get("partially_loaded_keys") or []
                if partial:
                    summary.append(f"partially loaded {len(partial)} tensors (tokenizer alignment)")
                reinit = load_info.get("reinitialized_keys") or []
                if reinit:
                    summary.append("reinitialised tensors: " + ", ".join(reinit))
                if summary:
                    print(f"[Stage1] {component_identifier}: " + "; ".join(summary))
                return True
            except FileNotFoundError as exc:
                raise FileNotFoundError(
                    f"Stage-1 checkpoint not found at '{stage1_checkpoint_path}'."
                ) from exc
            except Exception as exc:
                warnings.warn(
                    f"Failed to load Stage-1 checkpoint '{stage1_checkpoint_path}': {exc}"
                )
                return False

        if component_type == "adapter":
            prefix_candidates = [
                "module.decoder.adapter.",
                "decoder.adapter.",
                "adapter.",
                "module.adapter.",
            ]
        else:
            prefix_candidates = [
                "module.decoder.bridge.",
                "decoder.bridge.",
                "bridge.",
                "module.bridge.",
                "module.model.decoder.bridge.",
            ]
        metadata_prefix = self.stage1_metadata.get("bridge_prefix")
        if isinstance(metadata_prefix, str) and metadata_prefix:
            prefix_candidates = [metadata_prefix] + [
                prefix for prefix in prefix_candidates if prefix != metadata_prefix
            ]

        try:
            checkpoint = torch.load(stage1_checkpoint_path, map_location="cpu", weights_only=False)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Stage-1 checkpoint not found at '{stage1_checkpoint_path}'."
            ) from exc
        except Exception as exc:
            warnings.warn(
                f"Failed to load Stage-1 checkpoint '{stage1_checkpoint_path}': {exc}"
            )
            return False

        state_dict = checkpoint.get("model_state_dict", checkpoint)
        subset: dict[str, torch.Tensor] = {}
        used_prefix: str | None = None
        for prefix in prefix_candidates:
            filtered = {
                key[len(prefix):]: value
                for key, value in state_dict.items()
                if key.startswith(prefix)
            }
            if filtered:
                subset = filtered
                used_prefix = prefix
                break

        if not subset:
            print(
                f"[Stage1] WARNING: checkpoint '{stage1_checkpoint_path}' did not contain weights for {component_identifier} "
                f"(checked prefixes: {prefix_candidates})"
            )
            return False

        component_state = stage1_component.state_dict()
        loadable_state: dict[str, torch.Tensor] = {}
        unexpected_keys: list[str] = []
        partial_keys: list[str] = []
        shape_mismatched_keys: list[tuple[str, tuple[int, ...], tuple[int, ...]]] = []

        for key, tensor in subset.items():
            if key not in component_state:
                unexpected_keys.append(key)
                continue
            target_tensor = component_state[key]
            if tensor.shape == target_tensor.shape:
                loadable_state[key] = tensor
                continue

            if tensor.ndim == target_tensor.ndim and tensor.shape[1:] == target_tensor.shape[1:]:
                rows = min(tensor.shape[0], target_tensor.shape[0])
                if rows > 0:
                    new_tensor = target_tensor.clone()
                    new_tensor[:rows] = tensor[:rows]
                    loadable_state[key] = new_tensor
                    partial_keys.append(key)
                    continue

            shape_mismatched_keys.append((key, tuple(tensor.shape), tuple(target_tensor.shape)))

        if not loadable_state:
            print(
                f"[Stage1] WARNING: checkpoint '{stage1_checkpoint_path}' had no compatible tensors for {component_identifier} "
                f"(checked prefixes: {prefix_candidates})"
            )
            self.bridge_load_info = {
                "missing_keys": [],
                "unexpected_keys": unexpected_keys,
                "shape_mismatched_keys": shape_mismatched_keys,
                "partially_loaded_keys": partial_keys,
                "loaded_prefix": used_prefix,
                "loaded_tensor_count": 0,
            }
            return False

        print(
            f"[Stage1] load plan for {component_identifier}: tensors={len(loadable_state)} "
            f"(partial={len(partial_keys)}, mismatched={len(shape_mismatched_keys)}, unexpected={len(unexpected_keys)})"
        )

        incompatible = stage1_component.load_state_dict(loadable_state, strict=False)
        missing_keys = list(getattr(incompatible, "missing_keys", []))
        unexpected_after = list(getattr(incompatible, "unexpected_keys", []))
        unexpected_keys.extend(unexpected_after)
        self.bridge_load_info = {
            "missing_keys": missing_keys,
            "unexpected_keys": unexpected_keys,
            "shape_mismatched_keys": shape_mismatched_keys,
            "partially_loaded_keys": partial_keys,
            "loaded_prefix": used_prefix,
            "loaded_tensor_count": len(loadable_state),
        }

        summary_parts: list[str] = []
        if missing_keys:
            summary_parts.append(f"missing tensors: {', '.join(missing_keys[:5])}")
        if unexpected_keys:
            summary_parts.append(f"unexpected tensors: {', '.join(unexpected_keys[:5])}")
        if shape_mismatched_keys:
            preview = ", ".join(
                f"{name} (ckpt={ckpt_shape}, model={model_shape})"
                for name, ckpt_shape, model_shape in shape_mismatched_keys[:5]
            )
            summary_parts.append(f"shape mismatches: {preview}")
        if partial_keys:
            summary_parts.append(f"partially loaded tensors: {', '.join(partial_keys[:5])}")

        if summary_parts:
            print(
                f"[Stage1] checkpoint partially loaded for {component_identifier} using prefix '{used_prefix}'; "
                + "; ".join(summary_parts)
            )
        else:
            print(
                f"[Stage1] checkpoint fully loaded for {component_identifier} using prefix '{used_prefix}'."
            )

        return True

    def _initialize_ecg_tokens_semantically(self, start_id: int, num_tokens: int) -> None:
        embedding_layer = self.llm_model.get_input_embeddings()
        with torch.no_grad():
            embeddings = embedding_layer.weight
            device = embeddings.device
            dtype = embeddings.dtype

            base_tokens = torch.arange(0, min(2048, start_id), device=device)
            base_embeddings = embeddings[base_tokens]
            mean_embedding = base_embeddings.mean(dim=0)
            std_embedding = base_embeddings.std(dim=0)
            noise = torch.randn((num_tokens, embeddings.size(1)), device=device, dtype=dtype)
            init_embeddings = mean_embedding + 0.01 * std_embedding * noise
            embeddings[start_id:start_id + num_tokens].copy_(init_embeddings)

    def _apply_ecg_embedding_mask(self) -> None:
        if self._ecg_embedding_hook_handle is not None:
            try:
                self._ecg_embedding_hook_handle.remove()
            except RuntimeError:
                pass
            finally:
                self._ecg_embedding_hook_handle = None

        mask_base = getattr(self, "_ecg_embedding_train_mask", None)
        if mask_base is None:
            return

        embedding_weight = self.llm_model.get_input_embeddings().weight

        def _mask_gradients(grad: torch.Tensor) -> torch.Tensor:
            mask = mask_base.to(device=grad.device, dtype=grad.dtype).unsqueeze(1)
            return grad * mask

        self._ecg_embedding_hook_handle = embedding_weight.register_hook(_mask_gradients)
        embedding_weight.requires_grad_(True)

    def _strip_prefix_tokens(
        self,
        generated: Union[torch.Tensor, Any],
        prefix_len: int,
    ) -> Union[torch.Tensor, Any]:
        """Remove synthetic prefix tokens from generated sequences when prefix tuning is active."""
        if prefix_len <= 0:
            return generated

        if isinstance(generated, torch.Tensor):
            if generated.size(-1) <= prefix_len:
                return generated[:, 0:0]
            return generated[:, prefix_len:]

        sequences = getattr(generated, "sequences", None)
        if isinstance(sequences, torch.Tensor):
            if sequences.size(-1) <= prefix_len:
                stripped = sequences[:, 0:0]
            else:
                stripped = sequences[:, prefix_len:]
            generated.sequences = stripped
        return generated

    def _should_train_input_embeddings(self) -> bool:
        """Train input embeddings only when textual ECG placeholder rows exist.

        In Q-Former or prefix-tuning modes there are no dedicated ECG rows in the
        embedding matrix; training the entire embedding table destabilizes decoding.
        """
        if getattr(self, "prefix_tuning", False):
            return False
        if getattr(self, "ecg_token_start_id", None) is None:
            return False
        mask = getattr(self, "_ecg_embedding_train_mask", None)
        try:
            return bool(mask is not None and mask.any().item())
        except Exception:
            return False

    def _count_leading_ecg_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        """Count leading ECG placeholder tokens on each row.

        This mirrors runner logic and makes decoder robust to legacy prompts
        that still contain <|ecg_pos_*|> tokens when the bridge already injects
        soft ECG prompts. Returns a 1D tensor of per-row counts to strip.
        """
        if ids is None or ids.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=ids.device if ids is not None else None)
        if getattr(self, "ecg_token_start_id", None) is None:
            return torch.zeros(ids.size(0), dtype=torch.long, device=ids.device)
        try:
            total = int(getattr(self, "num_ecg_tokens", 0))
        except Exception:
            total = 0
        if total <= 0:
            return torch.zeros(ids.size(0), dtype=torch.long, device=ids.device)

        start = int(self.ecg_token_start_id)
        end = start + int(total)
        out: list[int] = []
        for row in ids:
            k = 0
            for t in row.tolist():
                ti = int(t)
                if start <= ti < end:
                    k += 1
                else:
                    break
            out.append(k)
        return torch.tensor(out, dtype=torch.long, device=ids.device)

    def _start_image_token_id(self) -> Optional[int]:
        """Return the token id that actually appears for the image anchor.

        We derive it via encoding instead of relying on a hard-coded string.
        """
        cached = getattr(self, "_start_image_id_cache", None)
        if cached is not None:
            return cached

        candidates = []

        # 1. If tokenizer exposes a boi_token (Gemma 3 style), use it
        try:
            if hasattr(self.tokenizer, "special_tokens_map"):
                boi = self.tokenizer.special_tokens_map.get("boi_token", None)
                if boi:
                    candidates.append(boi)
        except Exception:
            pass

        # 2. Your custom sentinel
        candidates.append("<start_of_image>")

        tok_id: Optional[int] = None
        for s in candidates:
            if not s:
                continue
            try:
                enc = self.tokenizer(
                    s,
                    add_special_tokens=True,
                    return_tensors="pt",
                )
                ids = enc.input_ids[0].tolist()
                # Typically [BOS, anchor, ...]; pick the first non-BOS
                bos_id = getattr(self.tokenizer, "bos_token_id", None)
                for tid in ids:
                    if bos_id is not None and tid == bos_id:
                        continue
                    tok_id = int(tid)
                    break
                if tok_id is not None:
                    break
            except Exception:
                continue

        self._start_image_id_cache = tok_id
        return tok_id

    def _inject_ecg_after_image_token(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor],
        ecg_embeddings: torch.Tensor,
        embed_layer: torch.nn.Embedding,
        ecg_counts: Optional[torch.Tensor] = None,
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], torch.Tensor]]:
        """Insert ECG embeddings immediately after the <start_of_image> token.

        Returns padded (inputs_embeds, attention_mask, input_ids, labels, token_type_ids) or None if no <start_of_image> is found.
        """
        start_img_id = self._start_image_token_id()
        if start_img_id is None or input_ids is None or ecg_embeddings is None:
            return None

        batch_size = input_ids.size(0)
        device = input_ids.device
        pad_id = getattr(self, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if pad_id is None:
            pad_id = getattr(self, "eos_token_id", None)
        if isinstance(pad_id, (list, tuple)):
            pad_id = pad_id[0] if pad_id else None
        if pad_id is None:
            pad_id = 0
        pad_id = int(pad_id)
        pad_vec = embed_layer(torch.tensor([pad_id], device=device)).squeeze(0)

        model_dtype = embed_layer.weight.dtype
        ecg_embeddings = ecg_embeddings.to(device=device)
        if ecg_embeddings.dtype != model_dtype:
            ecg_embeddings = ecg_embeddings.to(model_dtype)

        new_embeds: list[torch.Tensor] = []
        new_masks: list[torch.Tensor] = []
        new_ids: list[torch.Tensor] = []
        new_labels: list[torch.Tensor] = []
        new_ttids: list[torch.Tensor] = []
        need_labels = labels is not None

        # Multi-ECG: ecg_embeddings is flat [sum(N_b), num_tokens, hidden] and ecg_counts[b]
        # gives the number of ECG blocks (and <start_of_image> anchors) for row b.
        # Legacy single-ECG (ecg_counts is None): one block per row inserted after the
        # FIRST anchor — byte-identical to the previous behavior.
        counts_list = ecg_counts.to("cpu").tolist() if ecg_counts is not None else None
        offset = 0

        for b in range(batch_size):
            ids_b = input_ids[b]
            mask_b = attention_mask[b]
            emb_b = embed_layer(ids_b)
            lbl_b = labels[b] if need_labels else None
            positions = (ids_b == start_img_id).nonzero(as_tuple=False).flatten().tolist()

            if counts_list is None:
                if len(positions) == 0:
                    return None  # legacy: fall back if any row lacks the token
                positions = positions[:1]
                blocks = ecg_embeddings[b : b + 1]
            else:
                n_b = int(counts_list[b])
                blocks = ecg_embeddings[offset : offset + n_b]
                offset += n_b
                k = min(len(positions), int(blocks.size(0)))
                if k != len(positions) and self.debug_ecg_injection:
                    print(f"[ECG Injection] row {b}: anchors={len(positions)} but blocks={blocks.size(0)}; using {k}")
                positions = positions[:k]
                blocks = blocks[:k]

            # No anchor / no ECG (e.g. refusal rows): pass the row through unchanged.
            if len(positions) == 0:
                new_ids.append(ids_b)
                new_masks.append(mask_b)
                new_embeds.append(emb_b)
                if need_labels and lbl_b is not None:
                    new_labels.append(lbl_b)
                new_ttids.append(torch.zeros(ids_b.size(0), device=device, dtype=torch.long))
                continue

            seg_ids, seg_mask, seg_emb, seg_lbl, seg_ttid = [], [], [], [], []
            prev = 0
            for j, p in enumerate(positions):
                seg_ids.append(ids_b[prev : p + 1])
                seg_mask.append(mask_b[prev : p + 1])
                seg_emb.append(emb_b[prev : p + 1])
                if need_labels and lbl_b is not None:
                    seg_lbl.append(lbl_b[prev : p + 1])
                seg_ttid.append(torch.zeros(p + 1 - prev, device=device, dtype=torch.long))

                blk = blocks[j]
                if blk.dim() == 1:
                    blk = blk.unsqueeze(0)
                elen = blk.size(0)
                seg_ids.append(torch.full((elen,), pad_id, device=device, dtype=ids_b.dtype))
                seg_mask.append(torch.ones(elen, device=device, dtype=mask_b.dtype))
                seg_emb.append(blk)
                if need_labels and lbl_b is not None:
                    seg_lbl.append(torch.full((elen,), -100, device=device, dtype=lbl_b.dtype))
                seg_ttid.append(torch.ones(elen, device=device, dtype=torch.long))
                prev = p + 1

            # tail after the last anchor
            seg_ids.append(ids_b[prev:])
            seg_mask.append(mask_b[prev:])
            seg_emb.append(emb_b[prev:])
            if need_labels and lbl_b is not None:
                seg_lbl.append(lbl_b[prev:])
            seg_ttid.append(torch.zeros(ids_b.size(0) - prev, device=device, dtype=torch.long))

            new_ids.append(torch.cat(seg_ids, dim=0))
            new_masks.append(torch.cat(seg_mask, dim=0))
            new_embeds.append(torch.cat(seg_emb, dim=0))
            if need_labels and lbl_b is not None:
                new_labels.append(torch.cat(seg_lbl, dim=0))
            new_ttids.append(torch.cat(seg_ttid, dim=0))

        max_len = max(x.size(0) for x in new_ids)
        padded_embeds: list[torch.Tensor] = []
        padded_masks: list[torch.Tensor] = []
        padded_ids: list[torch.Tensor] = []
        padded_labels: list[torch.Tensor] = []
        padded_ttids: list[torch.Tensor] = []

        for i in range(len(new_ids)):
            diff = max_len - new_ids[i].size(0)
            if diff > 0:
                pad_ids = torch.full((diff,), pad_id, device=device, dtype=new_ids[i].dtype)
                pad_masks = torch.zeros(diff, device=device, dtype=new_masks[i].dtype)
                pad_embs = pad_vec.unsqueeze(0).expand(diff, -1)
                pad_ttid = torch.zeros(diff, device=device, dtype=torch.long)
                new_ids[i] = torch.cat([new_ids[i], pad_ids], dim=0)
                new_masks[i] = torch.cat([new_masks[i], pad_masks], dim=0)
                new_embeds[i] = torch.cat([new_embeds[i], pad_embs], dim=0)
                new_ttids[i] = torch.cat([new_ttids[i], pad_ttid], dim=0)
                if need_labels and len(new_labels) > i:
                    pad_labs = torch.full((diff,), -100, device=device, dtype=new_labels[i].dtype)
                    new_labels[i] = torch.cat([new_labels[i], pad_labs], dim=0)
            padded_ids.append(new_ids[i])
            padded_masks.append(new_masks[i])
            padded_embeds.append(new_embeds[i])
            padded_ttids.append(new_ttids[i])
            if need_labels and len(new_labels) > i:
                padded_labels.append(new_labels[i])

        inputs_embeds = torch.stack(padded_embeds, dim=0)
        attention_mask = torch.stack(padded_masks, dim=0)
        input_ids = torch.stack(padded_ids, dim=0)
        token_type_ids = torch.stack(padded_ttids, dim=0)
        labels_out = None
        if need_labels and padded_labels:
            labels_out = torch.stack(padded_labels, dim=0)

        return inputs_embeds, attention_mask, input_ids, labels_out, token_type_ids

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def _toggle_lora_trainable(self, enabled: bool) -> None:
        lora_flag = False
        if hasattr(self.llm_model, "named_parameters"):
            for name, param in self.llm_model.named_parameters():
                if "lora_" in name:
                    param.requires_grad = enabled
                    lora_flag = True
        if enabled and not lora_flag:
            warnings.warn(
                "LoRA was requested but no parameters matching 'lora_' were found on the MedGemma model."
            )

    def freeze_llm_parameters(self) -> None:
        for param in self.llm_model.parameters():
            param.requires_grad = False
        # Re-enable LoRA adapters so Stage-2 can still update them while the base remains frozen.
        self._toggle_lora_trainable(True)
        # Train input embeddings ONLY if we have ECG placeholder rows
        try:
            embedding_weight = self.llm_model.get_input_embeddings().weight
            if self._should_train_input_embeddings():
                embedding_weight.requires_grad_(True)
                self._apply_ecg_embedding_mask()
            else:
                embedding_weight.requires_grad_(False)
        except Exception:
            pass

    def unfreeze_llm_parameters(self) -> None:
        for param in self.llm_model.parameters():
            param.requires_grad = True

    def _compute_ecg_embeddings(
        self,
        quantized_features: Optional[torch.Tensor],
        quantized_codes: Optional[torch.Tensor],
        *,
        prompt_input_ids: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        detach_soft_prompts: bool = False,
        continuous_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        if self.bridge is None:
            raise RuntimeError("ECG bridge is not initialized")

        embed_layer = self.llm_model.get_input_embeddings()
        device = embed_layer.weight.device

        def _append_continuous(code_embeddings: torch.Tensor) -> torch.Tensor:
            """P1: concat the continuous-feature Perceiver tokens after the code-bridge tokens."""
            if self.continuous_bridge is None:
                return code_embeddings
            cont_in = continuous_features if continuous_features is not None else self._continuous_features
            if cont_in is None:
                raise ValueError(
                    "continuous_features must be provided when the continuous Perceiver bridge is enabled"
                )
            cont_in = cont_in.to(device=device, dtype=next(self.continuous_bridge.parameters()).dtype)
            cont_tokens = self.continuous_bridge(cont_in)
            if cont_tokens.dim() == 2:
                cont_tokens = cont_tokens.unsqueeze(1)
            if code_embeddings.dim() == 2:
                code_embeddings = code_embeddings.unsqueeze(1)
            cont_tokens = cont_tokens.to(dtype=code_embeddings.dtype)
            return torch.cat([code_embeddings, cont_tokens], dim=1)

        def _prepare_ecg_ids(codes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            pad_id = getattr(self.bridge, "pad_id", None)
            if codes.dim() == 4 and codes.size(0) == 1:
                codes = codes.squeeze(0)
            if codes.dim() == 3 and codes.size(-1) == 1:
                codes = codes.squeeze(-1)
            if codes.dim() not in (2, 3):
                raise ValueError(
                    f"quantized_codes must be [batch, seq] or [batch, seq, depth]; got {codes.shape}"
                )
            codes = codes.to(device=device)
            if codes.dim() == 3:
                valid_levels = codes >= 0
                if pad_id is not None and pad_id >= 0:
                    valid_levels = valid_levels & (codes != pad_id)
                attn_mask = valid_levels.any(dim=-1)
            else:
                if pad_id is not None and pad_id >= 0:
                    attn_mask = codes != pad_id
                else:
                    attn_mask = codes >= 0
            codes = codes.clamp_min(0).to(dtype=torch.long)
            attn_mask = attn_mask.to(device=device, dtype=torch.bool)
            return codes, attn_mask

        # Phase B: continuous-only. Skip the discrete code/Q-Former path entirely; emit ONLY the
        # continuous Perceiver tokens. The code bridge is still built (clean checkpoint/LoRA load)
        # but never executed, since the encoder no longer matches the VQ codebook.
        if getattr(self, "continuous_only", False):
            if self.continuous_bridge is None:
                raise RuntimeError(
                    "continuous_only requires use_continuous_features=True (continuous_bridge missing)."
                )
            cont_in = continuous_features if continuous_features is not None else self._continuous_features
            if cont_in is None:
                raise ValueError(
                    "continuous_features must be provided when continuous_only is enabled."
                )
            batch_size = cont_in.size(0)
            empty = torch.zeros(
                batch_size, 0, embed_layer.weight.size(-1), device=device, dtype=embed_layer.weight.dtype
            )
            embeddings = _append_continuous(empty)
            return embeddings, None

        if hasattr(self.bridge, "forward_instruction_hidden"):
            if not getattr(self.bridge, "uses_codes", False):
                raise RuntimeError(
                    "Instruction-aware bridge currently requires discrete ECG codes."
                )
            if quantized_codes is None:
                raise ValueError(
                    "quantized_codes must be provided for instruction-aware Q-Former bridge."
                )
            ecg_ids = quantized_codes
            if isinstance(ecg_ids, (tuple, list)):
                ecg_ids = ecg_ids[0]
            if not isinstance(ecg_ids, torch.Tensor):
                ecg_ids = torch.as_tensor(ecg_ids)
            ecg_ids, ecg_mask = _prepare_ecg_ids(ecg_ids)

            instruction_hidden: Optional[torch.Tensor]
            instruction_mask: Optional[torch.Tensor]
            if prompt_input_ids is not None:
                prompt_ids = prompt_input_ids.to(device)
                instruction_mask = (
                    prompt_attention_mask.to(device)
                    if prompt_attention_mask is not None
                    else (prompt_ids != self.pad_token_id).long()
                )
                # Ignore ECG sentinel tokens in instruction fusion
                for tok in ("<|start_ecg|>", "<|end_ecg|>"):
                    try:
                        tid = self.tokenizer.convert_tokens_to_ids(tok)
                    except Exception:
                        tid = None
                    if tid is not None:
                        try:
                            tid = int(tid)
                            if tid >= 0:
                                instruction_mask = instruction_mask * (prompt_ids != tid).long()
                        except Exception:
                            pass
                instruction_embeds = embed_layer(prompt_ids)
                target_dtype = self.bridge.queries.dtype if hasattr(self.bridge, "queries") else instruction_embeds.dtype
                if instruction_embeds.dtype != target_dtype:
                    instruction_embeds = instruction_embeds.to(dtype=target_dtype)
                if self.qformer_text_projection is None or self.qformer_text_norm is None:
                    raise RuntimeError(
                        "Instruction-aware bridge requires qformer_text_projection and qformer_text_norm."
                    )
                instruction_hidden = self.qformer_text_projection(instruction_embeds)
                instruction_hidden = self.qformer_text_norm(instruction_hidden)
                if self.qformer_instruction_dropout is not None:
                    instruction_hidden = self.qformer_instruction_dropout(instruction_hidden)
            else:
                instruction_hidden = None
                instruction_mask = None

            bridge_outputs = self.bridge.forward_instruction_hidden(
                ecg_ids=ecg_ids,
                instruction_hidden=instruction_hidden,
                instruction_attention_mask=instruction_mask,
                ecg_attn_mask=ecg_mask,
                detach_soft_prompts=detach_soft_prompts,
            )
            embeddings = bridge_outputs["token_embeddings"]
            if embeddings.dim() == 2:
                embeddings = embeddings.unsqueeze(1)
            embeddings = _append_continuous(embeddings)
            return embeddings, bridge_outputs

        if getattr(self.bridge, "uses_codes", False):
            if quantized_codes is None:
                raise ValueError("quantized_codes must be provided when using the ECG code bridge")
            ecg_ids = quantized_codes
            if isinstance(ecg_ids, (tuple, list)):
                ecg_ids = ecg_ids[0]
            if not isinstance(ecg_ids, torch.Tensor):
                ecg_ids = torch.as_tensor(ecg_ids)
            ecg_ids, ecg_mask = _prepare_ecg_ids(ecg_ids)
            bridge_outputs = self.bridge(ecg_ids, attn_mask=ecg_mask)
            if isinstance(bridge_outputs, tuple):
                embeddings = bridge_outputs[0]
            elif isinstance(bridge_outputs, dict):
                embeddings = bridge_outputs.get("token_embeddings") or bridge_outputs.get("embeddings")
                if embeddings is None:
                    raise ValueError("Bridge dictionary output missing token embeddings.")
            else:
                embeddings = bridge_outputs
        else:
            if quantized_features is None:
                raise ValueError("quantized_features must be provided when using the ECG projection bridge")
            embeddings = self.bridge(quantized_features.to(device))

        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(1)
        embeddings = _append_continuous(embeddings)
        return embeddings, None

    # ------------------------------------------------------------------
    # LVEF soft-decoding loss
    # ------------------------------------------------------------------
    def _ensure_digit_token_ids(self):
        """Build digit-to-token-id mapping from the tokenizer. Called once lazily."""
        if self._digit_token_ids is not None:
            return
        self._digit_token_ids = []
        for d in range(10):
            ids = self.tokenizer.encode(str(d), add_special_tokens=False)
            assert len(ids) == 1, f"Digit {d} encodes to multiple tokens: {ids}"
            self._digit_token_ids.append(ids[0])
        pct_ids = self.tokenizer.encode('%', add_special_tokens=False)
        self._pct_token_id = pct_ids[0] if pct_ids else None

    def _compute_lvef_soft_loss(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        lvef_gt: torch.Tensor,
    ) -> Optional[torch.Tensor]:
        """
        Compute differentiable Huber loss between soft-decoded LVEF and ground truth.

        For each LVEF sample: find '%' in labels, extract logits at the two
        preceding digit positions, softmax over digit token IDs, compute expected
        value (tens*10 + units), and Huber loss vs ground truth.
        """
        self._ensure_digit_token_ids()

        device = logits.device
        batch_size = logits.size(0)
        digit_ids = torch.tensor(self._digit_token_ids, device=device)
        digit_values = torch.arange(10, dtype=torch.float32, device=device)

        losses = []

        for b in range(batch_size):
            gt = lvef_gt[b].item()
            if gt != gt:  # NaN check
                continue

            sample_labels = labels[b]
            pct_positions = (sample_labels == self._pct_token_id).nonzero(as_tuple=True)[0]

            if len(pct_positions) == 0:
                continue

            pct_pos = pct_positions[0].item()
            if pct_pos < 2:
                continue

            units_pos = pct_pos - 1
            tens_pos = pct_pos - 2

            # Verify these are actual digit tokens in labels
            tens_label = sample_labels[tens_pos].item()
            units_label = sample_labels[units_pos].item()
            if tens_label not in self._digit_token_ids or units_label not in self._digit_token_ids:
                continue

            # logits[t] predicts labels[t+1] in HF causal LM
            tens_logit_pos = tens_pos - 1
            units_logit_pos = units_pos - 1

            if tens_logit_pos < 0 or units_logit_pos >= logits.size(1):
                continue

            tens_logits = logits[b, tens_logit_pos, digit_ids]
            units_logits = logits[b, units_logit_pos, digit_ids]

            tens_probs = torch.softmax(tens_logits.float(), dim=0)
            units_probs = torch.softmax(units_logits.float(), dim=0)

            predicted_tens = (tens_probs * digit_values).sum()
            predicted_units = (units_probs * digit_values).sum()
            predicted_lvef = predicted_tens * 10.0 + predicted_units

            gt_tensor = torch.tensor(gt, dtype=torch.float32, device=device)

            loss = torch.nn.functional.huber_loss(
                predicted_lvef, gt_tensor, reduction='none', delta=5.0
            )
            losses.append(loss)

        if not losses:
            return None

        return torch.stack(losses).mean()

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------
    def forward(
        self,
        quantized_features: Optional[torch.Tensor] = None,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        quantized_codes: Optional[torch.Tensor] = None,
        ecg_embeddings: Optional[torch.Tensor] = None,
        prompt_input_ids: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        pattern_targets: Optional[torch.Tensor] = None,
        ecg_counts: Optional[torch.Tensor] = None,
        continuous_features: Optional[torch.Tensor] = None,
        **unused_kwargs: Any,
    ) -> Dict[str, torch.Tensor]:
        if input_ids is None:
            raise ValueError("input_ids must be provided for MedGemmaDecoder forward pass")
        # P1: stash pre-quant features so generation helpers reuse the same continuous path.
        self._continuous_features = continuous_features

        embed_layer = self.llm_model.get_input_embeddings()
        device = embed_layer.weight.device
        model_dtype = embed_layer.weight.dtype

        bridge_outputs: Optional[Dict[str, torch.Tensor]] = None
        if ecg_embeddings is not None:
            ecg_embeddings = ecg_embeddings.to(device=device)
            if ecg_embeddings.dim() == 2:
                ecg_embeddings = ecg_embeddings.unsqueeze(1)
            prefix_embeddings = ecg_embeddings
        else:
            # Multi-ECG: ecg features are flat-concatenated to [sum(N_b), ...]. Repeat the
            # per-row prompt to align with each ECG so the (optionally instruction-aware)
            # bridge fuses each ECG with its own prompt. count=0 rows drop out (0 copies).
            _bridge_pii, _bridge_pam = prompt_input_ids, prompt_attention_mask
            if ecg_counts is not None and prompt_input_ids is not None:
                _counts = ecg_counts.to(prompt_input_ids.device).clamp_min(0)
                if int(_counts.sum().item()) > 0:
                    _bridge_pii = prompt_input_ids.repeat_interleave(_counts, dim=0)
                    if prompt_attention_mask is not None:
                        _bridge_pam = prompt_attention_mask.repeat_interleave(_counts, dim=0)
            prefix_embeddings, bridge_outputs = self._compute_ecg_embeddings(
                quantized_features,
                quantized_codes,
                prompt_input_ids=_bridge_pii,
                prompt_attention_mask=_bridge_pam,
                continuous_features=continuous_features,
            )
            prefix_embeddings = prefix_embeddings.to(device=device)

        prefix_len = int(prefix_embeddings.size(1))
        input_ids = input_ids.to(device)
        
        # Legacy slicing removed per new architecture: input_ids contains pure text (plus <image>) 
        # and no placeholder prefixes.
        text_input_ids = input_ids
        base_mask = attention_mask.to(device) if attention_mask is not None else torch.ones_like(text_input_ids, dtype=torch.long, device=device)
        base_labels = labels.to(device) if labels is not None else None

        if prefix_embeddings.dtype != model_dtype:
            prefix_embeddings = prefix_embeddings.to(model_dtype)

        # Prefer inserting ECG embeddings after <start_of_image> anchor
        merged = self._inject_ecg_after_image_token(
            input_ids=text_input_ids,
            attention_mask=base_mask,
            labels=base_labels,
            ecg_embeddings=prefix_embeddings,
            embed_layer=embed_layer,
            ecg_counts=ecg_counts,
        )

        if self.debug_ecg_injection:
            start_img_id = self._start_image_token_id()
            if start_img_id is not None and (text_input_ids == start_img_id).any().item():
                if merged is None:
                    raise ValueError(
                        "ECG injection debug: <start_of_image> present but injection failed in forward()."
                    )

        token_type_ids = None
        if merged is not None:
            inputs_embeds, attn_mask, input_ids, prepared_labels, token_type_ids = merged
        else:
            text_embeddings = embed_layer(text_input_ids)
            if text_embeddings.dtype != model_dtype:
                text_embeddings = text_embeddings.to(model_dtype)

            inputs_embeds = torch.cat([prefix_embeddings, text_embeddings], dim=1)

            text_mask = base_mask
            prefix_mask = torch.ones(
                text_mask.size(0),
                prefix_len,
                dtype=text_mask.dtype,
                device=device,
            ) if prefix_len > 0 else torch.zeros(
                text_mask.size(0), 0, dtype=text_mask.dtype, device=device
            )
            attn_mask = torch.cat([prefix_mask, text_mask], dim=1)

            # token_type_ids: ECG prefix = 1 (image-like), text = 0
            prefix_ttid = torch.ones(
                text_mask.size(0), prefix_len, device=device, dtype=torch.long
            ) if prefix_len > 0 else torch.zeros(
                text_mask.size(0), 0, device=device, dtype=torch.long
            )
            text_ttid = torch.zeros(
                text_mask.size(0), text_input_ids.size(1), device=device, dtype=torch.long
            )
            token_type_ids = torch.cat([prefix_ttid, text_ttid], dim=1)

            prepared_labels = None
            if base_labels is not None:
                prepared_labels = base_labels
                if prepared_labels.size(1) != inputs_embeds.size(1):
                    if prepared_labels.size(1) == text_input_ids.size(1):
                        ignore_pad = torch.full(
                            (prepared_labels.size(0), prefix_len),
                            self.label_ignore_index,
                            dtype=prepared_labels.dtype,
                            device=device,
                        )
                        prepared_labels = torch.cat([ignore_pad, prepared_labels], dim=1)
                    else:
                        raise ValueError(
                            "labels length does not match combined ECG/text sequence length."
                        )

        # Build forward kwargs — only pass token_type_ids when available
        # (Gemma3-based models require it during training).
        fwd_kwargs: Dict[str, Any] = dict(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            labels=prepared_labels,
            return_dict=True,
        )
        if token_type_ids is not None and self.pass_token_type_ids:
            fwd_kwargs["token_type_ids"] = token_type_ids
        outputs = self.llm_model(**fwd_kwargs)

        result: Dict[str, torch.Tensor] = {"logits": outputs.logits}
        if hasattr(outputs, "loss") and outputs.loss is not None:
            base_loss = outputs.loss
            # Keep the base LM/CF loss separately for logging alongside the combined loss
            result["loss"] = base_loss
            result["cf_loss"] = base_loss
        if bridge_outputs is not None:
            for key in ("query_hidden", "instruction_hidden", "pooled_queries"):
                value = bridge_outputs.get(key)
                if value is not None:
                    result[key] = value

        if pattern_targets is not None and self.pattern_loss_fn is not None and bridge_outputs is not None:
            pooled = bridge_outputs.get("pooled_queries")
            classifier = getattr(self.bridge, "pattern_classifier", None)
            if pooled is not None and classifier is not None:
                logits_pattern = classifier(pooled)
                target = pattern_targets
                if not isinstance(target, torch.Tensor):
                    target = torch.as_tensor(target)
                target = target.to(device=logits_pattern.device, dtype=logits_pattern.dtype)
                if target.dim() == 1:
                    target = target.unsqueeze(0)
                if target.size(0) != logits_pattern.size(0):
                    raise ValueError(
                        f"pattern_targets batch mismatch: got {target.size(0)}, expected {logits_pattern.size(0)}"
                    )
                if target.size(-1) != logits_pattern.size(-1):
                    if target.size(-1) < logits_pattern.size(-1):
                        pad_len = logits_pattern.size(-1) - target.size(-1)
                        pad = torch.zeros(target.size(0), pad_len, device=target.device, dtype=target.dtype)
                        target = torch.cat([target, pad], dim=-1)
                    else:
                        target = target[..., :logits_pattern.size(-1)]
                # Compute BCE loss in float32 for numerical stability
                pattern_loss_unscaled = self.pattern_loss_fn(
                    logits_pattern.float(),
                    target.float(),
                )
                if self.pattern_loss_weight > 0:
                    pattern_loss = pattern_loss_unscaled * self.pattern_loss_weight
                    if "loss" in result:
                        result["loss"] = result["loss"] + pattern_loss
                    else:
                        result["loss"] = pattern_loss
                    result["pattern_loss"] = pattern_loss
                else:
                    result["pattern_loss"] = pattern_loss_unscaled
                result["pattern_logits"] = logits_pattern
                result["pattern_targets"] = target
                result["pattern_loss_unscaled"] = pattern_loss_unscaled

        # --- Auxiliary scalar heads on the bridge pooled output (LVEF / SHD / AFib) ---
        pooled_for_aux = bridge_outputs.get("pooled_queries") if bridge_outputs is not None else None
        if pooled_for_aux is not None and self.aux_endpoint_specs:
            aux_gts = {
                "lvef": unused_kwargs.get("aux_lvef_gt"),
                "shd": unused_kwargs.get("aux_shd_gt"),
                "afib": unused_kwargs.get("aux_afib_gt"),
            }
            for name, spec in self.aux_endpoint_specs.items():
                head = getattr(self.bridge, f"{name}_head", None)
                gt = aux_gts.get(name)
                if head is None or gt is None:
                    continue
                target = gt if isinstance(gt, torch.Tensor) else torch.as_tensor(gt)
                target = target.to(device=pooled_for_aux.device, dtype=torch.float32).reshape(-1)
                logits_aux = head(pooled_for_aux).reshape(-1).float()
                if target.numel() != logits_aux.numel():
                    continue
                # Expose full logits + targets (NaN preserved) so the runner can compute AUROC.
                result[f"{name}_head_logits"] = logits_aux.detach()
                result[f"{name}_head_targets"] = target.detach()
                mask = torch.isfinite(target)
                n_valid = int(mask.sum().item())
                result[f"{name}_head_n"] = float(n_valid)
                if n_valid == 0:
                    # No labels for this head in this micro-batch. Still attach a
                    # zero-valued term so the head's parameters receive a (zero)
                    # gradient on EVERY rank. Under DDP the set of parameters that
                    # get gradients must be identical across ranks each backward;
                    # otherwise the gradient all-reduce buckets mismatch and NCCL
                    # hangs (masked aux labels are sparse, so ranks routinely
                    # disagree on which heads are active). Value is exactly 0.
                    zero_loss = 0.0 * logits_aux.sum()
                    result["loss"] = result["loss"] + zero_loss if "loss" in result else zero_loss
                    continue
                lm_, tm_ = logits_aux[mask], target[mask]
                if spec["kind"] == "binary":
                    head_loss = F.binary_cross_entropy_with_logits(lm_, tm_)
                else:  # regression on EF, scaled to [0,1] via /100
                    head_loss = F.smooth_l1_loss(lm_, tm_ / 100.0)
                result[f"{name}_head_loss_unscaled"] = head_loss.detach()
                scaled = head_loss * spec["weight"]
                result["loss"] = result["loss"] + scaled if "loss" in result else scaled
                result[f"{name}_head_loss"] = scaled.detach()

        # --- LVEF soft-decoding loss ---
        lvef_gt = unused_kwargs.get("lvef_gt")
        if (
            self.lvef_loss_weight > 0
            and lvef_gt is not None
            and "logits" in result
        ):
            lvef_gt_tensor = lvef_gt
            if not isinstance(lvef_gt_tensor, torch.Tensor):
                lvef_gt_tensor = torch.as_tensor(lvef_gt_tensor, dtype=torch.float32)
            lvef_gt_tensor = lvef_gt_tensor.to(device=result["logits"].device)

            if not torch.isnan(lvef_gt_tensor).all():
                lvef_loss = self._compute_lvef_soft_loss(
                    logits=result["logits"],
                    labels=prepared_labels if prepared_labels is not None else base_labels,
                    lvef_gt=lvef_gt_tensor,
                )
                if lvef_loss is not None:
                    scaled_lvef_loss = lvef_loss * self.lvef_loss_weight
                    if "loss" in result:
                        result["loss"] = result["loss"] + scaled_lvef_loss
                    else:
                        result["loss"] = scaled_lvef_loss
                    result["lvef_loss"] = scaled_lvef_loss
                    result["lvef_loss_unscaled"] = lvef_loss

        return result

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------
    def _binary_bad_words_ids(self) -> Optional[list[list[int]]]:
        if not self._binary_allowed_token_ids:
            return None
        if self._binary_bad_words_cache is not None:
            return self._binary_bad_words_cache

        vocab_size = len(self.tokenizer)
        eos_ok: set[int] = set()
        if getattr(self, "eos_token_id", None) is not None:
            eos_ok.add(int(self.eos_token_id))
        # Include all EOS tokens (MedGemma's <end_of_turn> + LLaMA's <|eot_id|>)
        for eos_id in getattr(self, "_eos_token_ids", []):
            if eos_id is not None:
                eos_ok.add(int(eos_id))
        if getattr(self, "_eot_token_id", None) is not None and self._eot_token_id is not None:
            eos_ok.add(int(self._eot_token_id))
        allowed = set(self._binary_allowed_token_ids)
        bad_words: list[list[int]] = []
        for token_id in range(vocab_size):
            if token_id not in allowed and token_id not in eos_ok:
                bad_words.append([token_id])

        if self.bad_ecg_token_ids:
            existing_single = {seq[0] for seq in bad_words if len(seq) == 1}
            for seq in self.bad_ecg_token_ids:
                if len(seq) == 1:
                    tid = seq[0]
                    if tid in allowed or tid in existing_single:
                        continue
                    bad_words.append([tid])
                    existing_single.add(tid)
                else:
                    bad_words.append(seq)

        self._binary_bad_words_cache = bad_words
        return self._binary_bad_words_cache

    def _build_default_prompt(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        system_message = (
            "You are an expert cardiologist. Read the ECG patches and answer concisely."
        )
        default_user_content = "<start_of_image> Review the ECG data and provide the clinical answer."
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": default_user_content},
        ]
        prompt_text = cast(
            str,
            self.tokenizer.apply_chat_template(  # type: ignore[attr-defined]
                messages,
                tokenize=False,
                add_generation_prompt=True,
            ),
        )
        encoding = self.tokenizer.encode_plus(
            prompt_text,
            add_special_tokens=False,
            return_tensors=None,
        )
        prompt_ids = encoding.input_ids
        if not prompt_ids:
            prompt_ids = [self.pad_token_id]
        prompt_tensor = torch.tensor(prompt_ids, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)
        prompt_mask = torch.ones_like(prompt_tensor, dtype=torch.long)
        return prompt_tensor, prompt_mask

    def generate_report(
        self,
        quantized_features: Optional[torch.Tensor] = None,
        quantized_codes: Optional[torch.Tensor] = None,
        max_token_length: int = 256,
        continuous_features: Optional[torch.Tensor] = None,
        **generate_kwargs: Any,
    ) -> torch.Tensor:
        # P1: stash pre-quant features for the continuous Perceiver path used in _compute_ecg_embeddings.
        self._continuous_features = continuous_features
        embed_layer = self.llm_model.get_input_embeddings()
        device = embed_layer.weight.device

        if getattr(self.bridge, "uses_codes", False) and quantized_codes is None:
            raise ValueError("quantized_codes must be provided when using the ECG code bridge")
        if not getattr(self.bridge, "uses_codes", False) and quantized_features is None:
            raise ValueError("quantized_features must be provided when using the ECG projection bridge")

        batch_size = 1
        if getattr(self.bridge, "uses_codes", False) and quantized_codes is not None:
            batch_size = quantized_codes.size(0)
        elif quantized_features is not None:
            batch_size = quantized_features.size(0)

        prompt_tensor, prompt_mask = self._build_default_prompt(batch_size, device)

        ecg_embeddings, _ = self._compute_ecg_embeddings(
            quantized_features,
            quantized_codes,
            prompt_input_ids=prompt_tensor,
            prompt_attention_mask=prompt_mask,
            detach_soft_prompts=True,
        )
        ecg_embeddings = ecg_embeddings.to(device)
        model_dtype = embed_layer.weight.dtype

        prompt_embeddings = embed_layer(prompt_tensor)
        if prompt_embeddings.dtype != model_dtype:
            prompt_embeddings = prompt_embeddings.to(model_dtype)

        if ecg_embeddings.dim() == 2:
            ecg_embeddings = ecg_embeddings.unsqueeze(1)

        if ecg_embeddings.dtype != model_dtype:
            ecg_embeddings = ecg_embeddings.to(model_dtype)

        merged = self._inject_ecg_after_image_token(
            input_ids=prompt_tensor,
            attention_mask=prompt_mask,
            labels=None,
            ecg_embeddings=ecg_embeddings,
            embed_layer=embed_layer,
        )
        if merged is not None:
            prompt_embeddings, prompt_mask, prompt_tensor, _, _ttids = merged
            prefix_len = 0
        else:
            prompt_embeddings = torch.cat([ecg_embeddings, prompt_embeddings], dim=1)
            prefix_len = ecg_embeddings.size(1)
            prefix_mask = torch.ones(batch_size, prefix_len, device=device, dtype=torch.long)
            prompt_mask = torch.cat([prefix_mask, prompt_mask], dim=1)

        if not self.prefix_tuning and self.ecg_token_start_id is not None and prefix_len > 0:
            prefix_token_ids = torch.arange(
                self.ecg_token_start_id,
                self.ecg_token_start_id + prefix_len,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0).expand(batch_size, -1)
            input_ids = torch.cat([prefix_token_ids, prompt_tensor], dim=1)
        else:
            input_ids = prompt_tensor

        generate_args = dict(self.default_generation_params)
        generate_args.update(generate_kwargs)
        generate_args.setdefault("max_new_tokens", max_token_length)
        if self.bad_ecg_token_ids:
            generate_args.setdefault("bad_words_ids", self.bad_ecg_token_ids)
        else:
            generate_args.pop("bad_words_ids", None)
        generate_args.pop("input_ids", None)
        generate_args.pop("attention_mask", None)

        # Task-aware routing based on default prompt and optional category hint
        try:
            prompt_text = self.tokenizer.decode(prompt_tensor[0].tolist(), skip_special_tokens=True)
        except Exception:
            prompt_text = ""
        # Coerce category hint if provided (supports str or list[str])
        task = self._infer_task(prompt_text)
        route = self._decoding_profile(task)
        for k, v in route.items():
            generate_args.setdefault(k, v)

        generate_args = self._sanitize_generate_args(generate_args)

        eos_token_id = generate_args.pop("eos_token_id", None)
        if eos_token_id is None:
            # Use the list of EOS tokens (includes <end_of_turn> for MedGemma)
            eos_token_id = self._eos_token_ids if self._eos_token_ids else self._eot_token_id

        # Extract custom routing guards and remove unsupported kwargs
        force_json_flag = bool(generate_args.pop("force_json", False))
        min_tokens_guard = int(generate_args.pop("min_tokens_guard", 12))

        generated = self.llm_model.generate(
            inputs_embeds=prompt_embeddings,
            attention_mask=prompt_mask,
            logits_processor=self._default_logits_processors(
                force_json=force_json_flag,
                min_tokens=min_tokens_guard,
            ),
            eos_token_id=eos_token_id,
            **generate_args,
        )

        if self.prefix_tuning:
            generated = self._strip_prefix_tokens(generated, prefix_len)

        return generated

    def _prepare_inputs_for_generation(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        quantized_features: Optional[torch.Tensor],
        quantized_codes: Optional[torch.Tensor],
        *,
        detach_soft_prompts: bool = True,
        continuous_features: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Build inputs_embeds + attention_mask for a batch, injecting ECG once.

        Returns:
            inputs_embeds: [B, L, D]
            attention_mask: [B, L]
            prefix_len: number of ECG tokens prepended (0 when using <start_of_image>).
        """
        embed_layer = self.llm_model.get_input_embeddings()
        device = embed_layer.weight.device
        model_dtype = embed_layer.weight.dtype

        prompt_input_ids = prompt_input_ids.to(device)
        prompt_attention_mask = prompt_attention_mask.to(device)

        # Compute ECG embeddings for the whole batch
        ecg_embeddings, _ = self._compute_ecg_embeddings(
            quantized_features,
            quantized_codes,
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            detach_soft_prompts=detach_soft_prompts,
            continuous_features=continuous_features,
        )
        if ecg_embeddings.dim() == 2:
            ecg_embeddings = ecg_embeddings.unsqueeze(1)
        ecg_embeddings = ecg_embeddings.to(model_dtype)

        # Try MedGemma-style injection after <start_of_image>
        merged = self._inject_ecg_after_image_token(
            input_ids=prompt_input_ids,
            attention_mask=prompt_attention_mask,
            labels=None,
            ecg_embeddings=ecg_embeddings,
            embed_layer=embed_layer,
        )

        if self.debug_ecg_injection:
            start_img_id = self._start_image_token_id()
            if start_img_id is not None and (prompt_input_ids == start_img_id).any().item():
                if merged is None:
                    raise ValueError(
                        "ECG injection debug: <start_of_image> present but injection failed in generation path."
                    )

        if merged is not None:
            inputs_embeds, attention_mask, _, _, _ttids = merged
            prefix_len = 0
        else:
            # Fallback: prepend ECG embeddings as a soft prefix
            prompt_embeddings = embed_layer(prompt_input_ids)
            if prompt_embeddings.dtype != model_dtype:
                prompt_embeddings = prompt_embeddings.to(model_dtype)

            inputs_embeds = torch.cat([ecg_embeddings, prompt_embeddings], dim=1)

            batch_size, ecg_len, _ = ecg_embeddings.shape
            prefix_len = ecg_len
            prefix_mask = torch.ones(
                batch_size,
                ecg_len,
                dtype=prompt_attention_mask.dtype,
                device=device,
            )
            attention_mask = torch.cat([prefix_mask, prompt_attention_mask], dim=1)

        return inputs_embeds, attention_mask, prefix_len

    def _build_generation_args_for_task(
        self,
        base_args: Dict[str, Any],
        task: str,
    ) -> Tuple[Dict[str, Any], bool, int, Union[int, Sequence[int], None]]:
        """
        Overlay task-specific decoding profile on top of base_args and
        return sanitized args plus routing flags.

        Uses setdefault to avoid overwriting user-provided values (e.g., temperature).
        """
        args = dict(base_args)
        for k, v in self._decoding_profile(task).items():
            args.setdefault(k, v)
        args = self._sanitize_generate_args(args)

        eos_token_id = args.pop("eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = self._eos_token_ids if self._eos_token_ids else self._eot_token_id

        force_json_flag = bool(args.pop("force_json", False))
        min_tokens_guard = int(args.pop("min_tokens_guard", 12))

        return args, force_json_flag, min_tokens_guard, eos_token_id

    def _generate_with_microbatch(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
        batch_args: Dict[str, Any],
        force_json_flag: bool,
        min_tokens_guard: int,
        eos_token_id: Union[int, Sequence[int], None],
        prefix_len: int,
    ) -> torch.Tensor:
        """
        Run generation in small micro-batches to avoid MedGemma's buggy batched generate() with inputs_embeds.
        Defaults to micro-batch size 1 for correctness; encoder/bridge work remains batched.
        """
        micro = max(1, int(getattr(self, "generation_microbatch_size", 1)))
        outputs: list[torch.Tensor] = []
        bsz = inputs_embeds.size(0)
        for start in range(0, bsz, micro):
            end = min(start + micro, bsz)
            gen = self.llm_model.generate(
                inputs_embeds=inputs_embeds[start:end],
                attention_mask=attention_mask[start:end],
                logits_processor=self._default_logits_processors(
                    force_json=force_json_flag,
                    min_tokens=min_tokens_guard,
                ),
                eos_token_id=eos_token_id,
                **batch_args,
            )
            if self.prefix_tuning and prefix_len > 0:
                gen = self._strip_prefix_tokens(gen, prefix_len)
            outputs.append(gen)

        if len(outputs) == 1:
            return outputs[0]

        # Pad outputs to same length before concatenating (different samples may finish at different times)
        max_len = max(out.size(1) for out in outputs)
        pad_token_id = int(batch_args.get("pad_token_id", 0))

        padded_outputs = []
        for out in outputs:
            if out.size(1) < max_len:
                padding = torch.full(
                    (out.size(0), max_len - out.size(1)),
                    pad_token_id,
                    dtype=out.dtype,
                    device=out.device
                )
                out = torch.cat([out, padding], dim=1)
            padded_outputs.append(out)

        return torch.cat(padded_outputs, dim=0)

    def generate_report_with_question(
        self,
        quantized_features: Optional[torch.Tensor] = None,
        quantized_codes: Optional[torch.Tensor] = None,
        prompt_input_ids: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        max_token_length: int = 256,
        continuous_features: Optional[torch.Tensor] = None,
        **generate_kwargs: Any,
    ) -> torch.Tensor:
        # P1: stash pre-quant features for the continuous Perceiver path.
        self._continuous_features = continuous_features
        if prompt_input_ids is None:
            return self.generate_report(
                quantized_features=quantized_features,
                quantized_codes=quantized_codes,
                max_token_length=max_token_length,
                continuous_features=continuous_features,
                **generate_kwargs,
            )

        embed_layer = self.llm_model.get_input_embeddings()
        device = embed_layer.weight.device

        prompt_input_ids = prompt_input_ids.to(device)
        if prompt_attention_mask is None:
            prompt_attention_mask = (prompt_input_ids != self.pad_token_id).long()
        else:
            prompt_attention_mask = prompt_attention_mask.to(device)

        # Strip any legacy leading ECG placeholder tokens to prevent double-prefixing
        try:
            to_strip = self._count_leading_ecg_tokens(prompt_input_ids)
        except Exception:
            to_strip = None
        if to_strip is not None and to_strip.numel() > 0 and int(to_strip.max().item()) > 0:
            rows_ids = []
            rows_mask = []
            for b in range(prompt_input_ids.size(0)):
                s = int(to_strip[b].item())
                rows_ids.append(prompt_input_ids[b, s:])
                rows_mask.append(prompt_attention_mask[b, s:])
            prompt_input_ids = torch.nn.utils.rnn.pad_sequence(
                rows_ids, batch_first=True, padding_value=self.pad_token_id
            )
            prompt_attention_mask = torch.nn.utils.rnn.pad_sequence(
                rows_mask, batch_first=True, padding_value=0
            )

        # CRITICAL: Save raw prompt IDs BEFORE any injection to prevent double-injection bug
        raw_prompt_ids = prompt_input_ids.clone()
        raw_prompt_mask = prompt_attention_mask.clone()

        batch_size = prompt_input_ids.size(0)
        model_dtype = embed_layer.weight.dtype

        # Build base generation args
        generate_args = dict(self.default_generation_params)
        generate_args.update(generate_kwargs)
        generate_args.pop("input_ids", None)
        generate_args.pop("attention_mask", None)

        generate_args["max_new_tokens"] = min(
            int(generate_args.get("max_new_tokens", max_token_length)),
            max_token_length,
        )
        generate_args.setdefault("min_new_tokens", 0)

        # Infer task for EACH sample using raw (uninjected) prompt IDs
        tasks = []
        for b in range(batch_size):
            try:
                prompt_text = self.tokenizer.decode(raw_prompt_ids[b].tolist(), skip_special_tokens=True)
            except Exception:
                prompt_text = ""

            # Apply category hint if provided
            task = self._infer_task(prompt_text)
            tasks.append(task)

        # Compute prompt lengths for grouping
        prompt_lengths = [int(raw_prompt_mask[b].sum().item()) for b in range(batch_size)]

        # Group samples by (task, prompt_length) for efficient batched processing
        # This allows TRUE batched generation for groups with same task + same prompt length
        from collections import defaultdict
        groups = defaultdict(list)
        for b in range(batch_size):
            key = (tasks[b], prompt_lengths[b])
            groups[key].append(b)

        # If only one group, use fast path
        if len(groups) == 1:
            # All samples have same task AND same prompt length - TRUE batched generation
            task = tasks[0]

            inputs_embeds, attention_mask, prefix_len = self._prepare_inputs_for_generation(
                raw_prompt_ids,
                raw_prompt_mask,
                quantized_features,
                quantized_codes,
                detach_soft_prompts=True,
                continuous_features=self._continuous_features,
            )

            batch_args, force_json_flag, min_tokens_guard, eos_token_id = \
                self._build_generation_args_for_task(generate_args, task)

            generated = self._generate_with_microbatch(
                inputs_embeds,
                attention_mask,
                batch_args,
                force_json_flag,
                min_tokens_guard,
                eos_token_id,
                prefix_len,
            )

            return generated

        else:
            # Multiple groups: process each group as a batch, then reassemble
            generated_by_idx = {}

            for (task, prompt_len), indices in groups.items():
                group_size = len(indices)

                # Gather UNPADDED data for this group
                group_ids_list = []
                group_masks_list = []
                group_features_list = []
                group_codes_list = []
                group_continuous_list = []

                for b in indices:
                    # Extract unpadded prompt
                    sample_mask = raw_prompt_mask[b]
                    valid_len = int(sample_mask.sum().item())
                    group_ids_list.append(raw_prompt_ids[b, :valid_len])
                    group_masks_list.append(sample_mask[:valid_len])

                    # Extract ECG data
                    if quantized_features is not None:
                        group_features_list.append(quantized_features[b])
                    if quantized_codes is not None:
                        group_codes_list.append(quantized_codes[b])
                    # P1: slice pre-quant continuous features the same way as codes/features
                    if self._continuous_features is not None:
                        group_continuous_list.append(self._continuous_features[b])

                # Stack into batch (no padding needed - all same length!)
                group_ids = torch.stack(group_ids_list, dim=0)
                group_masks = torch.stack(group_masks_list, dim=0)
                group_features = torch.stack(group_features_list, dim=0) if group_features_list else None
                group_codes = torch.stack(group_codes_list, dim=0) if group_codes_list else None
                group_continuous = torch.stack(group_continuous_list, dim=0) if group_continuous_list else None

                # Process this group as a TRUE batch
                inputs_embeds, attention_mask, prefix_len = self._prepare_inputs_for_generation(
                    group_ids,
                    group_masks,
                    group_features,
                    group_codes,
                    detach_soft_prompts=True,
                    continuous_features=group_continuous,
                )

                group_args, force_json_flag, min_tokens_guard, eos_token_id = \
                    self._build_generation_args_for_task(generate_args, task)

                gen = self._generate_with_microbatch(
                    inputs_embeds,
                    attention_mask,
                    group_args,
                    force_json_flag,
                    min_tokens_guard,
                    eos_token_id,
                    prefix_len,
                )

                # Store results by original index
                for i, b in enumerate(indices):
                    generated_by_idx[b] = gen[i:i+1]

            # Reassemble in original order and pad
            all_generated = [generated_by_idx[b] for b in range(batch_size)]
            max_length = max(g.size(1) for g in all_generated)
            padded_generated = []
            for g in all_generated:
                if g.size(1) < max_length:
                    padding = torch.full(
                        (g.size(0), max_length - g.size(1)),
                        self.pad_token_id,
                        dtype=g.dtype,
                        device=g.device
                    )
                    g = torch.cat([g, padding], dim=1)
                padded_generated.append(g)

            return torch.cat(padded_generated, dim=0)


__all__ = ["MedGemmaDecoder"]
