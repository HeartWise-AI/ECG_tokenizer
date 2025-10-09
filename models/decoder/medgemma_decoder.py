"""MedGemma decoder that mirrors the LLaMA ECG token integration path."""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple, Union, cast

import torch
import torch.nn as nn

from transformers import AutoModelForImageTextToText, AutoTokenizer, PreTrainedModel

from models.adapter.bridge import ECGCodeBridge, ECGProjectionBridge
from utils.enums import AdapterName, ModelName
from utils.registry import ModelRegistry


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
    for loader in loaders:
        try:
            return loader.from_pretrained(
                model_name,
                dtype=torch_dtype,
                torch_dtype=torch_dtype,
                trust_remote_code=True,
            )
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

    def __init__(
        self,
        huggingface_model_name: str = "google/medgemma-4b-it",
        llm_input_embedding_size: int = 4096,
        quantized_feature_shape: Tuple[int, int] = (128, 82),
        adapter_name: Union[AdapterName, str] = AdapterName.LLAMA32_ECG_PROJECTION_BRIDGE,
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
        **unused_kwargs: Any,
    ) -> None:
        super().__init__()

        self.quantizer = quantizer
        self.prefix_tuning = prefix_tuning
        self.label_ignore_index = int(label_ignore_index)

        if isinstance(adapter_name, str):
            try:
                adapter_name = AdapterName(adapter_name)
            except ValueError:
                pass
        self.adapter_name = adapter_name

        adapter_name_str = adapter_name.value if hasattr(adapter_name, "value") else str(adapter_name)

        self.bridge: Optional[Union[ECGCodeBridge, ECGProjectionBridge]] = None
        self.bridge_config: Optional[Dict[str, Any]] = None

        visual_tokens = num_visual_tokens if num_visual_tokens is not None else quantized_feature_shape[0]

        if adapter_name in {
            AdapterName.LLAMA32_ECG_CODE_BRIDGE,
            AdapterName.LLAMA32_ECG_PROJECTION_BRIDGE,
        }:
            if adapter_name == AdapterName.LLAMA32_ECG_CODE_BRIDGE:
                self.bridge = ECGCodeBridge(
                    vocab_size=ecg_codebook_size,
                    d_mid=bridge_mid_dim,
                    d_model=llm_input_embedding_size,
                    num_output_tokens=visual_tokens,
                    num_heads=bridge_num_heads,
                    num_special_tokens=bridge_num_special_tokens,
                    dropout=bridge_dropout,
                )
                self.bridge_config = {
                    "style": "code",
                    "vocab_size": ecg_codebook_size,
                    "mid_dim": bridge_mid_dim,
                    "output_tokens": visual_tokens,
                    "num_heads": bridge_num_heads,
                    "dropout": bridge_dropout,
                    "num_special_tokens": bridge_num_special_tokens,
                }
            else:
                feature_dim = quantized_feature_shape[1] if len(quantized_feature_shape) > 1 else llm_input_embedding_size
                self.bridge = ECGProjectionBridge(
                    input_dim=feature_dim,
                    d_model=llm_input_embedding_size,
                    num_tokens=visual_tokens,
                    dropout=adapter_dropout,
                )
                self.bridge_config = {
                    "style": "projection",
                    "input_dim": feature_dim,
                    "d_model": llm_input_embedding_size,
                    "num_tokens": visual_tokens,
                    "dropout": adapter_dropout,
                }

        if self.bridge is None:
            raise ValueError(
                "MedGemmaDecoder currently requires an ECG bridge (projection or code)."
            )

        self.num_ecg_tokens = getattr(self.bridge, "num_tokens", visual_tokens)

        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(huggingface_model_name)

        ecg_tokens = [f"<|ecg_pos_{i}|>" for i in range(self.num_ecg_tokens)]
        first_ecg_id = self.tokenizer.convert_tokens_to_ids(ecg_tokens[0])  # type: ignore[attr-defined]
        unk_id = getattr(self.tokenizer, 'unk_token_id', None)  # type: ignore[attr-defined]

        if not self.prefix_tuning:
            if first_ecg_id is None or first_ecg_id == -1 or (unk_id is not None and int(first_ecg_id) == int(unk_id)):
                base_vocab_size = len(self.tokenizer)  # type: ignore[arg-type]
                self.tokenizer.add_tokens(ecg_tokens, special_tokens=True)  # type: ignore[attr-defined]
                self.ecg_token_start_id = base_vocab_size
            else:
                self.ecg_token_start_id = int(first_ecg_id)

            if ecg_token_start_id is not None and self.ecg_token_start_id != int(ecg_token_start_id):
                raise ValueError(
                    f"Tokenizer ECG token start id ({self.ecg_token_start_id}) does not match dataset-configured "
                    f"ecg_token_start_id ({ecg_token_start_id}). Ensure the config uses the tokenizer's vocabulary size."
                )
        else:
            if ecg_token_start_id is not None:
                self.ecg_token_start_id = int(ecg_token_start_id)
            elif first_ecg_id is not None and first_ecg_id != -1 and not (
                unk_id is not None and int(first_ecg_id) == int(unk_id)
            ):
                self.ecg_token_start_id = int(first_ecg_id)
            else:
                self.ecg_token_start_id = None

        self.llm_model: PreTrainedModel = _load_medgemma_model(
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
            raise ValueError(
                f"Embedding size {llm_input_embedding_size} does not match MedGemma hidden size {hidden_size}"
            )

        self.llm = self.llm_model
        self.llm_model.resize_token_embeddings(len(self.tokenizer), mean_resizing=True)  # type: ignore[arg-type]

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

        base_defaults: Dict[str, Any] = {
            "do_sample": True,
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 160,
            "min_new_tokens": 24,
            "repetition_penalty": 1.05,
        }
        if default_generation_kwargs:
            base_defaults.update(default_generation_kwargs)
        self._eot_token_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
        if not self.prefix_tuning and self.ecg_token_start_id is not None:
            self.bad_ecg_token_ids: Optional[list[list[int]]] = [[tid] for tid in range(
                self.ecg_token_start_id,
                self.ecg_token_start_id + self.num_ecg_tokens,
            )]
        else:
            self.bad_ecg_token_ids = None
        self.default_generation_params = base_defaults
        if self.bad_ecg_token_ids:
            self.default_generation_params.setdefault("bad_words_ids", self.bad_ecg_token_ids)

    # ------------------------------------------------------------------
    # Token initialization helpers (borrowed from LLaMA decoder)
    # ------------------------------------------------------------------
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

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------
    def freeze_llm_parameters(self) -> None:
        for param in self.llm_model.parameters():
            param.requires_grad = False

    def unfreeze_llm_parameters(self) -> None:
        for param in self.llm_model.parameters():
            param.requires_grad = True

    def _compute_ecg_embeddings(
        self,
        quantized_features: Optional[torch.Tensor],
        quantized_codes: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        if self.bridge is None:
            raise RuntimeError("ECG bridge is not initialized")

        if getattr(self.bridge, "uses_codes", False):
            if quantized_codes is None:
                raise ValueError("quantized_codes must be provided when using the ECG code bridge")
            ecg_ids = quantized_codes
            if isinstance(ecg_ids, tuple):
                ecg_ids = ecg_ids[0]
            if ecg_ids.dim() == 3 and ecg_ids.size(-1) == 1:
                ecg_ids = ecg_ids.squeeze(-1)
            elif ecg_ids.dim() == 3:
                ecg_ids = ecg_ids[..., 0]
            elif ecg_ids.dim() != 2:
                raise ValueError(
                    f"quantized_codes must be [batch, seq] or [batch, seq, depth]; got {ecg_ids.shape}"
                )
            ecg_mask = (ecg_ids >= 0).to(device)
            ecg_ids = ecg_ids.clamp_min(0).to(device=device, dtype=torch.long)
            embeddings = self.bridge(ecg_ids, attn_mask=ecg_mask)
        else:
            if quantized_features is None:
                raise ValueError("quantized_features must be provided when using the ECG projection bridge")
            embeddings = self.bridge(quantized_features.to(device))

        if embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(1)
        return embeddings

    def _prepare_inputs_with_bridge(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        quantized_features: Optional[torch.Tensor],
        quantized_codes: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], int]:
        embed_layer = self.llm_model.get_input_embeddings()
        device = embed_layer.weight.device
        model_dtype = embed_layer.weight.dtype

        ecg_embeddings = self._compute_ecg_embeddings(quantized_features, quantized_codes, device)
        prefix_len = int(ecg_embeddings.size(1))

        if not self.prefix_tuning:
            if input_ids.size(1) < prefix_len:
                raise ValueError(
                    f"Input sequence too short for {prefix_len} ECG tokens: {input_ids.shape}"
                )
            text_input_ids = input_ids[:, prefix_len:].to(device)
        else:
            text_input_ids = input_ids.to(device)

        text_embeddings = embed_layer(text_input_ids)

        if ecg_embeddings.dtype != model_dtype:
            ecg_embeddings = ecg_embeddings.to(model_dtype)
        if text_embeddings.dtype != model_dtype:
            text_embeddings = text_embeddings.to(model_dtype)

        input_embeddings = torch.cat([ecg_embeddings, text_embeddings], dim=1)

        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            if self.prefix_tuning:
                prefix_mask = torch.ones(
                    attention_mask.size(0),
                    prefix_len,
                    dtype=attention_mask.dtype,
                    device=device,
                )
                attention_mask = torch.cat([prefix_mask, attention_mask], dim=1)
        elif self.prefix_tuning:
            attention_mask = torch.ones(
                input_embeddings.size(0),
                prefix_len + text_input_ids.size(1),
                dtype=torch.long,
                device=device,
            )

        return input_embeddings, attention_mask, prefix_len

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
        **_: Any,
    ) -> Dict[str, torch.Tensor]:
        if input_ids is None:
            raise ValueError("input_ids must be provided for MedGemmaDecoder forward pass")

        inputs_embeds, attn_mask, prefix_len = self._prepare_inputs_with_bridge(
            input_ids,
            attention_mask,
            quantized_features,
            quantized_codes,
        )

        prepared_labels = labels
        if labels is not None:
            prepared_labels = labels.to(inputs_embeds.device)
            if self.prefix_tuning:
                ignore_pad = torch.full(
                    (prepared_labels.size(0), prefix_len),
                    self.label_ignore_index,
                    dtype=prepared_labels.dtype,
                    device=prepared_labels.device,
                )
                prepared_labels = torch.cat([ignore_pad, prepared_labels], dim=1)

        outputs = self.llm_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attn_mask,
            labels=prepared_labels,
            return_dict=True,
        )

        result: Dict[str, torch.Tensor] = {"logits": outputs.logits}
        if hasattr(outputs, "loss") and outputs.loss is not None:
            result["loss"] = outputs.loss
        return result

    # ------------------------------------------------------------------
    # Generation helpers
    # ------------------------------------------------------------------
    def _build_default_prompt(self, batch_size: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        system_message = (
            "You are a medical expert specialized in ECG interpretation. Provide a concise list "
            "of clinical findings separated by semicolons, similar to standard ECG reports."
        )
        default_user_content = "Analyze this ECG and list the clinical findings."
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
        **generate_kwargs: Any,
    ) -> torch.Tensor:
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

        ecg_embeddings = self._compute_ecg_embeddings(quantized_features, quantized_codes, device)
        model_dtype = embed_layer.weight.dtype

        prompt_embeddings = embed_layer(prompt_tensor)
        if prompt_embeddings.dtype != model_dtype:
            prompt_embeddings = prompt_embeddings.to(model_dtype)

        if ecg_embeddings.dim() == 2:
            ecg_embeddings = ecg_embeddings.unsqueeze(1)

        if ecg_embeddings.dtype != model_dtype:
            ecg_embeddings = ecg_embeddings.to(model_dtype)

        prompt_embeddings = torch.cat([ecg_embeddings, prompt_embeddings], dim=1)
        prefix_len = ecg_embeddings.size(1)
        prefix_mask = torch.ones(batch_size, prefix_len, device=device, dtype=torch.long)
        prompt_mask = torch.cat([prefix_mask, prompt_mask], dim=1)

        if not self.prefix_tuning and self.ecg_token_start_id is not None:
            prefix_token_ids = torch.arange(
                self.ecg_token_start_id,
                self.ecg_token_start_id + prefix_len,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            prefix_token_ids = torch.full(
                (batch_size, prefix_len),
                self.pad_token_id,
                dtype=torch.long,
                device=device,
            )

        input_ids = torch.cat([prefix_token_ids, prompt_tensor], dim=1)

        generate_args = dict(self.default_generation_params)
        generate_args.update(generate_kwargs)
        generate_args.setdefault("max_new_tokens", max_token_length)
        if self.bad_ecg_token_ids:
            generate_args.setdefault("bad_words_ids", self.bad_ecg_token_ids)
        else:
            generate_args.pop("bad_words_ids", None)
        generate_args.pop("input_ids", None)
        generate_args.pop("attention_mask", None)

        eos_token_id = generate_args.pop("eos_token_id", None)
        if eos_token_id is None and self._eot_token_id is not None:
            eos_token_id = self._eot_token_id

        generated = self.llm_model.generate(
            inputs_embeds=prompt_embeddings,
            input_ids=input_ids,
            attention_mask=prompt_mask,
            eos_token_id=eos_token_id,
            **generate_args,
        )

        if self.prefix_tuning:
            generated = self._strip_prefix_tokens(generated, prefix_len)

        return generated

    def generate_report_with_question(
        self,
        quantized_features: Optional[torch.Tensor] = None,
        quantized_codes: Optional[torch.Tensor] = None,
        prompt_input_ids: Optional[torch.Tensor] = None,
        prompt_attention_mask: Optional[torch.Tensor] = None,
        max_token_length: int = 256,
        **generate_kwargs: Any,
    ) -> torch.Tensor:
        if prompt_input_ids is None:
            return self.generate_report(
                quantized_features=quantized_features,
                quantized_codes=quantized_codes,
                max_token_length=max_token_length,
                **generate_kwargs,
            )

        embed_layer = self.llm_model.get_input_embeddings()
        device = embed_layer.weight.device

        prompt_input_ids = prompt_input_ids.to(device)
        if prompt_attention_mask is None:
            prompt_attention_mask = (prompt_input_ids != self.pad_token_id).long()
        else:
            prompt_attention_mask = prompt_attention_mask.to(device)

        model_dtype = embed_layer.weight.dtype

        prompt_embeddings = embed_layer(prompt_input_ids)
        if prompt_embeddings.dtype != model_dtype:
            prompt_embeddings = prompt_embeddings.to(model_dtype)
        ecg_embeddings = self._compute_ecg_embeddings(quantized_features, quantized_codes, device)
        if ecg_embeddings.dim() == 2:
            ecg_embeddings = ecg_embeddings.unsqueeze(1)

        if ecg_embeddings.dtype != model_dtype:
            ecg_embeddings = ecg_embeddings.to(model_dtype)

        inputs_embeds = torch.cat([ecg_embeddings, prompt_embeddings], dim=1)
        prefix_len = ecg_embeddings.size(1)
        prefix_mask = torch.ones(prompt_embeddings.size(0), prefix_len, device=device, dtype=torch.long)
        attention_mask = torch.cat([prefix_mask, prompt_attention_mask], dim=1)

        batch_size = prompt_embeddings.size(0)
        if not self.prefix_tuning and self.ecg_token_start_id is not None:
            prefix_token_ids = torch.arange(
                self.ecg_token_start_id,
                self.ecg_token_start_id + prefix_len,
                dtype=torch.long,
                device=device,
            ).unsqueeze(0).expand(batch_size, -1)
        else:
            prefix_token_ids = torch.full(
                (batch_size, prefix_len),
                self.pad_token_id,
                dtype=torch.long,
                device=device,
            )

        input_ids = torch.cat([prefix_token_ids, prompt_input_ids], dim=1)

        generate_args = dict(self.default_generation_params)
        generate_args.update(generate_kwargs)
        generate_args.setdefault("max_new_tokens", max_token_length)
        if self.bad_ecg_token_ids:
            generate_args.setdefault("bad_words_ids", self.bad_ecg_token_ids)
        else:
            generate_args.pop("bad_words_ids", None)
        generate_args.pop("input_ids", None)
        generate_args.pop("attention_mask", None)

        eos_token_id = generate_args.pop("eos_token_id", None)
        if eos_token_id is None and self._eot_token_id is not None:
            eos_token_id = self._eot_token_id

        generated = self.llm_model.generate(
            inputs_embeds=inputs_embeds,
            input_ids=input_ids,
            attention_mask=attention_mask,
            eos_token_id=eos_token_id,
            **generate_args,
        )

        if self.prefix_tuning:
            generated = self._strip_prefix_tokens(generated, prefix_len)

        return generated


__all__ = ["MedGemmaDecoder"]
