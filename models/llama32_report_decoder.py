import torch
import torch.nn as nn

from typing import Optional, Dict, Any, Tuple

from transformers import LlamaForCausalLM, PreTrainedModel
from peft import LoraConfig, get_peft_model, TaskType

from utils.enums import (
    ModelName,
    BridgeName,
)
from utils.registry import ModelRegistry
from models.types import ModelT, ModelClassT


@ModelRegistry.register(ModelName.LLAMA32_DECODER)
class Llama32ReportDecoder(nn.Module):
    """Llama-3.2-1B report-generation head attached to the ECG tokenizer.

    Parallel to GPT2Decoder but with LoRA (q_proj + v_proj, r=16, alpha=32,
    matching ECG-Byte's PEFT defaults). Conditioning is done the GPT-2 way:
    a single <ecg> special token whose embedding row is replaced at runtime
    by the SequenceAdapter output, fed through inputs_embeds.

    Trainable parameters:
      - SequenceAdapter (~0.5-1M)
      - LoRA q_proj/v_proj across all 16 layers (~1.7M)
      - The <ecg> row of embed_tokens (2,048)

    Generation uses a manual autoregressive loop because
    `LlamaForCausalLM.generate(inputs_embeds=...)` is broken in transformers>=4.45
    (re-feeds original embeds instead of newly generated tokens).
    """

    def __init__(
        self,
        huggingface_model_name: str = 'meta-llama/Llama-3.2-1B',
        llm_input_embedding_size: int = 2048,
        quantized_feature_shape: Tuple[int, int] = (128, 82),
        adapter_name: BridgeName = BridgeName.LLAMA32_SEQUENCE_BRIDGE,
        adapter_dropout: float = 0.2,
        label_ignore_index: int = -100,
        # LoRA (matches ECG-Byte exactly)
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        lora_target_modules: Tuple[str, ...] = ("q_proj", "v_proj"),
        lora_bias: str = "none",
        # Default generation params
        default_do_sample: bool = True,
        default_top_p: float = 0.92,
        default_temperature: float = 0.85,
    ):
        super().__init__()

        self.label_ignore_index: int = label_ignore_index
        self.default_generation_params: Dict[str, Any] = {
            "do_sample": default_do_sample,
            "top_p": default_top_p,
            "temperature": default_temperature,
        }

        # Adapter
        self.adapter_class: ModelClassT = ModelRegistry.get(adapter_name)
        if self.adapter_class is None:
            raise ValueError(f"Adapter {adapter_name} not found in ModelRegistry")
        self.adapter_name = adapter_name
        self.adapter: ModelT = self.adapter_class(
            input_shape=quantized_feature_shape,
            output_size=llm_input_embedding_size,
            dropout=adapter_dropout,
        )

        # Base Llama model
        base: PreTrainedModel = LlamaForCausalLM.from_pretrained(huggingface_model_name)

        if llm_input_embedding_size != base.config.hidden_size:
            raise ValueError(
                f"Embedding size {llm_input_embedding_size} does not match "
                f"Llama hidden size {base.config.hidden_size}"
            )

        # Add <ecg> special token
        vocab_before: int = base.get_input_embeddings().weight.shape[0]
        base.resize_token_embeddings(vocab_before + 1)
        self.ecg_token_id: int = vocab_before  # the newly added row
        self.eos_token_id: int = int(base.config.eos_token_id) if base.config.eos_token_id is not None else 0

        # Apply LoRA (PEFT auto-freezes all base params)
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=list(lora_target_modules),
            bias=lora_bias,
            task_type=TaskType.CAUSAL_LM,
        )
        self.llm_model: PreTrainedModel = get_peft_model(base, lora_config)

        # Unfreeze only the <ecg> embedding row: turn requires_grad on for the full
        # embedding table, then zero-mask the gradient on every row except <ecg>.
        self._register_ecg_row_grad_mask()

    def _register_ecg_row_grad_mask(self) -> None:
        """Enable training on the <ecg> embedding row only.

        PEFT has frozen the base model including embed_tokens. We re-enable
        requires_grad on the embedding weight and install a gradient hook that
        zeros every row except self.ecg_token_id.
        """
        embed: nn.Embedding = self.llm_model.get_input_embeddings()
        embed.weight.requires_grad = True

        ecg_id = int(self.ecg_token_id)

        def _mask_non_ecg_rows(grad: torch.Tensor) -> torch.Tensor:
            masked = torch.zeros_like(grad)
            masked[ecg_id] = grad[ecg_id]
            return masked

        embed.weight.register_hook(_mask_non_ecg_rows)

    def forward(
        self,
        quantized_features: torch.Tensor,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        **_unused: Any,
    ) -> Dict[str, Any]:
        """Teacher-forced training forward.

        Prepends the <ecg> token to input_ids, replaces its embedding with the
        adapter output, and runs LlamaForCausalLM with labels for CE loss.
        """
        quantized_features = quantized_features.to(dtype=torch.float32)
        ecg_embedding: torch.Tensor = self.adapter(quantized_features)

        batch_size: int = input_ids.size(0)
        device = input_ids.device

        ecg_token = torch.full(
            (batch_size, 1),
            self.ecg_token_id,
            dtype=input_ids.dtype,
            device=device,
        )
        input_ids = torch.cat([ecg_token, input_ids], dim=1)

        if attention_mask is not None:
            ecg_mask = torch.ones(
                (batch_size, 1),
                device=attention_mask.device,
                dtype=attention_mask.dtype,
            )
            attention_mask = torch.cat([ecg_mask, attention_mask], dim=1)
        else:
            attention_mask = torch.ones_like(input_ids)

        if labels is not None:
            label_ignore = torch.full(
                (batch_size, 1),
                self.label_ignore_index,
                dtype=labels.dtype,
                device=labels.device,
            )
            labels = torch.cat([label_ignore, labels], dim=1)

        input_embedding: torch.Tensor = self.llm_model.get_input_embeddings()(input_ids)
        input_embedding[:, 0, :] = ecg_embedding

        outputs = self.llm_model(
            inputs_embeds=input_embedding,
            attention_mask=attention_mask,
            labels=labels,
        )
        return outputs

    @torch.no_grad()
    def generate_report(
        self,
        quantized_features: torch.Tensor,
        max_token_length: int = 512,
        **generate_kwargs: Any,
    ) -> torch.Tensor:
        """Generate a clinical report via manual autoregressive loop.

        Workaround for transformers>=4.45 bug where `LlamaForCausalLM.generate
        (inputs_embeds=...)` re-feeds the original embeds at every step.

        Args:
            quantized_features: ECG features from the tokenizer (B, 128, 82) or
                (B, 1, 128, 82).
            max_token_length: Maximum total tokens to generate (counting the
                first token after the <ecg> seed).
            **generate_kwargs: `do_sample`, `temperature`, `top_p`, `top_k`,
                `eos_token_id`.

        Returns:
            Generated token IDs, shape (B, generated_length).
        """
        if quantized_features.dim() == 3:
            adapter_input = quantized_features.unsqueeze(1)
        else:
            adapter_input = quantized_features

        ecg_embedding: torch.Tensor = self.adapter(adapter_input)
        batch_size: int = ecg_embedding.size(0)
        device = ecg_embedding.device

        params = {**self.default_generation_params, **generate_kwargs}
        do_sample: bool = bool(params.get("do_sample", False))
        temperature: float = float(params.get("temperature", 1.0))
        top_p: float = float(params.get("top_p", 1.0))
        top_k: int = int(params.get("top_k", 0))
        eos_token_id: int = int(params.get("eos_token_id", self.eos_token_id))

        embed_layer: nn.Embedding = self.llm_model.get_input_embeddings()
        ecg_token = torch.full((batch_size, 1), self.ecg_token_id, dtype=torch.long, device=device)
        input_embedding: torch.Tensor = embed_layer(ecg_token)
        input_embedding[:, 0, :] = ecg_embedding

        outputs = self.llm_model(inputs_embeds=input_embedding, use_cache=True)
        past_kv = outputs.past_key_values
        logits = outputs.logits[:, -1, :]
        next_tokens = self._sample_next(logits, do_sample, temperature, top_p, top_k)

        generated = [next_tokens]
        finished = next_tokens.eq(eos_token_id)

        for _ in range(max_token_length - 1):
            if finished.all():
                break
            out = self.llm_model(
                input_ids=next_tokens.unsqueeze(-1),
                past_key_values=past_kv,
                use_cache=True,
            )
            past_kv = out.past_key_values
            logits = out.logits[:, -1, :]
            sampled = self._sample_next(logits, do_sample, temperature, top_p, top_k)
            next_tokens = torch.where(
                finished,
                torch.full_like(sampled, eos_token_id),
                sampled,
            )
            finished = finished | next_tokens.eq(eos_token_id)
            generated.append(next_tokens)

        return torch.stack(generated, dim=1)

    @staticmethod
    def _sample_next(
        logits: torch.Tensor,
        do_sample: bool,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> torch.Tensor:
        if not do_sample:
            return logits.argmax(dim=-1)
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-8)
        if top_k > 0:
            topk_vals, _ = logits.topk(top_k, dim=-1)
            logits = torch.where(
                logits < topk_vals[:, [-1]],
                torch.full_like(logits, float("-inf")),
                logits,
            )
        if 0.0 < top_p < 1.0:
            sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
            cum_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            mask = cum_probs > top_p
            mask[..., 0] = False
            sorted_logits = sorted_logits.masked_fill(mask, float("-inf"))
            logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)
        probs = logits.softmax(dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)
