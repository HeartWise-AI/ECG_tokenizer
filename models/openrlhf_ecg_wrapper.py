"""Wraps ECG_Tokenizer_Wrapper as a HuggingFace-compatible model so
OpenRLHF's PolicyModelActor can load and train it.

STATUS: STUB. This file sketches the API surface OpenRLHF requires; the
real implementation has TODOs at each method. See
`services/README_OPENRLHF.md` for the full plan.

OpenRLHF's actor expects to call:
    model = AutoModelForCausalLM.from_pretrained(model_path)
    model.forward(input_ids, attention_mask, **mm_inputs)
    model.generate(input_ids, attention_mask, ...)
    model.save_pretrained(path)

Our ECG_Tokenizer_Wrapper is a custom nn.Module that does:
    out = wrapper.generate_report_with_question(
        x=ecg_signal, prompt_input_ids=..., prompt_attention_mask=..., ...)

So we need to bridge:
- `from_pretrained` -> load our .pt checkpoint (state_dict + config)
- `forward(input_ids, attention_mask, **mm_inputs)` -> dispatch to the
  MedGemma decoder, inserting pre-encoded ECG soft tokens at the position
  marked by `<image>` placeholder (or equivalent) in the prompt.
- `generate(...)` -> same dispatch, calling MedGemma's generate.
- `save_pretrained` -> dump state_dict + a small config.json.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn

try:
    from transformers import PretrainedConfig, PreTrainedModel
    from transformers.modeling_outputs import CausalLMOutputWithPast
except ImportError:
    raise RuntimeError(
        "transformers >= 4.52 required for OpenRLHF integration") from None


class ECGCausalLMConfig(PretrainedConfig):
    """Minimal config so transformers' from_pretrained machinery is happy.

    Stores the same fields our YAML configs use, so we can reconstruct the
    underlying ECG_Tokenizer_Wrapper from it.
    """
    model_type = "ecg_causal_lm"

    def __init__(
        self,
        # ECG encoder
        num_quantizers: int = 8,
        ecg_codebook_size: int = 512,
        num_codebooks_kept: int = 8,
        ecg_waveform_length: int = 2500,
        ecg_num_leads: int = 12,
        # Bridge / LM
        bridge_name: str = "InstructionAwareECGQFormerBridge",
        num_query_tokens: int = 32,
        bridge_mid_dim: int = 768,
        bridge_qformer_layers: int = 10,
        llm_input_embedding_size: int = 2560,
        huggingface_model_name: str = "google/medgemma-4b-it",
        decoder_name: str = "MedGemma_Decoder",
        # LoRA
        use_lora: bool = True,
        lora_r: int = 32,
        lora_alpha: int = 64,
        lora_dropout: float = 0.05,
        lora_target_modules: Optional[list] = None,
        # Vision config stub so OpenRLHF's VLM detection fires
        vision_config: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_quantizers = num_quantizers
        self.ecg_codebook_size = ecg_codebook_size
        self.num_codebooks_kept = num_codebooks_kept
        self.ecg_waveform_length = ecg_waveform_length
        self.ecg_num_leads = ecg_num_leads
        self.bridge_name = bridge_name
        self.num_query_tokens = num_query_tokens
        self.bridge_mid_dim = bridge_mid_dim
        self.bridge_qformer_layers = bridge_qformer_layers
        self.llm_input_embedding_size = llm_input_embedding_size
        self.huggingface_model_name = huggingface_model_name
        self.decoder_name = decoder_name
        self.use_lora = use_lora
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules or ["q_proj", "k_proj", "v_proj", "o_proj"]
        # Trigger VLM auto-detection in OpenRLHF
        self.vision_config = vision_config or {"hidden_size": llm_input_embedding_size}


class ECGCausalLM(PreTrainedModel):
    """HuggingFace-compatible wrapper around ECG_Tokenizer_Wrapper.

    TODO list to make this fully functional:
    1. `__init__`: instantiate ECG_Tokenizer_Wrapper from config, load its
       state_dict from the .pt checkpoint that lives next to config.json.
    2. `forward(input_ids, attention_mask, ecg_soft_tokens=None, labels=None, ...)`:
       a. If ecg_soft_tokens is None, raise — we must have pre-encoded the
          signal before reaching the actor.
       b. Take the MedGemma input embeddings, splice the ecg_soft_tokens in
          at the position marked by the special image-placeholder token in
          input_ids.
       c. Run the underlying MedGemma forward, return logits + loss.
    3. `generate(...)`: same splicing trick, then delegate to MedGemma's
       generate. vLLM will use this.
    4. `save_pretrained(path)`: dump config.json + best_model.pt; OpenRLHF
       checkpoint resumption uses this.
    5. `from_pretrained(path)`: load config.json + best_model.pt, rebuild
       the underlying wrapper.
    6. Register with transformers' AutoModel registry so
       `AutoModelForImageTextToText.from_pretrained(...)` works:
           from transformers import AutoConfig, AutoModelForImageTextToText
           AutoConfig.register("ecg_causal_lm", ECGCausalLMConfig)
           AutoModelForImageTextToText.register(
               ECGCausalLMConfig, ECGCausalLM)
    """

    config_class = ECGCausalLMConfig

    def __init__(self, config: ECGCausalLMConfig):
        super().__init__(config)
        # TODO: instantiate ECG_Tokenizer_Wrapper here.
        # Lazy import to avoid heavy dependencies at module load time:
        # from models import ECG_Tokenizer_Wrapper
        # self.ecg_wrapper = ECG_Tokenizer_Wrapper(
        #     num_quantizers=config.num_quantizers,
        #     ecg_codebook_size=config.ecg_codebook_size,
        #     ... )
        raise NotImplementedError(
            "ECGCausalLM is a stub. See TODO list in class docstring; full "
            "implementation requires ~150 LOC of glue plus an "
            "`encode_to_soft_tokens(signal)` public method on "
            "ECG_Tokenizer_Wrapper.")

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ecg_soft_tokens: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:
        raise NotImplementedError("see class TODO")

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        ecg_soft_tokens: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        raise NotImplementedError("see class TODO")
