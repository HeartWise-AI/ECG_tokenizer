"""ECG Decoder modules for language model integration."""

from .gpt2_decoder import GPT2Decoder
# NOTE: main's Llama32Decoder is intentionally NOT imported here. PR #120's
# Llama32ReportDecoder (models/llama32_report_decoder.py) owns
# ModelName.LLAMA32_DECODER; importing llama32_decoder too would run its
# @ModelRegistry.register decorator and raise a duplicate-registration error.
from .medgemma_decoder import MedGemmaDecoder
from .qwen_decoder import QwenDecoder

__all__ = [
    "GPT2Decoder",
    "MedGemmaDecoder",
    "QwenDecoder",
]
