from models.bridge import (
    ECGCodeBridge,
    ECGProjectionBridge,
    PerceiverProjectionBridge,
    ECGQFormerBridge,
    LinearBridge,
    EmbeddingBridge,
    SimpleEmbeddingBridge,
    SequenceBridge,
    SequenceTokenBridge,
    SimpleTokenBridge,
    CrossModalSequenceTokenBridge,
    CrossAttentionLayer,
)
from models.bert_classifier import BertClassifier
from models.decoder import (
    GPT2Decoder,
    MedGemmaDecoder,
    QwenDecoder,
)
# PR #120 report-generation head supersedes main's Llama32Decoder for
# ModelName.LLAMA32_DECODER (see models/decoder/__init__.py where the old
# registration is dropped to avoid a duplicate-registration ValueError).
from models.llama32_report_decoder import Llama32ReportDecoder
from models.ecg_tokenizer_wrapper import (
    ECG_Tokenizer_Wrapper,
    Conv_Encoder,
    Conv_Decoder,
    ECG_Tokenizer_Quantizer,
    ECG_Tokenizer_Quantizer_RVQ
)
from models.local_residual_vq import ResidualVQ
from models.text_encoder import TextEncoder

__all__ = [
    "ECGCodeBridge",
    "ECGProjectionBridge",
    "PerceiverProjectionBridge",
    "ECGQFormerBridge",
    "LinearBridge",
    "EmbeddingBridge",
    "SimpleEmbeddingBridge",
    "SequenceBridge",
    "SequenceTokenBridge",
    "SimpleTokenBridge",
    "CrossModalSequenceTokenBridge",
    # Cross-attention
    "CrossAttentionLayer",
    # Decoders
    "GPT2Decoder",
    "Llama32ReportDecoder",
    "MedGemmaDecoder",
    "QwenDecoder",
    # Other models
    "BertClassifier",
    "ECG_Tokenizer_Wrapper",
    "Conv_Encoder",
    "Conv_Decoder",
    "ECG_Tokenizer_Quantizer",
    "ECG_Tokenizer_Quantizer_RVQ",
    "ResidualVQ",
    "TextEncoder",
]
