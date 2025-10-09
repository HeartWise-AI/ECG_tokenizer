"""ECG Adapter modules for connecting ECG representations to language models."""

from .bridge import ECGCodeBridge, ECGProjectionBridge, PerceiverProjectionBridge
from .adapter import (
    LinearAdapter,
    EmbeddingAdapter,
    SimpleEmbeddingAdapter,
    SequenceAdapter,
    SequenceTokenAdapter,
    SimpleTokenAdapter,
)
from .cross_attention import (
    CrossModalSequenceTokenAdapter,
    CrossAttentionLayer,
)

__all__ = [
    # Bridge modules
    "ECGCodeBridge",
    "ECGProjectionBridge",
    "PerceiverProjectionBridge",
    # Basic adapters
    "LinearAdapter",
    "EmbeddingAdapter",
    "SimpleEmbeddingAdapter",
    "SequenceAdapter",
    "SequenceTokenAdapter",
    "SimpleTokenAdapter",
    # Cross-attention modules
    "CrossModalSequenceTokenAdapter",
    "CrossAttentionLayer",
]