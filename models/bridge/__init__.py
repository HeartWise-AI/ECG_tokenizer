"""ECG Bridge modules for connecting ECG representations to language models."""

from .bridge import (
    ECGCodeBridge,
    ECGProjectionBridge,
    PerceiverProjectionBridge,
    LinearBridge,
    EmbeddingBridge,
    SimpleEmbeddingBridge,
    SequenceBridge,
    SequenceTokenBridge,
    SimpleTokenBridge,
    CrossModalSequenceTokenBridge,
    CrossAttentionLayer,
    calibrate_bridge_scale,
)

__all__ = [
    "ECGCodeBridge",
    "ECGProjectionBridge",
    "PerceiverProjectionBridge",
    "LinearBridge",
    "EmbeddingBridge",
    "SimpleEmbeddingBridge",
    "SequenceBridge",
    "SequenceTokenBridge",
    "SimpleTokenBridge",
    "CrossModalSequenceTokenBridge",
    "CrossAttentionLayer",
    "calibrate_bridge_scale",
]
