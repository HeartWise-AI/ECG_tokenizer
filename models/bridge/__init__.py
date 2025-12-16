"""ECG Bridge modules for connecting ECG representations to language models."""

from .bridge import (
    ECGCodeBridge,
    ECGProjectionBridge,
    PerceiverProjectionBridge,
    ECGQFormerBridge,
    ECGQFormerBridgeStage1,
    InstructionAwareECGQFormerBridge,
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
    "ECGQFormerBridge",
    "ECGQFormerBridgeStage1",
    "InstructionAwareECGQFormerBridge",
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
