import torch.nn as nn
from typing import Union, Type, Protocol, Any, Iterator, TYPE_CHECKING
from abc import abstractmethod
from transformers import AutoProcessor, AutoTokenizer, PreTrainedTokenizerBase
try:  # transformers < 4.43 fallback
    from transformers import ProcessorMixin
except ImportError:  # pragma: no cover
    ProcessorMixin = PreTrainedTokenizerBase  # type: ignore[misc,assignment]

if TYPE_CHECKING:
    from models import (
        LinearAdapter, 
        EmbeddingAdapter, 
        SequenceAdapter,
        GPT2Decoder,
        SimpleEmbeddingAdapter,
        BertClassifier,
        ECG_Tokenizer_Wrapper,
        Conv_Encoder,
        Conv_Decoder,
        ECG_Tokenizer_Quantizer
    )

class ModelProtocol(Protocol):
    """Protocol defining the interface for all models."""
    
    # Properties that PyTorch modules have
    training: bool
    
    @abstractmethod
    def forward(self, *args, **kwargs) -> Any:
        """Forward pass - must be implemented by all models."""
        ...
    
    def __call__(self, *args, **kwargs) -> Any:
        """Make the model callable (PyTorch convention)."""
        ...
    
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]:
        """Return an iterator over module parameters."""
        ...
    
    def named_parameters(self, prefix: str = '', recurse: bool = True) -> Iterator[tuple[str, nn.Parameter]]:
        """Return an iterator over module parameters, yielding both the name and parameter."""
        ...
    
    def state_dict(self, *args, **kwargs) -> dict[str, Any]:
        """Return a dictionary containing the whole state of the module."""
        ...
    
    def load_state_dict(self, state_dict: dict[str, Any], strict: bool = True) -> Any:
        """Copy parameters and buffers from state_dict into this module and its descendants."""
        ...
    
    def train(self, mode: bool = True) -> "ModelProtocol":
        """Set the module in training mode."""
        ...
    
    def eval(self) -> "ModelProtocol":
        """Set the module in evaluation mode."""
        ...
    
    def to(self, *args, **kwargs) -> "ModelProtocol":
        """Move and/or cast the parameters and buffers."""
        ...
    
    def cuda(self, device: Any = None) -> "ModelProtocol":
        """Move all model parameters and buffers to the GPU."""
        ...
    
    def cpu(self) -> "ModelProtocol":
        """Move all model parameters and buffers to the CPU."""
        ...

# Protocol-based types (for type safety)
ModelT = ModelProtocol
ModelClassT = Type[ModelProtocol]

# Union-based types (for IDE navigation to concrete classes)
ModelUnionT = Union[
    "LinearAdapter", 
    "EmbeddingAdapter", 
    "SequenceAdapter",
    "GPT2Decoder",
    "SimpleEmbeddingAdapter",
    "BertClassifier",
    "ECG_Tokenizer_Wrapper",
    "Conv_Encoder",
    "Conv_Decoder",
    "ECG_Tokenizer_Quantizer"
]
ModelClassUnionT = Type[ModelUnionT]

# Tokenizer types for type safety across different LLM tokenizers
AutoTokenizerT = Union[AutoTokenizer, PreTrainedTokenizerBase]
AutoProcessorT = Union[AutoProcessor, ProcessorMixin]
