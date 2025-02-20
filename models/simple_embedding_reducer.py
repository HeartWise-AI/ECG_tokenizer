import torch.nn as nn
from utils.registry import ModelRegistry

@ModelRegistry.register("GPT2_SimpleEmbeddingReducer")
class SimpleEmbeddingReducer(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int] = (8, 128, 160),
        output_size: int = 768,
        dropout: float = 0.2
    ):
        super(SimpleEmbeddingReducer, self).__init__()
        # Global average pooling over the height and width dimensions.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Map from the number of channels (first element in input_shape) to the desired output size.
        self.fc = nn.Linear(input_shape[0], output_size)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x shape: (batch, channels, height, width)
        x = self.global_pool(x)   # -> shape: (batch, channels, 1, 1)
        x = x.view(x.size(0), -1)   # -> shape: (batch, channels)
        # Map directly to GPT-2's embedding size.
        x = self.fc(x)
        x = self.dropout(x)
        return x  # Output shape: (batch, output_size) 