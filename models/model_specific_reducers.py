import torch.nn as nn
from utils.registry import ModelRegistry

@ModelRegistry.register("BLOOM_SimpleEmbeddingReducer")
class BloomSimpleEmbeddingReducer(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int] = (8, 128, 160),
        output_size: int = 1024,  # BLOOM-560m hidden size
        dropout: float = 0.2
    ):
        super(BloomSimpleEmbeddingReducer, self).__init__()
        # Global average pooling over the height and width dimensions.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Map from the number of channels (first element in input_shape) to the desired output size.
        self.fc = nn.Linear(input_shape[0], output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        x = self.global_pool(x)   # -> shape: (batch, channels, 1, 1)
        x = x.view(x.size(0), -1)   # -> shape: (batch, channels)
        # Map directly to BLOOM's embedding size.
        x = self.fc(x)
        x = self.dropout(x)
        return x  # Output shape: (batch, output_size)


@ModelRegistry.register("OPT_SimpleEmbeddingReducer")
class OPTSimpleEmbeddingReducer(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int] = (8, 128, 160),
        output_size: int = 768,  # OPT-125m hidden size (350m model is broken)
        dropout: float = 0.2
    ):
        super(OPTSimpleEmbeddingReducer, self).__init__()
        # Global average pooling over the height and width dimensions.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Map from the number of channels (first element in input_shape) to the desired output size.
        self.fc = nn.Linear(input_shape[0], output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        x = self.global_pool(x)   # -> shape: (batch, channels, 1, 1)
        x = x.view(x.size(0), -1)   # -> shape: (batch, channels)
        # Map directly to OPT's embedding size.
        x = self.fc(x)
        x = self.dropout(x)
        return x  # Output shape: (batch, output_size)


@ModelRegistry.register("Mistral_SimpleEmbeddingReducer")
class MistralSimpleEmbeddingReducer(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int] = (8, 128, 160),
        output_size: int = 4096,  # Mistral-7B hidden size
        dropout: float = 0.2
    ):
        super(MistralSimpleEmbeddingReducer, self).__init__()
        # Global average pooling over the height and width dimensions.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Map from the number of channels (first element in input_shape) to the desired output size.
        self.fc = nn.Linear(input_shape[0], output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        x = self.global_pool(x)   # -> shape: (batch, channels, 1, 1)
        x = x.view(x.size(0), -1)   # -> shape: (batch, channels)
        # Map directly to Mistral's embedding size.
        x = self.fc(x)
        x = self.dropout(x)
        return x  # Output shape: (batch, output_size)


@ModelRegistry.register("GPTNeo_SimpleEmbeddingReducer")
class GPTNeoSimpleEmbeddingReducer(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int] = (8, 128, 160),
        output_size: int = 768,  # GPT-Neo-125M hidden size
        dropout: float = 0.2
    ):
        super(GPTNeoSimpleEmbeddingReducer, self).__init__()
        # Global average pooling over the height and width dimensions.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Map from the number of channels (first element in input_shape) to the desired output size.
        self.fc = nn.Linear(input_shape[0], output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        x = self.global_pool(x)   # -> shape: (batch, channels, 1, 1)
        x = x.view(x.size(0), -1)   # -> shape: (batch, channels)
        # Map directly to GPT-Neo's embedding size.
        x = self.fc(x)
        x = self.dropout(x)
        return x  # Output shape: (batch, output_size)


@ModelRegistry.register("GPTJ_SimpleEmbeddingReducer")
class GPTJSimpleEmbeddingReducer(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int] = (8, 128, 160),
        output_size: int = 4096,  # GPT-J-6B hidden size
        dropout: float = 0.2
    ):
        super(GPTJSimpleEmbeddingReducer, self).__init__()
        # Global average pooling over the height and width dimensions.
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        # Map from the number of channels (first element in input_shape) to the desired output size.
        self.fc = nn.Linear(input_shape[0], output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: (batch, channels, height, width)
        x = self.global_pool(x)   # -> shape: (batch, channels, 1, 1)
        x = x.view(x.size(0), -1)   # -> shape: (batch, channels)
        # Map directly to GPT-J's embedding size.
        x = self.fc(x)
        x = self.dropout(x)
        return x  # Output shape: (batch, output_size)
