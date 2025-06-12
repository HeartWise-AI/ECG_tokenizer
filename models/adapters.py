import torch.nn as nn
from utils.registry import ModelRegistry
from utils.enums import AdapterName

@ModelRegistry.register(AdapterName.GPT2_LINEAR_ADAPTER)
class LinearAdapter(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int] = (8, 128, 160),
        output_size: int = 768,
        dropout: float = 0.0  # Set dropout > 0 to enable dropout regularization
    ):
        super(LinearAdapter, self).__init__()
        # Calculate the flattened input size (e.g. 8 * 128 * 160 = 163840)
        self.flatten_dim: int = input_shape[0] * input_shape[1] * input_shape[2]
        
        self.adapter: nn.Sequential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(1024, output_size)
        )
    
    def forward(self, x):
        return self.adapter(x) 
    
@ModelRegistry.register(AdapterName.GPT2_EMBEDDING_ADAPTER)
class EmbeddingAdapter(nn.Module):
    def __init__(
        self, 
        input_shape: tuple[int, int, int] = (8, 128, 160), 
        output_size: int = 768,
        dropout: float = 0.2
    ):
        super(EmbeddingAdapter, self).__init__()
        self.input_shape: tuple[int, int, int] = input_shape
        self.output_size: int = output_size
        self.conv_layers: nn.Sequential = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape[0], 
                out_channels=32, 
                kernel_size=3, 
                stride=2, 
                padding=1
            ),  # -> (32, 64, 80)
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32, 
                out_channels=64, 
                kernel_size=3, 
                stride=2, 
                padding=1
            ),  # -> (64, 32, 40)
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64, 
                out_channels=128, 
                kernel_size=3, 
                stride=2, 
                padding=1
            ),  # -> (128, 16, 20)
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()  # -> 128 x 1 x 1 = 128
        self.fc_layers = nn.Sequential(
            nn.Linear(128, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(1024, output_size)
        )
    
    def forward(self, x):
        x = x / 1024.0
        x = self.conv_layers(x)
        x = self.avgpool(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x  # Shape: (batch_size, 768)
    
import torch.nn as nn
from utils.registry import ModelRegistry

@ModelRegistry.register(AdapterName.GPT2_SIMPLE_EMBEDDING_ADAPTER)
class SimpleEmbeddingAdapter(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int] = (8, 128, 160),
        output_size: int = 768,
        dropout: float = 0.2
    ):
        super(SimpleEmbeddingAdapter, self).__init__()
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