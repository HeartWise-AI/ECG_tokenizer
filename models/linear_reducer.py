import torch.nn as nn
from utils.registry import ModelRegistry

@ModelRegistry.register("GPT2_LinearReducer")
class LinearReducer(nn.Module):
    def __init__(
        self,
        input_shape: tuple[int, int, int] = (8, 128, 160),
        output_size: int = 768,
        dropout: float = 0.0  # Set dropout > 0 to enable dropout regularization
    ):
        super(LinearReducer, self).__init__()
        # Calculate the flattened input size (e.g. 8 * 128 * 160 = 163840)
        self.flatten_dim: int = input_shape[0] * input_shape[1] * input_shape[2]
        
        self.reducer: nn.Sequential = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flatten_dim, 1024),
            nn.ReLU(),
            nn.Dropout(p=dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(1024, output_size)
        )
    
    def forward(self, x):
        return self.reducer(x) 