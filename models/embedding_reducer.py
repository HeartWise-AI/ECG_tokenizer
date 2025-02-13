import torch.nn as nn

class EmbeddingReducer(nn.Module):
    def __init__(
        self, 
        input_shape: tuple[int, int, int] = (8, 128, 160), 
        output_size: int = 768
    ):
        super(EmbeddingReducer, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels=input_shape[0], out_channels=32, kernel_size=3, stride=2, padding=1),  # -> (32, 64, 80)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),  # -> (64, 32, 40)
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # -> (128, 16, 20)
            nn.ReLU()
        )
        self.flatten = nn.Flatten()  # -> 128 * 16 * 20 = 40960
        self.fc_layers = nn.Sequential(
            nn.Linear(128 * 16 * 20, 1024),
            nn.ReLU(),
            nn.Linear(1024, output_size)
        )
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.flatten(x)
        x = self.fc_layers(x)
        return x  # Shape: (batch_size, 768)