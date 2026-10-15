"""Original dense IR autoencoder architecture and initialization."""

import torch
from torch import nn


class DenseAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(1024, 128), nn.ReLU())
        self.decoder = nn.Sequential(nn.Linear(128, 1024), nn.ReLU())
        # Keras Dense defaults: GlorotUniform kernel and zero bias.
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(inputs))
