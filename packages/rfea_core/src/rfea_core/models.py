"""Classifier network for spectrogram inputs."""

import torch
from torch import nn


def _conv_block(in_ch: int, out_ch: int, pool: bool) -> nn.Sequential:
    layers: list[nn.Module] = [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]
    if pool:
        layers.append(nn.MaxPool2d(2))
    return nn.Sequential(*layers)


class SpectrogramCNN(nn.Module):
    """Small CNN: 4 conv blocks (3x3 conv, batch norm, ReLU, max pool) and a linear head.

    Input: (B, 1, 64, 64) spectrogram. Output: raw logits (B, num_classes); softmax
    (optionally with a temperature) is applied outside the model.
    """

    def __init__(self, num_classes: int = 11, in_channels: int = 1, dropout: float = 0.5):
        super().__init__()
        self.features = nn.Sequential(
            _conv_block(in_channels, 32, pool=True),
            _conv_block(32, 64, pool=True),
            _conv_block(64, 128, pool=True),
            _conv_block(128, 128, pool=False),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(128, num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))
