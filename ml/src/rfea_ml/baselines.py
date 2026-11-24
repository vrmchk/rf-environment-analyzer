"""Baseline models for comparison with the literature."""

import torch
from torch import nn


class OSheaCNN(nn.Module):
    """CNN on raw IQ samples after O'Shea, Corgan, Clancy (2016), "Convolutional Radio
    Modulation Recognition Networks" (arXiv:1602.04105), the "CNN2" network.

    Input (B, 2, 128) is treated as a 1-channel 2x128 image:
    zero-pad -> conv 256 x (1x3) -> ReLU -> dropout ->
    zero-pad -> conv 80 x (2x3) -> ReLU -> dropout ->
    dense 256 -> ReLU -> dropout -> dense num_classes (logits).
    """

    def __init__(self, num_classes: int = 11, dropout: float = 0.5, n_samples: int = 128):
        super().__init__()
        width = n_samples + 4  # two pads of 2 on each side, two valid 3-wide convs
        self.net = nn.Sequential(
            nn.ZeroPad2d((2, 2, 0, 0)),
            nn.Conv2d(1, 256, kernel_size=(1, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.ZeroPad2d((2, 2, 0, 0)),
            nn.Conv2d(256, 80, kernel_size=(2, 3)),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Flatten(),
            nn.Linear(80 * width, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )
        # Initialization of the original Keras code: Glorot-uniform convolutions,
        # He-normal dense layers, zero biases. PyTorch defaults leave the network stuck
        # at chance level, because RadioML samples are small (RMS ~ 0.007).
        for m in self.net:
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, iq: torch.Tensor) -> torch.Tensor:
        return self.net(iq.unsqueeze(1))
