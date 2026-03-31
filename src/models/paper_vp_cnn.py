from __future__ import annotations

import torch
from torch import nn


class PaperVPCNN(nn.Module):
    """CNN based on the ICCP 2017 paper's channel layout.

    The paper specifies filter counts and kernel sizes but does not state
    strides/padding in text. We use stride-2 convolutions to realize the
    required downsampling before the FC layers while keeping the architecture
    faithful to the published channel progression.
    """

    def __init__(self, input_channels: int = 3) -> None:
        super().__init__()
        self.input_bn = nn.BatchNorm2d(input_channels)
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 24, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(24, 36, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(36, 48, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 48, kernel_size=2, stride=2, padding=0),
            nn.ReLU(inplace=True),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.LazyLinear(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(512, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_bn(x)
        x = self.features(x)
        return self.regressor(x)
