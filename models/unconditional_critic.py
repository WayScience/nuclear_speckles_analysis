import torch
from torch import nn


class UnconditionalCritic(nn.Module):
    """Small convolutional critic for single-channel image scoring."""

    def __init__(self, in_channels: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=4, stride=2, padding=1),
            nn.InstanceNorm2d(base_channels * 2, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(
                base_channels * 2,
                base_channels * 4,
                kernel_size=4,
                stride=2,
                padding=1,
            ),
            nn.InstanceNorm2d(base_channels * 4, affine=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_channels * 4, 1, kernel_size=4, stride=1, padding=1),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)
