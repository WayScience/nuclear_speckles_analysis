"""Global image critic used for adversarial training.

This module defines a compact convolutional discriminator (critic) suitable for
Wasserstein-style GAN training (for example WGAN-GP). The network downsamples
the full input image through strided convolutions, aggregates global context
with adaptive average pooling, and produces one unconstrained scalar score per
sample.

Notes:
    - The output is a critic score, not a probability.
    - No sigmoid activation is applied at the end.
    - Inputs are expected in ``(N, C, H, W)`` format.
"""

import torch
from torch import nn

class GlobalDiscriminator(nn.Module):
    """Convolutional global critic for image-level real/fake scoring.

    Architecture:
        1. Four ``Conv2d + LeakyReLU`` blocks with stride 2 for progressive
           spatial downsampling.
        2. ``AdaptiveAvgPool2d(1)`` to collect global features independent of
           input spatial size.
        3. Linear projection to a single scalar critic score per image.

    Args:
        in_channels: Number of channels in the input image tensor.
        base_channels: Number of feature channels in the first convolutional
            block. Later blocks scale this as ``x2``, ``x4``, and ``x8``.
    """

    def __init__(self, in_channels: int = 1, base_channels: int = 64):
        """Initialize the global discriminator network."""
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels, base_channels * 2, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 2, base_channels * 4, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(base_channels * 4, base_channels * 8, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels * 8, 1),
        )

    def forward(self, x):
        """Compute critic scores for a batch of images.

        Args:
            x: Input tensor of shape ``(batch, in_channels, height, width)``.

        Returns:
            Tensor of shape ``(batch, 1)`` with unconstrained critic scores.
            Larger values indicate samples judged as more real by the critic.
        """

        return self.classifier(self.features(x))
