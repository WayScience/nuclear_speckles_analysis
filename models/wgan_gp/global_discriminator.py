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
        num_blocks: Number of strided convolutional blocks used for
            downsampling.
        max_channels: Optional upper bound for feature channel width in deeper
            blocks. If ``None``, channels are uncapped.
    """

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 64,
        num_blocks: int = 4,
        max_channels: int | None = None,
    ):
        """Initialize the global discriminator network.

        Args:
            in_channels: Number of channels in each input image.
            base_channels: Channel width used by the first convolutional block.
            num_blocks: Number of stride-2 convolutional feature blocks.
            max_channels: Optional cap on block output channels.
        """
        super().__init__()

        if num_blocks < 1:
            raise ValueError(f"Expected num_blocks >= 1, got {num_blocks}.")
        if base_channels < 1:
            raise ValueError(f"Expected base_channels >= 1, got {base_channels}.")
        if max_channels is not None and max_channels < 1:
            raise ValueError(f"Expected max_channels >= 1, got {max_channels}.")

        feature_blocks: list[nn.Module] = []
        in_ch = in_channels
        out_ch = base_channels

        for block_idx in range(num_blocks):
            if block_idx > 0:
                out_ch = base_channels * (2 ** block_idx)
            if max_channels is not None:
                out_ch = min(out_ch, max_channels)

            feature_blocks.extend(
                [
                    nn.Conv2d(in_ch, out_ch, 4, 2, 1),
                    nn.LeakyReLU(0.2),
                ]
            )
            in_ch = out_ch

        self.features = nn.Sequential(*feature_blocks)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_ch, 1),
        )

    def forward(self, x):
        """Compute critic scores for a batch of images.

        Args:
            x: Input tensor of shape ``(batch, in_channels, height, width)``.

        Returns:
            Tensor of shape ``(batch,)`` with unconstrained critic scores.
            Larger values indicate samples judged as more real by the critic.
        """

        return self.classifier(self.features(x)).squeeze(-1)
