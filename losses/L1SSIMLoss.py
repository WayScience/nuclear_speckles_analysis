from typing import Optional

import torch
from torch import nn

from .l1_ssim import compute_l1_ssim_mean_components


class L1SSIMLoss(nn.Module):
    """Composite training loss combining L1 and MS-SSIM terms."""

    loss_name = "l1_ssim"

    def __init__(self, ssim_weight: float, data_range: Optional[float] = None) -> None:
        """Store MS-SSIM weighting and optional fixed intensity range.

        Args:
            ssim_weight: Multiplier applied to ``-1 * ms_ssim``.
            data_range: Optional fixed data range for MS-SSIM. If omitted, the
                range is derived from the current batch.

        Raises:
            ValueError: If ``ssim_weight`` is negative or ``data_range`` is not
                positive when provided.
        """

        super().__init__()
        if ssim_weight < 0:
            raise ValueError("ssim_weight must be non-negative")
        if data_range is not None and data_range <= 0:
            raise ValueError("data_range must be positive when provided")
        self.ssim_weight = float(ssim_weight)
        self.data_range = None if data_range is None else float(data_range)

    def forward(
        self,
        generated_predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Compute composite L1 and MS-SSIM batch loss for one optimization step.

        Args:
            generated_predictions: Model predictions.
            targets: Ground-truth targets with matching shape.
            **kwargs: Additional unused loss arguments.

        Returns:
            Dictionary containing scalar ``l1``, ``ssim``, and ``total`` loss
            components, where ``ssim`` stores ``-1 * MS-SSIM``.

        Raises:
            ValueError: If prediction and target shapes differ.
        """
        return compute_l1_ssim_mean_components(
            generated_predictions=generated_predictions,
            targets=targets,
            ssim_weight=self.ssim_weight,
            data_range=self.data_range,
        )
