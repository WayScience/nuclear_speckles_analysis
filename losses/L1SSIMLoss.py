from typing import Optional

import torch
from torch import nn
from torchmetrics.functional.image import structural_similarity_index_measure


class L1SSIMLoss(nn.Module):
    """Composite training loss combining L1 and SSIM terms."""

    loss_name = "l1_ssim"

    def __init__(self, ssim_weight: float, data_range: Optional[float] = None) -> None:
        """Store SSIM weighting and optional fixed intensity range.

        Args:
            ssim_weight: Multiplier applied to ``1 - ssim``.
            data_range: Optional fixed data range for SSIM. If omitted, the
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
        """Compute composite L1 and SSIM batch loss for one optimization step.

        Args:
            generated_predictions: Model predictions.
            targets: Ground-truth targets with matching shape.
            **kwargs: Additional unused loss arguments.

        Returns:
            Dictionary containing scalar ``l1``, ``ssim``, and ``total`` loss
            components, where ``ssim`` stores ``1 - SSIM``.

        Raises:
            ValueError: If prediction and target shapes differ.
        """

        if generated_predictions.shape != targets.shape:
            raise ValueError("The generated predictions and targets must be the same shape.")

        l1 = torch.nn.functional.l1_loss(generated_predictions, targets, reduction="mean")
        if self.data_range is None:
            # Derive a positive SSIM range from the current batch so the loss can
            # operate directly in z-score space without a fixed intensity bound.
            batch_max = torch.maximum(generated_predictions.max(), targets.max())
            batch_min = torch.minimum(generated_predictions.min(), targets.min())
            ssim_data_range = (batch_max - batch_min).clamp_min(torch.finfo(torch.float32).eps)
            ssim_data_range = ssim_data_range.to(generated_predictions.dtype)
        else:
            ssim_data_range = self.data_range

        ssim = structural_similarity_index_measure(
            preds=generated_predictions,
            target=targets,
            data_range=ssim_data_range,
        )
        ssim_loss = 1.0 - ssim
        total = l1 + self.ssim_weight * ssim_loss
        return {"l1": l1, "ssim": ssim_loss, "total": total}
