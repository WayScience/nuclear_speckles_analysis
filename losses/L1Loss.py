import torch
from torch import nn


class L1Loss(nn.Module):
    """Training loss wrapper with trainer-compatible call signature."""

    def forward(
        self,
        generated_predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute mean L1 training loss for one batch.

        Args:
            generated_predictions: Model predictions.
            targets: Ground-truth targets with matching shape.
            **kwargs: Additional unused loss arguments.

        Returns:
            Scalar mean absolute error used for optimization.

        Raises:
            ValueError: If prediction and target shapes differ.
        """

        if generated_predictions.shape != targets.shape:
            raise ValueError("The generated predictions and targets must be the same shape.")
        return torch.nn.functional.l1_loss(generated_predictions, targets, reduction="mean")
