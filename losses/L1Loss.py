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
        """Return mean L1 loss for backpropagation."""

        if generated_predictions.shape != targets.shape:
            raise ValueError("The generated predictions and targets must be the same shape.")
        return torch.nn.functional.l1_loss(generated_predictions, targets, reduction="mean")
