import torch
from torch import nn


class L1Loss(nn.Module):
    """Training loss wrapper with trainer-compatible call signature."""

    loss_name = "l1"  # Stable MLflow metric namespace for this loss family.

    def forward(
        self,
        generated_predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Compute mean L1 training loss for one batch.

        Args:
            generated_predictions: Model predictions.
            targets: Ground-truth targets with matching shape.
            **kwargs: Additional unused loss arguments.

        Returns:
            Dictionary with scalar mean absolute error under ``total``.

        Raises:
            ValueError: If prediction and target shapes differ.
        """

        if generated_predictions.shape != targets.shape:
            raise ValueError("The generated predictions and targets must be the same shape.")
        total = torch.nn.functional.l1_loss(generated_predictions, targets, reduction="mean")
        return {"total": total}
