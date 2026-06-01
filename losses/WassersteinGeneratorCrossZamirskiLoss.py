import torch
from torch import nn


class WassersteinGeneratorCrossZamirskiLoss(nn.Module):
    """Generator loss combining L1 reconstruction and Wasserstein term."""

    def __init__(self, reconstruction_importance: float = 100.0) -> None:
        """Configure weighting for the reconstruction component.

        Args:
            reconstruction_importance: Multiplier applied to mean L1 reconstruction loss.
        """

        super().__init__()
        self.reconstruction_importance = reconstruction_importance

    def forward(
        self,
        fake_classification_outputs: torch.Tensor,
        generated_predictions: torch.Tensor,
        targets: torch.Tensor,
        epoch: int = 0,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Compute Zamirski-style generator objective for one batch.

        Args:
            fake_classification_outputs: Critic outputs for generated samples.
            generated_predictions: Generator predictions.
            targets: Ground-truth targets with matching shape.
            epoch: Zero-based epoch index used to down-weight adversarial term over time.
            **kwargs: Additional unused loss arguments.

        Returns:
            Dictionary with scalar generator loss under ``total`` plus
            reconstruction and adversarial components.

        Raises:
            ValueError: If prediction and target shapes differ.
            ValueError: If critic output batch size does not match predictions batch size.
        """

        if generated_predictions.shape != targets.shape:
            raise ValueError("generated_predictions and targets must have the same shape.")

        batch_size = generated_predictions.size(0)
        if fake_classification_outputs.size(0) != batch_size:
            raise ValueError(
                "fake_classification_outputs batch size must match generated_predictions."
            )

        reconstruction_loss = torch.nn.functional.l1_loss(
            generated_predictions, targets, reduction="mean"
        )
        adversarial_term = torch.mean(fake_classification_outputs) / (epoch + 1)
        total = self.reconstruction_importance * reconstruction_loss - adversarial_term
        return {
            "total": total,
            "reconstruction": reconstruction_loss,
            "adversarial": adversarial_term,
        }
