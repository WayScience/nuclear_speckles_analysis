import torch
from torch import nn


class WassersteinGeneratorCrossZamirskiLoss(nn.Module):
    """Generator loss combining L1 reconstruction and Wasserstein term."""

    def __init__(self, reconstruction_importance: float = 100.0) -> None:
        super().__init__()
        self.reconstruction_importance = reconstruction_importance

    def forward(
        self,
        fake_classification_outputs: torch.Tensor,
        generated_predictions: torch.Tensor,
        targets: torch.Tensor,
        epoch: int = 0,
        **kwargs,
    ) -> torch.Tensor:
        """Return Zamirski-style generator loss with epoch-weighted critic term."""

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
        return self.reconstruction_importance * reconstruction_loss - adversarial_term
