import torch
from torch import nn

from trainers.utils.wgan_gp import compute_generator_components


class WassersteinGeneratorCrossZamirskiLoss(nn.Module):
    """Generator loss combining weighted L1 reconstruction and Wasserstein term."""

    loss_name = "wasserstein_generator"  # Stable MLflow namespace for this loss family.

    def __init__(
        self,
        reconstruction_importance: float = 100.0,
        adversarial_importance: float = 1.0,
    ) -> None:
        """Configure weighting for the reconstruction and adversarial components.

        Args:
            reconstruction_importance: Multiplier applied to mean L1 reconstruction loss.
            adversarial_importance: Multiplier applied to mean critic score term.
        """

        super().__init__()
        self.reconstruction_importance = reconstruction_importance
        self.adversarial_importance = adversarial_importance

    def forward(
        self,
        fake_classification_outputs: torch.Tensor,
        generated_predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Compute Zamirski-style generator objective for one batch.

        Args:
            fake_classification_outputs: Critic outputs for generated samples.
            generated_predictions: Generator predictions.
            targets: Ground-truth targets with matching shape.
            **kwargs: Additional unused loss arguments.

        Returns:
            Dictionary with scalar generator loss under ``total`` plus
            reconstruction and adversarial components.

        Raises:
            ValueError: If prediction and target shapes differ.
            ValueError: If critic output batch size does not match predictions batch size.
        """

        components = compute_generator_components(
            fake_classification_outputs=fake_classification_outputs,
            generated_predictions=generated_predictions,
            targets=targets,
        )
        reconstruction_loss = components["reconstruction_term"]
        adversarial_term = components["adversarial_term"]
        total = (
            self.reconstruction_importance * reconstruction_loss
            - self.adversarial_importance * adversarial_term
        )
        return {
            "total": total,
            "reconstruction": reconstruction_loss,
            "adversarial": adversarial_term,
        }
