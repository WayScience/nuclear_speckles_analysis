import torch
from torch import nn

from trainers.utils.wgan_gp import compute_generator_components


class WassersteinGeneratorCrossZamirskiLoss(nn.Module):
    """Generator loss combining L1 reconstruction and Wasserstein term."""

    loss_name = "wasserstein_generator"  # Stable MLflow namespace for this loss family.

    def __init__(
        self,
        reconstruction_importance: float = 100.0,
        use_adversarial_decay: bool = True,
    ) -> None:
        """Configure weighting for the reconstruction component.

        Args:
            reconstruction_importance: Multiplier applied to mean L1 reconstruction loss.
        """

        super().__init__()
        self.reconstruction_importance = reconstruction_importance
        self.use_adversarial_decay = use_adversarial_decay

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

        components = compute_generator_components(
            fake_classification_outputs=fake_classification_outputs,
            generated_predictions=generated_predictions,
            targets=targets,
            epoch=epoch,
            use_adversarial_decay=self.use_adversarial_decay,
        )
        reconstruction_loss = components["reconstruction_term"]
        adversarial_term = components["adversarial_term"]
        total = self.reconstruction_importance * reconstruction_loss - adversarial_term
        return {
            "total": total,
            "reconstruction": reconstruction_loss,
            "adversarial": adversarial_term,
        }
