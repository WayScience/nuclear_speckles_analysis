import torch
from torch import nn

from trainers.utils.wgan_gp import compute_wgan_gp_terms


class WassersteinGradientPenaltyLoss(nn.Module):
    """WGAN-GP loss wrapper with trainer-compatible call signature."""

    loss_name = "wasserstein_gp"  # Stable MLflow namespace for this loss family.

    def __init__(self, gradient_penalty_importance: float = 10.0) -> None:
        """Configure weighting for the gradient penalty term.

        Args:
            gradient_penalty_importance: Multiplier applied to gradient penalty.
        """

        super().__init__()
        self.gradient_penalty_importance = gradient_penalty_importance

    def forward(
        self,
        critic: nn.Module,
        real_samples: torch.Tensor,
        fake_samples: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Compute critic loss with Wasserstein distance and gradient penalty.

        Args:
            critic: Critic network used to score real/fake samples.
            real_samples: Real target samples.
            fake_samples: Generated samples.
            **kwargs: Additional unused loss arguments.

        Returns:
            Dictionary with scalar critic loss under ``total`` plus
            Wasserstein and gradient-penalty components.

        Raises:
            ValueError: If real critic output batch size does not match gradients batch size.
            ValueError: If fake critic output batch size does not match gradients batch size.
        """

        components = compute_wgan_gp_terms(
            critic=critic,
            real_samples=real_samples,
            fake_samples=fake_samples,
        )
        gradient_penalty = components["gradient_penalty_unweighted"]
        wasserstein = components["wasserstein_term"]
        total = wasserstein + gradient_penalty * self.gradient_penalty_importance
        return {
            "total": total,
            "wasserstein": wasserstein,
            "gradient_penalty": gradient_penalty,
        }
