import torch
from torch import nn


class WassersteinGradientPenaltyLoss(nn.Module):
    """WGAN-GP loss wrapper with trainer-compatible call signature."""

    def __init__(self, gradient_penalty_importance: float = 10.0) -> None:
        """Configure weighting for the gradient penalty term.

        Args:
            gradient_penalty_importance: Multiplier applied to gradient penalty.
        """

        super().__init__()
        self.gradient_penalty_importance = gradient_penalty_importance

    def forward(
        self,
        gradients: torch.Tensor,
        real_classification_outputs: torch.Tensor,
        fake_classification_outputs: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Compute critic loss with Wasserstein distance and gradient penalty.

        Args:
            gradients: Gradients of critic outputs w.r.t. interpolated inputs.
            real_classification_outputs: Critic outputs for real samples.
            fake_classification_outputs: Critic outputs for generated samples.
            **kwargs: Additional unused loss arguments.

        Returns:
            Scalar critic loss equal to
            ``mean(fake_classification_outputs) - mean(real_classification_outputs) + gradient_penalty_importance * penalty``.

        Raises:
            ValueError: If real critic output batch size does not match gradients batch size.
            ValueError: If fake critic output batch size does not match gradients batch size.
        """

        batch_size = gradients.size(0)
        if real_classification_outputs.size(0) != batch_size:
            raise ValueError("real_classification_outputs batch size must match gradients.")
        if fake_classification_outputs.size(0) != batch_size:
            raise ValueError("fake_classification_outputs batch size must match gradients.")

        gradients = gradients.view(batch_size, -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return (
            torch.mean(fake_classification_outputs)
            - torch.mean(real_classification_outputs)
            + gradient_penalty * self.gradient_penalty_importance
        )
