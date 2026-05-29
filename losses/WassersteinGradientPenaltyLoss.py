import torch
from torch import nn


class WassersteinGradientPenaltyLoss(nn.Module):
    """WGAN-GP loss wrapper with trainer-compatible call signature."""

    def __init__(self, gradient_penalty_importance: float = 10.0) -> None:
        super().__init__()
        self.gradient_penalty_importance = gradient_penalty_importance

    def forward(
        self,
        gradients: torch.Tensor,
        real_classification_outputs: torch.Tensor,
        fake_classification_outputs: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Return Wasserstein loss with gradient penalty term."""

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
