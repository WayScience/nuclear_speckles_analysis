import torch
from torch import nn

from .gamma_weighted_l1 import compute_gamma_weighted_l1_mean_components


class GammaWeightedL1Loss(nn.Module):
    """Training loss using gamma-weighted absolute error."""

    loss_name = "gamma_weighted_l1"

    def __init__(self, gamma: float, epsilon: float = 1e-8) -> None:
        """Store gamma and epsilon for the training objective."""

        super().__init__()
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")
        self.gamma = float(gamma)
        self.epsilon = float(epsilon)

    def forward(
        self,
        generated_predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> dict[str, torch.Tensor]:
        """Compute the mean gamma-weighted absolute-error training loss."""

        return compute_gamma_weighted_l1_mean_components(
            generated_predictions=generated_predictions,
            targets=targets,
            gamma=self.gamma,
            epsilon=self.epsilon,
        )
