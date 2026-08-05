from typing import Union

import torch

from losses.gamma_weighted_l1 import compute_gamma_weighted_l1_per_sample

from .AbstractMetric import AbstractMetric
from .utils.streaming_stats import StreamingScalarStats


class GammaWeightedL1LossMetric(AbstractMetric):
    """Accumulate gamma-weighted absolute-error loss for evaluation."""

    def __init__(
        self,
        gamma: float,
        epsilon: float = 1e-8,
        use_logits: bool = True,
        device: Union[str, torch.device] = "cuda",
    ):
        """Configure epoch-level accumulation of the gamma-weighted loss."""

        super().__init__()
        if epsilon <= 0:
            raise ValueError("epsilon must be positive")

        self.gamma = float(gamma)
        self.epsilon = float(epsilon)
        self.use_logits = use_logits
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.reset()

    def reset(self):
        """Reset running loss accumulators used for epoch-level logging."""

        self.total_stats = StreamingScalarStats(device=self.device)

    def forward(
        self,
        generated_predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> None:
        """Accumulate per-sample gamma-weighted loss statistics."""

        per_sample_total = compute_gamma_weighted_l1_per_sample(
            generated_predictions=generated_predictions,
            targets=targets,
            gamma=self.gamma,
            epsilon=self.epsilon,
        )
        self.total_stats.update(per_sample_total)
        return None

    def update(self, generated_predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> None:
        """Alias for state updates to align with TorchMetrics-like API."""

        self.forward(generated_predictions=generated_predictions, targets=targets, **kwargs)

    def compute(self) -> dict[str, float]:
        """Compute averaged gamma-weighted loss values and standard deviation."""

        total_stats = self.total_stats.compute()
        return {
            self.metric_name: total_stats["mean"],
            f"{self.metric_name}_std": total_stats["std"],
        }

    @property
    def metric_name(self) -> str:
        """Base metric key used for model selection and logging."""

        return "loss_total"

    def get_metric_data(self) -> dict[str, float]:
        """Compute accumulated loss stats and reset state."""

        metric_data = self.compute()
        self.reset()
        return metric_data
