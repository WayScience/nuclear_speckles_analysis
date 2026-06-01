from typing import Union

import torch

from .AbstractMetric import AbstractMetric
from .utils.streaming_stats import StreamingScalarStats


class L2(AbstractMetric):
    """L2 (MSE) metric with epoch accumulation support."""

    def __init__(
        self,
        use_logits: bool = False,
        device: Union[str, torch.device] = "cuda",
    ):
        """Configure L2 (MSE) metric accumulation.

        Args:
            use_logits: Whether caller should provide logits instead of postprocessed outputs.
            device: Device for accumulation buffers.
        """

        super().__init__()
        self.use_logits = use_logits
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.reset()

    def reset(self):
        """Reset running squared-error accumulators."""

        self.stats = StreamingScalarStats(device=self.device)

    def forward(
        self,
        generated_predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> None:
        """Accumulate batch L2 statistics for split-level logging.

        Args:
            generated_predictions: Model predictions.
            targets: Ground-truth targets with matching shape.
            **kwargs: Additional unused metric arguments.

        Raises:
            ValueError: If shapes mismatch.
        """

        if generated_predictions.shape != targets.shape:
            raise ValueError("The generated predictions and targets must be the same shape.")

        sq_error = (generated_predictions - targets) ** 2
        sq_error = sq_error.reshape(sq_error.shape[0], -1)
        per_sample_l2 = sq_error.mean(dim=1)

        self.stats.update(per_sample_l2)
        return None

    def update(self, generated_predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> None:
        """Alias for state updates to align with TorchMetrics-like API."""

        self.forward(generated_predictions=generated_predictions, targets=targets, **kwargs)

    def compute(self) -> dict[str, float]:
        """Compute averaged L2 and population std for current state.

        Returns:
            Dictionary containing mean and std metric values.
        """

        stats = self.stats.compute()

        return {
            self.metric_name: stats["mean"],
            f"{self.metric_name}_std": stats["std"],
        }

    @property
    def metric_name(self) -> str:
        """Base metric key for logging."""

        return "l2_total"

    def get_metric_data(self) -> dict[str, float]:
        """Compute metric stats and reset state."""

        metric_data = self.compute()
        self.reset()
        return metric_data
