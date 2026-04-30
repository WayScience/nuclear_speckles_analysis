from typing import Union

import torch

from .AbstractMetric import AbstractMetric


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

        self.total_squared_error = torch.tensor(0.0, device=self.device)
        self.total_elements = torch.tensor(0.0, device=self.device)

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
        self.total_squared_error += sq_error.sum().detach().to(self.device)
        self.total_elements += torch.tensor(
            sq_error.numel(),
            dtype=torch.float32,
            device=self.device,
        )
        return None

    def update(self, generated_predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> None:
        """Alias for state updates to align with TorchMetrics-like API."""

        self.forward(generated_predictions=generated_predictions, targets=targets, **kwargs)

    def compute(self) -> torch.Tensor:
        """Compute averaged L2 value for currently accumulated state.

        Returns:
            Mapping from metric name to scalar value.

        Raises:
            Scalar tensor with current L2 value.
        """

        average_l2 = torch.where(
            self.total_elements > 0,
            self.total_squared_error / self.total_elements,
            torch.tensor(0.0, device=self.device),
        )
        if not torch.isfinite(average_l2):
            average_l2 = torch.tensor(0.0, device=self.device)
        return average_l2

    @property
    def metric_name(self) -> str:
        """Base metric key for logging."""

        return "l2_total"

    def get_metric_data(self) -> dict[str, float]:
        """Backward-compatible helper that computes and resets state."""

        metric_data = {self.metric_name: self.compute().item()}
        self.reset()
        return metric_data
