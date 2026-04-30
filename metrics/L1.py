from typing import Union

import torch

from .AbstractMetric import AbstractMetric


class L1(AbstractMetric):
    """L1 metric/loss with epoch accumulation support."""

    def __init__(
        self,
        use_logits: bool = False,
        device: Union[str, torch.device] = "cuda",
    ):
        """Configure L1 behavior for optimization and/or split-level logging.

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
        """Reset running L1 accumulators used for epoch-level logging."""

        self.total_abs_error = torch.tensor(0.0, device=self.device)
        self.total_elements = torch.tensor(0.0, device=self.device)

    def forward(
        self,
        generated_predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor | None:
        """Compute batch L1 loss or accumulate split statistics.

        Args:
            generated_predictions: Model predictions.
            targets: Ground-truth targets with matching shape.
            **kwargs: Additional unused metric arguments.

        Returns:
            ``None``.

        Raises:
            ValueError: If prediction and target shapes differ.
        """

        if generated_predictions.shape != targets.shape:
            raise ValueError("The generated predictions and targets must be the same shape.")

        abs_error = torch.abs(generated_predictions - targets)
        self.total_abs_error += abs_error.sum().detach().to(self.device)
        self.total_elements += torch.tensor(
            abs_error.numel(),
            dtype=torch.float32,
            device=self.device,
        )
        return None

    def update(self, generated_predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> None:
        """Alias for state updates to align with TorchMetrics-like API."""

        self.forward(generated_predictions=generated_predictions, targets=targets, **kwargs)

    def compute(self) -> torch.Tensor:
        """Compute averaged L1 value for currently accumulated state.

        Returns:
            Mapping from metric name to scalar value.

        Raises:
            Scalar tensor with current L1 value.
        """

        average_l1 = torch.where(
            self.total_elements > 0,
            self.total_abs_error / self.total_elements,
            torch.tensor(0.0, device=self.device),
        )
        if not torch.isfinite(average_l1):
            average_l1 = torch.tensor(0.0, device=self.device)
        return average_l1

    @property
    def metric_name(self) -> str:
        """Base metric key for logging."""

        return "l1_total"

    def get_metric_data(self) -> dict[str, float]:
        """Backward-compatible helper that computes and resets state."""

        metric_data = {self.metric_name: self.compute().item()}
        self.reset()
        return metric_data
