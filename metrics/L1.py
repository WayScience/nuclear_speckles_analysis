from typing import Optional, Union

import torch

from .AbstractMetric import AbstractMetric


class L1(AbstractMetric):
    """L1 metric/loss with epoch accumulation support."""

    def __init__(
        self,
        is_loss: bool = False,
        use_logits: bool = False,
        device: Union[str, torch.device] = "cuda",
    ):
        """Configure L1 behavior for optimization and/or split-level logging.

        Args:
            is_loss: Whether this instance is used directly as optimization loss.
            use_logits: Whether caller should provide logits instead of postprocessed outputs.
            device: Device for accumulation buffers.
        """

        super().__init__()
        self.is_loss = is_loss
        self.use_logits = use_logits
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.reset()

    def reset(self):
        """Reset running L1 accumulators used for epoch-level logging."""

        self.total_abs_error = torch.tensor(0.0, device=self.device)
        self.total_elements = torch.tensor(0.0, device=self.device)
        self.data_split_logging: Optional[str] = None

    def forward(
        self,
        generated_predictions: torch.Tensor,
        targets: torch.Tensor,
        data_split_logging: Optional[str] = None,
        **kwargs,
    ) -> torch.Tensor | None:
        """Compute batch L1 loss or accumulate split statistics.

        Args:
            generated_predictions: Model predictions.
            targets: Ground-truth targets with matching shape.
            data_split_logging: Split name used for deferred metric logging.
            **kwargs: Additional unused metric arguments.

        Returns:
            Mean L1 loss when used as optimization loss, else ``None``.

        Raises:
            ValueError: If prediction and target shapes differ, or split name is missing
                for logging-only usage.
        """

        if generated_predictions.shape != targets.shape:
            raise ValueError("The generated predictions and targets must be the same shape.")

        abs_error = torch.abs(generated_predictions - targets)

        if data_split_logging is None:
            if not self.is_loss:
                raise ValueError(
                    "If the metric is not a loss, then it must be used for logging."
                )
            return abs_error.mean()

        self.data_split_logging = data_split_logging
        self.total_abs_error += abs_error.sum().detach().to(self.device)
        self.total_elements += torch.tensor(
            abs_error.numel(),
            dtype=torch.float32,
            device=self.device,
        )
        return None

    def get_metric_data(self) -> dict[str, float]:
        """Return averaged L1 value for the accumulated split and reset state.

        Returns:
            Mapping from metric name to scalar value.

        Raises:
            ValueError: If no split data has been accumulated.
        """

        if self.data_split_logging is None:
            raise ValueError("No accumulated split data found for L1 metric logging.")

        average_l1 = torch.where(
            self.total_elements > 0,
            self.total_abs_error / self.total_elements,
            torch.tensor(0.0, device=self.device),
        )
        if not torch.isfinite(average_l1):
            average_l1 = torch.tensor(0.0, device=self.device)

        if self.is_loss:
            metric_key = f"l1_total_loss_{self.data_split_logging}"
        else:
            metric_key = f"l1_total_{self.data_split_logging}"

        metric_data = {metric_key: average_l1.item()}
        self.reset()
        return metric_data
