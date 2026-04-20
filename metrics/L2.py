from typing import Optional, Union

import torch

from .AbstractMetric import AbstractMetric


class L2(AbstractMetric):
    """L2 (MSE) metric with epoch accumulation support."""

    def __init__(
        self,
        use_logits: bool = False,
        device: Union[str, torch.device] = "cuda",
    ):
        super().__init__()
        self.use_logits = use_logits
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.reset()

    def reset(self):
        self.total_squared_error = torch.tensor(0.0, device=self.device)
        self.total_elements = torch.tensor(0.0, device=self.device)
        self.data_split_logging: Optional[str] = None

    def forward(
        self,
        generated_predictions: torch.Tensor,
        targets: torch.Tensor,
        data_split_logging: Optional[str] = None,
        **kwargs,
    ) -> None:
        if generated_predictions.shape != targets.shape:
            raise ValueError("The generated predictions and targets must be the same shape.")

        if data_split_logging is None:
            raise ValueError("L2 is logging-only and requires data_split_logging.")

        self.data_split_logging = data_split_logging
        sq_error = (generated_predictions - targets) ** 2
        self.total_squared_error += sq_error.sum().detach().to(self.device)
        self.total_elements += torch.tensor(
            sq_error.numel(),
            dtype=torch.float32,
            device=self.device,
        )
        return None

    def get_metric_data(self) -> dict[str, float]:
        if self.data_split_logging is None:
            raise ValueError("No accumulated split data found for L2 metric logging.")

        average_l2 = torch.where(
            self.total_elements > 0,
            self.total_squared_error / self.total_elements,
            torch.tensor(0.0, device=self.device),
        )
        metric_data = {f"l2_total_{self.data_split_logging}": average_l2.item()}
        self.reset()
        return metric_data
