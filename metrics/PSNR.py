from typing import Union

import torch
from torchmetrics.image import PeakSignalNoiseRatio

from .AbstractMetric import AbstractMetric


class PSNR(AbstractMetric):
    """PSNR metric with epoch accumulation support."""

    def __init__(
        self,
        max_pixel_value: float = 1.0,
        nonfinite_cap: float = 100.0,
        use_logits: bool = False,
        device: Union[str, torch.device] = "cuda",
    ):
        """Configure PSNR accumulation settings.

        Args:
            max_pixel_value: Peak pixel value used in PSNR calculation.
            nonfinite_cap: Finite fallback logged when PSNR is non-finite.
            use_logits: Whether caller should provide logits instead of postprocessed outputs.
            device: Device for accumulation buffers.
        """

        super().__init__()
        self.max_pixel_value = max_pixel_value
        self.nonfinite_cap = nonfinite_cap
        self.use_logits = use_logits
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.psnr_metric = PeakSignalNoiseRatio(
            data_range=max_pixel_value,
            reduction="elementwise_mean",
            dim=(1, 2, 3),
        ).to(self.device)
        self.reset()

    def reset(self):
        """Reset running PSNR accumulators."""

        self.psnr_metric.reset()

    def forward(
        self,
        generated_predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> None:
        """Accumulate per-sample PSNR values for a split.

        Args:
            generated_predictions: Model predictions.
            targets: Ground-truth targets with matching shape.
            **kwargs: Additional unused metric arguments.

        Raises:
            ValueError: If shapes mismatch.
        """

        if generated_predictions.shape != targets.shape:
            raise ValueError("The generated predictions and targets must be the same shape.")

        self.psnr_metric.update(generated_predictions, targets)
        return None

    def update(self, generated_predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> None:
        """Alias for state updates to align with TorchMetrics-like API."""

        self.forward(generated_predictions=generated_predictions, targets=targets, **kwargs)

    def compute(self) -> torch.Tensor:
        """Compute averaged PSNR for currently accumulated state.

        Returns:
            Scalar tensor with current PSNR value.
        """

        average_psnr = self.psnr_metric.compute().to(self.device)
        if not torch.isfinite(average_psnr):
            average_psnr = torch.tensor(self.nonfinite_cap, device=self.device)
        return average_psnr

    @property
    def metric_name(self) -> str:
        """Base metric key for logging."""

        return "psnr_total"

    def get_metric_data(self) -> dict[str, float]:
        """Backward-compatible helper that computes and resets state."""

        metric_data = {self.metric_name: self.compute().item()}
        self.reset()
        return metric_data
