from typing import Optional, Union

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
        self.data_split_logging: Optional[str] = None

    def forward(
        self,
        generated_predictions: torch.Tensor,
        targets: torch.Tensor,
        data_split_logging: Optional[str] = None,
        **kwargs,
    ) -> None:
        """Accumulate per-sample PSNR values for a split.

        Args:
            generated_predictions: Model predictions.
            targets: Ground-truth targets with matching shape.
            data_split_logging: Split name used in final metric key.
            **kwargs: Additional unused metric arguments.

        Raises:
            ValueError: If shapes mismatch or split name is missing.
        """

        if generated_predictions.shape != targets.shape:
            raise ValueError("The generated predictions and targets must be the same shape.")

        if data_split_logging is None:
            raise ValueError("PSNR is logging-only and requires data_split_logging.")

        self.data_split_logging = data_split_logging
        self.psnr_metric.update(generated_predictions, targets)
        return None

    def get_metric_data(self) -> dict[str, float]:
        """Return averaged PSNR for the accumulated split and reset state.

        Returns:
            Mapping from metric name to scalar value.

        Raises:
            ValueError: If no split data has been accumulated.
        """

        if self.data_split_logging is None:
            raise ValueError("No accumulated split data found for PSNR metric logging.")

        average_psnr = self.psnr_metric.compute().to(self.device)
        if not torch.isfinite(average_psnr):
            average_psnr = torch.tensor(self.nonfinite_cap, device=self.device)
        metric_data = {f"psnr_total_{self.data_split_logging}": average_psnr.item()}
        self.reset()
        return metric_data
