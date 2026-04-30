from typing import Union

import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

from .AbstractMetric import AbstractMetric


class SSIM(AbstractMetric):
    """SSIM metric with epoch accumulation support."""

    def __init__(
        self,
        max_pixel_value: float = 1.0,
        use_logits: bool = False,
        device: Union[str, torch.device] = "cuda",
    ):
        """Configure SSIM accumulation settings.

        Args:
            max_pixel_value: Peak pixel value used in SSIM constants.
            use_logits: Whether caller should provide logits instead of postprocessed outputs.
            device: Device for accumulation buffers.
        """

        super().__init__()
        self.max_pixel_value = max_pixel_value
        self.use_logits = use_logits
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.ssim_metric = StructuralSimilarityIndexMeasure(
            data_range=max_pixel_value,
            reduction="elementwise_mean",
        ).to(self.device)
        self.reset()

    def reset(self):
        """Reset running SSIM accumulators."""

        self.ssim_metric.reset()

    def forward(
        self,
        generated_predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> None:
        """Accumulate per-sample SSIM values for a split.

        Args:
            generated_predictions: Model predictions.
            targets: Ground-truth targets with matching shape.
            **kwargs: Additional unused metric arguments.

        Raises:
            ValueError: If shapes mismatch.
        """

        if generated_predictions.shape != targets.shape:
            raise ValueError("The generated predictions and targets must be the same shape.")

        self.ssim_metric.update(generated_predictions, targets)
        return None

    def update(self, generated_predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> None:
        """Alias for state updates to align with TorchMetrics-like API."""

        self.forward(generated_predictions=generated_predictions, targets=targets, **kwargs)

    def compute(self) -> torch.Tensor:
        """Compute averaged SSIM for currently accumulated state.

        Returns:
            Scalar tensor with current SSIM value.
        """

        average_ssim = self.ssim_metric.compute().to(self.device)
        if not torch.isfinite(average_ssim):
            average_ssim = torch.tensor(0.0, device=self.device)
        return average_ssim

    @property
    def metric_name(self) -> str:
        """Base metric key for logging."""

        return "ssim_total"

    def get_metric_data(self) -> dict[str, float]:
        """Backward-compatible helper that computes and resets state."""

        metric_data = {self.metric_name: self.compute().item()}
        self.reset()
        return metric_data
