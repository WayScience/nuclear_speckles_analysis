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
            reduction="none",
        ).to(self.device)
        self.reset()

    def reset(self):
        """Reset running SSIM accumulators."""

        self.ssim_metric.reset()
        self.total_ssim = torch.tensor(0.0, device=self.device)
        self.total_ssim_sq = torch.tensor(0.0, device=self.device)
        self.total_examples = torch.tensor(0.0, device=self.device)

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
        per_sample_ssim = self.ssim_metric.compute().to(self.device).reshape(-1)
        self.ssim_metric.reset()
        finite_ssim = torch.where(
            torch.isfinite(per_sample_ssim),
            per_sample_ssim,
            torch.tensor(0.0, device=self.device),
        )
        self.total_ssim += finite_ssim.sum().detach()
        self.total_ssim_sq += finite_ssim.pow(2).sum().detach()
        self.total_examples += torch.tensor(
            finite_ssim.numel(),
            dtype=torch.float32,
            device=self.device,
        )
        return None

    def update(self, generated_predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> None:
        """Alias for state updates to align with TorchMetrics-like API."""

        self.forward(generated_predictions=generated_predictions, targets=targets, **kwargs)

    def compute(self) -> dict[str, float]:
        """Compute averaged SSIM and population std for current state.

        Returns:
            Dictionary containing mean and std metric values.
        """

        average_ssim = torch.where(
            self.total_examples > 0,
            self.total_ssim / self.total_examples,
            torch.tensor(0.0, device=self.device),
        )
        variance_ssim = torch.where(
            self.total_examples > 0,
            (self.total_ssim_sq / self.total_examples) - average_ssim.pow(2),
            torch.tensor(0.0, device=self.device),
        )
        std_ssim = torch.sqrt(torch.clamp(variance_ssim, min=0.0))
        if not torch.isfinite(average_ssim):
            average_ssim = torch.tensor(0.0, device=self.device)
        if not torch.isfinite(std_ssim):
            std_ssim = torch.tensor(0.0, device=self.device)

        return {
            self.metric_name: average_ssim.item(),
            f"{self.metric_name}_std": std_ssim.item(),
        }

    @property
    def metric_name(self) -> str:
        """Base metric key for logging."""

        return "ssim_total"

    def get_metric_data(self) -> dict[str, float]:
        """Compute metric stats and reset state."""

        metric_data = self.compute()
        self.reset()
        return metric_data
