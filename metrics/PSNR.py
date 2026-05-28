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
            reduction="none",
            dim=(1, 2, 3),
        ).to(self.device)
        self.reset()

    def reset(self):
        """Reset running PSNR accumulators."""

        self.psnr_metric.reset()
        self.total_psnr = torch.tensor(0.0, device=self.device)
        self.total_psnr_sq = torch.tensor(0.0, device=self.device)
        self.total_examples = torch.tensor(0.0, device=self.device)

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
        per_sample_psnr = self.psnr_metric.compute().to(self.device).reshape(-1)
        self.psnr_metric.reset()
        finite_psnr = torch.where(
            torch.isfinite(per_sample_psnr),
            per_sample_psnr,
            torch.tensor(self.nonfinite_cap, device=self.device),
        )
        self.total_psnr += finite_psnr.sum().detach()
        self.total_psnr_sq += finite_psnr.pow(2).sum().detach()
        self.total_examples += torch.tensor(
            finite_psnr.numel(),
            dtype=torch.float32,
            device=self.device,
        )
        return None

    def update(self, generated_predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> None:
        """Alias for state updates to align with TorchMetrics-like API."""

        self.forward(generated_predictions=generated_predictions, targets=targets, **kwargs)

    def compute(self) -> dict[str, float]:
        """Compute averaged PSNR and population std for current state.

        Returns:
            Dictionary containing mean and std metric values.
        """

        average_psnr = torch.where(
            self.total_examples > 0,
            self.total_psnr / self.total_examples,
            torch.tensor(0.0, device=self.device),
        )
        variance_psnr = torch.where(
            self.total_examples > 0,
            (self.total_psnr_sq / self.total_examples) - average_psnr.pow(2),
            torch.tensor(0.0, device=self.device),
        )
        std_psnr = torch.sqrt(torch.clamp(variance_psnr, min=0.0))
        if not torch.isfinite(average_psnr):
            average_psnr = torch.tensor(self.nonfinite_cap, device=self.device)
        if not torch.isfinite(std_psnr):
            std_psnr = torch.tensor(0.0, device=self.device)

        return {
            self.metric_name: average_psnr.item(),
            f"{self.metric_name}_std": std_psnr.item(),
        }

    @property
    def metric_name(self) -> str:
        """Base metric key for logging."""

        return "psnr_total"

    def get_metric_data(self) -> dict[str, float]:
        """Compute metric stats and reset state."""

        metric_data = self.compute()
        self.reset()
        return metric_data
