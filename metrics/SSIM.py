from typing import Union

import torch

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
        self.reset()

    def reset(self):
        """Reset running SSIM accumulators."""

        self.total_ssim = torch.tensor(0.0, device=self.device)
        self.total_samples = torch.tensor(0.0, device=self.device)

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

        mu_x = generated_predictions.mean(dim=(2, 3), keepdim=True)
        mu_y = targets.mean(dim=(2, 3), keepdim=True)

        sigma_x = ((generated_predictions - mu_x) ** 2).mean(dim=(2, 3), keepdim=True)
        sigma_y = ((targets - mu_y) ** 2).mean(dim=(2, 3), keepdim=True)
        sigma_xy = ((generated_predictions - mu_x) * (targets - mu_y)).mean(
            dim=(2, 3), keepdim=True
        )

        c1 = (0.01 * self.max_pixel_value) ** 2
        c2 = (0.03 * self.max_pixel_value) ** 2

        ssim_map = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / (
            (mu_x**2 + mu_y**2 + c1) * (sigma_x + sigma_y + c2)
        )

        batch_ssim = ssim_map.mean(dim=(1, 2, 3))
        self.total_ssim += batch_ssim.sum().detach().to(self.device)
        self.total_samples += torch.tensor(
            batch_ssim.numel(),
            dtype=torch.float32,
            device=self.device,
        )
        return None

    def update(self, generated_predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> None:
        """Alias for state updates to align with TorchMetrics-like API."""

        self.forward(generated_predictions=generated_predictions, targets=targets, **kwargs)

    def compute(self) -> torch.Tensor:
        """Compute averaged SSIM for currently accumulated state.

        Returns:
            Scalar tensor with current SSIM value.
        """

        average_ssim = torch.where(
            self.total_samples > 0,
            self.total_ssim / self.total_samples,
            torch.tensor(0.0, device=self.device),
        )
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
