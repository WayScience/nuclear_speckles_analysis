from typing import Literal, Union

import torch
from torchmetrics.functional.image.lpips import (
    learned_perceptual_image_patch_similarity,
)

from .AbstractMetric import AbstractMetric


class LPIPS(AbstractMetric):
    """LPIPS metric with epoch accumulation support."""

    def __init__(
        self,
        net: Literal["alex", "vgg", "squeeze"] = "vgg",
        use_logits: bool = False,
        device: Union[str, torch.device] = "cuda",
    ):
        """Configure LPIPS accumulation settings.

        Args:
            net: Backbone network variant for LPIPS (``alex``, ``vgg``, or ``squeeze``).
            use_logits: Whether caller should provide logits instead of postprocessed outputs.
            device: Device for accumulation buffers.
        """

        super().__init__()
        self.net = net
        self.use_logits = use_logits
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.reset()

    def reset(self):
        """Reset running LPIPS accumulators."""

        self.total_lpips = torch.tensor(0.0, device=self.device)
        self.total_lpips_sq = torch.tensor(0.0, device=self.device)
        self.total_examples = torch.tensor(0.0, device=self.device)

    def _prepare_tensor(self, x: torch.Tensor) -> torch.Tensor:
        """Prepare image tensor for LPIPS input conventions."""

        x = x.to(self.device)
        if x.ndim != 4:
            raise ValueError(
                "LPIPS expects image tensors with shape [N, C, H, W]."
            )
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError("LPIPS supports only 1-channel or 3-channel inputs.")

        return (x * 2.0) - 1.0

    def forward(
        self,
        generated_predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> None:
        """Accumulate per-sample LPIPS values for a split.

        Args:
            generated_predictions: Model predictions.
            targets: Ground-truth targets with matching shape.
            **kwargs: Additional unused metric arguments.

        Raises:
            ValueError: If shapes mismatch.
        """

        if generated_predictions.shape != targets.shape:
            raise ValueError("The generated predictions and targets must be the same shape.")

        predictions_lpips = self._prepare_tensor(generated_predictions)
        targets_lpips = self._prepare_tensor(targets)

        per_sample_lpips = learned_perceptual_image_patch_similarity(
            predictions_lpips,
            targets_lpips,
            net_type=self.net,
            reduction="none",
            normalize=False,
        ).reshape(-1)
        self.total_lpips += per_sample_lpips.sum().detach().to(self.device)
        self.total_lpips_sq += per_sample_lpips.pow(2).sum().detach().to(self.device)
        self.total_examples += torch.tensor(
            per_sample_lpips.numel(),
            dtype=torch.float32,
            device=self.device,
        )
        return None

    def update(self, generated_predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> None:
        """Alias for state updates to align with TorchMetrics-like API."""

        self.forward(generated_predictions=generated_predictions, targets=targets, **kwargs)

    def compute(self) -> dict[str, float]:
        """Compute averaged LPIPS and population std for current state.

        Returns:
            Dictionary containing mean and std metric values.
        """

        average_lpips = torch.where(
            self.total_examples > 0,
            self.total_lpips / self.total_examples,
            torch.tensor(0.0, device=self.device),
        )
        variance_lpips = torch.where(
            self.total_examples > 0,
            (self.total_lpips_sq / self.total_examples) - average_lpips.pow(2),
            torch.tensor(0.0, device=self.device),
        )
        std_lpips = torch.sqrt(torch.clamp(variance_lpips, min=0.0))
        if not torch.isfinite(average_lpips):
            average_lpips = torch.tensor(0.0, device=self.device)
        if not torch.isfinite(std_lpips):
            std_lpips = torch.tensor(0.0, device=self.device)
        return {
            self.metric_name: average_lpips.item(),
            f"{self.metric_name}_std": std_lpips.item(),
        }

    @property
    def metric_name(self) -> str:
        """Base metric key for logging."""

        return "lpips_total"

    def get_metric_data(self) -> dict[str, float]:
        """Compute metric stats and reset state."""

        metric_data = self.compute()
        self.reset()
        return metric_data
