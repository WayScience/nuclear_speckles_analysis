from typing import Literal, Union

import torch
from torchmetrics.functional.image.lpips import (
    learned_perceptual_image_patch_similarity,
)

from .AbstractMetric import AbstractMetric
from .utils.streaming_stats import StreamingScalarStats


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

        self.stats = StreamingScalarStats(device=self.device)

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
        self.stats.update(per_sample_lpips)
        return None

    def update(self, generated_predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> None:
        """Alias for state updates to align with TorchMetrics-like API."""

        self.forward(generated_predictions=generated_predictions, targets=targets, **kwargs)

    def compute(self) -> dict[str, float]:
        """Compute averaged LPIPS and population std for current state.

        Returns:
            Dictionary containing mean and std metric values.
        """

        stats = self.stats.compute()
        return {
            self.metric_name: stats["mean"],
            f"{self.metric_name}_std": stats["std"],
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
