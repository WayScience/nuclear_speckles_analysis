from typing import Union

import torch
from torchmetrics.functional.image.dists import (
    deep_image_structure_and_texture_similarity,
)

from .AbstractMetric import AbstractMetric
from .utils.streaming_stats import StreamingScalarStats


class DISTS(AbstractMetric):
    """DISTS metric with epoch accumulation support."""

    def __init__(
        self,
        use_logits: bool = False,
        device: Union[str, torch.device] = "cuda",
    ):
        """Configure DISTS accumulation settings.

        Args:
            use_logits: Whether caller should provide logits instead of postprocessed outputs.
            device: Device for accumulation buffers.
        """

        super().__init__()
        self.use_logits = use_logits
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.reset()

    def reset(self):
        """Reset running DISTS accumulators."""

        self.stats = StreamingScalarStats(device=self.device)

    def _to_three_channels(self, tensor: torch.Tensor) -> torch.Tensor:
        """Convert grayscale tensors to 3-channel tensors for DISTS."""

        if tensor.ndim != 4:
            raise ValueError("Expected tensor with shape (N, C, H, W).")
        if tensor.shape[1] == 3:
            return tensor
        if tensor.shape[1] == 1:
            return tensor.repeat(1, 3, 1, 1)
        raise ValueError("DISTS expects images with 1 or 3 channels.")

    def forward(
        self,
        generated_predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> None:
        """Accumulate per-sample DISTS values for a split.

        Args:
            generated_predictions: Model predictions.
            targets: Ground-truth targets with matching shape.
            **kwargs: Additional unused metric arguments.

        Raises:
            ValueError: If shapes mismatch.
        """

        if generated_predictions.shape != targets.shape:
            raise ValueError("The generated predictions and targets must be the same shape.")

        preds_rgb = self._to_three_channels(generated_predictions)
        targets_rgb = self._to_three_channels(targets)
        per_sample_dists = deep_image_structure_and_texture_similarity(
            preds_rgb,
            targets_rgb,
            reduction="none",
        ).reshape(-1)
        self.stats.update(per_sample_dists)
        return None

    def update(self, generated_predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> None:
        """Alias for state updates to align with TorchMetrics-like API."""

        self.forward(generated_predictions=generated_predictions, targets=targets, **kwargs)

    def compute(self) -> dict[str, float]:
        """Compute averaged DISTS and population std for current state.

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

        return "dists_total"

    def get_metric_data(self) -> dict[str, float]:
        """Compute metric stats and reset state."""

        metric_data = self.compute()
        self.reset()
        return metric_data
