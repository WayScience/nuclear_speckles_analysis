from typing import Optional, Union

import torch

from losses.l1_ssim import compute_l1_per_sample, compute_ssim_loss_per_sample

from .AbstractMetric import AbstractMetric
from .utils.streaming_stats import StreamingScalarStats


class L1SSIMLossMetric(AbstractMetric):
    """Accumulate normalized-space L1, SSIM loss, and total loss for evaluation."""

    def __init__(
        self,
        ssim_weight: float,
        data_range: Optional[float] = None,
        use_logits: bool = True,
        device: Union[str, torch.device] = "cuda",
    ):
        """Configure epoch-level mixed-loss accumulation.

        Args:
            ssim_weight: Multiplier applied to ``-1 * ssim``.
            data_range: Optional fixed intensity range for SSIM. If omitted, the
                range is derived from each evaluation batch.
            use_logits: Whether caller should provide logits instead of
                postprocessed outputs. This defaults to ``True`` so model
                selection matches the normalized training objective.
            device: Device for accumulation buffers.

        Raises:
            ValueError: If ``ssim_weight`` is negative or ``data_range`` is not
                positive when provided.
        """

        super().__init__()
        if ssim_weight < 0:
            raise ValueError("ssim_weight must be non-negative")
        if data_range is not None and data_range <= 0:
            raise ValueError("data_range must be positive when provided")

        self.ssim_weight = float(ssim_weight)
        self.data_range = None if data_range is None else float(data_range)
        self.use_logits = use_logits
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.reset()

    def reset(self):
        """Reset running loss accumulators used for epoch-level logging."""

        self.l1_stats = StreamingScalarStats(device=self.device)
        self.ssim_stats = StreamingScalarStats(device=self.device)
        self.total_stats = StreamingScalarStats(device=self.device)

    def forward(
        self,
        generated_predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> None:
        """Accumulate per-sample mixed-loss statistics for one evaluation batch.

        Args:
            generated_predictions: Model predictions.
            targets: Ground-truth targets with matching shape.
            **kwargs: Additional unused metric arguments.

        Raises:
            ValueError: If prediction and target shapes differ.
        """

        per_sample_l1 = compute_l1_per_sample(
            generated_predictions=generated_predictions,
            targets=targets,
        )
        per_sample_ssim = compute_ssim_loss_per_sample(
            generated_predictions=generated_predictions,
            targets=targets,
            data_range=self.data_range,
        )
        per_sample_total = per_sample_l1 + self.ssim_weight * per_sample_ssim

        self.l1_stats.update(per_sample_l1)
        self.ssim_stats.update(per_sample_ssim)
        self.total_stats.update(per_sample_total)
        return None

    def update(self, generated_predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> None:
        """Alias for state updates to align with TorchMetrics-like API."""

        self.forward(generated_predictions=generated_predictions, targets=targets, **kwargs)

    def compute(self) -> dict[str, float]:
        """Compute averaged mixed-loss values and standard deviations.

        Returns:
            Dictionary containing mean and std values for the L1, SSIM-derived, and
            total-loss components.
        """

        l1_stats = self.l1_stats.compute()
        ssim_stats = self.ssim_stats.compute()
        total_stats = self.total_stats.compute()
        return {
            "loss_l1": l1_stats["mean"],
            "loss_l1_std": l1_stats["std"],
            "loss_ssim": ssim_stats["mean"],
            "loss_ssim_std": ssim_stats["std"],
            self.metric_name: total_stats["mean"],
            f"{self.metric_name}_std": total_stats["std"],
        }

    @property
    def metric_name(self) -> str:
        """Base metric key used for model selection and logging."""

        return "loss_total"

    def get_metric_data(self) -> dict[str, float]:
        """Compute accumulated loss stats and reset state."""

        metric_data = self.compute()
        self.reset()
        return metric_data
