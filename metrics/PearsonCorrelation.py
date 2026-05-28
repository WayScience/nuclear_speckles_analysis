from typing import Union

import torch

from .AbstractMetric import AbstractMetric


class PearsonCorrelation(AbstractMetric):
    """Pearson correlation metric with epoch accumulation support."""

    def __init__(
        self,
        use_logits: bool = False,
        device: Union[str, torch.device] = "cuda",
    ):
        """Configure Pearson correlation accumulation settings.

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
        """Reset running Pearson correlation accumulators."""

        self.total_pearson = torch.tensor(0.0, device=self.device)
        self.total_examples = torch.tensor(0.0, device=self.device)

    def forward(
        self,
        generated_predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> None:
        """Accumulate per-sample Pearson correlation values for a split.

        Args:
            generated_predictions: Model predictions.
            targets: Ground-truth targets with matching shape.
            **kwargs: Additional unused metric arguments.

        Raises:
            ValueError: If shapes mismatch.
        """

        if generated_predictions.shape != targets.shape:
            raise ValueError("The generated predictions and targets must be the same shape.")

        if generated_predictions.ndim < 2:
            raise ValueError("Expected batched tensor with shape (N, ...).")

        preds_flat = generated_predictions.reshape(generated_predictions.shape[0], -1).to(
            self.device
        )
        targets_flat = targets.reshape(targets.shape[0], -1).to(self.device)

        preds_centered = preds_flat - preds_flat.mean(dim=1, keepdim=True)
        targets_centered = targets_flat - targets_flat.mean(dim=1, keepdim=True)

        numerator = (preds_centered * targets_centered).sum(dim=1)
        denominator = torch.sqrt(
            (preds_centered.pow(2).sum(dim=1) * targets_centered.pow(2).sum(dim=1))
        )
        per_sample_pearson = torch.where(
            denominator > 0,
            numerator / denominator,
            torch.tensor(0.0, device=self.device),
        )

        self.total_pearson += per_sample_pearson.sum().detach()
        self.total_examples += torch.tensor(
            per_sample_pearson.shape[0],
            dtype=torch.float32,
            device=self.device,
        )
        return None

    def update(self, generated_predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> None:
        """Alias for state updates to align with TorchMetrics-like API."""

        self.forward(generated_predictions=generated_predictions, targets=targets, **kwargs)

    def compute(self) -> torch.Tensor:
        """Compute averaged Pearson correlation for currently accumulated state.

        Returns:
            Scalar tensor with current Pearson correlation value.
        """

        average_pearson = torch.where(
            self.total_examples > 0,
            self.total_pearson / self.total_examples,
            torch.tensor(0.0, device=self.device),
        )
        if not torch.isfinite(average_pearson):
            average_pearson = torch.tensor(0.0, device=self.device)
        return average_pearson

    @property
    def metric_name(self) -> str:
        """Base metric key for logging."""

        return "pearson_total"

    def get_metric_data(self) -> dict[str, float]:
        """Backward-compatible helper that computes and resets state."""

        metric_data = {self.metric_name: self.compute().item()}
        self.reset()
        return metric_data
