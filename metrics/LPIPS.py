from typing import Literal, Union

import lpips
import torch

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
        self.lpips_metric = lpips.LPIPS(net=self.net).to(self.device)
        self.lpips_metric.eval()
        self.reset()

    def reset(self):
        """Reset running LPIPS accumulators."""

        self.total_lpips = torch.tensor(0.0, device=self.device)
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

        lpips_values = self.lpips_metric(predictions_lpips, targets_lpips)
        lpips_values = lpips_values.view(-1)

        self.total_lpips += lpips_values.sum().detach().to(self.device)
        self.total_examples += torch.tensor(
            lpips_values.numel(),
            dtype=torch.float32,
            device=self.device,
        )
        return None

    def update(self, generated_predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> None:
        """Alias for state updates to align with TorchMetrics-like API."""

        self.forward(generated_predictions=generated_predictions, targets=targets, **kwargs)

    def compute(self) -> torch.Tensor:
        """Compute averaged LPIPS for currently accumulated state.

        Returns:
            Scalar tensor with current LPIPS value.
        """

        average_lpips = torch.where(
            self.total_examples > 0,
            self.total_lpips / self.total_examples,
            torch.tensor(0.0, device=self.device),
        )
        if not torch.isfinite(average_lpips):
            average_lpips = torch.tensor(0.0, device=self.device)
        return average_lpips

    @property
    def metric_name(self) -> str:
        """Base metric key for logging."""

        return "lpips_total"

    def get_metric_data(self) -> dict[str, float]:
        """Backward-compatible helper that computes and resets state."""

        metric_data = {self.metric_name: self.compute().item()}
        self.reset()
        return metric_data
