from typing import Union

import torch
from torch import nn

from .AbstractMetric import AbstractMetric
from .utils.streaming_stats import StreamingScalarStats


class ValidationGeneratorLoss(AbstractMetric):
    """Stable validation-time generator loss used for model selection."""

    def __init__(
        self,
        discriminator: nn.Module,
        reconstruction_importance: float = 100.0,
        adversarial_importance: float = 1.0,
        device: Union[str, torch.device] = "cuda",
    ):
        super().__init__()
        self.discriminator = discriminator
        self.reconstruction_importance = reconstruction_importance
        self.adversarial_importance = adversarial_importance
        self.use_logits = False
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.reset()

    def reset(self):
        """Reset running validation generator loss accumulators."""

        self.total_stats = StreamingScalarStats(device=self.device)
        self.reconstruction_stats = StreamingScalarStats(device=self.device)
        self.adversarial_stats = StreamingScalarStats(device=self.device)

    def forward(
        self,
        generated_predictions: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> None:
        """Accumulate per-sample generator loss statistics for one eval batch."""

        if generated_predictions.shape != targets.shape:
            raise ValueError("The generated predictions and targets must be the same shape.")

        was_training = self.discriminator.training
        self.discriminator.eval()
        with torch.no_grad():
            # Reuse the current critic so validation model selection reflects the
            # same adversarial signal the generator is training against.
            fake_classification_outputs = self.discriminator(generated_predictions)
        if was_training:
            self.discriminator.train()

        abs_error = torch.abs(generated_predictions - targets)
        reconstruction = abs_error.reshape(abs_error.shape[0], -1).mean(dim=1)
        adversarial = fake_classification_outputs.reshape(fake_classification_outputs.shape[0], -1).mean(
            dim=1
        )
        total = (
            self.reconstruction_importance * reconstruction
            - self.adversarial_importance * adversarial
        )

        self.total_stats.update(total)
        self.reconstruction_stats.update(reconstruction)
        self.adversarial_stats.update(adversarial)
        return None

    def update(self, generated_predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> None:
        """Alias for state updates to align with TorchMetrics-like API."""

        self.forward(generated_predictions=generated_predictions, targets=targets, **kwargs)

    def compute(self) -> dict[str, float]:
        """Compute averaged validation generator loss stats for current state."""

        total_stats = self.total_stats.compute()
        reconstruction_stats = self.reconstruction_stats.compute()
        adversarial_stats = self.adversarial_stats.compute()
        return {
            self.metric_name: total_stats["mean"],
            f"{self.metric_name}_std": total_stats["std"],
            "generator_loss_reconstruction": reconstruction_stats["mean"],
            "generator_loss_reconstruction_std": reconstruction_stats["std"],
            "generator_loss_adversarial": adversarial_stats["mean"],
            "generator_loss_adversarial_std": adversarial_stats["std"],
        }

    @property
    def metric_name(self) -> str:
        """Base metric key used for checkpoint selection."""

        return "generator_loss_total"

    def get_metric_data(self) -> dict[str, float]:
        """Compute metric stats and reset state."""

        metric_data = self.compute()
        self.reset()
        return metric_data
