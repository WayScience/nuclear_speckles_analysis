from abc import ABC, abstractmethod

import torch
from torch import nn


class AbstractMetric(nn.Module, ABC):
    """
    Abstract base class for all metrics.
    """

    def __init__(self):
        """Initialize metric modules as standard ``torch.nn.Module`` objects."""

        super().__init__()

    @abstractmethod
    def update(self, generated_predictions: torch.Tensor, targets: torch.Tensor, **kwargs) -> None:
        """Update metric state for one batch."""

        pass

    @abstractmethod
    def compute(self) -> torch.Tensor:
        """Compute the current aggregated metric value without resetting state."""

        pass

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Base metric key used by logging callbacks."""

        pass
