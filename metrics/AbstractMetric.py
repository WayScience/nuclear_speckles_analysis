from abc import ABC, abstractmethod
from typing import Optional

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
    def forward(self) -> torch.Tensor | None:
        """Update metric state for a batch and optionally return a scalar loss."""

        pass

    @abstractmethod
    def get_metric_data(self) -> dict[str, torch.Tensor]:
        """Return accumulated metric values for logging and reset internal state."""

        pass
