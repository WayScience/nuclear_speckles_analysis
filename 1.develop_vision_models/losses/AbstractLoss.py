from abc import ABC, abstractmethod

import torch.nn as nn


class AbstractLoss(nn.Module, ABC):
    """This loss should be inherited by other losses."""

    @property
    @abstractmethod
    def metric_name(self):
        """Defines the mertic name returned by the class."""
        pass

    @abstractmethod
    def forward(self):
        """Computes the metric given information about the data."""
        pass
