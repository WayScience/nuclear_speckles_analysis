import torch.nn as nn
from abc import ABC, abstractmethod

class AbstractLoss(nn.Module, ABC):

    @property
    @abstractmethod
    def metric_name(self):
        pass

    @abstractmethod
    def forward(self, x):
        pass
