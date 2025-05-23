import torch
from AbstractLoss import AbstractLoss


class WassersteinGeneratorCrossZamirskiLoss(AbstractLoss):

    def __init__(self, _metric_name, _reconstruction_importance: int = 100):
        super(WassersteinGeneratorCrossZamirskiLoss, self).__init__()

        self.__metric_name = _metric_name
        self.__reconstruction_importance = _reconstruction_importance
        self.__metric_func = torch.nn.L1Loss(reduction="mean")

    def forward(
        self,
        _fake_classification_outputs: torch.Tensor,
        _generated_outputs: torch.Tensor,
        _targets: torch.Tensor,
        _epoch: int,
    ):

        return self.__reconstruction_importance * self.__metric_func(
            _generated_outputs, _targets
        ) - torch.mean(_fake_classification_outputs) / (_epoch + 1)

    @property
    def metric_name(self):
        return self.__metric_name
