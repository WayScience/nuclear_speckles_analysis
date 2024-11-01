import torch
from AbstractLoss import AbstractLoss


class L1Loss(AbstractLoss):

    def __init__(
        self,
        _metric_name
    ):
        super(L1Loss, self).__init__()

        self.__metric_func = torch.nn.L1Loss(reduction="mean")
        self.__metric_name = _metric_name

    def forward(self, _generated_outputs: torch.Tensor, _targets: torch.Tensor):
        return self.__metric_func(_generated_outputs, _targets)

    @property
    def metric_name(self):
        return self.__metric_name
