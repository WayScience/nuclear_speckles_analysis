import torch
from AbstractLoss import AbstractLoss


class WassersteinGradientPenaltyLoss(AbstractLoss):

    def __init__(self, _metric_name, _gradient_penalty_importance):
        super(WassersteinGradientPenaltyLoss, self).__init__()

        self.__metric_name = _metric_name
        self.__gradient_penalty_importance = _gradient_penalty_importance

    def forward(
        self,
        _gradients: torch.Tensor,
        _real_classification_outputs: torch.Tensor,
        _fake_classification_outputs: torch.Tensor,
    ):
        _gradients = _gradients.view(_gradients.size(0), -1)
        return (
            torch.mean(_fake_classification_outputs)
            - torch.mean(_real_classification_outputs)
            + ((_gradients.norm(2, dim=1) - 1) ** 2).mean()
            * self.__gradient_penalty_importance
        )

    @property
    def metric_name(self):
        return self.__metric_name
