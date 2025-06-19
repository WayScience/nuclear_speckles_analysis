import torch
from AbstractLoss import AbstractLoss
from CPLoss import CPLoss
from WassersteinGeneratorLoss import WassersteinGeneratorLoss


class WassersteinCPTextureVarianceGeneratorLoss(AbstractLoss):

    def __init__(
        self,
        _metric_name: str,
        _cp_loss: CPLoss,
        _cp_loss_importance: int,
        _reconstruction_importance: int,
    ):
        super(WassersteinCPTextureVarianceGeneratorLoss, self).__init__()

        self._metric_name = _metric_name
        self.wasserstein_generator_loss = WassersteinGeneratorLoss(
            _metric_name="wgangp_discriminator_loss"
        )
        self._cp_loss = _cp_loss
        self._cp_loss_importance = _cp_loss_importance
        self._metric_func = torch.nn.L1Loss(reduction="mean")
        self._reconstruction_importance = _reconstruction_importance

    @property
    def metric_name(self):
        return self._metric_name

    def forward(
        self,
        _fake_classification_outputs: torch.Tensor,
        _generator_outputs: torch.Tensor,
        _targets: torch.Tensor,
        _target_names: list[str],
    ):

        return (
            self.wasserstein_generator_loss(_fake_classification_outputs)
            + self._cp_loss_importance
            * self._cp_loss(_generator_outputs, _target_names)
            + self._reconstruction_importance
            * self._metric_func(_generator_outputs, _targets)
        )
