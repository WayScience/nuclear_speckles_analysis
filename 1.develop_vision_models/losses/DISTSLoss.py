import torch
from AbstractLoss import AbstractLoss
from DISTS_pytorch import DISTS


class DISTSLoss(AbstractLoss):
    def __init__(self, _metric_name, _device="cuda"):
        super(DISTSLoss, self).__init__()

        self.__metric_name = _metric_name

        # Initialize DISTS model
        self.__metric_func = DISTS()
        self.__metric_func.eval().to(_device)

    def forward(
        self,
        _generated_outputs: torch.Tensor,
        _targets: torch.Tensor,
    ):
        # Convert from 1-channel to 3-channel
        gen_rgb = _generated_outputs.repeat(1, 3, 1, 1)
        tgt_rgb = _targets.repeat(1, 3, 1, 1)

        # No need to rescale — DISTS expects inputs in [0, 1]

        # DISTS returns similarity; convert to loss
        return self.__metric_func(
            gen_rgb, tgt_rgb, require_grad=False, batch_average=False
        ).mean()

    @property
    def metric_name(self):
        return self.__metric_name
