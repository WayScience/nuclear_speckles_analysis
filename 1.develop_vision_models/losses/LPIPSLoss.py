import lpips
import torch
from AbstractLoss import AbstractLoss


class LPIPSLoss(AbstractLoss):
    def __init__(self, _metric_name, _device="cuda"):
        super(LPIPSLoss, self).__init__()

        self.__metric_name = _metric_name

        # Initialize LPIPS with a pretrained network
        self.__metric_func = lpips.LPIPS(net="vgg")
        self.__metric_func.eval().to(_device)

    def forward(
        self,
        _generated_outputs: torch.Tensor,
        _targets: torch.Tensor,
    ):
        # Convert from 1-channel to 3-channel
        gen_rgb = _generated_outputs.repeat(1, 3, 1, 1)
        tgt_rgb = _targets.repeat(1, 3, 1, 1)

        # Normalize to [-1, 1] if in [0, 1]
        gen_rgb = gen_rgb * 2 - 1
        tgt_rgb = tgt_rgb * 2 - 1

        # Compute LPIPS loss (mean over batch)
        return self.__metric_func(gen_rgb, tgt_rgb).mean()

    @property
    def metric_name(self):
        return self.__metric_name
