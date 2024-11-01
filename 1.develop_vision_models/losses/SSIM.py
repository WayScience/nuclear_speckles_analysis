import torch
from AbstractLoss import AbstractLoss

class SSIM(AbstractLoss):
    def __init__(self, _metric_name, _max_pixel_value = 1):
        super(SSIM, self).__init__()

        self.__metric_name = _metric_name
        self.__max_pixel_value = _max_pixel_value

    def forward(self, _generated_outputs: torch.Tensor, _targets: torch.Tensor):
        mu1 = _generated_outputs.mean(dim=[2, 3], keepdim=True)
        mu2 = _targets.mean(dim=[2, 3], keepdim=True)

        sigma1_sq = ((_generated_outputs - mu1) ** 2).mean(dim=[2, 3], keepdim=True)
        sigma2_sq = ((_targets - mu2) ** 2).mean(dim=[2, 3], keepdim=True)
        sigma12 = ((_generated_outputs - mu1) * (_targets - mu2)).mean(dim=[2, 3], keepdim=True)

        C1 = (self.__max_pixel_value * 0.01) ** 2
        C2 = (self.__max_pixel_value * 0.03) ** 2

        ssim_value = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq ** 2 + sigma2_sq ** 2 + C2))

        return ssim_value.mean()

    @property
    def metric_name(self):
        return self.__metric_name
