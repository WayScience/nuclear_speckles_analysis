import torch
import torch.nn as nn


class Pix2PixDiscriminator(nn.Module):
    def __init__(self, _number_input_channels: int, _number_output_channels: int, _conv_depth: int = 4):
        super(Pix2PixDiscriminator, self).__init__()

        self.__conv_depth = _conv_depth
        self.__leaky_relu = nn.LeakyReLU()

        if _conv_depth > 0:
            self.__conv = nn.Conv2d(_number_input_channels, _number_output_channels, kernel_size=3, padding=1)
            _number_input_channels = _number_output_channels
            self.__conv_layers = Pix2PixDiscriminator(_number_input_channels, _number_output_channels * 2, _conv_depth - 1)

        else:
            self.flatten = nn.Flatten()
            self.llinear0 = nn.LazyLinear(512)
            self.llinear1 = nn.LazyLinear(1)
            self.sigmoid = nn.Sigmoid()

    def forward(self, _img_tensor: torch.Tensor):

        if self.__conv_depth > 0:
            _img_tensor = self.__conv(_img_tensor)
            _img_tensor = self.__leaky_relu(_img_tensor)
            return self.__conv_layers(_img_tensor)

        else:
            _img_tensor = self.flatten(_img_tensor)
            _img_tensor = self.llinear0(_img_tensor)
            _img_tensor = self.__leaky_relu(_img_tensor)
            _img_tensor = self.llinear1(_img_tensor)
            return _img_tensor
