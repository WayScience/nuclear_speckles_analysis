import pathlib

import pandas as pd
import torch
from CPLoss import CPLoss


class CPTextureVarianceLoss(CPLoss):
    """Computes the Texture Variance CP Loss."""

    def __init__(
        self,
        _metric_name: str,
        _pipeline_path: pathlib.Path,
        _targets_path: pathlib.Path,
    ):
        super().__init__(_metric_name, _pipeline_path, _targets_path)
        self.__feature_name = "Texture_Variance_GOLD_3_00_256"

    def forward(self, _generated_outputs: torch.Tensor, _target_names: list[str]):

        targetdf = self.create_target_features(_target_names)
        outputdf = self.create_features_from_pytorch(_generated_outputs, _target_names)

        return (targetdf[self.__feature_name] - outputdf[self.__feature_name]).abs().mean()
