import pathlib

import pandas as pd
import torch
from CPLoss import CPLoss
from sklearn.preprocessing import StandardScaler


class CombinedBatchNormCPLoss(CPLoss):
    """Computes the Combined Batch Norm CP Loss."""

    def __init__(
        self,
        _metric_name: str,
        _pipeline_path: pathlib.Path,
        _targets_path: pathlib.Path,
        _feature_names: list[str],
    ):
        super().__init__(_metric_name, _pipeline_path, _targets_path)
        self.__feature_names = _feature_names

    def forward(self, _generated_outputs: torch.Tensor, _target_names: list[str]):

        targetdf = self.create_target_features(_target_names)[self.__feature_names]
        outputdf = self.create_features_from_pytorch(_generated_outputs, _target_names)[self.__feature_names]

        generated_scaler = StandardScaler()
        generateddf_standardized = pd.DataFrame(
            generated_scaler.fit_transform(targetdf), columns=targetdf.columns
        )

        target_scaler = StandardScaler()
        targetdf_standardized = pd.DataFrame(
            target_scaler.fit_transform(outputdf), columns=outputdf.columns
        )

        diff_featuredf = (
            targetdf - outputdf
        ).abs()

        total_loss = 0

        for feature in self.__feature_names:
            total_loss += diff_featuredf[feature].mean()

        return total_loss / len(self.__feature_names)
