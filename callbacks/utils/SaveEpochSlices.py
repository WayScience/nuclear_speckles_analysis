import pathlib
from typing import Any, Optional

import numpy as np
import torch

from .save_utils import save_image_mlflow


class SaveEpochSlices:
    """Save crop-level input/target/prediction images during training."""

    def __init__(
        self,
        image_dataset: torch.utils.data.Dataset,
        image_postprocessor: Any = lambda x: x,
        image_dataset_idxs: Optional[list[int]] = None,
    ) -> None:
        self.image_dataset = image_dataset
        self.image_postprocessor = image_postprocessor
        self.image_dataset_idxs = (
            range(len(image_dataset)) if image_dataset_idxs is None else image_dataset_idxs
        )
        self.epoch = None
        self.metadata = None

    def save_image(
        self,
        image_path: pathlib.Path,
        image_type: str,
        image: torch.Tensor,
    ) -> None:
        if image.ndim == 3 and image.shape[0] == 1:
            image = image[0]

        if image.ndim != 2:
            raise ValueError(f"Expected image shape (H, W), got {tuple(image.shape)}")

        image = image.clamp(0.0, 1.0)
        image_np = (image * 255).byte().cpu().numpy()

        if np.max(image_np) == 0:
            return

        plate = self.metadata["Metadata_Plate"]
        well = self.metadata["Metadata_Well"]
        site = self.metadata["Metadata_Site"]
        sample_id = self.metadata["Metadata_Sample_ID"]

        image_suffix = ".tiff" if ".tif" in image_path.suffix else image_path.suffix
        save_image_path_folder = (
            f"cropped_images/epoch_{self.epoch:02}/{plate}/{well}_{site}/{sample_id}"
        )
        image_filename = f"{image_type}_{image_path.stem}{image_suffix}"

        save_image_mlflow(
            image=image_np,
            save_image_path_folder=save_image_path_folder,
            image_filename=image_filename,
        )

    def predict_target(self, image: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        with torch.no_grad():
            prediction = model(image.unsqueeze(0)).squeeze(0)
        return self.image_postprocessor(prediction)

    def __call__(self, model: torch.nn.Module, epoch: int) -> None:
        self.epoch = epoch
        for sample_idx in self.image_dataset_idxs:
            sample = self.image_dataset[sample_idx]
            self.metadata = sample["metadata"]

            self.save_image(
                image_path=sample["input_path"],
                image_type="input",
                image=sample["input"],
            )

            self.save_image(
                image_path=sample["target_path"],
                image_type="target",
                image=sample["target"],
            )

            generated_prediction = self.predict_target(image=sample["input"], model=model)
            self.save_image(
                image_path=sample["target_path"],
                image_type="generated_prediction",
                image=generated_prediction,
            )
