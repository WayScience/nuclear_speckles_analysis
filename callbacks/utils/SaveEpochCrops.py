import pathlib
from typing import Any, Optional

import numpy as np
import torch

from .save_utils import save_image_mlflow


class SaveEpochCrops:
    """Save crop-level input/target/prediction images during training."""

    def __init__(
        self,
        image_dataset: torch.utils.data.Dataset,
        image_postprocessor: Any = lambda x: x,
        image_dataset_idxs: Optional[list[int]] = None,
    ) -> None:
        """Initialize epoch-wise crop saving configuration.

        Args:
            image_dataset: Dataset used to fetch fixed samples for visualization.
            image_postprocessor: Transform applied to model predictions before saving.
            image_dataset_idxs: Optional dataset indices to save each epoch.
        """

        self.image_dataset = image_dataset
        self.image_postprocessor = image_postprocessor
        self.image_dataset_idxs = (
            range(len(image_dataset)) if image_dataset_idxs is None else image_dataset_idxs
        )

    def save_image(
        self,
        image_path: pathlib.Path,
        image_type: str,
        image: torch.Tensor,
        metadata: dict[str, Any],
        epoch: int,
    ) -> None:
        """Convert a tensor image to uint8 and log it as an MLflow artifact.

        Args:
            image_path: Source path used to derive filename metadata.
            image_type: Prefix describing the image role (input/target/prediction).
            image: Image tensor with shape ``(H, W)`` or ``(1, H, W)``.
            metadata: Per-image metadata used for artifact path construction.
            epoch: Current epoch index used in artifact paths.

        Raises:
            ValueError: If image is not convertible to a single 2D crop.
        """

        if image.ndim == 3 and image.shape[0] == 1:
            image = image[0]

        if image.ndim != 2:
            raise ValueError(f"Expected image shape (H, W), got {tuple(image.shape)}")

        image = image.clamp(0.0, 1.0)
        image_np = (image * 255).byte().cpu().numpy()

        if np.max(image_np) == 0:
            return

        plate = metadata["Metadata_Plate"]
        well = metadata["Metadata_Well"]
        site = metadata["Metadata_Site"]
        sample_id = metadata["Metadata_Sample_ID"]

        image_suffix = ".tiff" if ".tif" in image_path.suffix else image_path.suffix
        save_image_path_folder = (
            f"cropped_images/epoch_{epoch:02}/{plate}/{well}_{site}/{sample_id}"
        )
        image_filename = f"{image_type}_{image_path.stem}{image_suffix}"

        save_image_mlflow(
            image=image_np,
            save_image_path_folder=save_image_path_folder,
            image_filename=image_filename,
        )

    def predict_target(self, image: torch.Tensor, model: torch.nn.Module) -> torch.Tensor:
        """Run model inference for one sample and apply postprocessing.

        Args:
            image: Input sample tensor with channel-first layout.
            model: Trained model used for prediction.

        Returns:
            Postprocessed prediction tensor.
        """

        with torch.no_grad():
            prediction = model(image.unsqueeze(0)).squeeze(0)
        return self.image_postprocessor(prediction)

    def __call__(self, model: torch.nn.Module, epoch: int) -> None:
        """Save input, target, and generated prediction images for one epoch.

        Args:
            model: Model used to generate predictions.
            epoch: Current epoch index used in artifact paths.
        """

        for sample_idx in self.image_dataset_idxs:
            sample = self.image_dataset[sample_idx]
            metadata = sample["metadata"]

            self.save_image(
                image_path=sample["input_path"],
                image_type="input",
                image=sample["input"],
                metadata=metadata,
                epoch=epoch,
            )

            self.save_image(
                image_path=sample["target_path"],
                image_type="target",
                image=sample["target"],
                metadata=metadata,
                epoch=epoch,
            )

            generated_prediction = self.predict_target(image=sample["input"], model=model)
            self.save_image(
                image_path=sample["target_path"],
                image_type="generated_prediction",
                image=generated_prediction,
                metadata=metadata,
                epoch=epoch,
            )
