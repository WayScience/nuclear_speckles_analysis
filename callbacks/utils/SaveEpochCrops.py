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
        split_name: str = "validation",
    ) -> None:
        """Initialize epoch-wise crop saving configuration.

        Args:
            image_dataset: Dataset used to fetch fixed samples for visualization.
            image_postprocessor: Transform applied to model predictions before saving.
            image_dataset_idxs: Optional dataset indices to save each epoch.
            split_name: Split label used in artifact paths (for example, training).
        """

        self.image_dataset = image_dataset
        self.image_postprocessor = image_postprocessor
        self.image_dataset_idxs = (
            range(len(image_dataset)) if image_dataset_idxs is None else image_dataset_idxs
        )
        self.split_name = split_name

    def save_image(
        self,
        image_path: pathlib.Path,
        image_type: str,
        image: torch.Tensor,
        metadata: dict[str, Any],
        epoch: int,
        display_bounds: tuple[float, float] | None = None,
    ) -> None:
        """Convert a denormalized tensor image to uint8 and log it as an artifact.

        Args:
            image_path: Source path used to derive filename metadata.
            image_type: Prefix describing the image role (input/target/prediction).
            image: Denormalized image tensor with shape ``(H, W)`` or ``(1, H, W)``.
            metadata: Per-image metadata used for artifact path construction.
            epoch: Current epoch index used in artifact paths.
            display_bounds: Optional lower/upper clipping bounds applied before
                uint8 rescaling.

        Raises:
            ValueError: If image is not convertible to a single 2D crop.
        """

        if image.ndim == 3 and image.shape[0] == 1:
            image = image[0]

        if image.ndim != 2:
            raise ValueError(f"Expected image shape (H, W), got {tuple(image.shape)}")

        image_np = self._to_display_uint8(image=image, display_bounds=display_bounds)

        if np.max(image_np) == 0:
            return

        plate = metadata["Metadata_Plate"]
        well = metadata["Metadata_Well"]
        site = metadata["Metadata_Site"]
        sample_id = metadata["Metadata_Sample_ID"]

        image_suffix = ".tiff" if ".tif" in image_path.suffix else image_path.suffix
        save_image_path_folder = (
            f"cropped_images/{self.split_name}/epoch_{epoch:02}/{plate}/{well}_{site}/{sample_id}"
        )
        image_filename = f"{image_type}_{image_path.stem}{image_suffix}"

        save_image_mlflow(
            image=image_np,
            save_image_path_folder=save_image_path_folder,
            image_filename=image_filename,
        )

    def _compute_percentile_bounds(
        self,
        image: torch.Tensor,
        lower_percentile: float = 1.0,
        upper_percentile: float = 99.0,
    ) -> tuple[float, float]:
        """Compute robust display bounds from one denormalized image tensor."""

        image_np = image.detach().float().cpu().numpy()
        lower = float(np.percentile(image_np, lower_percentile))
        upper = float(np.percentile(image_np, upper_percentile))
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            lower = float(np.min(image_np))
            upper = float(np.max(image_np))
        if lower >= upper:
            upper = lower + 1.0
        return lower, upper

    def _to_display_uint8(
        self,
        image: torch.Tensor,
        display_bounds: tuple[float, float] | None = None,
    ) -> np.ndarray:
        """Clip a denormalized image and rescale it to displayable uint8 values."""

        lower, upper = (
            self._compute_percentile_bounds(image=image)
            if display_bounds is None
            else display_bounds
        )
        image = image.detach().float().clamp(lower, upper)
        image = (image - lower) / (upper - lower)
        return (image * 255).byte().cpu().numpy()

    def _compute_shared_pair_bounds(
        self,
        target_image: torch.Tensor,
        prediction_image: torch.Tensor,
    ) -> tuple[float, float]:
        """Compute shared percentile bounds for one target/prediction pair."""

        pair_values = torch.cat(
            [target_image.detach().reshape(-1), prediction_image.detach().reshape(-1)]
        )
        pair_np = pair_values.float().cpu().numpy()
        lower = float(np.percentile(pair_np, 1.0))
        upper = float(np.percentile(pair_np, 99.0))
        if not np.isfinite(lower) or not np.isfinite(upper) or lower >= upper:
            lower = float(np.min(pair_np))
            upper = float(np.max(pair_np))
        if lower >= upper:
            upper = lower + 1.0
        return lower, upper

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
            denormalized_input = self.image_postprocessor.denormalize_input(sample["input"])
            denormalized_target = self.image_postprocessor.denormalize_target(sample["target"])
            input_display_bounds = self._compute_percentile_bounds(image=denormalized_input)

            self.save_image(
                image_path=sample["input_path"],
                image_type="input",
                image=denormalized_input,
                metadata=metadata,
                epoch=epoch,
                display_bounds=input_display_bounds,
            )

            generated_prediction = self.predict_target(image=sample["input"], model=model)
            denormalized_prediction = self.image_postprocessor.denormalize_target(
                generated_prediction
            )
            pair_display_bounds = self._compute_shared_pair_bounds(
                target_image=denormalized_target,
                prediction_image=denormalized_prediction,
            )

            self.save_image(
                image_path=sample["target_path"],
                image_type="target",
                image=denormalized_target,
                metadata=metadata,
                epoch=epoch,
                display_bounds=pair_display_bounds,
            )
            self.save_image(
                image_path=sample["target_path"],
                image_type="generated_prediction",
                image=denormalized_prediction,
                metadata=metadata,
                epoch=epoch,
                display_bounds=pair_display_bounds,
            )
