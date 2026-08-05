from typing import Any, Optional, Union

import numpy as np
import torch


class ImagePreProcessor:
    """Normalize paired 2D crops and format tensors for training."""

    def __init__(
        self,
        image_specs: dict[str, Any],
        device: Union[str, torch.device] = "cuda",
        input_transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
    ):
        """Configure normalization constants and optional augmentations.

        Args:
            image_specs: Image metadata dictionary used by ``set_image_specs``.
            device: Device where output tensors should be placed.
            input_transform: Optional albumentations-style transform for inputs.
            target_transform: Optional albumentations-style transform for targets.
        """

        self.image_specs = image_specs
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.input_transform = input_transform
        self.target_transform = target_transform

    def set_image_specs(
        self,
        input_percentile_lower_value: float | None = None,
        input_percentile_upper_value: float | None = None,
        target_percentile_lower_value: float | None = None,
        target_percentile_upper_value: float | None = None,
        **kwargs,
    ) -> None:
        """Store robust percentile normalization bounds inferred from training data.

        Args:
            input_percentile_lower_value: Lower train-split clipping bound for inputs.
            input_percentile_upper_value: Upper train-split clipping bound for inputs.
            target_percentile_lower_value: Lower train-split clipping bound for targets.
            target_percentile_upper_value: Upper train-split clipping bound for targets.
            **kwargs: Additional image spec keys ignored by this preprocessor.
        """

        self.input_lower_value = (
            None
            if input_percentile_lower_value is None
            else float(input_percentile_lower_value)
        )
        self.input_upper_value = (
            None
            if input_percentile_upper_value is None
            else float(input_percentile_upper_value)
        )
        self.target_lower_value = (
            None
            if target_percentile_lower_value is None
            else float(target_percentile_lower_value)
        )
        self.target_upper_value = (
            None
            if target_percentile_upper_value is None
            else float(target_percentile_upper_value)
        )

    def format_img(self, img: np.ndarray) -> torch.Tensor:
        """Convert a normalized 2D numpy image into a channel-first tensor.

        Args:
            img: 2D image array.

        Returns:
            Tensor with shape ``(1, H, W)`` on configured device.

        Raises:
            ValueError: If ``img`` is not a 2D array.
        """

        if img.ndim != 2:
            raise ValueError(f"Expected 2D image, got shape {img.shape}")

        return torch.from_numpy(img).unsqueeze(0).to(dtype=torch.float32)

    def __call__(self, input_img: np.ndarray, target_img: np.ndarray) -> dict[str, Any]:
        """Apply transforms, normalize, and format paired images.

        Args:
            input_img: Raw input image array.
            target_img: Raw target image array.

        Returns:
            Dictionary containing formatted ``input_image`` and ``target_image`` tensors.

        Raises:
            ValueError: If robust percentile bounds are missing or invalid.
        """

        if self.input_transform is not None:
            input_img = self.input_transform(image=input_img)["image"]

        if self.target_transform is not None:
            target_img = self.target_transform(image=target_img)["image"]

        if None in (
            self.input_lower_value,
            self.input_upper_value,
            self.target_lower_value,
            self.target_upper_value,
        ):
            raise ValueError(
                "Robust percentile normalization bounds must be set before loading data"
            )
        if (
            self.input_lower_value >= self.input_upper_value
            or self.target_lower_value >= self.target_upper_value
        ):
            raise ValueError("Robust percentile normalization bounds must be increasing")

        input_img = np.clip(input_img, self.input_lower_value, self.input_upper_value)
        input_img = (input_img - self.input_lower_value) / (
            self.input_upper_value - self.input_lower_value
        )
        input_img = np.clip(input_img, 0.0, 1.0)

        target_img = np.clip(target_img, self.target_lower_value, self.target_upper_value)
        target_img = (target_img - self.target_lower_value) / (
            self.target_upper_value - self.target_lower_value
        )
        target_img = np.clip(target_img, 0.0, 1.0)

        return {
            "input_image": self.format_img(input_img),
            "target_image": self.format_img(target_img),
        }
