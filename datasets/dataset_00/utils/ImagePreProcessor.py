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
        input_max_pixel_value: float,
        target_max_pixel_value: float,
        **kwargs,
    ) -> None:
        """Store normalization constants inferred from cached images.

        Args:
            input_max_pixel_value: Max pixel value used to normalize inputs.
            target_max_pixel_value: Max pixel value used to normalize targets.
            **kwargs: Additional image spec keys ignored by this preprocessor.
        """

        self.input_max_pixel_value = float(input_max_pixel_value)
        self.target_max_pixel_value = float(target_max_pixel_value)

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

        return torch.from_numpy(img).unsqueeze(0).to(
            dtype=torch.float32,
            device=self.device,
        )

    def __call__(self, input_img: np.ndarray, target_img: np.ndarray) -> dict[str, Any]:
        """Apply transforms, normalize, and format paired images.

        Args:
            input_img: Raw input image array.
            target_img: Raw target image array.

        Returns:
            Dictionary containing formatted ``input_image`` and ``target_image`` tensors.

        Raises:
            ValueError: If normalization constants are non-positive.
        """

        if self.input_transform is not None:
            input_img = self.input_transform(image=input_img)["image"]

        if self.target_transform is not None:
            target_img = self.target_transform(image=target_img)["image"]

        if self.input_max_pixel_value <= 0 or self.target_max_pixel_value <= 0:
            raise ValueError("Pixel value normalization constants must be positive")

        input_img = input_img / self.input_max_pixel_value
        target_img = target_img / self.target_max_pixel_value

        return {
            "input_image": self.format_img(input_img),
            "target_image": self.format_img(target_img),
        }
