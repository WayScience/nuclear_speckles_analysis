import torch


class ImagePostProcessor:
    """Postprocess model outputs and denormalize tensors for logging/export."""

    def __init__(
        self,
        input_lower_value: float,
        input_upper_value: float,
        target_lower_value: float,
        target_upper_value: float,
    ):
        """Store train-split robust percentile bounds for inverse transforms.

        Args:
            input_lower_value: Lower train-split clipping bound for inputs.
            input_upper_value: Upper train-split clipping bound for inputs.
            target_lower_value: Lower train-split clipping bound for targets.
            target_upper_value: Upper train-split clipping bound for targets.

        Raises:
            ValueError: If any percentile bounds are not strictly increasing.
        """

        if input_lower_value >= input_upper_value or target_lower_value >= target_upper_value:
            raise ValueError("Robust percentile normalization bounds must be increasing")

        self.input_lower_value = float(input_lower_value)
        self.input_upper_value = float(input_upper_value)
        self.target_lower_value = float(target_lower_value)
        self.target_upper_value = float(target_upper_value)

    def __call__(self, generated_prediction: torch.Tensor) -> torch.Tensor:
        """Return model predictions in normalized training space.

        Args:
            generated_prediction: Raw model output tensor.

        Returns:
            Prediction tensor in robust percentile-normalized space.
        """

        return generated_prediction

    def denormalize_input(self, image: torch.Tensor) -> torch.Tensor:
        """Map normalized input tensors back to original intensity space."""

        image = image.float().clamp(0.0, 1.0)
        return image * (self.input_upper_value - self.input_lower_value) + self.input_lower_value

    def denormalize_target(self, image: torch.Tensor) -> torch.Tensor:
        """Map normalized target or prediction tensors back to original intensity space."""

        image = image.float().clamp(0.0, 1.0)
        return image * (self.target_upper_value - self.target_lower_value) + self.target_lower_value
