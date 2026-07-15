import torch


class ImagePostProcessor:
    """Postprocess model outputs and denormalize tensors for logging/export."""

    def __init__(
        self,
        input_mean: float,
        input_std: float,
        target_mean: float,
        target_std: float,
    ):
        """Store train-split normalization statistics for inverse transforms.

        Args:
            input_mean: Mean intensity used to z-score inputs.
            input_std: Standard deviation used to z-score inputs.
            target_mean: Mean intensity used to z-score targets.
            target_std: Standard deviation used to z-score targets.

        Raises:
            ValueError: If any standard deviation is non-positive.
        """

        if input_std <= 0 or target_std <= 0:
            raise ValueError("Z-score standard deviations must be positive")

        self.input_mean = float(input_mean)
        self.input_std = float(input_std)
        self.target_mean = float(target_mean)
        self.target_std = float(target_std)

    def __call__(self, generated_prediction: torch.Tensor) -> torch.Tensor:
        """Return model predictions in training space.

        Args:
            generated_prediction: Raw model output tensor.

        Returns:
            Prediction tensor in z-score space.
        """

        return generated_prediction

    def denormalize_input(self, image: torch.Tensor) -> torch.Tensor:
        """Map z-scored input tensors back to original intensity space."""

        return image.float() * self.input_std + self.input_mean

    def denormalize_target(self, image: torch.Tensor) -> torch.Tensor:
        """Map z-scored target or prediction tensors back to original intensity space."""

        return image.float() * self.target_std + self.target_mean
