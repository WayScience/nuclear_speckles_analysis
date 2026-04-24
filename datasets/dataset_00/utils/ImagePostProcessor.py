from typing import Any, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import autocast


class ImagePostProcessor:
    """
    Processes generated predictions computed from the model.
    """

    def __init__(self):
        """Initialize postprocessing behavior for generated predictions."""

        pass

    def __call__(self, generated_prediction: torch.Tensor) -> torch.Tensor:
        """Apply output activation to model logits.

        Args:
            generated_prediction: Raw model output tensor.

        Returns:
            Sigmoid-transformed prediction tensor.
        """

        return torch.sigmoid(generated_prediction)
