"""Utility for streaming scalar metric aggregation.

This module provides a reusable accumulator for per-sample scalar metric values
so metric classes can share one implementation of running mean and standard
deviation computation.
"""

from typing import Union

import torch


class StreamingScalarStats:
    """Accumulate mean and std for a stream of scalar observations.

    By default, variance is computed as population variance:
        var = E[x^2] - E[x]^2
    If ``ddof=1``, the variance estimate is scaled for sample variance.
    """

    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float64,
        ddof: int = 0,
    ):
        """Initialize streaming scalar statistics state.

        Args:
            device: Device used to store accumulation tensors.
            dtype: Floating dtype used for accumulation precision.
            ddof: Delta degrees of freedom for variance correction.

        Raises:
            ValueError: If ``ddof`` is negative.
        """

        if ddof < 0:
            raise ValueError("ddof must be non-negative")

        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.dtype = dtype
        self.ddof = ddof
        self.reset()

    def reset(self) -> None:
        """Reset accumulated sums and count back to empty state."""

        self.sum_x = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.sum_x_sq = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        # Keep count as a Python int for exact increment semantics.
        self.n = 0

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        """Update state with a tensor of scalar observations.

        Args:
            values: Tensor containing scalar observations. Any shape is accepted
                and flattened to 1D.
        """

        values = values.detach().to(device=self.device, dtype=self.dtype).reshape(-1)
        if values.numel() == 0:
            return

        self.sum_x += values.sum()
        self.sum_x_sq += values.pow(2).sum()
        self.n += values.numel()

    def compute(self) -> dict[str, float]:
        """Compute current mean and std without resetting state.

        Returns:
            Dictionary with ``mean`` and ``std`` as Python floats. Returns zeros
            when no observations were accumulated.
        """

        if self.n == 0:
            return {
                "mean": 0.0,
                "std": 0.0,
            }

        n = torch.tensor(float(self.n), device=self.device, dtype=self.dtype)
        mean = self.sum_x / n

        denom = self.n - self.ddof
        if denom <= 0:
            variance = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        else:
            denom_t = torch.tensor(float(denom), device=self.device, dtype=self.dtype)
            correction = n / denom_t
            variance = correction * (self.sum_x_sq / n - mean.pow(2))

        # Protect against tiny negative values from floating-point roundoff.
        variance = torch.clamp(variance, min=0.0)
        std = torch.sqrt(variance)

        # Guard downstream logging from NaN/Inf propagation.
        if not torch.isfinite(mean):
            mean = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        if not torch.isfinite(std):
            std = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        return {
            "mean": mean.item(),
            "std": std.item(),
        }
