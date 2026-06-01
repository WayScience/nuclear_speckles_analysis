from typing import Union

import torch


class StreamingScalarStats:
    """Accumulates mean and std for a stream of scalar observations."""

    def __init__(
        self,
        device: Union[str, torch.device] = "cuda",
        dtype: torch.dtype = torch.float64,
        ddof: int = 0,
    ):
        if ddof < 0:
            raise ValueError("ddof must be non-negative")

        self.device = device if isinstance(device, torch.device) else torch.device(device)
        self.dtype = dtype
        self.ddof = ddof
        self.reset()

    def reset(self) -> None:
        self.sum_x = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.sum_x_sq = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        self.n = 0

    @torch.no_grad()
    def update(self, values: torch.Tensor) -> None:
        values = values.detach().to(device=self.device, dtype=self.dtype).reshape(-1)
        if values.numel() == 0:
            return

        self.sum_x += values.sum()
        self.sum_x_sq += values.pow(2).sum()
        self.n += values.numel()

    def compute(self) -> dict[str, float]:
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

        variance = torch.clamp(variance, min=0.0)
        std = torch.sqrt(variance)

        if not torch.isfinite(mean):
            mean = torch.tensor(0.0, device=self.device, dtype=self.dtype)
        if not torch.isfinite(std):
            std = torch.tensor(0.0, device=self.device, dtype=self.dtype)

        return {
            "mean": mean.item(),
            "std": std.item(),
        }
