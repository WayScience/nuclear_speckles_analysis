from typing import Any

from callbacks.base import BaseCallback


class ProgressLoggerCallback(BaseCallback):
    """Print progress messages at epoch and configurable batch intervals."""

    def __init__(self, batch_log_every_n: int = 50) -> None:
        if batch_log_every_n <= 0:
            raise ValueError("batch_log_every_n must be a positive integer")
        self.batch_log_every_n = batch_log_every_n

    def on_epoch_start(self, context: dict[str, Any]) -> None:
        print(f"Starting epoch {context['epoch']}")

    def on_batch_start(self, context: dict[str, Any]) -> None:
        batch = context["batch"]
        if batch % self.batch_log_every_n == 0:
            print(f"Starting batch {batch}")
