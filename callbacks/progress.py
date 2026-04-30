from typing import Any

from callbacks.base import BaseCallback


class ProgressLoggerCallback(BaseCallback):
    """Print progress messages at epoch and configurable batch intervals."""

    def __init__(self, batch_log_every_n: int = 50) -> None:
        """Initialize progress logging frequency.

        Args:
            batch_log_every_n: Batch interval for progress messages.

        Raises:
            ValueError: If ``batch_log_every_n`` is not positive.
        """

        if batch_log_every_n <= 0:
            raise ValueError("batch_log_every_n must be a positive integer")
        self.batch_log_every_n = batch_log_every_n

    def on_epoch_start(self, hook_data: dict[str, Any]) -> None:
        """Print epoch start progress.

        Args:
            hook_data: Shared hook payload containing current epoch.
        """

        print(f"Starting epoch {hook_data['epoch']}")

    def on_batch_start(self, hook_data: dict[str, Any]) -> None:
        """Print batch progress at configured intervals.

        Args:
            hook_data: Shared hook payload containing current batch index.
        """

        batch = hook_data["batch"]
        if batch % self.batch_log_every_n == 0:
            print(f"Starting batch {batch}")
