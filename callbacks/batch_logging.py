from typing import Any

import mlflow

from callbacks.base import BaseCallback


class BatchLossMlflowLoggerCallback(BaseCallback):
    """Log per-batch training loss components to MLflow."""

    def __init__(self, batch_metric_log_every_n: int = 1) -> None:
        if batch_metric_log_every_n <= 0:
            raise ValueError("batch_metric_log_every_n must be a positive integer")
        self.batch_metric_log_every_n = batch_metric_log_every_n
        self.global_batch_step = 0

    def on_batch_end(self, hook_data: dict[str, Any]) -> None:
        if self.global_batch_step % self.batch_metric_log_every_n != 0:
            self.global_batch_step += 1
            return

        batch_loss_components = hook_data.get("batch_loss_components", {})
        batch_loss_name = hook_data.get("batch_loss_name", "unknown_loss")
        for loss_name, loss_value in batch_loss_components.items():
            mlflow.log_metric(
                f"batch/train/{batch_loss_name}/{loss_name}",
                loss_value,
                step=self.global_batch_step,
            )

        self.global_batch_step += 1
