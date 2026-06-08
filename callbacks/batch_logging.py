from typing import Any

import mlflow

from callbacks.base import BaseCallback


class BatchLossMlflowLoggerCallback(BaseCallback):
    """Log per-batch training loss components to MLflow."""

    def __init__(self, batch_metric_log_every_n: int = 1) -> None:
        """Configure per-batch MLflow logging frequency.

        Args:
            batch_metric_log_every_n: Log interval for batch metrics. A value of
                ``1`` logs every batch, ``n`` logs every n-th batch.

        Raises:
            ValueError: If ``batch_metric_log_every_n`` is not positive.
        """
        if batch_metric_log_every_n <= 0:
            raise ValueError("batch_metric_log_every_n must be a positive integer")
        self.batch_metric_log_every_n = batch_metric_log_every_n
        self.global_batch_step = 0

    def on_batch_end(self, hook_data: dict[str, Any]) -> None:
        """Log batch loss components to MLflow using stable metric names.

        Expected hook payload:
            - ``batch_loss_name``: Loss identifier (for example ``"l1"``).
            - ``batch_loss_components``: Mapping of component name to scalar
              float values (for example ``{"total": 0.12}``).

        Metrics are logged under ``batch/train/<loss_name>/<component>``.

        Args:
            hook_data: Shared callback payload for the current batch.
        """
        if self.global_batch_step % self.batch_metric_log_every_n != 0:
            self.global_batch_step += 1
            return

        batch_loss_groups = hook_data.get("batch_loss_groups")
        if batch_loss_groups is not None:
            for group_name, batch_loss_components in batch_loss_groups.items():
                for loss_name, loss_value in batch_loss_components.items():
                    mlflow.log_metric(
                        f"batch/train/{group_name}/{loss_name}",
                        loss_value,
                        step=self.global_batch_step,
                    )
        else:
            batch_loss_components = hook_data.get("batch_loss_components", {})
            batch_loss_name = hook_data.get("batch_loss_name", "unknown_loss")
            for loss_name, loss_value in batch_loss_components.items():
                mlflow.log_metric(
                    f"batch/train/{batch_loss_name}/{loss_name}",
                    loss_value,
                    step=self.global_batch_step,
                )

        self.global_batch_step += 1
