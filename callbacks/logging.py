from typing import Any, List

import mlflow

from callbacks.base import BaseCallback


class MetricsMlflowLoggerCallback(BaseCallback):
    """Log aggregated loss and metric data to MLflow."""

    def __init__(self, metrics: List, loss: Any) -> None:
        """Store metric objects used for MLflow logging.

        Args:
            metrics: Non-loss metric objects exposing ``get_metric_data``.
            loss: Loss metric object exposing ``get_metric_data``.
        """
        self.metrics = metrics
        self.loss = loss

    def on_epoch_end(self, hook_data: dict[str, Any]) -> None:
        """Log epoch-level metrics and store loss for downstream callbacks.

        Args:
            hook_data: Shared hook payload containing epoch index and outputs.

        Raises:
            ValueError: If a primary loss metric key cannot be identified.
        """

        time_step = hook_data["epoch"]
        epoch_metric_data = hook_data["epoch_metric_data"]

        for name, metric_value in epoch_metric_data.items():
            mlflow.log_metric(name, metric_value, step=time_step)

        validation_loss_key = f"{self.loss.metric_name}_validation"
        if validation_loss_key not in epoch_metric_data:
            raise ValueError(f"Missing expected validation loss key: {validation_loss_key}")
        hook_data["loss_value"] = epoch_metric_data[validation_loss_key]
