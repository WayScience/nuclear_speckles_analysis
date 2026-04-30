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

        loss_value = None
        for name, current_value in self.loss.get_metric_data().items():
            # Early stopping expects a single primary loss value.
            if "loss" in name and "component" not in name:
                loss_value = current_value

            mlflow.log_metric(name, current_value, step=time_step)

        if loss_value is None:
            raise ValueError(
                "The loss name should contain the string 'loss' and shouldn't contain the string 'component'"
            )

        hook_data["loss_value"] = loss_value

        for metric in self.metrics:
            for name, metric_value in metric.get_metric_data().items():
                mlflow.log_metric(name, metric_value, step=time_step)
