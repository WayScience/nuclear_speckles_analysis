from typing import Any

import mlflow
import torch
from mlflow.models import ModelSignature
from mlflow.models.signature import infer_signature
from torch.nn import Module

from callbacks.base import BaseCallback


class EarlyStoppingAndCheckpointCallback(BaseCallback):
    """Track best loss, save checkpoints, and control early stopping."""

    def __init__(
        self,
        early_stopping_counter_threshold: int,
        image_postprocessor: Any = lambda x: x,
    ) -> None:
        self.early_stopping_counter_threshold = early_stopping_counter_threshold
        self.image_postprocessor = image_postprocessor
        self.best_loss_value = float("inf")
        self.early_stopping_counter = 0

    def on_epoch_end(self, context: dict[str, Any]) -> None:
        epoch = context["epoch"]
        model = context["model"]
        val_dataloader = context["val_dataloader"]
        loss_value = context["loss_value"]

        val_sample = next(iter(val_dataloader))
        signature = self._prepare_signature(input_example=val_sample["input"], model=model)

        if self.best_loss_value > loss_value:
            self.best_loss_value = loss_value
            self.early_stopping_counter = 0

            mlflow.pytorch.log_model(
                model,
                name="model",
                signature=signature,
                step=epoch,
            )
            context["continue_training"] = True
            return

        self.early_stopping_counter += 1
        if self.early_stopping_counter >= self.early_stopping_counter_threshold:
            print(f"Early stopping triggered at epoch {epoch}")
            mlflow.log_param("early_stopping_epoch", epoch)
            context["continue_training"] = False

    def _prepare_signature(
        self, input_example: torch.Tensor, model: Module
    ) -> ModelSignature:
        model.eval()
        with torch.no_grad():
            output_example = (
                self.image_postprocessor(model(input_example)).detach().cpu().numpy()
            )

        input_numpy = input_example.detach().cpu().numpy().astype("float32")
        return infer_signature(input_numpy, output_example)
