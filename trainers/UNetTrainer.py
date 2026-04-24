from typing import Any, Union

import torch
from torch.utils.data import DataLoader

from metrics.AbstractMetric import AbstractMetric


class UNetTrainer:
    """
    Orchestrates training and evaluation of image-to-image translation models.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_optimizer: torch.optim.Optimizer,
        model_loss: AbstractMetric,
        train_dataloader: Union[torch.utils.data.Dataset, DataLoader],
        val_dataloader: Union[torch.utils.data.Dataset, DataLoader],
        callbacks: Any,
        image_postprocessor: Any = lambda x: x,
        epochs: int = 10,
        device: Union[str, torch.device] = "cuda",
        use_amp: bool = True,
        max_train_batches: int | None = None,
    ) -> None:
        """Initialize trainer state and optional AMP scaler.

        Args:
            model: Trainable image-to-image model.
            model_optimizer: Optimizer used for parameter updates.
            model_loss: Loss module used for backpropagation.
            train_dataloader: Training dataloader.
            val_dataloader: Validation dataloader used by callbacks.
            callbacks: Callback dispatcher used for hooks and logging.
            image_postprocessor: Postprocessor applied to model outputs.
            epochs: Maximum number of training epochs.
            device: Target device for model and tensors.
            use_amp: Whether to use automatic mixed precision.
            max_train_batches: Optional cap on train batches per epoch.
        """

        self.model = model
        self.model_optimizer = model_optimizer
        self.model_loss = model_loss
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.callbacks = callbacks
        self.image_postprocessor = image_postprocessor
        self.epochs = epochs
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.use_amp = use_amp  # Automatic Mixed Precision (AMP)
        self.max_train_batches = max_train_batches

        if self.use_amp:
            if self.device.type == "cuda":
                self.scaler = torch.amp.GradScaler("cuda")
            else:
                self.scaler = torch.amp.GradScaler("cpu")
        else:
            self.scaler = None

    @property
    def best_loss_value(self):
        """Expose best validation loss tracked by callbacks."""

        return self.callbacks.best_loss_value

    def train(self) -> None:
        """Run the training loop with callback hooks and optional early stopping."""

        train_data = {}
        train_data["continue_training"] = True
        train_data["device"] = self.device

        self.model = self.model.to(self.device)

        for epoch in range(self.epochs):
            train_data["epoch"] = epoch
            train_data["callback_hook"] = "on_epoch_start"
            self.callbacks(**train_data)

            self.model.train()

            for batch, batch_data in enumerate(self.train_dataloader):
                train_data["callback_hook"] = "on_batch_start"
                train_data["batch"] = batch
                train_data["batch_data"] = batch_data
                self.callbacks(**train_data)

                inputs = batch_data["input"].to(self.device)
                targets = batch_data["target"].to(self.device)

                with torch.amp.autocast(
                    enabled=self.use_amp, device_type=self.device.type
                ):
                    generated_predictions = self.image_postprocessor(self.model(inputs))
                    loss = self.model_loss(
                        targets=targets,
                        generated_predictions=generated_predictions,
                        loss_mask=batch_data.get("loss_mask"),
                    )

                train_data["generated_predictions"] = generated_predictions
                train_data["model_update_loss"] = loss

                self.model_optimizer.zero_grad()
                if self.use_amp and self.scaler is not None:
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.model_optimizer)
                    self.scaler.update()
                else:
                    loss.backward()
                    self.model_optimizer.step()

                train_data["model"] = self.model
                train_data["callback_hook"] = "on_batch_end"

                self.callbacks(**train_data)

                if not train_data["continue_training"]:
                    break

                if (
                    self.max_train_batches is not None
                    and (batch + 1) >= self.max_train_batches
                ):
                    break

            train_data["callback_hook"] = "on_epoch_end"
            train_data["continue_training"] = self.callbacks(
                train_dataloader=self.train_dataloader,
                val_dataloader=self.val_dataloader,
                **train_data,
            )

            if not train_data["continue_training"]:
                break

    def __call__(self) -> None:
        """Execute training when the trainer is called like a function."""

        self.train()
