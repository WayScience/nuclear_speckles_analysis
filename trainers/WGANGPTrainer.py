from typing import Any, Union

import torch
from torch.utils.data import DataLoader


class WGANGPTrainer:
    """Train a generator-discriminator pair with unconditional WGAN-GP."""

    def __init__(
        self,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        generator_optimizer: torch.optim.Optimizer,
        discriminator_optimizer: torch.optim.Optimizer,
        generator_loss: torch.nn.Module,
        discriminator_loss: torch.nn.Module,
        train_dataloader: Union[torch.utils.data.Dataset, DataLoader],
        val_dataloader: Union[torch.utils.data.Dataset, DataLoader],
        callbacks: Any,
        image_postprocessor: Any = lambda x: x,
        epochs: int = 10,
        device: Union[str, torch.device] = "cuda",
        max_train_batches: int | None = None,
        discriminator_updates_per_generator_update: int = 1,
    ) -> None:
        if discriminator_updates_per_generator_update <= 0:
            raise ValueError(
                "discriminator_updates_per_generator_update must be a positive integer"
            )

        self.generator = generator
        self.discriminator = discriminator
        self.generator_optimizer = generator_optimizer
        self.discriminator_optimizer = discriminator_optimizer
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.callbacks = callbacks
        self.image_postprocessor = image_postprocessor
        self.epochs = epochs
        self.device = (
            device if isinstance(device, torch.device) else torch.device(device)
        )
        self.max_train_batches = max_train_batches
        self.discriminator_updates_per_generator_update = (
            discriminator_updates_per_generator_update
        )
        self.discriminator_steps_since_generator_update = 0

    @property
    def best_loss_value(self):
        return self.callbacks.best_loss_value

    @staticmethod
    def _detach_components(
        batch_loss_components: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if not isinstance(batch_loss_components, dict):
            raise TypeError("Loss modules must return dict[str, torch.Tensor].")
        if "total" not in batch_loss_components:
            raise ValueError("Loss module output must include a 'total' key.")

        loss = batch_loss_components["total"]
        if not torch.is_tensor(loss) or loss.ndim != 0:
            raise ValueError("Loss module output 'total' must be a scalar torch.Tensor.")

        detached_loss_components: dict[str, float] = {}
        for name, value in batch_loss_components.items():
            if not torch.is_tensor(value) or value.ndim != 0:
                raise ValueError(f"Loss module output '{name}' must be a scalar torch.Tensor.")
            detached_loss_components[name] = value.detach().item()

        return loss, detached_loss_components

    def train(self) -> None:
        train_data = {"continue_training": True, "device": self.device}

        self.generator = self.generator.to(self.device)
        self.discriminator = self.discriminator.to(self.device)

        for epoch in range(self.epochs):
            train_data["epoch"] = epoch
            train_data["callback_hook"] = "on_epoch_start"
            self.callbacks(**train_data)

            self.generator.train()
            self.discriminator.train()

            for batch, batch_data in enumerate(self.train_dataloader):
                train_data["callback_hook"] = "on_batch_start"
                train_data["batch"] = batch
                train_data["batch_data"] = batch_data
                self.callbacks(**train_data)

                inputs = batch_data["input"].to(self.device)
                targets = batch_data["target"].to(self.device)

                with torch.no_grad():
                    fake_targets_for_discriminator = self.image_postprocessor(
                        self.generator(inputs)
                    )
                discriminator_outputs = self.discriminator_loss(
                    critic=self.discriminator,
                    real_samples=targets,
                    fake_samples=fake_targets_for_discriminator,
                )
                discriminator_loss, discriminator_components = self._detach_components(
                    discriminator_outputs
                )

                self.discriminator_optimizer.zero_grad()
                discriminator_loss.backward()
                self.discriminator_optimizer.step()
                self.discriminator_steps_since_generator_update += 1

                train_data.pop("generated_predictions", None)
                train_data.pop("model_update_loss", None)
                train_data.pop("batch_loss_name", None)
                train_data["batch_loss_components"] = discriminator_components
                train_data["batch_loss_groups"] = {
                    "discriminator": discriminator_components,
                }

                if (
                    self.discriminator_steps_since_generator_update
                    >= self.discriminator_updates_per_generator_update
                ):
                    generated_predictions = self.image_postprocessor(
                        self.generator(inputs)
                    )
                    fake_classification_outputs = self.discriminator(
                        generated_predictions
                    )
                    generator_outputs = self.generator_loss(
                        fake_classification_outputs=fake_classification_outputs,
                        generated_predictions=generated_predictions,
                        targets=targets,
                        epoch=epoch,
                        loss_mask=batch_data.get("loss_mask"),
                    )
                    generator_loss, generator_components = self._detach_components(
                        generator_outputs
                    )

                    self.generator_optimizer.zero_grad()
                    generator_loss.backward()
                    self.generator_optimizer.step()
                    self.discriminator_steps_since_generator_update = 0

                    train_data["generated_predictions"] = generated_predictions
                    train_data["model_update_loss"] = generator_loss
                    train_data["batch_loss_components"] = generator_components
                    train_data["batch_loss_name"] = getattr(
                        self.generator_loss,
                        "loss_name",
                        self.generator_loss.__class__.__name__,
                    )
                    train_data["batch_loss_groups"] = {
                        "generator": generator_components,
                        "discriminator": discriminator_components,
                    }

                train_data["model"] = self.generator
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
        self.train()
