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
        discriminator_steps_since_generator_update: int = 0,
        start_epoch: int = 0,
        checkpoint_manager: Any = None,
        trial_metadata: dict[str, Any] | None = None,
        use_amp: bool = False,
        eval_train_dataloader: Union[torch.utils.data.Dataset, DataLoader, None] = None,
        eval_val_dataloader: Union[torch.utils.data.Dataset, DataLoader, None] = None,
    ) -> None:
        """Store one training run's modules, loaders, and resume state.

        Args:
            generator: Generator module optimized during training.
            discriminator: Discriminator module optimized during training.
            generator_optimizer: Optimizer paired with the generator.
            discriminator_optimizer: Optimizer paired with the discriminator.
            generator_loss: Generator loss module.
            discriminator_loss: Discriminator loss module.
            train_dataloader: Training dataloader.
            val_dataloader: Validation dataloader.
            callbacks: Callback pipeline triggered during training.
            image_postprocessor: Postprocessor applied to generator outputs.
            epochs: Total epoch budget for the logical trial.
            device: Device used for training.
            max_train_batches: Optional cap on train batches per epoch.
            discriminator_updates_per_generator_update: Number of discriminator
                steps per generator step.
            discriminator_steps_since_generator_update: Restored cadence state
                when resuming an interrupted trial.
            start_epoch: Epoch index to resume from.
            checkpoint_manager: Optional resumable checkpoint writer.
            trial_metadata: Trial metadata persisted alongside checkpoints.
            use_amp: Whether to run standard training forwards under AMP.
            eval_train_dataloader: Optional dataloader used for epoch-end train metrics.
            eval_val_dataloader: Optional dataloader used for epoch-end validation metrics.
        """
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
        self.discriminator_steps_since_generator_update = (
            discriminator_steps_since_generator_update
        )
        self.start_epoch = start_epoch
        self.checkpoint_manager = checkpoint_manager
        self.trial_metadata = trial_metadata or {}
        self.last_completed_epoch = start_epoch - 1
        self.use_amp = use_amp
        self.amp_dtype = torch.bfloat16
        self.eval_train_dataloader = (
            train_dataloader if eval_train_dataloader is None else eval_train_dataloader
        )
        self.eval_val_dataloader = (
            val_dataloader if eval_val_dataloader is None else eval_val_dataloader
        )
        self.scaler = self._build_scaler()

    @property
    def best_loss_value(self):
        return self.callbacks.best_loss_value

    def _build_scaler(self) -> None:
        return None

    def _backward_and_step(
        self,
        *,
        loss: torch.Tensor,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        if self.scaler is not None:
            self.scaler.scale(loss).backward()
            self.scaler.step(optimizer)
            self.scaler.update()
            return

        loss.backward()
        optimizer.step()

    def amp_state_dict(self) -> dict[str, Any]:
        scaler_state = self.scaler.state_dict() if self.scaler is not None else None
        return {
            "use_amp": self.use_amp,
            "scaler_state_dict": scaler_state,
        }

    def load_amp_state_dict(self, state_dict: dict[str, Any]) -> None:
        if self.scaler is None:
            return

        scaler_state_dict = state_dict.get("scaler_state_dict")
        if scaler_state_dict is not None:
            self.scaler.load_state_dict(scaler_state_dict)

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

        for epoch in range(self.start_epoch, self.epochs):
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
                    with torch.amp.autocast(
                        enabled=self.use_amp,
                        device_type=self.device.type,
                        dtype=self.amp_dtype,
                    ):
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
                self._backward_and_step(
                    loss=discriminator_loss,
                    optimizer=self.discriminator_optimizer,
                )
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
                    with torch.amp.autocast(
                        enabled=self.use_amp,
                        device_type=self.device.type,
                        dtype=self.amp_dtype,
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
                            loss_mask=batch_data.get("loss_mask"),
                        )
                    generator_loss, generator_components = self._detach_components(
                        generator_outputs
                    )

                    self.generator_optimizer.zero_grad()
                    self._backward_and_step(
                        loss=generator_loss,
                        optimizer=self.generator_optimizer,
                    )
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
                eval_train_dataloader=self.eval_train_dataloader,
                eval_val_dataloader=self.eval_val_dataloader,
                **train_data,
            )
            self.last_completed_epoch = epoch

            if self.checkpoint_manager is not None:
                # Persist full trial state after each completed epoch so restarts
                # can continue without discarding completed work.
                self.checkpoint_manager.save(
                    generator=self.generator,
                    discriminator=self.discriminator,
                    generator_optimizer=self.generator_optimizer,
                    discriminator_optimizer=self.discriminator_optimizer,
                    next_epoch=epoch + 1,
                    callbacks_state=self.callbacks.state_dict(),
                    discriminator_steps_since_generator_update=(
                        self.discriminator_steps_since_generator_update
                    ),
                    amp_state=self.amp_state_dict(),
                    trial_metadata={
                        **self.trial_metadata,
                        "last_completed_epoch": epoch,
                        "best_loss_value": self.callbacks.best_loss_value,
                    },
                )

            if not train_data["continue_training"]:
                break

    def __call__(self) -> None:
        self.train()
