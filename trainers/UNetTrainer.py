from typing import Any, Union

import torch
from torch import nn
from torch.utils.data import DataLoader


class UNetTrainer:
    """
    Orchestrates training and evaluation of image-to-image translation models.

    Optimization batches and epoch-end evaluation can use different dataloaders
    so training throughput and metric aggregation can be tuned independently.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        model_optimizer: torch.optim.Optimizer,
        model_loss: nn.Module,
        train_dataloader: Union[torch.utils.data.Dataset, DataLoader],
        val_dataloader: Union[torch.utils.data.Dataset, DataLoader],
        callbacks: Any,
        image_postprocessor: Any = lambda x: x,
        epochs: int = 10,
        device: Union[str, torch.device] = "cuda",
        use_amp: bool = True,
        eval_train_dataloader: Union[torch.utils.data.Dataset, DataLoader, None] = None,
        eval_val_dataloader: Union[torch.utils.data.Dataset, DataLoader, None] = None,
        max_train_batches: int | None = None,
    ) -> None:
        """Initialize trainer state and optional AMP-enabled forward passes.

        Args:
            model: Trainable image-to-image model.
            model_optimizer: Optimizer used for parameter updates.
            model_loss: Loss module used for backpropagation. Must return
                ``dict[str, torch.Tensor]`` with required scalar key
                ``"total"`` and optional additional scalar components.
            train_dataloader: Training dataloader.
            val_dataloader: Validation dataloader used by callbacks.
            callbacks: Callback dispatcher used for hooks and logging.
            image_postprocessor: Postprocessor applied to model outputs before
                the optimization loss consumes them.
            epochs: Maximum number of training epochs.
            device: Target device for model and tensors.
            use_amp: Whether to use automatic mixed precision autocast during
                training forward and loss computation.
            eval_train_dataloader: Optional dataloader used for epoch-end train metrics.
            eval_val_dataloader: Optional dataloader used for epoch-end validation metrics.
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
        self.amp_dtype = torch.bfloat16
        # Evaluation loaders can be non-shuffled or use a different batch size
        # without affecting the optimization dataloaders.
        self.eval_train_dataloader = (
            train_dataloader if eval_train_dataloader is None else eval_train_dataloader
        )
        self.eval_val_dataloader = (
            val_dataloader if eval_val_dataloader is None else eval_val_dataloader
        )
        self.max_train_batches = max_train_batches
        # Stable loss identifier used to namespace batch metrics in MLflow.
        self.loss_name = getattr(self.model_loss, "loss_name", self.model_loss.__class__.__name__)

    @property
    def best_loss_value(self):
        """Expose best validation loss tracked by callbacks."""

        return self.callbacks.best_loss_value

    def train(self) -> None:
        """Run the training loop with callback hooks and optional early stopping.

        The trainer enforces a strict loss output contract:
        ``model_loss(...) -> dict[str, torch.Tensor]`` with required scalar key
        ``"total"`` used for optimization.

        At each batch end, callback hook data includes:
            - ``model_update_loss``: Scalar tensor used for backward pass.
            - ``batch_loss_name``: Stable loss identifier for metric naming.
            - ``batch_loss_components``: Detached scalar loss components as
              ``dict[str, float]`` for logging.
        """

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
                    enabled=self.use_amp,
                    device_type=self.device.type,
                    dtype=self.amp_dtype,
                ):
                    generated_predictions = self.image_postprocessor(self.model(inputs))
                    batch_loss_components = self.model_loss(
                        targets=targets,
                        generated_predictions=generated_predictions,
                        loss_mask=batch_data.get("loss_mask"),
                    )

                if not isinstance(batch_loss_components, dict):
                    raise TypeError("model_loss must return dict[str, torch.Tensor].")
                if "total" not in batch_loss_components:
                    raise ValueError("model_loss output must include a 'total' key.")

                loss = batch_loss_components["total"]
                if not torch.is_tensor(loss) or loss.ndim != 0:
                    raise ValueError("model_loss['total'] must be a scalar torch.Tensor.")

                detached_loss_components: dict[str, float] = {}
                for name, value in batch_loss_components.items():
                    if not torch.is_tensor(value) or value.ndim != 0:
                        raise ValueError(
                            f"model_loss['{name}'] must be a scalar torch.Tensor."
                        )
                    detached_loss_components[name] = value.detach().item()

                train_data["generated_predictions"] = generated_predictions
                train_data["model_update_loss"] = loss
                train_data["batch_loss_components"] = detached_loss_components
                train_data["batch_loss_name"] = self.loss_name

                # This training path uses autocast for forward/loss computation
                # but performs a standard backward/update step.
                self.model_optimizer.zero_grad()
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
                eval_train_dataloader=self.eval_train_dataloader,
                eval_val_dataloader=self.eval_val_dataloader,
                **train_data,
            )

            if not train_data["continue_training"]:
                break

    def __call__(self) -> None:
        """Execute training when the trainer is called like a function."""

        self.train()
