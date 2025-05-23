import pathlib
from collections import defaultdict
from inspect import signature
from typing import Optional

import mlflow
import torch
import utils.visualize_model_performance as vm
from farmhash import Fingerprint64
from losses.AbstractLoss import AbstractLoss
from torch.utils.data import DataLoader, random_split
import copy


class WGANGPCPPix2PixMetaTrainer:
    """
    Orchestrates training and evaluation of paired stain-to-stain translational modeling with the WGAN-GP.
    This trainer saves image samples after each epoch.
    """

    def __init__(
        self,
        _generator_model: torch.nn.Module,
        _discriminator_model: torch.nn.Module,
        _image_dataset: torch.utils.data.Dataset,
        _generator_optimizer: torch.optim.Optimizer,
        _discriminator_optimizer: torch.optim.Optimizer,
        _discriminator_loss: AbstractLoss,
        _generator_loss: AbstractLoss,
        _auxiliary_metrics: Optional[list[AbstractLoss]] = None,
        _discriminator_update_frequency: int = 1,
        _example_images_per_epoch: int = 0,
        _batch_size: int = 4,
        _epochs: int = 10,
        _patience: int = 5,
        _save_pretrained_generated_imgs: bool = False
    ):

        self._generator_model = _generator_model
        self._discriminator_model = _discriminator_model
        self._generator_optimizer = _generator_optimizer
        self._discriminator_optimizer = _discriminator_optimizer

        self._discriminator_loss = _discriminator_loss
        self._generator_loss = _generator_loss
        self._auxiliary_metrics = _auxiliary_metrics

        self._discriminator_update_frequency = _discriminator_update_frequency
        self._example_images_per_epoch = _example_images_per_epoch
        self._batch_size = _batch_size
        self._epochs = _epochs

        # Also known as an early stopping counter threshold
        self._patience = _patience

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._best_model = None
        self._best_loss = float("inf")
        self._early_stop_counter = 0
        self._save_pretrained_generated_imgs = _save_pretrained_generated_imgs

        if self._save_pretrained_generated_imgs:
            self._pretrained_model = copy.deepcopy(_generator_model)

        # Set the fixed datasplits
        train_size = int(0.7 * len(_image_dataset))
        val_size = int(0.15 * len(_image_dataset))
        test_size = len(_image_dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(
            _image_dataset, [train_size, val_size, test_size]
        )

        # Create dataLoaders for each dataset
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=_batch_size, shuffle=True
        )
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=_batch_size, shuffle=False
        )

        self._generated_images_path = pathlib.Path("generated_image_epoch_montage")
        self._generated_images_path.mkdir(parents=True, exist_ok=True)

    def _log_metrics(self, _metrics: defaultdict[str, float], _datasplit: str):

        for metric_name, metric_value in _metrics.items():
            mlflow.log_metric(
                f"{_datasplit}_batch_averaged_{metric_name}_per_epoch",
                metric_value,
                step=self._epoch,
            )

    def train(self):
        self._generator_model.to(self._device)
        self._discriminator_model.to(self._device)

        for epoch in range(self._epochs):
            training_losses = defaultdict(float)
            auxiliary_metric_values = defaultdict(float)

            self._generator_model.train()
            self._discriminator_model.train()
            self._epoch = epoch
            print(f"Starting epoch {self._epoch}")

            # Run to see if this works
            for idx, (inputs, targets, names) in enumerate(self.train_loader):
                inputs, targets = inputs.to(self._device), targets.to(self._device)

                # Generators Loss
                generated_outputs = self._generator_model(inputs)
                fake_classification_outputs = self._discriminator_model(inputs)
                generator_loss = self._generator_loss(
                    _fake_classification_outputs=fake_classification_outputs,
                    _generator_outputs=generated_outputs,
                    _targets=targets,
                    _target_names=names["target_name"],
                )

                # Update the Generator
                self._generator_optimizer.zero_grad()
                generator_loss.backward()
                self._generator_optimizer.step()

                # Discriminators loss
                generated_outputs = self._generator_model(inputs).detach()

                # Gradient penalty
                alpha = torch.rand(targets.shape[0], 1, 1, 1, device=self._device)

                interpolates = (
                    alpha * targets + (1 - alpha) * generated_outputs
                ).requires_grad_(True)
                interpolates_classification_outputs = self._discriminator_model(
                    interpolates
                )

                if epoch % self._discriminator_update_frequency == 0:

                    gradients = torch.autograd.grad(
                        outputs=interpolates_classification_outputs,
                        inputs=interpolates,
                        grad_outputs=torch.ones_like(
                            interpolates_classification_outputs
                        ),
                        create_graph=True,
                    )[0]

                    real_classification_outputs = self._discriminator_model(targets)
                    fake_classification_outputs = self._discriminator_model(
                        generated_outputs
                    )

                    discriminator_loss = self._discriminator_loss(
                        _gradients=gradients,
                        _real_classification_outputs=real_classification_outputs,
                        _fake_classification_outputs=fake_classification_outputs,
                    )

                    # Update the Discriminator
                    self._discriminator_optimizer.zero_grad()
                    discriminator_loss.backward()
                    self._discriminator_optimizer.step()

                else:

                    gradients = torch.autograd.grad(
                        outputs=interpolates_classification_outputs,
                        inputs=interpolates,
                        grad_outputs=torch.ones_like(
                            interpolates_classification_outputs
                        ),
                        create_graph=True,
                    )[0].detach()

                    real_classification_outputs = self._discriminator_model(
                        targets
                    ).detach()
                    fake_classification_outputs = self._discriminator_model(
                        generated_outputs
                    ).detach()

                    discriminator_loss = self._discriminator_loss(
                        _gradients=gradients,
                        _real_classification_outputs=real_classification_outputs,
                        _fake_classification_outputs=fake_classification_outputs,
                    )

                training_losses[self._generator_loss.metric_name] += generator_loss
                training_losses[
                    self._discriminator_loss.metric_name
                ] += discriminator_loss

                if self._auxiliary_metrics:

                    auxiliary_metric_parameters = {
                        "_generated_outputs": generated_outputs,
                        "_fake_classification_outputs": fake_classification_outputs,
                        "_real_classification_outputs": real_classification_outputs,
                        "_targets": targets,
                    }

                    for metric in self._auxiliary_metrics:
                        param_names = signature(metric.forward).parameters.keys()
                        parameters = {
                            k: v
                            for k, v in auxiliary_metric_parameters.items()
                            if k in list(param_names)
                        }
                        auxiliary_metric_values[metric.metric_name] += metric(
                            **parameters
                        )

                if idx > 40:
                    break

            if self._auxiliary_metrics:
                auxiliary_metric_values = {
                    loss_name: loss / len(self.train_loader)
                    for loss_name, loss in auxiliary_metric_values.items()
                }
                training_losses = training_losses | auxiliary_metric_values

            self._log_metrics(training_losses, "train")

            validation_losses = self.evaluation_losses(_data_loader=self.val_loader)
            self._log_metrics(validation_losses, "validation")

            # Define the validation loss to use for early stopping
            if validation_losses[self._generator_loss.metric_name] < self._best_loss:
                self._best_loss = validation_losses[self._generator_loss.metric_name]
                self._early_stop_counter = 0
                self._best_generator_model = self._generator_model
                self._best_discriminator_model = self._discriminator_model

            else:
                self._early_stop_counter += 1

            mlflow.log_metric(
                "early_stopping_counter_per_epoch",
                self._early_stop_counter,
                step=self._epoch,
            )

            if self._early_stop_counter >= self._patience:
                break

        return (
            self._best_loss,
            self._best_generator_model,
            self._best_discriminator_model,
        )

    def visualize_per_epoch_examples(
        self,
        _inputs: torch.Tensor,
        _outputs: torch.Tensor,
        _targets: torch.Tensor,
        _data_loader: DataLoader,
        _input_name: str,
    ):
        """Use the image's unique name to consistently decide if the image should be sampled based on the predetermined target frequency."""

        if self._example_images_per_epoch > 0:

            num_cells = len(_data_loader.dataset)
            divisor = 10_000
            mod_cutoff = (self._example_images_per_epoch / num_cells) * divisor
            normalization_factor = self.val_dataset.dataset.input_transform[
                0
            ].normalization_factor

            if self._save_pretrained_generated_imgs:
                _pretrained_outputs = self._pretrained_model(_inputs).detach()

            pretrained_outputs = None

            for input_idx, input_name in enumerate(_input_name):

                # Remove channel component of name
                start_ch_idx = input_name.find("CH")
                input_name = (
                    input_name[: start_ch_idx - 1] + input_name[start_ch_idx + 3 :]
                )

                input_name = input_name.replace("_illumcorrect.tiff", "")
                if mod_cutoff > (Fingerprint64(input_name) % divisor):

                    imgs_path = self._generated_images_path / input_name
                    imgs_path.mkdir(parents=True, exist_ok=True)
                    img_path = imgs_path / f"epoch_{self._epoch}_{input_name}"

                    input = vm.format_img(
                        _tensor_img=_inputs[input_idx].unsqueeze(1),
                        _normalization_factor=normalization_factor,
                    )

                    output = vm.format_img(
                        _tensor_img=_outputs[input_idx],
                        _normalization_factor=normalization_factor,
                    )

                    if self._save_pretrained_generated_imgs:
                        pretrained_outputs = vm.format_img(
                            _tensor_img=_pretrained_outputs[input_idx],
                            _normalization_factor=normalization_factor,
                        )

                    target = vm.format_img(
                        _tensor_img=_targets[input_idx],
                        _normalization_factor=normalization_factor,
                    )

                    vm.visualize_stains(
                        _input=input,
                        _output=output,
                        _target=target,
                        _image_path=img_path,
                        _title=input_name,
                        _pretrained_output=pretrained_outputs
                    )

    def evaluation_losses(self, _data_loader: torch.utils.data.DataLoader):
        """Computes the loss for an evaluation datasplit, e.g. validation or testing."""

        self._generator_model.eval()
        self._discriminator_model.eval()
        losses = defaultdict(float)
        auxiliary_metric_values = defaultdict(float)

        for inputs, targets, metadata in _data_loader:
            inputs, targets = inputs.to(self._device), targets.to(self._device)

            generated_outputs = self._generator_model(inputs).detach()
            fake_classification_outputs = self._discriminator_model(inputs).detach()
            generator_loss = self._generator_loss(
                _fake_classification_outputs=fake_classification_outputs,
                _generator_outputs=generated_outputs,
                _targets=targets,
                _target_names=metadata["target_name"],
            )

            self.visualize_per_epoch_examples(
                _inputs=inputs,
                _outputs=generated_outputs,
                _targets=targets,
                _data_loader=_data_loader,
                _input_name=metadata["input_name"],
            )

            real_classification_outputs = self._discriminator_model(targets).detach()

            discriminator_loss = self._discriminator_loss(
                _gradients=torch.zeros_like(targets),
                _real_classification_outputs=real_classification_outputs,
                _fake_classification_outputs=fake_classification_outputs,
            )

            losses[self._generator_loss.metric_name] += generator_loss
            losses[self._discriminator_loss.metric_name] += discriminator_loss

            if self._auxiliary_metrics:

                auxiliary_metric_parameters = {
                    "_generated_outputs": generated_outputs,
                    "_fake_classification_outputs": fake_classification_outputs,
                    "_real_classification_outputs": real_classification_outputs,
                    "_targets": targets,
                }

                for metric in self._auxiliary_metrics:
                    param_names = signature(metric.forward).parameters.keys()
                    parameters = {
                        k: v
                        for k, v in auxiliary_metric_parameters.items()
                        if k in list(param_names)
                    }
                    auxiliary_metric_values[metric.metric_name] += metric(**parameters)

        losses = {
            loss_name: loss / len(_data_loader) for loss_name, loss in losses.items()
        }

        if auxiliary_metric_values:
            auxiliary_metric_values = {
                loss_name: loss / len(self.train_loader)
                for loss_name, loss in auxiliary_metric_values.items()
            }
            losses = losses | auxiliary_metric_values

        return losses
