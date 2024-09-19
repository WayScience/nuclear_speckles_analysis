from collections import defaultdict

import mlflow
import torch
from torch.utils.data import DataLoader, random_split


class ModelTrainer:
    """
    Orchestrates training and evaluation of paired stain-to-stain translational modeling.
    """

    def __init__(
        self,
        _model: torch.nn.Module,
        _image_dataset: torch.utils.data.Dataset,
        _optimizer: torch.optim.Optimizer,
        _tracked_losses: dict[str, torch.nn.Module],
        _backprop_loss_name: str,
        _batch_size: int = 4,
        _epochs: int = 10,
        _patience: int = 5
    ):

        self.__model = _model
        self.__optimizer = _optimizer
        self.__tracked_losses = _tracked_losses

        # The name of the loss used for backpropagation
        self.__backprop_loss_name = _backprop_loss_name
        self.__batch_size = _batch_size
        self.__epochs = _epochs
        self.__patience = _patience

        self.__device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.__best_model = None
        self.__best_loss = float('inf')
        self.__early_stop_counter = 0

        # Set the fixed datasplits
        train_size = int(0.7 * len(_image_dataset))
        val_size = int(0.15 * len(_image_dataset))
        test_size = len(_image_dataset) - train_size - val_size
        self.train_dataset, self.val_dataset, self.test_dataset = random_split(_image_dataset, [train_size, val_size, test_size])

        # Create dataLoaders for each dataset
        self.train_loader = DataLoader(self.train_dataset, batch_size=_batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=_batch_size, shuffle=False)

    def __log_losses(self, _losses, _datasplit):

        for loss_name in self.__tracked_losses.keys():
            mlflow.log_metric(
                f"{_datasplit}_batch_averaged_{loss_name}_per_epoch",
                _losses[loss_name],
                step=self.__epoch
            )

    def __update_evaluation_losses(self, _losses, _outputs, _targets):
        """Updates losses for any evaluation datasplit e.g. validation and testing."""

        return {loss_name: _losses[loss_name] + loss(_outputs, _targets).item() for loss_name, loss in self.__tracked_losses.items()}

    @property
    def model(self):
        return self.__model

    def train(self):
        self.__model.to(self.__device)

        for epoch in range(self.__epochs):
            training_losses = defaultdict(float)

            self.__model.train()
            self.__epoch = epoch

            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.__device), targets.to(self.__device)

                # Forward Pass
                self.__optimizer.zero_grad()
                outputs = self.__model(inputs)

                backprop_loss = self.__tracked_losses[self.__backprop_loss_name](outputs, targets)

                # Backward Pass
                backprop_loss.backward()
                self.__optimizer.step()

                training_losses = {loss_name: training_losses[loss_name] + loss(outputs, targets).item() for loss_name, loss in self.__tracked_losses.items()}


            # Normalize training losses by the number of training samples
            training_losses = {loss_name: loss / len(self.train_loader) for loss_name, loss in training_losses.items()}
            self.__log_losses(training_losses, "train")

            validation_losses = self.evaluate_loss(self.val_loader)
            self.__log_losses(validation_losses, "validation")

            if validation_losses[self.__backprop_loss_name] < self.__best_loss:
                self.__best_loss = validation_losses[self.__backprop_loss_name]
                self.__early_stop_counter = 0
                self.__best_model = self.__model
            else:
                self.__early_stop_counter += 1

            mlflow.log_metric(
                "early_stopping_counter_per_epoch",
                self.__early_stop_counter,
                step=self.__epoch
            )

            if self.__early_stop_counter >= self.__patience:
                break

    def evaluate_loss(self, _data_loader: torch.utils.data.DataLoader):
        """Computes the loss for an evaluation datasplit, e.g. validation or testing."""

        self.__model.eval()
        losses = defaultdict(int)

        with torch.no_grad():
            for inputs, targets in _data_loader:
                inputs, targets = inputs.to(self.__device), targets.to(self.__device)
                outputs = self.__model(inputs)
                losses = self.__update_evaluation_losses(losses, outputs, targets)

        return {loss_name: running_loss / len(_data_loader) for loss_name, running_loss in losses.items()}
