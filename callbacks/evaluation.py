from typing import Any, List

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from callbacks.base import BaseCallback


class EpochEvaluatorCallback(BaseCallback):
    """Run epoch-end evaluation for configured data splits.

    Train and validation metrics can be computed with dedicated evaluation
    dataloaders so the metric pass is deterministic and independent from the
    optimization loaders.
    """

    def __init__(
        self,
        metrics: List,
        loss: Module,
        image_postprocessor: Any = lambda x: x,
        max_eval_batches: int | None = None,
        use_amp: bool = False,
    ) -> None:
        """Initialize evaluation dependencies.

        Args:
            metrics: Metrics updated on each evaluation batch.
            loss: Loss metric object updated on each evaluation batch.
            image_postprocessor: Prediction postprocessor and inverse-transform
                helper used to denormalize tensors for logging.
            max_eval_batches: Optional cap on evaluation batches per split.
            use_amp: Whether to run evaluation inference under AMP autocast.
        """
        self.metrics = metrics
        self.loss = loss
        self.image_postprocessor = image_postprocessor
        self.max_eval_batches = max_eval_batches
        self.use_amp = use_amp
        self.amp_dtype = torch.bfloat16

    def on_epoch_end(self, hook_data: dict[str, Any]) -> None:
        """Run evaluation on train and validation splits.

        Args:
            hook_data: Shared hook payload containing model and dataloaders.
        """

        model = hook_data["model"]
        epoch_metric_data: dict[str, float] = {}
        for data_split, dataloader in [
            (
                "train",
                hook_data.get("eval_train_dataloader", hook_data["train_dataloader"]),
            ),
            (
                "validation",
                hook_data.get("eval_val_dataloader", hook_data["val_dataloader"]),
            ),
        ]:
            split_metric_data = self._evaluate_split(
                model=model, dataloader=dataloader, data_split=data_split
            )
            epoch_metric_data.update(split_metric_data)

        hook_data["epoch_metric_data"] = epoch_metric_data

    def _evaluate_split(
        self,
        model: Module,
        dataloader: DataLoader,
        data_split: str,
    ) -> dict[str, float]:
        """Evaluate one split and update running metric/loss accumulators.

        Args:
            model: Model evaluated for current split.
            dataloader: Split dataloader iterated for evaluation.
            data_split: Split name used for downstream metric keys.
        """

        model.eval()

        with torch.no_grad():
            for batch_idx, samples in enumerate(dataloader):
                with torch.amp.autocast(
                    enabled=self.use_amp,
                    device_type=samples["input"].device.type,
                    dtype=self.amp_dtype,
                ):
                    generated_predictions = model(samples["input"])
                    postprocessed_predictions = self.image_postprocessor(
                        generated_predictions
                    )

                denormalized_predictions = self.image_postprocessor.denormalize_target(
                    postprocessed_predictions
                )
                denormalized_targets = self.image_postprocessor.denormalize_target(
                    samples["target"]
                )

                self.loss.update(
                    generated_predictions=(
                        generated_predictions
                        if self.loss.use_logits
                        else denormalized_predictions
                    ),
                    targets=(samples["target"] if self.loss.use_logits else denormalized_targets),
                    loss_mask=samples.get("loss_mask"),
                )

                for metric in self.metrics:
                    metric.update(
                        generated_predictions=(
                            generated_predictions
                            if metric.use_logits
                            else denormalized_predictions
                        ),
                        targets=(samples["target"] if metric.use_logits else denormalized_targets),
                        loss_mask=samples.get("loss_mask"),
                    )

                if (
                    self.max_eval_batches is not None
                    and (batch_idx + 1) >= self.max_eval_batches
                ):
                    break

        loss_metric_data = self.loss.compute()
        split_metric_data = {
            f"{metric_name}_{data_split}": metric_value
            for metric_name, metric_value in loss_metric_data.items()
        }
        self.loss.reset()
        for metric in self.metrics:
            metric_data = metric.compute()
            for metric_name, metric_value in metric_data.items():
                split_metric_data[f"{metric_name}_{data_split}"] = metric_value
            metric.reset()

        return split_metric_data
