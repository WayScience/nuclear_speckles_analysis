from typing import Any, List

import torch
from torch.nn import Module
from torch.utils.data import DataLoader

from callbacks.base import BaseCallback


class EpochEvaluatorCallback(BaseCallback):
    """Run epoch-end evaluation for configured data splits."""

    def __init__(
        self,
        metrics: List,
        loss: Module,
        image_postprocessor: Any = lambda x: x,
        max_eval_batches: int | None = None,
    ) -> None:
        """Initialize evaluation dependencies.

        Args:
            metrics: Metrics updated on each evaluation batch.
            loss: Loss metric object updated on each evaluation batch.
            image_postprocessor: Postprocessor applied when logits are not used.
            max_eval_batches: Optional cap on evaluation batches per split.
        """
        self.metrics = metrics
        self.loss = loss
        self.image_postprocessor = image_postprocessor
        self.max_eval_batches = max_eval_batches
        self.compute_sigmoid = any(not metric.use_logits for metric in [*metrics, loss])

    def on_epoch_end(self, hook_data: dict[str, Any]) -> None:
        """Run evaluation on train and validation splits.

        Args:
            hook_data: Shared hook payload containing model and dataloaders.
        """

        model = hook_data["model"]
        epoch_metric_data: dict[str, float] = {}
        for data_split, dataloader in [
            ("train", hook_data["train_dataloader"]),
            ("validation", hook_data["val_dataloader"]),
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
                generated_predictions = model(samples["input"])
                sigmoid_generated_predictions = generated_predictions.clone()

                # Only postprocess if any metric/loss expects non-logit values.
                if self.compute_sigmoid:
                    sigmoid_generated_predictions = self.image_postprocessor(
                        generated_predictions
                    )

                self.loss(
                    generated_predictions=(
                        generated_predictions
                        if self.loss.use_logits
                        else sigmoid_generated_predictions
                    ),
                    targets=samples["target"],
                    loss_mask=samples.get("loss_mask"),
                )

                for metric in self.metrics:
                    metric.update(
                        generated_predictions=(
                            generated_predictions
                            if metric.use_logits
                            else sigmoid_generated_predictions
                        ),
                        targets=samples["target"],
                        loss_mask=samples.get("loss_mask"),
                    )

                if (
                    self.max_eval_batches is not None
                    and (batch_idx + 1) >= self.max_eval_batches
                ):
                    break

        split_metric_data = {
            f"{self.loss.metric_name}_{data_split}": self.loss.compute().item(),
        }
        self.loss.reset()
        for metric in self.metrics:
            split_metric_data[f"{metric.metric_name}_{data_split}"] = metric.compute().item()
            metric.reset()

        return split_metric_data
