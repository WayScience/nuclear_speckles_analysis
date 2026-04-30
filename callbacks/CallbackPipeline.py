from typing import Any, List, Optional, Union

from torch.nn import Module

from callbacks.base import Callback
from callbacks.early_stopping import EarlyStoppingAndCheckpointCallback
from callbacks.evaluation import EpochEvaluatorCallback
from callbacks.image_saving import ImageSaverCallback
from callbacks.logging import MetricsMlflowLoggerCallback
from callbacks.progress import ProgressLoggerCallback


class CallbackComposer:
    """Execute callback hooks in registration order."""

    def __init__(self, callbacks: list[Callback]) -> None:
        self.callbacks = callbacks

    def trigger(self, callback_hook: str, context: dict[str, Any]) -> None:
        for callback in self.callbacks:
            getattr(callback, callback_hook)(context)


class CallbackPipeline:
    """Compose lifecycle callbacks behind a single trainer-facing object."""

    def __init__(
        self,
        metrics: List,
        loss: Module,
        early_stopping_counter_threshold: int,
        image_savers: Optional[Union[Any, List[Any]]] = None,
        image_postprocessor: Any = lambda x: x,
        batch_log_every_n: int = 50,
        max_eval_batches: int | None = None,
    ) -> None:
        self.progress = ProgressLoggerCallback(batch_log_every_n=batch_log_every_n)
        self.evaluator = EpochEvaluatorCallback(
            metrics=metrics,
            loss=loss,
            image_postprocessor=image_postprocessor,
            max_eval_batches=max_eval_batches,
        )
        self.metrics_logger = MetricsMlflowLoggerCallback(metrics=metrics, loss=loss)
        self.image_saver = ImageSaverCallback(image_savers=image_savers)
        self.early_stopping = EarlyStoppingAndCheckpointCallback(
            early_stopping_counter_threshold=early_stopping_counter_threshold,
            image_postprocessor=image_postprocessor,
        )

        self.composer = CallbackComposer(
            callbacks=[
                self.progress,
                self.evaluator,
                self.metrics_logger,
                self.image_saver,
                self.early_stopping,
            ]
        )

    @property
    def best_loss_value(self) -> float:
        return self.early_stopping.best_loss_value

    def __call__(self, callback_hook: str, **kwargs) -> Any:
        context = kwargs
        context.setdefault("continue_training", True)
        self.composer.trigger(callback_hook=callback_hook, context=context)
        return context.get("continue_training")
