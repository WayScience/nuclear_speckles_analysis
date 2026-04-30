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
        """Store ordered callbacks.

        Args:
            callbacks: Callback instances invoked for each hook.
        """
        self.callbacks = callbacks

    def trigger(self, callback_hook: str, hook_data: dict[str, Any]) -> None:
        """Invoke one hook on every registered callback.

        Args:
            callback_hook: Hook method name (for example ``on_epoch_end``).
            hook_data: Shared mutable payload for this hook invocation.
        """

        # Dynamic dispatch keeps the trainer-side callback API compact while
        # preserving strict execution order across callback instances.
        for callback in self.callbacks:
            getattr(callback, callback_hook)(hook_data)


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
        """Initialize composed callbacks used during training.

        Args:
            metrics: Metrics updated during epoch-end evaluation.
            loss: Loss object used for callback-side logging and early stopping.
            early_stopping_counter_threshold: Number of non-improving epochs before stop.
            image_savers: Optional saver callable or list of saver callables.
            image_postprocessor: Postprocessor applied when needed before logging.
            batch_log_every_n: Batch interval for progress logging.
            max_eval_batches: Optional cap on batches during callback evaluation.
        """
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
        """Return the best validation loss tracked by early stopping."""

        return self.early_stopping.best_loss_value

    def __call__(self, callback_hook: str, **kwargs) -> Any:
        """Execute a lifecycle hook across composed callbacks.

        Args:
            callback_hook: Hook method name to dispatch.
            **kwargs: Hook payload forwarded to callbacks.

        Returns:
            Continue-training flag when present in hook payload.
        """

        hook_data = kwargs
        # Callbacks communicate via this shared payload across one hook call.
        hook_data.setdefault("continue_training", True)
        self.composer.trigger(callback_hook=callback_hook, hook_data=hook_data)
        return hook_data.get("continue_training")
