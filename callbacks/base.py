from typing import Any, Protocol


class Callback(Protocol):
    """Callback hook protocol for trainer lifecycle events."""

    def on_epoch_start(self, context: dict[str, Any]) -> None:
        ...

    def on_batch_start(self, context: dict[str, Any]) -> None:
        ...

    def on_batch_end(self, context: dict[str, Any]) -> None:
        ...

    def on_epoch_end(self, context: dict[str, Any]) -> None:
        ...


class BaseCallback:
    """No-op callback base class for selective hook overrides."""

    def on_epoch_start(self, context: dict[str, Any]) -> None:
        pass

    def on_batch_start(self, context: dict[str, Any]) -> None:
        pass

    def on_batch_end(self, context: dict[str, Any]) -> None:
        pass

    def on_epoch_end(self, context: dict[str, Any]) -> None:
        pass
