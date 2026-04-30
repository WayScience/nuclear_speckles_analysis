from typing import Any, Protocol


class Callback(Protocol):
    """Protocol for trainer lifecycle callback hooks.

    Implementers may read from and write to ``hook_data`` so callbacks can share
    state within a single hook execution.
    """

    def on_epoch_start(self, hook_data: dict[str, Any]) -> None:
        """Handle start-of-epoch hook.

        Args:
            hook_data: Shared hook payload for the current lifecycle event.
        """
        ...

    def on_batch_start(self, hook_data: dict[str, Any]) -> None:
        """Handle start-of-batch hook.

        Args:
            hook_data: Shared hook payload for the current lifecycle event.
        """
        ...

    def on_batch_end(self, hook_data: dict[str, Any]) -> None:
        """Handle end-of-batch hook.

        Args:
            hook_data: Shared hook payload for the current lifecycle event.
        """
        ...

    def on_epoch_end(self, hook_data: dict[str, Any]) -> None:
        """Handle end-of-epoch hook.

        Args:
            hook_data: Shared hook payload for the current lifecycle event.
        """
        ...


class BaseCallback:
    """No-op base callback for selective hook overrides."""

    def on_epoch_start(self, hook_data: dict[str, Any]) -> None:
        """Handle start-of-epoch hook.

        Args:
            hook_data: Shared hook payload for the current lifecycle event.
        """
        pass

    def on_batch_start(self, hook_data: dict[str, Any]) -> None:
        """Handle start-of-batch hook.

        Args:
            hook_data: Shared hook payload for the current lifecycle event.
        """
        pass

    def on_batch_end(self, hook_data: dict[str, Any]) -> None:
        """Handle end-of-batch hook.

        Args:
            hook_data: Shared hook payload for the current lifecycle event.
        """
        pass

    def on_epoch_end(self, hook_data: dict[str, Any]) -> None:
        """Handle end-of-epoch hook.

        Args:
            hook_data: Shared hook payload for the current lifecycle event.
        """
        pass
