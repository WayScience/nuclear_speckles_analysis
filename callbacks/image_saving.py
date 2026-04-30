from typing import Any, Optional, Union

from callbacks.base import BaseCallback


class ImageSaverCallback(BaseCallback):
    """Persist epoch outputs through one or more saver callables."""

    def __init__(self, image_savers: Optional[Union[Any, list[Any]]] = None) -> None:
        """Initialize image saver callback.

        Args:
            image_savers: Optional saver callable or list of saver callables.
        """

        self.image_savers = image_savers

    def on_epoch_end(self, hook_data: dict[str, Any]) -> None:
        """Save images at epoch end using configured saver callables.

        Args:
            hook_data: Shared hook payload containing model and epoch.
        """

        if self.image_savers is None:
            return

        model = hook_data["model"]
        epoch = hook_data["epoch"]

        if isinstance(self.image_savers, list):
            for image_saver in self.image_savers:
                image_saver(model=model, epoch=epoch)
        else:
            self.image_savers(model=model, epoch=epoch)
