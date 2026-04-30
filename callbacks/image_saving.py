from typing import Any, Optional, Union

from callbacks.base import BaseCallback


class ImageSaverCallback(BaseCallback):
    """Persist epoch outputs through one or more saver callables."""

    def __init__(self, image_savers: Optional[Union[Any, list[Any]]] = None) -> None:
        self.image_savers = image_savers

    def on_epoch_end(self, context: dict[str, Any]) -> None:
        if self.image_savers is None:
            return

        model = context["model"]
        epoch = context["epoch"]

        if isinstance(self.image_savers, list):
            for image_saver in self.image_savers:
                image_saver(model=model, epoch=epoch)
        else:
            self.image_savers(model=model, epoch=epoch)
