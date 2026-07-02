import json
import pathlib
import random
from typing import Any

import numpy as np
import torch


class TrialCheckpointManager:
    """Persist resumable per-trial training state on shared storage."""

    def __init__(self, checkpoint_dir: pathlib.Path) -> None:
        """Store checkpoint file locations for one logical Optuna trial.

        Args:
            checkpoint_dir: Trial-specific directory on shared storage.
        """
        self.checkpoint_dir = checkpoint_dir
        self.latest_checkpoint_path = self.checkpoint_dir / "latest.pt"
        self.metadata_path = self.checkpoint_dir / "metadata.json"

    def exists(self) -> bool:
        """Return whether a resumable trial checkpoint is already present."""

        return self.latest_checkpoint_path.exists()

    def save(
        self,
        *,
        generator: torch.nn.Module,
        discriminator: torch.nn.Module,
        generator_optimizer: torch.optim.Optimizer,
        discriminator_optimizer: torch.optim.Optimizer,
        next_epoch: int,
        callbacks_state: dict[str, Any],
        discriminator_steps_since_generator_update: int,
        trial_metadata: dict[str, Any],
        amp_state: dict[str, Any] | None = None,
    ) -> None:
        """Atomically write the latest resumable checkpoint for a trial.

        Args:
            generator: Generator module whose weights should be resumed.
            discriminator: Discriminator module whose weights should be resumed.
            generator_optimizer: Optimizer paired with the generator.
            discriminator_optimizer: Optimizer paired with the discriminator.
            next_epoch: Epoch index to start from on the next launch.
            callbacks_state: Serializable callback-side training state.
            discriminator_steps_since_generator_update: WGAN-GP update cadence state.
            trial_metadata: Human-readable metadata mirrored into ``metadata.json``.
            amp_state: Optional serialized AMP/scaler state required for resume.
        """
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        payload = {
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "generator_optimizer_state_dict": generator_optimizer.state_dict(),
            "discriminator_optimizer_state_dict": discriminator_optimizer.state_dict(),
            "next_epoch": next_epoch,
            "callbacks_state": callbacks_state,
            "discriminator_steps_since_generator_update": (
                discriminator_steps_since_generator_update
            ),
            "trial_metadata": trial_metadata,
            "rng_state": self._capture_rng_state(),
            "amp_state": amp_state or {},
        }

        temp_checkpoint_path = self.latest_checkpoint_path.with_suffix(".pt.tmp")
        torch.save(payload, temp_checkpoint_path)
        temp_checkpoint_path.replace(self.latest_checkpoint_path)

        temp_metadata_path = self.metadata_path.with_suffix(".json.tmp")
        temp_metadata_path.write_text(
            json.dumps(
                {
                    **trial_metadata,
                    "next_epoch": next_epoch,
                    "checkpoint_path": str(self.latest_checkpoint_path),
                },
                indent=2,
                sort_keys=True,
            )
        )
        temp_metadata_path.replace(self.metadata_path)

    def load(self, map_location: str | torch.device | None = None) -> dict[str, Any]:
        """Load the most recent resumable checkpoint payload.

        Args:
            map_location: Device mapping forwarded to ``torch.load``.

        Returns:
            Deserialized checkpoint payload.
        """
        return torch.load(
            self.latest_checkpoint_path,
            map_location=map_location,
            weights_only=False,
        )

    @staticmethod
    def _capture_rng_state() -> dict[str, Any]:
        """Snapshot Python, NumPy, and Torch RNG state for deterministic resume."""

        rng_state: dict[str, Any] = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        }
        if torch.cuda.is_available():
            rng_state["torch_cuda"] = torch.cuda.get_rng_state_all()
        return rng_state

    @staticmethod
    def restore_rng_state(rng_state: dict[str, Any]) -> None:
        """Restore Python, NumPy, and Torch RNG state from a checkpoint.

        Args:
            rng_state: Serialized RNG state captured by ``_capture_rng_state``.
        """
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        if torch.cuda.is_available() and "torch_cuda" in rng_state:
            torch.cuda.set_rng_state_all(rng_state["torch_cuda"])
