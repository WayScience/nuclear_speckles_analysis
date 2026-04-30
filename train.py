import argparse
import pathlib
import random
from typing import Any, Callable

import joblib
import mlflow
import numpy as np
import optuna
import torch

from callbacks.CallbackPipeline import CallbackPipeline
from callbacks.utils.SampleImages import SampleImages
from callbacks.utils.SaveEpochCrops import SaveEpochCrops
from datasets.dataset_00.CellCropToCropDataset import CellCropToCropDataset
from datasets.dataset_00.utils.CropCacheBuilder import (
    ensure_dapi_to_gold_cache,
    load_cache_manifest,
)
from datasets.dataset_00.utils.ImagePostProcessor import ImagePostProcessor
from datasets.dataset_00.utils.ImagePreProcessor import ImagePreProcessor
from metrics.L1 import L1
from metrics.L2 import L2
from metrics.PSNR import PSNR
from metrics.SSIM import SSIM
from models.UNet import UNet
from splitters.HashSplitter import HashSplitter
from trainers.UNetTrainer import UNetTrainer


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--n-trials", type=int, default=4)
parser.add_argument("--max-train-batches", type=int, default=-1)
parser.add_argument("--max-eval-batches", type=int, default=-1)
parser.add_argument("--enable-image-savers", type=int, choices=[0, 1], default=1)
args = parser.parse_args()

max_train_batches = None if args.max_train_batches <= 0 else args.max_train_batches
max_eval_batches = None if args.max_eval_batches <= 0 else args.max_eval_batches


class OptimizationManager:
    """Optuna objective function with MLflow logging."""

    def __init__(
        self,
        trainer: Any,
        hash_splitter: Any,
        dataset: Any,
        callbacks_args: dict[str, Any],
        model_factory: Callable[[], torch.nn.Module],
        **trainer_kwargs,
    ):
        """Store dependencies for Optuna-driven training trials.

        Args:
            trainer: Trainer class used to run one trial.
            hash_splitter: Callable that returns train/val/test dataloaders.
            dataset: Dataset associated with the optimization run.
            callbacks_args: Static callback arguments reused across trials.
            model_factory: Callable that creates a new model instance per trial.
            **trainer_kwargs: Shared trainer keyword arguments.
        """

        self.trainer = trainer
        self.hash_splitter = hash_splitter
        self.dataset = dataset
        self.callbacks_args = callbacks_args
        self.model_factory = model_factory
        self.trainer_kwargs = trainer_kwargs

    def __call__(self, trial: optuna.trial.Trial):
        """Execute one Optuna trial and return objective loss.

        Args:
            trial: Optuna trial used for hyperparameter suggestions.

        Returns:
            Best validation loss reported by the trainer.
        """

        batch_size = trial.suggest_int("batch_size", 8, 32)
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)

        train_dataloader, val_dataloader, _ = self.hash_splitter(batch_size=batch_size)
        self.trainer_kwargs["train_dataloader"] = train_dataloader
        self.trainer_kwargs["val_dataloader"] = val_dataloader

        model = self.model_factory()
        self.trainer_kwargs["model"] = model

        optimizer_params = {
            "params": model.parameters(),
            "lr": lr,
            "betas": (0.5, 0.999),
        }

        loss_trainer = L1(is_loss=True, device=device)
        loss_callbacks = L1(is_loss=True, device=device)
        metrics = [
            L2(device=device),
            PSNR(device=device, max_pixel_value=1.0),
            SSIM(device=device, max_pixel_value=1.0),
        ]

        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            optimizer = torch.optim.Adam(**optimizer_params)
            self.trainer_kwargs["model_optimizer"] = optimizer

            opt_params = optimizer.param_groups[0].copy()
            del opt_params["params"]
            mlflow.log_params({f"optimizer_{k}": v for k, v in opt_params.items()})
            mlflow.log_param("batch_size", batch_size)
            mlflow.set_tag("optimizer_class", optimizer.__class__.__name__.lower())

            self.trainer_kwargs["callbacks"] = CallbackPipeline(
                **self.callbacks_args | {"metrics": metrics, "loss": loss_callbacks}
            )

            trainer_obj = self.trainer(
                **self.trainer_kwargs | {"model_loss": loss_trainer}
            )
            trainer_obj.train()

            return trainer_obj.best_loss_value


data_dir = pathlib.Path("nuclear_speckles_data").resolve(strict=True)
cache_root = pathlib.Path("cached_nuclear_speckles_data")
crop_cache_path = cache_root / "dapi_to_gold_crop_cache"
tensor_cache_path = cache_root / "paired_tensor_cache"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
mlflow.log_param("random_seed", 0)

description = """
Optimization of a DAPI-to-Gold image-to-image translation model with:
- UNet Generator
- Single 2D crop input and single 2D crop target
- Cache-backed filtered nucleus crops generated from nuclear_speckles_data
- L1 optimization objective with L2, PSNR, and SSIM metric logging
"""
mlflow.set_tag("mlflow.note.content", description)

cache_result = ensure_dapi_to_gold_cache(data_dir=data_dir, cache_dir=crop_cache_path)
manifest_rows = load_cache_manifest(manifest_path=cache_result.manifest_path)
image_specs = cache_result.image_specs

mlflow.log_param("input_max_pixel_value", image_specs["input_max_pixel_value"])
mlflow.log_param("target_max_pixel_value", image_specs["target_max_pixel_value"])

image_preprocessor = ImagePreProcessor(image_specs=image_specs, device=device)
image_postprocessor = ImagePostProcessor()

crop_image_dataset = CellCropToCropDataset(
    manifest_rows=manifest_rows,
    image_specs=image_specs,
    image_preprocessor=image_preprocessor,
    image_cache_path=tensor_cache_path,
)

hash_splitter = HashSplitter(
    dataset=crop_image_dataset,
    train_frac=0.8,
    val_frac=0.1,
)

_, val_dataloader, _ = hash_splitter(batch_size=16)
crop_dataset_idxs = SampleImages(datastruct=val_dataloader, image_fraction=1 / 32)()

image_prediction_saver = SaveEpochCrops(
    image_dataset=val_dataloader.dataset.dataset,
    image_postprocessor=image_postprocessor,
    image_dataset_idxs=crop_dataset_idxs,
)

callbacks_args = {
    "early_stopping_counter_threshold": 5,
    "image_savers": [image_prediction_saver] if args.enable_image_savers == 1 else None,
    "image_postprocessor": image_postprocessor,
    "max_eval_batches": max_eval_batches,
}

optimization_manager = OptimizationManager(
    trainer=UNetTrainer,
    hash_splitter=hash_splitter,
    dataset=crop_image_dataset,
    callbacks_args=callbacks_args,
    model_factory=lambda: UNet(in_channels=1, out_channels=1),
    epochs=args.epochs,
    max_train_batches=max_train_batches,
)

study = optuna.create_study(study_name="model_training", direction="minimize")
study.optimize(optimization_manager, n_trials=args.n_trials)

joblib.dump(study, "optuna_study.joblib")
mlflow.log_artifact("optuna_study.joblib")
