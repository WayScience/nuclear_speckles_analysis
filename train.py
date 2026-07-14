"""Train and optimize DAPI-to-Gold image-to-image models.

This script prepares cached crop datasets, builds deterministic data splits,
launches Optuna trials, and logs run metadata and artifacts with MLflow.
"""

import argparse
import math
import pathlib
import random
from dataclasses import dataclass
from typing import Any, Callable

import joblib
import mlflow
import numpy as np
import optuna
import torch
import tifffile

from callbacks.CallbackPipeline import CallbackPipeline
from callbacks.utils.SampleImages import SampleImages
from callbacks.utils.SaveEpochCrops import SaveEpochCrops
from datasets.dataset_00.CellCropToCropDataset import CellCropToCropDataset
from datasets.dataset_00.utils.CropCacheBuilder import (
    ensure_dapi_to_gold_cache, load_cache_manifest)
from datasets.dataset_00.utils.ImagePostProcessor import ImagePostProcessor
from datasets.dataset_00.utils.ImagePreProcessor import ImagePreProcessor
from losses.L1Loss import L1Loss
from metrics.L1 import L1
from metrics.L2 import L2
from metrics.PearsonCorrelation import PearsonCorrelation
from metrics.PSNR import PSNR
from metrics.SSIM import SSIM
from models.convnext_unet.unext import ConvNeXtUNet
from splitters.HashSplitter import HashSplitter
from trainers.UNetTrainer import UNetTrainer


@dataclass(frozen=True)
class DatasetConfig:
    """Dataset-specific paths and column/channel normalization settings.

    Attributes:
        image_dir: Root directory containing source TIFF image files.
        parquet_path: Path to single-cell profile parquet data.
        cache_root: Base directory where dataset caches are written.
        input_channel: Source channel name used for model input crop selection.
        target_channel: Target channel name used for supervision crop selection.
        metadata_column_map: Optional source-to-canonical metadata renaming map
            applied before crop cache generation.
        holdout_plate: Optional plate identifier removed before train/val splits.
        input_resolution: Optional source microscope resolution in microns per
            pixel. Whole-image resampling is enabled only when this and
            ``target_resolution`` are both provided.
        target_resolution: Optional target microscope resolution in microns per
            pixel. Whole-image resampling is enabled only when this and
            ``input_resolution`` are both provided.
    """

    image_dir: pathlib.Path
    parquet_path: pathlib.Path
    cache_root: pathlib.Path
    input_channel: str
    target_channel: str
    metadata_column_map: dict[str, str] | None = None
    holdout_plate: str | None = None
    input_resolution: float | None = None
    target_resolution: float | None = None


# Shared root for dataset-specific image directories, profiles, and caches.
speckle_dataset_path = pathlib.Path("/mnt/big_drive/nuclear_speckle_data").resolve(
    strict=True
)
u2os_dataset_path = speckle_dataset_path / "u20s_dataset_jan_15_2026"
initial_dataset_path = speckle_dataset_path / "initial_dataset"

DATASET_CONFIGS = {
    # Only U2OS currently uses microscope-resolution harmonization before crop caching.
    "u2os": DatasetConfig(
        image_dir=u2os_dataset_path / "u20s_images/tiffs",
        parquet_path=u2os_dataset_path
        / "u20s_profiles/single_cell_profiles/u2os_per_nuclei_sc_feature_selected.parquet",
        cache_root=u2os_dataset_path / "model_cache",
        input_channel="CH01",
        target_channel="CH03",
        metadata_column_map={
            # U2OS profiles label imaging site as "Metadata_Position".
            "Metadata_Position": "Metadata_Site",
        },
        holdout_plate="Rep3",
        input_resolution=2.74,
        target_resolution=6.45,
    ),
    "initial": DatasetConfig(
        image_dir=initial_dataset_path / "IC_corrected_images",
        parquet_path=initial_dataset_path / "Preprocessed_data/cleaned_sc_profiles",
        cache_root=initial_dataset_path / "model_cache",
        input_channel="CH0",
        target_channel="CH2",
        metadata_column_map={
            "Image_Metadata_Plate": "Metadata_Plate",
            "Image_Metadata_Well": "Metadata_Well",
            "Image_Metadata_Site": "Metadata_Site",
            "Nuclei_AreaShape_BoundingBoxMinimum_X": "Metadata_Nuclei_AreaShape_BoundingBoxMinimum_X",
            "Nuclei_AreaShape_BoundingBoxMaximum_X": "Metadata_Nuclei_AreaShape_BoundingBoxMaximum_X",
            "Nuclei_AreaShape_BoundingBoxMinimum_Y": "Metadata_Nuclei_AreaShape_BoundingBoxMinimum_Y",
            "Nuclei_AreaShape_BoundingBoxMaximum_Y": "Metadata_Nuclei_AreaShape_BoundingBoxMaximum_Y",
        },
        holdout_plate="slide2",
        input_resolution=None,
        target_resolution=None,
    ),
}


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--n-trials", type=int, default=4)
parser.add_argument("--max-train-batches", type=int, default=-1)
parser.add_argument("--max-eval-batches", type=int, default=-1)
# Evaluation can use a different batch size than optimization to make
# epoch-end metric passes easier to fit on available hardware.
parser.add_argument("--eval-batch-size", type=int, default=-1)
parser.add_argument("--eval-use-amp", type=int, choices=[0, 1], default=0)
parser.add_argument("--train-use-amp", type=int, choices=[0, 1], default=1)
parser.add_argument("--enable-image-savers", type=int, choices=[0, 1], default=1)
parser.add_argument("--batch-metric-log-every-n", type=int, default=1)
parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS.keys()), default="u2os")
parser.add_argument("--crop-size", type=int, default=256)
# Study metadata is passed through to Optuna storage so repeated runs can target
# a stable study name and backing database.
parser.add_argument("--study-name", type=str, default=None)
parser.add_argument("--optuna-storage", type=str, default="sqlite:///optuna_study.db")
parser.add_argument(
    "--checkpoint-root", type=pathlib.Path, default=pathlib.Path("trial_checkpoints")
)
parser.add_argument("--resume", type=int, choices=[0, 1], default=1)
parser.add_argument("--parent-run-id", type=str, default=None)
args = parser.parse_args()
if args.parent_run_id == "":
    args.parent_run_id = None
if args.study_name == "":
    args.study_name = None

# Interpret non-positive limits as "use the full epoch" for trainer/eval loops.
max_train_batches = None if args.max_train_batches <= 0 else args.max_train_batches
max_eval_batches = None if args.max_eval_batches <= 0 else args.max_eval_batches
requested_eval_batch_size = None if args.eval_batch_size <= 0 else args.eval_batch_size
eval_use_amp = args.eval_use_amp == 1
train_use_amp = args.train_use_amp == 1
max_batch_size = 8


def compute_training_image_stats(
    manifest_rows: list[dict[str, str]],
    train_indices: list[int],
) -> dict[str, float]:
    """Compute train-split image mean and std for z-score normalization.

    Args:
        manifest_rows: Full manifest rows backing the crop dataset.
        train_indices: Dataset indices assigned to the training split.

    Returns:
        Dictionary containing train-split input/target means and standard deviations.

    Raises:
        ValueError: If the training split is empty or any variance is non-positive.
    """

    if not train_indices:
        raise ValueError("Training split is empty; cannot compute z-score statistics.")

    input_sum = 0.0
    input_sum_sq = 0.0
    target_sum = 0.0
    target_sum_sq = 0.0
    input_count = 0
    target_count = 0

    for idx in train_indices:
        sample = manifest_rows[idx]
        input_image = tifffile.imread(sample["input_path"]).astype(np.float64)
        target_image = tifffile.imread(sample["target_path"]).astype(np.float64)

        input_sum += float(input_image.sum())
        input_sum_sq += float(np.square(input_image).sum())
        target_sum += float(target_image.sum())
        target_sum_sq += float(np.square(target_image).sum())
        input_count += int(input_image.size)
        target_count += int(target_image.size)

    input_mean = input_sum / input_count
    target_mean = target_sum / target_count
    input_var = (input_sum_sq / input_count) - (input_mean**2)
    target_var = (target_sum_sq / target_count) - (target_mean**2)
    input_std = math.sqrt(max(input_var, 0.0))
    target_std = math.sqrt(max(target_var, 0.0))

    if input_std <= 0 or target_std <= 0:
        raise ValueError("Training-split z-score standard deviations must be positive.")

    return {
        "input_mean": input_mean,
        "input_std": input_std,
        "target_mean": target_mean,
        "target_std": target_std,
    }


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

        # Couple learning rate to batch size so Optuna searches a scaling factor
        # while the derived rate stays within the previous learning-rate bounds.
        batch_size = trial.suggest_int("batch_size", 1, max_batch_size)
        lr_factor = trial.suggest_float(
            "lr_factor",
            1e-5,
            1e-3 / math.sqrt(max_batch_size),
            log=True,
        )
        lr = lr_factor * math.sqrt(batch_size)
        eval_batch_size = batch_size if requested_eval_batch_size is None else requested_eval_batch_size

        # Optimization can tune the training batch size without forcing the same
        # setting on epoch-end evaluation passes.
        train_dataloader, val_dataloader, _ = self.hash_splitter(batch_size=batch_size)
        eval_train_dataloader, eval_val_dataloader, _ = self.hash_splitter.build_loaders(
            batch_size=eval_batch_size,
            train_shuffle=False,
        )
        self.trainer_kwargs["train_dataloader"] = train_dataloader
        self.trainer_kwargs["val_dataloader"] = val_dataloader
        self.trainer_kwargs["eval_train_dataloader"] = eval_train_dataloader
        self.trainer_kwargs["eval_val_dataloader"] = eval_val_dataloader

        model = self.model_factory()
        self.trainer_kwargs["model"] = model

        optimizer_params = {
            "params": model.parameters(),
            "lr": lr,
            "betas": (0.5, 0.999),
        }

        loss_trainer = L1Loss()
        loss_callbacks = L1(device=device)
        metrics = [
            L2(device=device),
            PSNR(device=device, max_pixel_value=image_specs["target_max_pixel_value"]),
            SSIM(device=device, max_pixel_value=image_specs["target_max_pixel_value"]),
            PearsonCorrelation(device=device),
        ]

        # Use a nested MLflow run so each Optuna trial has its own metrics/artifacts.
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            optimizer = torch.optim.Adam(**optimizer_params)
            self.trainer_kwargs["model_optimizer"] = optimizer

            opt_params = optimizer.param_groups[0].copy()
            del opt_params["params"]
            mlflow.log_params({f"optimizer_{k}": v for k, v in opt_params.items()})
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("lr_factor", lr_factor)
            mlflow.log_param("eval_batch_size", eval_batch_size)
            mlflow.log_param("eval_use_amp", int(eval_use_amp))
            mlflow.log_param("train_use_amp", int(train_use_amp))
            mlflow.set_tag("optimizer_class", optimizer.__class__.__name__.lower())

            self.trainer_kwargs["callbacks"] = CallbackPipeline(
                **self.callbacks_args | {"metrics": metrics, "loss": loss_callbacks}
            )

            trainer_obj = self.trainer(
                **self.trainer_kwargs | {"model_loss": loss_trainer}
            )
            trainer_obj.train()

            return trainer_obj.best_loss_value


dataset_config = DATASET_CONFIGS[args.dataset]
image_dir = dataset_config.image_dir.resolve(strict=True)
parquet_path = dataset_config.parquet_path.resolve(strict=True)
cache_root = dataset_config.cache_root
crop_cache_path = cache_root / "dapi_to_gold_crop_cache"
tensor_cache_path = cache_root / "paired_tensor_cache"

if args.crop_size <= 0:
    raise ValueError(f"crop_size must be positive, got {args.crop_size}")

# Keep all random sources fixed so trial-to-trial differences come from hyperparameters.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if train_use_amp and device.type == "cuda" and not torch.cuda.is_bf16_supported():
    raise ValueError("train_use_amp requires CUDA bfloat16 support on this device.")
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
mlflow.log_param("random_seed", 0)
mlflow.log_param("dataset", args.dataset)
mlflow.log_param("input_channel", dataset_config.input_channel)
mlflow.log_param("target_channel", dataset_config.target_channel)
mlflow.log_param("crop_size", args.crop_size)
mlflow.log_param("requested_eval_batch_size", args.eval_batch_size)
mlflow.log_param("requested_eval_use_amp", int(eval_use_amp))
mlflow.log_param("requested_train_use_amp", int(train_use_amp))
mlflow.log_param("optuna_storage", args.optuna_storage)
mlflow.log_param("checkpoint_root", str(args.checkpoint_root))
mlflow.log_param("resume", args.resume)
mlflow.log_param("amp_dtype", "bfloat16")
mlflow.log_param("input_resolution", dataset_config.input_resolution)
mlflow.log_param("target_resolution", dataset_config.target_resolution)

description = """
Optimization of a DAPI-to-Gold image-to-image translation model with:
- ConvNeXtUNet Generator
- Single 2D crop input and single 2D crop target
- Cache-backed filtered nucleus crops generated from the configured data directory
- Train-split z-score normalization for inputs and targets
- L1 optimization objective in z-score space with denormalized L2, PSNR, SSIM,
  and Pearson correlation metric logging
"""
mlflow.set_tag("mlflow.note.content", description)

# Build or reuse cropped-nuclei cache so training does not repeatedly parse raw image files.
# When crop_size changes, delete the existing crop and tensor caches before rerunning.
cache_result = ensure_dapi_to_gold_cache(
    image_dir=image_dir,
    parquet_path=parquet_path,
    cache_dir=crop_cache_path,
    input_channel=dataset_config.input_channel,
    target_channel=dataset_config.target_channel,
    crop_size=args.crop_size,
    metadata_column_map=dataset_config.metadata_column_map,
    input_resolution=dataset_config.input_resolution,
    target_resolution=dataset_config.target_resolution,
)
manifest_nuclei = load_cache_manifest(manifest_path=cache_result.manifest_path)
manifest_nuclei_before_holdout_filter = len(manifest_nuclei)
if dataset_config.holdout_plate is not None:
    # Keep one plate fully held out to prevent leakage across similar acquisition batches.
    manifest_nuclei = [
        nuclei
        for nuclei in manifest_nuclei
        if nuclei.get("plate") != dataset_config.holdout_plate
    ]
manifest_nuclei_after_holdout_filter = len(manifest_nuclei)

if not manifest_nuclei:
    raise ValueError(
        "No cropped nuclei remain after applying holdout plate filter. "
        f"dataset={args.dataset}, holdout_plate={dataset_config.holdout_plate}"
    )

mlflow.log_param("holdout_plate", dataset_config.holdout_plate)
mlflow.log_param(
    "manifest_nuclei_before_holdout_filter", manifest_nuclei_before_holdout_filter
)
mlflow.log_param(
    "manifest_nuclei_after_holdout_filter", manifest_nuclei_after_holdout_filter
)
image_specs = cache_result.image_specs

mlflow.log_param("input_max_pixel_value", image_specs["input_max_pixel_value"])
mlflow.log_param("target_max_pixel_value", image_specs["target_max_pixel_value"])

bootstrap_preprocessor = ImagePreProcessor(image_specs=image_specs, device=device)

bootstrap_dataset = CellCropToCropDataset(
    manifest_rows=manifest_nuclei,
    image_specs=image_specs,
    image_preprocessor=bootstrap_preprocessor,
    image_cache_path=tensor_cache_path,
)

# HashSplitter uses metadata-derived IDs, so splits stay stable across reruns.
bootstrap_hash_splitter = HashSplitter(
    dataset=bootstrap_dataset,
    train_frac=0.825,
    val_frac=0.125,
)
bootstrap_hash_splitter.split_by_hash()
training_stats = compute_training_image_stats(
    manifest_rows=manifest_nuclei,
    train_indices=bootstrap_hash_splitter.splits["train"],
)
image_specs = image_specs | training_stats

mlflow.log_param("input_mean", image_specs["input_mean"])
mlflow.log_param("input_std", image_specs["input_std"])
mlflow.log_param("target_mean", image_specs["target_mean"])
mlflow.log_param("target_std", image_specs["target_std"])

image_preprocessor = ImagePreProcessor(image_specs=image_specs, device=device)
image_postprocessor = ImagePostProcessor(
    input_mean=image_specs["input_mean"],
    input_std=image_specs["input_std"],
    target_mean=image_specs["target_mean"],
    target_std=image_specs["target_std"],
)

crop_image_dataset = CellCropToCropDataset(
    manifest_rows=manifest_nuclei,
    image_specs=image_specs,
    image_preprocessor=image_preprocessor,
    image_cache_path=tensor_cache_path,
)

hash_splitter = HashSplitter(
    dataset=crop_image_dataset,
    train_frac=0.825,
    val_frac=0.125,
)

train_dataloader, val_dataloader, _ = hash_splitter(batch_size=16)
train_crop_dataset_idxs = SampleImages(
    datastruct=train_dataloader, image_fraction=1 / 512
)()
val_crop_dataset_idxs = SampleImages(datastruct=val_dataloader, image_fraction=1 / 64)()

# Save a fixed subset of predictions each epoch for qualitative drift checks.
train_image_prediction_saver = SaveEpochCrops(
    image_dataset=train_dataloader.dataset.dataset,
    image_postprocessor=image_postprocessor,
    image_dataset_idxs=train_crop_dataset_idxs,
    split_name="training",
)

val_image_prediction_saver = SaveEpochCrops(
    image_dataset=val_dataloader.dataset.dataset,
    image_postprocessor=image_postprocessor,
    image_dataset_idxs=val_crop_dataset_idxs,
    split_name="validation",
)

callbacks_args = {
    "early_stopping_counter_threshold": 5,
    "image_savers": (
        [train_image_prediction_saver, val_image_prediction_saver]
        if args.enable_image_savers == 1
        else None
    ),
    "image_postprocessor": image_postprocessor,
    "batch_metric_log_every_n": args.batch_metric_log_every_n,
    "max_eval_batches": max_eval_batches,
    "eval_use_amp": eval_use_amp,
}

# The trainer optimizes with one loader pair while callbacks can use separate,
# non-shuffled loaders for more stable epoch-end metric aggregation.
optimization_manager = OptimizationManager(
    trainer=UNetTrainer,
    hash_splitter=hash_splitter,
    dataset=crop_image_dataset,
    callbacks_args=callbacks_args,
    model_factory=lambda: ConvNeXtUNet(
        in_channels=1,
        out_channels=1,
        decoder_up_block="convt",
    ),
    device=device,
    epochs=args.epochs,
    use_amp=train_use_amp,
    max_train_batches=max_train_batches,
)

study = optuna.create_study(
    study_name=args.study_name,
    direction="minimize",
    storage=args.optuna_storage,
    load_if_exists=args.resume == 1,
)
study.optimize(optimization_manager, n_trials=args.n_trials)

joblib.dump(study, "optuna_study.joblib")
mlflow.log_artifact("optuna_study.joblib")
