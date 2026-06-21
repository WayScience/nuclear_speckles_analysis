import argparse
import pathlib
import random
from dataclasses import dataclass
from typing import Any, Callable

import joblib
import mlflow
import numpy as np
import optuna
import torch
from models.unconditional_critic import UnconditionalCritic
from models.convnext_unet.unext import ConvNeXtUNet

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
from losses.WassersteinGeneratorCrossZamirskiLoss import (
    WassersteinGeneratorCrossZamirskiLoss,
)
from losses.WassersteinGradientPenaltyLoss import WassersteinGradientPenaltyLoss
from metrics.L1 import L1
from metrics.L2 import L2
from metrics.PearsonCorrelation import PearsonCorrelation
from metrics.PSNR import PSNR
from metrics.SSIM import SSIM
from splitters.HashSplitter import HashSplitter
from trainers.WGANGPTrainer import WGANGPTrainer

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
    """

    image_dir: pathlib.Path
    parquet_path: pathlib.Path
    cache_root: pathlib.Path
    input_channel: str
    target_channel: str
    metadata_column_map: dict[str, str] | None = None
    holdout_plate: str | None = None


DATASET_CONFIGS = {
    "u2os": DatasetConfig(
        image_dir=pathlib.Path(
            "/mnt/big_drive/nuclear_speckle_data/u20s_dataset_jan_15_2026/u20s_images/tiffs"
        ),
        parquet_path=pathlib.Path(
            "/mnt/big_drive/nuclear_speckle_data/u20s_dataset_jan_15_2026/u20s_profiles/single_cell_profiles/u2os_per_nuclei_sc_feature_selected.parquet"
        ),
        cache_root=pathlib.Path(
            "/mnt/big_drive/nuclear_speckle_data/u20s_dataset_jan_15_2026/model_cache"
        ),
        input_channel="CH01",
        target_channel="CH03",
        metadata_column_map={
            # U2OS profiles label imaging site as "Metadata_Position".
            "Metadata_Position": "Metadata_Site",
        },
        holdout_plate="Rep3",
    ),
    "initial": DatasetConfig(
        image_dir=pathlib.Path(
            "/mnt/big_drive/nuclear_speckle_data/initial_dataset/IC_corrected_images"
        ),
        parquet_path=pathlib.Path(
            "/mnt/big_drive/nuclear_speckle_data/initial_dataset/Preprocessed_data/cleaned_sc_profiles"
        ),
        cache_root=pathlib.Path(
            "/mnt/big_drive/nuclear_speckle_data/initial_dataset/model_cache"
        ),
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
    ),
}


parser = argparse.ArgumentParser()
parser.add_argument("--epochs", type=int, default=20)
parser.add_argument("--n-trials", type=int, default=4)
parser.add_argument("--max-train-batches", type=int, default=-1)
parser.add_argument("--max-eval-batches", type=int, default=-1)
parser.add_argument("--enable-image-savers", type=int, choices=[0, 1], default=1)
parser.add_argument("--batch-metric-log-every-n", type=int, default=1)
parser.add_argument("--dataset", choices=sorted(DATASET_CONFIGS.keys()), default="u2os")
parser.add_argument("--crop-size", type=int, default=256)
args = parser.parse_args()

# Interpret non-positive limits as "use the full epoch" for trainer/eval loops.
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
        generator_factory: Callable[[], torch.nn.Module],
        discriminator_factory: Callable[[], torch.nn.Module],
        **trainer_kwargs,
    ):
        """Store dependencies for Optuna-driven training trials.

        Args:
            trainer: Trainer class used to run one trial.
            hash_splitter: Callable that returns train/val/test dataloaders.
            dataset: Dataset associated with the optimization run.
            callbacks_args: Static callback arguments reused across trials.
            generator_factory: Callable that creates a new generator instance per trial.
            discriminator_factory: Callable that creates a new discriminator instance per trial.
            **trainer_kwargs: Shared trainer keyword arguments.
        """

        self.trainer = trainer
        self.hash_splitter = hash_splitter
        self.dataset = dataset
        self.callbacks_args = callbacks_args
        self.generator_factory = generator_factory
        self.discriminator_factory = discriminator_factory
        self.trainer_kwargs = trainer_kwargs

    def __call__(self, trial: optuna.trial.Trial):
        """Execute one Optuna trial and return objective loss.

        Args:
            trial: Optuna trial used for hyperparameter suggestions.

        Returns:
            Best validation loss reported by the trainer.
        """

        # Let Optuna choose core optimization and loss weights for this trial.
        batch_size = trial.suggest_int("batch_size", 1, 2)
        lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
        gradient_penalty_importance = trial.suggest_float(
            "gradient_penalty_importance", 1.0, 20.0
        )
        reconstruction_importance = trial.suggest_float(
            "reconstruction_importance", 10.0, 200.0
        )
        discriminator_updates_per_generator_update = trial.suggest_int(
            "discriminator_updates_per_generator_update", 1, 5
        )

        # Rebuild train/val loaders at the chosen batch size while keeping deterministic splits.
        train_dataloader, val_dataloader, _ = self.hash_splitter(batch_size=batch_size)
        self.trainer_kwargs["train_dataloader"] = train_dataloader
        self.trainer_kwargs["val_dataloader"] = val_dataloader

        generator = self.generator_factory()
        discriminator = self.discriminator_factory()
        self.trainer_kwargs["generator"] = generator
        self.trainer_kwargs["discriminator"] = discriminator

        generator_optimizer_params = {
            "params": generator.parameters(),
            "lr": lr,
            "betas": (0.5, 0.999),
        }
        discriminator_optimizer_params = {
            "params": discriminator.parameters(),
            "lr": lr,
            "betas": (0.5, 0.999),
        }

        generator_loss = WassersteinGeneratorCrossZamirskiLoss(
            reconstruction_importance=reconstruction_importance,
            use_adversarial_decay=True,
        )
        discriminator_loss = WassersteinGradientPenaltyLoss(
            gradient_penalty_importance=gradient_penalty_importance
        )
        loss_callbacks = L1(device=device)
        metrics = [
            L2(device=device),
            PSNR(device=device, max_pixel_value=1.0),
            SSIM(device=device, max_pixel_value=1.0),
            PearsonCorrelation(device=device),
        ]

        # Use a nested MLflow run so each Optuna trial has its own metrics/artifacts.
        with mlflow.start_run(nested=True, run_name=f"trial_{trial.number}"):
            generator_optimizer = torch.optim.Adam(**generator_optimizer_params)
            discriminator_optimizer = torch.optim.Adam(**discriminator_optimizer_params)
            self.trainer_kwargs["generator_optimizer"] = generator_optimizer
            self.trainer_kwargs["discriminator_optimizer"] = discriminator_optimizer

            opt_params = generator_optimizer.param_groups[0].copy()
            del opt_params["params"]
            mlflow.log_params({f"optimizer_{k}": v for k, v in opt_params.items()})
            mlflow.log_param("batch_size", batch_size)
            mlflow.log_param("gradient_penalty_importance", gradient_penalty_importance)
            mlflow.log_param("reconstruction_importance", reconstruction_importance)
            mlflow.log_param(
                "discriminator_updates_per_generator_update",
                discriminator_updates_per_generator_update,
            )
            mlflow.log_param("use_adversarial_decay", True)
            mlflow.set_tag(
                "optimizer_class", generator_optimizer.__class__.__name__.lower()
            )

            self.trainer_kwargs["callbacks"] = CallbackPipeline(
                **self.callbacks_args | {"metrics": metrics, "loss": loss_callbacks}
            )

            trainer_obj = self.trainer(
                **self.trainer_kwargs
                | {
                    "generator_loss": generator_loss,
                    "discriminator_loss": discriminator_loss,
                    "discriminator_updates_per_generator_update": (
                        discriminator_updates_per_generator_update
                    ),
                }
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
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
mlflow.log_param("random_seed", 0)
mlflow.log_param("dataset", args.dataset)
mlflow.log_param("input_channel", dataset_config.input_channel)
mlflow.log_param("target_channel", dataset_config.target_channel)
mlflow.log_param("crop_size", args.crop_size)

description = """
Optimization of an unconditional WGAN-GP DAPI-to-Gold image-to-image translation model with:
- ConvNeXtUNet Generator
- Unconditional convolutional discriminator
- Single 2D crop input and single 2D crop target
- Cache-backed filtered nucleus crops generated from the configured data directory
- Generator objective: reconstruction-weighted L1 plus epoch-decayed adversarial term
- Discriminator objective: Wasserstein loss with gradient penalty
- Validation uses L1, L2, PSNR, SSIM, and Pearson correlation metric logging
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
)
manifest_nuclei = load_cache_manifest(manifest_path=cache_result.manifest_path)
manifest_nuclei_before_holdout_filter = len(manifest_nuclei)
if dataset_config.holdout_plate is not None:
    # Keep one plate fully held out to prevent leakage across similar acquisition batches.
    manifest_nuclei = [
        nuclei for nuclei in manifest_nuclei if nuclei.get("plate") != dataset_config.holdout_plate
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

image_preprocessor = ImagePreProcessor(image_specs=image_specs, device=device)
image_postprocessor = ImagePostProcessor()

crop_image_dataset = CellCropToCropDataset(
    manifest_rows=manifest_nuclei,
    image_specs=image_specs,
    image_preprocessor=image_preprocessor,
    image_cache_path=tensor_cache_path,
)

# HashSplitter uses metadata-derived IDs, so splits stay stable across reruns.
hash_splitter = HashSplitter(
    dataset=crop_image_dataset,
    train_frac=0.825,
    val_frac=0.125,
)

# Use a fixed batch size here only to iterate splits while choosing preview images.
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
    "early_stopping_counter_threshold": 10,
    "image_savers": (
        [train_image_prediction_saver, val_image_prediction_saver]
        if args.enable_image_savers == 1
        else None
    ),
    "image_postprocessor": image_postprocessor,
    "batch_metric_log_every_n": args.batch_metric_log_every_n,
    "max_eval_batches": max_eval_batches,
}

optimization_manager = OptimizationManager(
    trainer=WGANGPTrainer,
    hash_splitter=hash_splitter,
    dataset=crop_image_dataset,
    callbacks_args=callbacks_args,
    generator_factory=lambda: ConvNeXtUNet(
        in_channels=1,
        out_channels=1,
        decoder_up_block="convt",
    ),
    discriminator_factory=lambda: UnconditionalCritic(in_channels=1),
    epochs=args.epochs,
    image_postprocessor=image_postprocessor,
    device=device,
    max_train_batches=max_train_batches,
)

study = optuna.create_study(study_name="model_training", direction="minimize")
study.optimize(optimization_manager, n_trials=args.n_trials)

joblib.dump(study, "optuna_study.joblib")
mlflow.log_artifact("optuna_study.joblib")
