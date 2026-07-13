# nuclear_speckles_analysis

## Overview

This repository trains image-to-image models that predict nuclear speckle signal from cropped single-cell nucleus images. The training workflow is cache-backed, uses deterministic dataset splitting, and can be launched directly with `uv run` or through MLflow-managed experiment runs.

- Input: cropped DAPI nucleus image
- Target: cropped Gold nucleus image
- Task: predict Gold crops from DAPI crops

Core pieces:

- `train.py` is the main training and Optuna optimization entrypoint.
- dataset utilities build crop caches so repeated runs do not need to reprocess raw images.
- callbacks handle epoch-end evaluation, metric logging, image export, and early stopping.

Dataset-specific channel mappings:

- `initial`: DAPI=`CH0`, Gold=`CH2`
- `u2os`: DAPI=`CH01`, Gold=`CH03`

Dataset-specific cache roots:

- `initial`: `/mnt/big_drive/nuclear_speckle_data/initial_dataset/model_cache`
- `u2os`: `/mnt/big_drive/nuclear_speckle_data/u20s_dataset_jan_15_2026/model_cache`

Within each cache root, training uses:

- `dapi_to_gold_crop_cache`
- `paired_tensor_cache`

### Normalization

Before samples are passed through the model, both input and target crops are normalized by their channel max intensity (dtype max, for example `255` for `uint8` and `65535` for `uint16`) in `datasets/dataset_00/utils/ImagePreProcessor.py`.

## Data Splitting Strategy

Data splitting happens in two stages:

1. One full plate is held out per dataset before hash splitting.
2. The remaining samples are assigned deterministically to train/validation/test splits using `farmhash.Fingerprint64(metadata["Metadata_ID"]) % 10**6` in `splitters/HashSplitter.py`.

Current holdout configuration in `train.py`:

- `u2os`: hold out plate `Rep3`
- `initial`: hold out plate `slide2`

Current split fractions after holdout filtering:

- train: `0.825`
- validation: `0.125`
- test: remaining `0.05`

The hash-based assignment keeps splits deterministic across reruns for the same cached sample identities. The held-out plate for each dataset was chosen as the plate with the fewest single cells in that dataset, so a full acquisition batch can be excluded while minimizing the reduction in training data.

## Training

Main entrypoint:

```bash
mlflow run . -e train_model
```

Fast smoke run:

```bash
uv run train.py --epochs 1 --n-trials 1 --max-train-batches 2 --max-eval-batches 2 --enable-image-savers 0

# Select dataset (defaults to u2os)
uv run train.py --dataset initial --epochs 1 --n-trials 1 --max-train-batches 2 --max-eval-batches 2 --enable-image-savers 0
```

The repository also includes `training_orchestration.sh`, which runs the main training entrypoint for each configured dataset with MLflow tracking enabled.

## Precision / AMP

- Dataset tensors are loaded and normalized as `float32` before they are passed to the model.
- Training forward passes can run under automatic mixed precision with `torch.amp.autocast(...)`.
- Epoch-end evaluation and MLflow signature inference can also run under autocast when evaluation AMP is enabled.
- When AMP is enabled, this training path uses `bfloat16` autocast.

In practice this allows training and evaluation inference behavior to be configured separately while keeping the rest of the training loop unchanged.

## Loss and Metrics

- Optimization loss: L1
- Logged metrics: L1, L2, PSNR, SSIM, Pearson correlation

Metrics are implemented in `metrics/` and logged through `callbacks/CallbackPipeline.py`.

## Outputs

- Crop-level prediction artifacts can be logged each epoch via `callbacks/utils/SaveEpochCrops.py`.
- Whole-FOV reconstruction and patch stitching are intentionally removed.

## Third-Party Attribution

- The code in `models/convnext_unet/` is adapted from `virtual_stain_flow` by WayScience (contributor: Weishan Li).
- See `THIRD_PARTY_NOTICES.md` for source and license details.
