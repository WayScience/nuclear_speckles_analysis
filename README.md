# nuclear_speckles_analysis

This repository trains a UNet-style image-to-image translation model on cropped nuclei from multiple nuclear speckle datasets (including the initial dataset and U2OS).

- Input: cropped DAPI nucleus image
- Target: cropped Gold nucleus image
- Task: predict Gold crops from DAPI crops

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

## Precision / AMP

This repository currently uses a split precision strategy:

- Dataset tensors are loaded and normalized as `float32` in `datasets/dataset_00/utils/ImagePreProcessor.py`.
- The main training forward pass and loss computation run under automatic mixed precision (AMP) in `trainers/UNetTrainer.py` via `torch.amp.autocast(...)` and `GradScaler`.
- Epoch-end metric computation in `callbacks/evaluation.py` runs under `torch.no_grad()` without `autocast(...)`.
- Example output generation in `callbacks/utils/SaveEpochCrops.py` and MLflow signature inference in `callbacks/early_stopping.py` also run without `autocast(...)`.

In practice this means:

- training forward/backward: AMP enabled
- validation and train-split epoch metrics: full precision evaluation path
- saved example outputs: full precision evaluation path
- raw crop storage: TIFF

This is a conservative and common engineering setup for image-to-image training:

- AMP is used where it usually provides the most value: lower memory pressure and faster model training on CUDA.
- Validation metrics and model-selection logic stay outside AMP to reduce small numerical differences in logged L1/L2/PSNR/SSIM/Pearson values and early-stopping decisions.
- Example outputs are generated outside AMP because they are primarily qualitative inspection artifacts rather than the training bottleneck.

This README intentionally describes the current repository policy. The use of AMP for training is well supported in the literature; keeping validation metrics outside AMP is better understood as a conservative reproducibility choice than as a single formal literature standard.

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
