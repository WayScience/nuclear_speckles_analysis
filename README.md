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
- Logged metrics: L1, L2, PSNR, SSIM

Metrics are implemented in `metrics/` and logged through `callbacks/Callbacks.py`.

## Outputs

- Crop-level prediction artifacts can be logged each epoch via `callbacks/utils/SaveEpochCrops.py`.
- Whole-FOV reconstruction and patch stitching are intentionally removed.
