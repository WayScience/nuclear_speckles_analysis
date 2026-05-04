# nuclear_speckles_analysis

This repository trains a UNet-style image-to-image translation model on cropped nuclei from `/mnt/big_drive/nuclear_speckle_data/initial_dataset/initial_dataset_raw`.

- Input: cropped DAPI (`CH0`) nucleus image
- Target: cropped Gold (`CH2`) nucleus image
- Task: predict Gold crops from DAPI crops

## Data Pipeline

Training uses cached, filtered single-cell crops generated from:

- `/mnt/big_drive/nuclear_speckle_data/initial_dataset/initial_dataset_raw/IC_corrected_images`
- `/mnt/big_drive/nuclear_speckle_data/initial_dataset/initial_dataset_raw/Preprocessed_data/single_cell_profiles/*annotated*.parquet`

`datasets/dataset_00/utils/CropCacheBuilder.py` builds the crop cache and manifest when needed. Plate, well, site, and channel metadata are parsed from `IC_corrected_images` filenames split by `_`:

- field 1: plate
- field 2: well
- field 3: site
- field 4: channel (`CH0` for DAPI input, `CH2` for Gold target)

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
```

## Loss and Metrics

- Optimization loss: L1
- Logged metrics: L1, L2, PSNR, SSIM

Metrics are implemented in `metrics/` and logged through `callbacks/Callbacks.py`.

## Outputs

- Crop-level prediction artifacts can be logged each epoch via `callbacks/utils/SaveEpochCrops.py`.
- Whole-FOV reconstruction and patch stitching are intentionally removed.
