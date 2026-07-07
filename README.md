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

When running through `training_orchestration.sh` on Alpine, the script now pre-creates
the MLflow experiment with an explicit artifact location under
`/pl/active/koala/nuclear_speckle_data/training_results/mlartifacts/`.
Without that step, MLflow can fall back to creating a repo-local `mlruns/` directory.

## MLflow Migration

Install the migration tooling into the existing `uv` environment:

```bash
uv sync --group migration
```

Inspect a SQLite-backed MLflow store before migration:

```bash
./scripts/inspect_mlflow_store.sh \
  --db-path /home/camo/projects/nuclear_speckles_analysis/mlflow.db
```

Migrate all MLflow data from `alpine` into the root repo on this machine:

```bash
./scripts/migrate_mlflow_from_alpine.sh \
  --source-host alpine \
  --source-db /pl/active/koala/nuclear_speckle_data/training_results/mlflow.db \
  --target-repo /home/camo/projects/nuclear_speckles_analysis
```

What the migration script does:

- `rsync`s the source `mlflow.db` from `alpine`
- inspects experiment artifact locations in the copied DB
- prints the exact `rsync` commands it will run for artifact roots referenced by the DB
- copies those artifact roots into a local staging area under `/tmp/opencode/mlflow_migration`
- rewrites the copied DB so its artifact paths point at the staged local copies
- starts temporary local source and target MLflow servers
- runs `export-all` and `import-all` from `mlflow-export-import`
- preserves source provenance with `--import-source-tags True`
- prints the imported destination run IDs at the end so they are easy to verify in the UI

Notes:

- Imported runs are recreated in the destination store, so they appear as recent runs in the target MLflow UI.
- The source run ID is preserved on imported runs as the `mlflow_exim.run_info.run_id` tag.
- Re-running the import can create duplicate runs.
- The target MLflow server must be writable during import. The server process can be running, but avoid concurrent writes into the same experiments while the migration is in progress.
- The migration script starts temporary servers on ports `6001` and `6002` by default; pick different ports if those are already in use.

## Loss and Metrics

- Optimization loss: L1
- Logged metrics: L1, L2, PSNR, SSIM

Metrics are implemented in `metrics/` and logged through `callbacks/Callbacks.py`.

## Outputs

- Crop-level prediction artifacts can be logged each epoch via `callbacks/utils/SaveEpochCrops.py`.
- Whole-FOV reconstruction and patch stitching are intentionally removed.

## Third-Party Attribution

- The code in `models/convnext_unet/` is adapted from `virtual_stain_flow` by WayScience (contributor: Weishan Li).
- See `THIRD_PARTY_NOTICES.md` for source and license details.
