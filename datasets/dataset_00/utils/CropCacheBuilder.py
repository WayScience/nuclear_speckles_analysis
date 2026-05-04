import csv
import pathlib
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import tifffile


@dataclass
class CropCacheResult:
    """Output metadata returned after building or validating a crop cache.

    Attributes:
        manifest_path: CSV manifest containing one row per cached crop pair.
        image_specs: Normalization and shape metadata inferred from cached images.
    """

    manifest_path: pathlib.Path
    image_specs: dict[str, Any]


def _parse_image_filename(filename: str) -> tuple[str, str, str, str]:
    """Extract plate, well, site, and channel tokens from an image filename.

    Args:
        filename: Image filename expected to contain underscore-delimited metadata.

    Returns:
        A tuple of (plate, well, site, channel).

    Raises:
        ValueError: If the filename stem does not contain at least four fields.
    """

    stem = pathlib.Path(filename).stem
    parts = stem.split("_")
    if len(parts) < 4:
        raise ValueError(f"Filename does not contain plate/well/site/channel fields: {filename}")
    return parts[0], parts[1], parts[2], parts[3]


def _build_image_index(image_dir: pathlib.Path) -> dict[tuple[str, str, str], dict[str, pathlib.Path]]:
    """Index source images by (plate, well, site) and channel.

    Args:
        image_dir: Root directory containing TIFF images.

    Returns:
        Nested mapping keyed by (plate, well, site) and then channel name.
    """

    image_index: dict[tuple[str, str, str], dict[str, pathlib.Path]] = {}

    for image_path in sorted(image_dir.glob("**/*.tiff")):
        plate, well, site, channel = _parse_image_filename(image_path.name)
        key = (plate, well, site)
        if key not in image_index:
            image_index[key] = {}
        image_index[key][channel] = image_path

    return image_index


def _filter_bounding_box_size(scdf: pd.DataFrame, bounding_box_col: str) -> pd.DataFrame:
    """Filter implausible single-cell bounding-box sizes using MAD.

    Single-cell crops are generated directly from CellProfiler nuclei bounding
    boxes. Occasional segmentation artifacts (e.g., merged objects or bad masks)
    produce extreme width/height values that lead to unusable crops and noisy
    cache entries. This filter removes those size outliers per axis before crop
    caching, using a robust median absolute deviation (MAD) threshold.

    Args:
        scdf: Single-cell profile rows.
        bounding_box_col: Bounding-box size column to filter on.

    Returns:
        Filtered DataFrame copy that excludes rows with robust z-score >= 3.
    """

    median = scdf[bounding_box_col].median()
    absolute_dev = (scdf[bounding_box_col] - median).abs()
    mad = absolute_dev.median()
    if mad == 0:
        # If all values are identical (or nearly so), robust scaling is undefined.
        return scdf
    # Robust z-score based on MAD keeps filtering stable against heavy tails.
    robust_z = (scdf[bounding_box_col] - median) / mad
    return scdf.loc[robust_z < 3].copy()


def _compute_shifted_window(start: int, end: int, target_size: int, axis_limit: int) -> tuple[int, int]:
    """Compute a fixed-size crop window and shift it inside image bounds.

    The window is centered on the provided [start, end) interval midpoint.
    If the window would extend outside the field of view, it is translated so
    it fits within [0, axis_limit) instead of introducing zero padding.

    Args:
        start: Bounding-box minimum coordinate (inclusive).
        end: Bounding-box maximum coordinate (exclusive).
        target_size: Desired crop size along this axis.
        axis_limit: Full image size along this axis.

    Returns:
        Tuple of (window_start, window_end), where window_end is exclusive.

    Raises:
        ValueError: If axis_limit is not positive.
    """

    if axis_limit <= 0:
        raise ValueError(f"axis_limit must be positive, got {axis_limit}")

    if target_size >= axis_limit:
        return 0, axis_limit

    center = (start + end) / 2.0
    window_start = int(round(center - target_size / 2.0))
    window_end = window_start + target_size

    if window_start < 0:
        window_end -= window_start
        window_start = 0

    if window_end > axis_limit:
        shift = window_end - axis_limit
        window_start -= shift
        window_end = axis_limit

    if window_start < 0:
        window_start = 0

    return window_start, window_end


def _build_filtered_profiles(data_dir: pathlib.Path) -> pd.DataFrame:
    """Load, align, and filter annotated single-cell profile tables.

    Args:
        data_dir: Dataset root directory.

    Returns:
        Profile DataFrame with required metadata and cleaned bounding-box columns.

    Raises:
        FileNotFoundError: If annotated parquet profile files are missing.
        ValueError: If required profile columns are missing.
    """

    profile_dir = data_dir / "Preprocessed_data" / "single_cell_profiles"
    profile_paths = sorted(profile_dir.glob("*annotated*.parquet"))
    if not profile_paths:
        raise FileNotFoundError(f"No annotated parquet files found in {profile_dir}")

    scdfs = [pd.read_parquet(path) for path in profile_paths]
    common_columns = set(scdfs[0].columns)
    for scdf in scdfs[1:]:
        common_columns &= set(scdf.columns)

    required_cols = {
        "Metadata_Plate",
        "Metadata_Well",
        "Metadata_Site",
        "Nuclei_AreaShape_BoundingBoxMinimum_X",
        "Nuclei_AreaShape_BoundingBoxMaximum_X",
        "Nuclei_AreaShape_BoundingBoxMinimum_Y",
        "Nuclei_AreaShape_BoundingBoxMaximum_Y",
    }
    missing_cols = sorted(required_cols - common_columns)
    if missing_cols:
        raise ValueError(f"Missing required profile columns: {missing_cols}")

    keep_cols = [c for c in scdfs[0].columns if c in common_columns and (c.startswith("Metadata_") or c in required_cols)]
    scdf = pd.concat(scdfs, axis=0, ignore_index=True)[keep_cols].copy()

    bbox_cols = [
        "Nuclei_AreaShape_BoundingBoxMinimum_X",
        "Nuclei_AreaShape_BoundingBoxMaximum_X",
        "Nuclei_AreaShape_BoundingBoxMinimum_Y",
        "Nuclei_AreaShape_BoundingBoxMaximum_Y",
    ]
    for col in bbox_cols:
        scdf[col] = scdf[col].astype(int)

    scdf["Nuclei_AreaShape_BoundingBoxDelta_X"] = (
        scdf["Nuclei_AreaShape_BoundingBoxMaximum_X"] - scdf["Nuclei_AreaShape_BoundingBoxMinimum_X"]
    )
    scdf["Nuclei_AreaShape_BoundingBoxDelta_Y"] = (
        scdf["Nuclei_AreaShape_BoundingBoxMaximum_Y"] - scdf["Nuclei_AreaShape_BoundingBoxMinimum_Y"]
    )

    # Filter bbox width/height outliers before generating fixed-size crops.
    scdf = _filter_bounding_box_size(scdf=scdf, bounding_box_col="Nuclei_AreaShape_BoundingBoxDelta_X")
    scdf = _filter_bounding_box_size(scdf=scdf, bounding_box_col="Nuclei_AreaShape_BoundingBoxDelta_Y")

    return scdf.reset_index(drop=True)


def _validate_manifest(manifest_path: pathlib.Path) -> tuple[bool, list[dict[str, str]]]:
    """Check whether an existing cache manifest can be reused safely.

    Args:
        manifest_path: Path to the cache CSV manifest.

    Returns:
        Tuple of (is_valid, rows). Rows are returned only when valid.
    """

    if not manifest_path.exists():
        return False, []

    with manifest_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        return False, []

    required_fields = {
        "sample_id",
        "group_id",
        "plate",
        "well",
        "site",
        "input_channel",
        "target_channel",
        "input_path",
        "target_path",
    }
    if not required_fields.issubset(rows[0].keys()):
        return False, []

    center_x_pattern = re.compile(r"(?:^|\|)center_x=-?\d+\.\d{6}(?:\||$)")
    center_y_pattern = re.compile(r"(?:^|\|)center_y=-?\d+\.\d{6}(?:\||$)")

    for row in rows:
        sample_id = row["sample_id"]
        if not center_x_pattern.search(sample_id) or not center_y_pattern.search(sample_id):
            return False, []

        # Reuse is only safe when paths and sample IDs still match on-disk files.
        input_path = pathlib.Path(row["input_path"])
        target_path = pathlib.Path(row["target_path"])
        if not input_path.exists() or not target_path.exists():
            return False, []

        is_legacy_layout = input_path.stem == sample_id and target_path.stem == sample_id
        is_identifier_dir_layout = (
            input_path.name == "dapi_cropped_image.tiff"
            and target_path.name == "gold_cropped_image.tiff"
            and input_path.parent.name == sample_id
            and target_path.parent.name == sample_id
        )

        if not (is_legacy_layout or is_identifier_dir_layout):
            return False, []

    return True, rows


def _infer_image_specs(rows: list[dict[str, str]]) -> dict[str, Any]:
    """Infer image normalization and shape metadata from cached crops.

    Args:
        rows: Manifest rows containing absolute crop file paths.

    Returns:
        Dictionary with max pixel values, image shape, and crop margin.

    Raises:
        ValueError: If cached images are not 2D crops.
    """

    input_example = tifffile.imread(rows[0]["input_path"])
    target_example = tifffile.imread(rows[0]["target_path"])

    if input_example.ndim != 2 or target_example.ndim != 2:
        raise ValueError("Cached crop images must be single 2D crops.")

    return {
        "input_max_pixel_value": float(np.iinfo(input_example.dtype).max),
        "target_max_pixel_value": float(np.iinfo(target_example.dtype).max),
        "image_shape": [1, int(input_example.shape[0]), int(input_example.shape[1])],
        "crop_margin": 0,
    }


def ensure_dapi_to_gold_cache(
    data_dir: pathlib.Path,
    cache_dir: pathlib.Path,
) -> CropCacheResult:
    """Build or reuse a CH0(DAPI) to CH2(Gold) crop cache.

    Args:
        data_dir: Dataset root containing masks, images, and profile tables.
        cache_dir: Destination directory for cached crop TIFFs and manifest.

    Returns:
        Manifest path plus inferred image specs for training configuration.

    Raises:
        ValueError: If no valid crop pairs can be produced.
    """

    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.csv"

    is_valid, existing_rows = _validate_manifest(manifest_path=manifest_path)
    if is_valid:
        # Fast path: manifest already points to valid, existing cached crops.
        return CropCacheResult(
            manifest_path=manifest_path,
            image_specs=_infer_image_specs(rows=existing_rows),
        )

    scdf = _build_filtered_profiles(data_dir=data_dir)

    target_width = int(scdf["Nuclei_AreaShape_BoundingBoxDelta_X"].max())
    target_height = int(scdf["Nuclei_AreaShape_BoundingBoxDelta_Y"].max())

    rows: list[dict[str, str]] = []

    nuclear_mask_dir = (data_dir / "Nuclear_masks").resolve(strict=True)
    image_dir = (data_dir / "IC_corrected_images").resolve(strict=True)

    image_index = _build_image_index(image_dir=image_dir)

    dapi_mask_paths = sorted(nuclear_mask_dir.glob("**/*CH0*.tiff"))

    for dapi_mask_path in dapi_mask_paths:
        dapi_filename = dapi_mask_path.name.replace("_MaskNuclei", "")
        try:
            plate_name, well_name, site_name, mask_channel = _parse_image_filename(
                dapi_filename
            )
        except ValueError:
            continue

        if mask_channel != "CH0":
            continue

        image_key = (plate_name, well_name, site_name)
        keyed_images = image_index.get(image_key)
        if keyed_images is None:
            continue

        dapi_img_path = keyed_images.get("CH0")
        gold_img_path = keyed_images.get("CH2")

        if dapi_img_path is None or gold_img_path is None:
            continue

        image_df = scdf.loc[
            (scdf["Metadata_Plate"] == plate_name)
            & (scdf["Metadata_Well"] == well_name)
            & (scdf["Metadata_Site"] == site_name)
        ]
        if image_df.empty:
            continue

        dapi_mask = tifffile.imread(dapi_mask_path)
        dapi_img = tifffile.imread(dapi_img_path)
        gold_img = tifffile.imread(gold_img_path)

        if dapi_mask.ndim != 2 or dapi_img.ndim != 2 or gold_img.ndim != 2:
            raise ValueError(
                "Expected 2D CH0 mask, DAPI image, and Gold image after selecting a single z-crop."
            )

        image_df = image_df.copy()
        bbox_cols = [
            "Nuclei_AreaShape_BoundingBoxMinimum_X",
            "Nuclei_AreaShape_BoundingBoxMaximum_X",
            "Nuclei_AreaShape_BoundingBoxMinimum_Y",
            "Nuclei_AreaShape_BoundingBoxMaximum_Y",
        ]
        for col in bbox_cols:
            image_df[col] = image_df[col].astype(int)

        for _, nucleus in image_df.iterrows():
            x0 = int(nucleus["Nuclei_AreaShape_BoundingBoxMinimum_X"])
            y0 = int(nucleus["Nuclei_AreaShape_BoundingBoxMinimum_Y"])
            x1 = int(nucleus["Nuclei_AreaShape_BoundingBoxMaximum_X"])
            y1 = int(nucleus["Nuclei_AreaShape_BoundingBoxMaximum_Y"])

            crop_x0, crop_x1 = _compute_shifted_window(
                start=x0,
                end=x1,
                target_size=target_width,
                axis_limit=dapi_img.shape[1],
            )
            crop_y0, crop_y1 = _compute_shifted_window(
                start=y0,
                end=y1,
                target_size=target_height,
                axis_limit=dapi_img.shape[0],
            )

            cropped_dapi = dapi_img[crop_y0:crop_y1, crop_x0:crop_x1].copy()
            cropped_mask = dapi_mask[crop_y0:crop_y1, crop_x0:crop_x1]
            cropped_gold = gold_img[crop_y0:crop_y1, crop_x0:crop_x1].copy()

            if cropped_dapi.size == 0 or cropped_gold.size == 0:
                continue

            # Skip empty-signal crops.
            if cropped_dapi.max() == 0 or cropped_gold.max() == 0:
                continue

            # Keep raw background intensity values; do not zero outside mask.
            if cropped_mask.size == 0:
                continue

            center_x = (x0 + x1) / 2
            center_y = (y0 + y1) / 2

            sample_id = (
                f"plate={plate_name}|well={well_name}|site={site_name}|center_x={center_x:.6f}|center_y={center_y:.6f}"
            )
            group_id = f"plate={plate_name}|well={well_name}|site={site_name}"

            sample_dir = cache_dir / sample_id
            sample_dir.mkdir(parents=True, exist_ok=True)

            input_path = sample_dir / "dapi_cropped_image.tiff"
            target_path = sample_dir / "gold_cropped_image.tiff"

            if not input_path.exists():
                tifffile.imwrite(input_path, cropped_dapi)
            if not target_path.exists():
                tifffile.imwrite(target_path, cropped_gold)

            rows.append(
                {
                    "sample_id": sample_id,
                    "group_id": group_id,
                    "plate": plate_name,
                    "well": well_name,
                    "site": site_name,
                    "input_channel": "CH0",
                    "target_channel": "CH2",
                    "input_path": str(input_path.resolve()),
                    "target_path": str(target_path.resolve()),
                }
            )

    if not rows:
        raise ValueError("No valid DAPI-to-Gold crops were cached.")

    with manifest_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "group_id",
                "plate",
                "well",
                "site",
                "input_channel",
                "target_channel",
                "input_path",
                "target_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    return CropCacheResult(manifest_path=manifest_path, image_specs=_infer_image_specs(rows=rows))


def load_cache_manifest(manifest_path: pathlib.Path) -> list[dict[str, str]]:
    """Load a cache manifest into memory.

    Args:
        manifest_path: Path to a cache manifest CSV.

    Returns:
        List of manifest rows as dictionaries.

    Raises:
        ValueError: If the manifest has no data rows.
    """

    with manifest_path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        raise ValueError(f"No rows found in cache manifest: {manifest_path}")

    return rows
