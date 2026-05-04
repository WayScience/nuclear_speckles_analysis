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

    image_paths = sorted(image_dir.glob("**/*.tiff")) + sorted(image_dir.glob("**/*.tif"))
    for image_path in image_paths:
        if "excluded" in image_path.parts:
            continue
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


def _add_padding(image: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
    """Pad a 2D crop to the target size with centered zero padding.

    Args:
        image: Input crop image.
        target_width: Desired output width.
        target_height: Desired output height.

    Returns:
        Padded image with shape (target_height, target_width).

    Raises:
        ValueError: If target size is smaller than the input crop.
    """

    height, width = image.shape[:2]
    # Split extra pixels across both sides to keep the crop centered.
    top = (target_height - height) // 2
    bottom = target_height - height - top
    left = (target_width - width) // 2
    right = target_width - width - left

    if min(top, bottom, left, right) < 0:
        raise ValueError("Target crop shape is smaller than a nucleus crop.")

    if top == 0 and bottom == 0 and left == 0 and right == 0:
        return image

    return np.pad(image, ((top, bottom), (left, right)), mode="constant", constant_values=0)


def _build_filtered_profiles(parquet_path: pathlib.Path) -> pd.DataFrame:
    """Load, align, and filter annotated single-cell profile tables.

    Args:
        parquet_path: Absolute path to single-cell profile parquet.

    Returns:
        Profile DataFrame with required metadata and cleaned bounding-box columns.

    Raises:
        FileNotFoundError: If annotated parquet profile files are missing.
        ValueError: If required profile columns are missing.
    """

    parquet_path = parquet_path.resolve(strict=True)
    scdf = pd.read_parquet(parquet_path)
    common_columns = set(scdf.columns)

    required_cols = {
        "Metadata_Plate",
        "Metadata_Well",
        "Metadata_Site",
        "Metadata_Nuclei_AreaShape_BoundingBoxMinimum_X",
        "Metadata_Nuclei_AreaShape_BoundingBoxMaximum_X",
        "Metadata_Nuclei_AreaShape_BoundingBoxMinimum_Y",
        "Metadata_Nuclei_AreaShape_BoundingBoxMaximum_Y",
    }
    missing_cols = sorted(required_cols - common_columns)
    if missing_cols:
        raise ValueError(f"Missing required profile columns: {missing_cols}")

    keep_cols = [c for c in scdf.columns if c in common_columns and (c.startswith("Metadata_") or c in required_cols)]
    scdf = scdf[keep_cols].copy()

    bbox_cols = [
        "Metadata_Nuclei_AreaShape_BoundingBoxMinimum_X",
        "Metadata_Nuclei_AreaShape_BoundingBoxMaximum_X",
        "Metadata_Nuclei_AreaShape_BoundingBoxMinimum_Y",
        "Metadata_Nuclei_AreaShape_BoundingBoxMaximum_Y",
    ]
    for col in bbox_cols:
        scdf[col] = scdf[col].astype(int)

    scdf["Nuclei_AreaShape_BoundingBoxDelta_X"] = (
        scdf["Metadata_Nuclei_AreaShape_BoundingBoxMaximum_X"]
        - scdf["Metadata_Nuclei_AreaShape_BoundingBoxMinimum_X"]
    )
    scdf["Nuclei_AreaShape_BoundingBoxDelta_Y"] = (
        scdf["Metadata_Nuclei_AreaShape_BoundingBoxMaximum_Y"]
        - scdf["Metadata_Nuclei_AreaShape_BoundingBoxMinimum_Y"]
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
    image_dir: pathlib.Path,
    parquet_path: pathlib.Path,
    cache_dir: pathlib.Path,
) -> CropCacheResult:
    """Build or reuse a CH0(DAPI) to CH2(Gold) crop cache.

    Args:
        image_dir: Directory containing source TIFF images.
        parquet_path: Path to single-cell profile parquet.
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

    scdf = _build_filtered_profiles(parquet_path=parquet_path)

    target_width = int(scdf["Nuclei_AreaShape_BoundingBoxDelta_X"].max())
    target_height = int(scdf["Nuclei_AreaShape_BoundingBoxDelta_Y"].max())

    rows: list[dict[str, str]] = []

    image_dir = image_dir.resolve(strict=True)

    image_index = _build_image_index(image_dir=image_dir)

    for image_key, keyed_images in image_index.items():
        plate_name, well_name, site_name = image_key

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

        dapi_img = tifffile.imread(dapi_img_path)
        gold_img = tifffile.imread(gold_img_path)

        if dapi_img.ndim != 2 or gold_img.ndim != 2:
            raise ValueError(
                "Expected 2D DAPI image and Gold image after selecting a single z-crop."
            )

        image_df = image_df.copy()
        bbox_cols = [
            "Metadata_Nuclei_AreaShape_BoundingBoxMinimum_X",
            "Metadata_Nuclei_AreaShape_BoundingBoxMaximum_X",
            "Metadata_Nuclei_AreaShape_BoundingBoxMinimum_Y",
            "Metadata_Nuclei_AreaShape_BoundingBoxMaximum_Y",
        ]
        for col in bbox_cols:
            image_df[col] = image_df[col].astype(int)

        for _, nucleus in image_df.iterrows():
            x0 = int(nucleus["Metadata_Nuclei_AreaShape_BoundingBoxMinimum_X"])
            y0 = int(nucleus["Metadata_Nuclei_AreaShape_BoundingBoxMinimum_Y"])
            x1 = int(nucleus["Metadata_Nuclei_AreaShape_BoundingBoxMaximum_X"])
            y1 = int(nucleus["Metadata_Nuclei_AreaShape_BoundingBoxMaximum_Y"])

            cropped_dapi = dapi_img[y0:y1, x0:x1].copy()
            cropped_gold = gold_img[y0:y1, x0:x1].copy()

            if cropped_dapi.size == 0 or cropped_gold.size == 0:
                continue

            # Skip empty-signal crops in either channel.
            if cropped_dapi.max() == 0 or cropped_gold.max() == 0:
                continue

            padded_dapi = _add_padding(cropped_dapi, target_width=target_width, target_height=target_height)
            padded_gold = _add_padding(cropped_gold, target_width=target_width, target_height=target_height)

            if padded_dapi.max() == 0 or padded_gold.max() == 0:
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
                tifffile.imwrite(input_path, padded_dapi)
            if not target_path.exists():
                tifffile.imwrite(target_path, padded_gold)

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
