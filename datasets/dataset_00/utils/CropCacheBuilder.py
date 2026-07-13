import csv
import pathlib
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import tifffile
from skimage.transform import resize


@dataclass
class CropCacheResult:
    """Output metadata returned after building or validating a crop cache.

    Attributes:
        manifest_path: CSV manifest containing one row per cached crop pair.
        image_specs: Normalization and shape metadata inferred from cached images.
    """

    manifest_path: pathlib.Path
    image_specs: dict[str, Any]


@dataclass(frozen=True)
class ResamplingGeometry:
    """Geometry needed to map full-image pixels and bbox coordinates together."""

    scale_factor: float
    resized_height: int
    resized_width: int
    output_height: int
    output_width: int
    crop_top: int
    crop_left: int
    pad_top: int
    pad_bottom: int
    pad_left: int
    pad_right: int


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
        Nested mapping keyed by (plate, well, site) and then uppercase
        channel name.
    """

    image_index: dict[tuple[str, str, str], dict[str, pathlib.Path]] = {}

    image_paths = sorted(image_dir.glob("**/*.tiff")) + sorted(image_dir.glob("**/*.tif"))
    for image_path in image_paths:
        if "excluded" in image_path.parts:
            continue
        plate, well, site, channel = _parse_image_filename(image_path.name)
        # Normalize channel token case so config values and filenames match reliably.
        channel = channel.upper()
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

    The window is, by default, centered on the provided [start, end) interval midpoint.
    If the window would extend outside the field of view, the window is translated so
    it fits within [0, axis_limit) instead of introducing zero padding.

    Args:
        start: Bounding-box minimum coordinate (inclusive).
        end: Bounding-box maximum coordinate (exclusive).
        target_size: Desired crop size along this axis.
        axis_limit: Full image size along this axis.

    Returns:
        Tuple of (window_start, window_end), where window_end is exclusive.

    Raises:
        ValueError: If axis_limit or target_size is not positive.
    """

    if axis_limit <= 0:
        raise ValueError(f"axis_limit must be positive, got {axis_limit}")

    if target_size <= 0:
        raise ValueError(f"target_size must be positive, got {target_size}")

    if target_size >= axis_limit:
        return 0, axis_limit

    center = (start + end) / 2.0
    window_start = int(round(center - target_size / 2.0))
    window_start = min(max(window_start, 0), axis_limit - target_size)
    window_end = window_start + target_size

    return window_start, window_end


def _build_resampling_geometry(
    input_shape: tuple[int, int],
    template_shape: tuple[int, int],
    input_resolution: float,
    target_resolution: float,
) -> ResamplingGeometry:
    """Compute centered crop/pad geometry for whole-image resampling."""

    if input_resolution <= 0 or target_resolution <= 0:
        raise ValueError(
            "input_resolution and target_resolution must both be positive "
            f"got input_resolution={input_resolution}, target_resolution={target_resolution}"
        )

    scale_factor = float(input_resolution) / float(target_resolution)
    input_height, input_width = input_shape
    output_height, output_width = template_shape

    resized_height = max(1, int(round(input_height * scale_factor)))
    resized_width = max(1, int(round(input_width * scale_factor)))

    crop_top = max(0, (resized_height - output_height) // 2)
    crop_left = max(0, (resized_width - output_width) // 2)

    cropped_height = min(resized_height, output_height)
    cropped_width = min(resized_width, output_width)

    pad_total_y = max(0, output_height - cropped_height)
    pad_total_x = max(0, output_width - cropped_width)
    pad_top = pad_total_y // 2
    pad_bottom = pad_total_y - pad_top
    pad_left = pad_total_x // 2
    pad_right = pad_total_x - pad_left

    return ResamplingGeometry(
        scale_factor=scale_factor,
        resized_height=resized_height,
        resized_width=resized_width,
        output_height=output_height,
        output_width=output_width,
        crop_top=crop_top,
        crop_left=crop_left,
        pad_top=pad_top,
        pad_bottom=pad_bottom,
        pad_left=pad_left,
        pad_right=pad_right,
    )


def _cast_resampled_image(image: np.ndarray, dtype: np.dtype) -> np.ndarray:
    """Cast interpolated image values back to the source dtype."""

    if np.issubdtype(dtype, np.integer):
        dtype_info = np.iinfo(dtype)
        return np.clip(np.rint(image), dtype_info.min, dtype_info.max).astype(dtype)

    return image.astype(dtype, copy=False)


def _register_and_resample_full_image(
    image: np.ndarray,
    geometry: ResamplingGeometry,
) -> np.ndarray:
    """Resample a full image, then center-crop or pad it to template size."""

    resized = resize(
        image,
        output_shape=(geometry.resized_height, geometry.resized_width),
        order=3,
        preserve_range=True,
        anti_aliasing=False,
    )
    resized = _cast_resampled_image(resized, image.dtype)

    cropped = resized[
        geometry.crop_top : geometry.crop_top + min(geometry.resized_height, geometry.output_height),
        geometry.crop_left : geometry.crop_left + min(geometry.resized_width, geometry.output_width),
    ]

    if (
        geometry.pad_top
        or geometry.pad_bottom
        or geometry.pad_left
        or geometry.pad_right
    ):
        cropped = np.pad(
            cropped,
            pad_width=(
                (geometry.pad_top, geometry.pad_bottom),
                (geometry.pad_left, geometry.pad_right),
            ),
            mode="constant",
            constant_values=0,
        )

    return cropped


def _transform_bbox_coordinates(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    geometry: ResamplingGeometry,
) -> tuple[int, int, int, int]:
    """Map a bbox from original full-image coordinates into resampled image space."""

    scaled_x0 = int(np.floor(x0 * geometry.scale_factor))
    scaled_y0 = int(np.floor(y0 * geometry.scale_factor))
    scaled_x1 = int(np.ceil(x1 * geometry.scale_factor))
    scaled_y1 = int(np.ceil(y1 * geometry.scale_factor))

    transformed_x0 = scaled_x0 - geometry.crop_left + geometry.pad_left
    transformed_y0 = scaled_y0 - geometry.crop_top + geometry.pad_top
    transformed_x1 = scaled_x1 - geometry.crop_left + geometry.pad_left
    transformed_y1 = scaled_y1 - geometry.crop_top + geometry.pad_top

    transformed_x0 = min(max(transformed_x0, 0), geometry.output_width)
    transformed_y0 = min(max(transformed_y0, 0), geometry.output_height)
    transformed_x1 = min(max(transformed_x1, 0), geometry.output_width)
    transformed_y1 = min(max(transformed_y1, 0), geometry.output_height)

    if transformed_x1 <= transformed_x0:
        transformed_x1 = min(geometry.output_width, transformed_x0 + 1)
        transformed_x0 = max(0, transformed_x1 - 1)
    if transformed_y1 <= transformed_y0:
        transformed_y1 = min(geometry.output_height, transformed_y0 + 1)
        transformed_y0 = max(0, transformed_y1 - 1)

    return transformed_x0, transformed_y0, transformed_x1, transformed_y1


def _build_filtered_profiles(
    parquet_path: pathlib.Path,
    metadata_column_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load, align, and filter annotated single-cell profile tables.

    Args:
        parquet_path: Absolute path to a single parquet file or a directory of parquets.
        metadata_column_map: Optional source-to-canonical metadata renaming map.

    Returns:
        Profile DataFrame with required metadata and cleaned bounding-box columns.

    Raises:
        FileNotFoundError: If annotated parquet profile files are missing.
        ValueError: If required profile columns are missing.
    """

    def _normalize_token(value: str) -> str:
        return re.sub(r"[^a-z0-9]", "", value.lower())

    def _resolve_column_by_substring(
        columns: list[str],
        required_name: str,
    ) -> str:
        required_norm = _normalize_token(required_name)
        matches = [
            column
            for column in columns
            if required_norm in _normalize_token(column)
        ]
        if not matches:
            raise ValueError(
                f"Missing required profile column containing '{required_name}'. "
                f"Available columns: {sorted(columns)}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous profile columns for '{required_name}': {sorted(matches)}"
            )
        return matches[0]

    parquet_path = parquet_path.resolve(strict=True)
    if parquet_path.is_dir():
        parquet_files = sorted(parquet_path.glob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files found in directory: {parquet_path}")
        scdf = pd.concat((pd.read_parquet(path) for path in parquet_files), ignore_index=True)
    else:
        scdf = pd.read_parquet(parquet_path)

    if metadata_column_map:
        available_renames = {
            source: target for source, target in metadata_column_map.items() if source in scdf.columns
        }
        scdf = scdf.rename(columns=available_renames)
    common_columns = set(scdf.columns)

    required_metadata_cols = {
        "Metadata_Plate",
        "Metadata_Well",
        "Metadata_Site",
    }
    required_bbox_cols = {
        "Metadata_Nuclei_AreaShape_BoundingBoxMinimum_X",
        "Metadata_Nuclei_AreaShape_BoundingBoxMaximum_X",
        "Metadata_Nuclei_AreaShape_BoundingBoxMinimum_Y",
        "Metadata_Nuclei_AreaShape_BoundingBoxMaximum_Y",
    }
    required_cols = required_metadata_cols | required_bbox_cols

    missing_metadata_cols = sorted(required_metadata_cols - common_columns)
    if missing_metadata_cols:
        raise ValueError(f"Missing required profile columns: {missing_metadata_cols}")

    missing_bbox_cols = sorted(required_bbox_cols - common_columns)
    if missing_bbox_cols:
        resolved_renames: dict[str, str] = {}
        source_columns = list(scdf.columns)
        for required_col in missing_bbox_cols:
            basename = required_col.removeprefix("Metadata_")
            source_column = _resolve_column_by_substring(
                columns=source_columns,
                required_name=basename,
            )
            resolved_renames[source_column] = required_col

        if resolved_renames:
            scdf = scdf.rename(columns=resolved_renames)
            common_columns = set(scdf.columns)

        still_missing_bbox_cols = sorted(required_bbox_cols - common_columns)
        if still_missing_bbox_cols:
            raise ValueError(f"Missing required profile columns: {still_missing_bbox_cols}")

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


def _validate_manifest(
    manifest_path: pathlib.Path,
    input_resolution: float | None,
    target_resolution: float | None,
) -> tuple[bool, list[dict[str, str]]]:
    """Check whether an existing cache manifest can be reused safely.

    Args:
        manifest_path: Path to the cache CSV manifest.
        input_resolution: Requested source microscope resolution.
        target_resolution: Requested target microscope resolution.

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
        "input_resolution",
        "target_resolution",
    }
    if not required_fields.issubset(rows[0].keys()):
        return False, []

    center_x_pattern = re.compile(r"(?:^|\|)center_x=-?\d+\.\d{6}(?:\||$)")
    center_y_pattern = re.compile(r"(?:^|\|)center_y=-?\d+\.\d{6}(?:\||$)")
    requested_input_resolution = (
        "" if input_resolution is None else f"{input_resolution:.6f}"
    )
    requested_target_resolution = (
        "" if target_resolution is None else f"{target_resolution:.6f}"
    )

    for row in rows:
        sample_id = row["sample_id"]
        if not center_x_pattern.search(sample_id) or not center_y_pattern.search(sample_id):
            return False, []
        if row["input_resolution"] != requested_input_resolution:
            return False, []
        if row["target_resolution"] != requested_target_resolution:
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
    input_channel: str,
    target_channel: str,
    crop_size: int = 256,
    metadata_column_map: dict[str, str] | None = None,
    input_resolution: float | None = None,
    target_resolution: float | None = None,
) -> CropCacheResult:
    """Build or reuse a DAPI-to-Gold crop cache for configured channels.

    Args:
        image_dir: Directory containing source TIFF images.
        parquet_path: Path to single-cell profile parquet.
        cache_dir: Destination directory for cached crop TIFFs and manifest.
        input_channel: Source channel name used for DAPI input crops. The value
            is normalized to uppercase before channel lookup and manifest writes.
        target_channel: Target channel name used for Gold target crops. The value
            is normalized to uppercase before channel lookup and manifest writes.
        crop_size: Fixed square crop size in pixels for cached nucleus crops.
        metadata_column_map: Optional source-to-canonical metadata renaming map.
        input_resolution: Source microscope resolution in microns per pixel.
        target_resolution: Target microscope resolution in microns per pixel.

    Returns:
        Manifest path plus inferred image specs for training configuration.

    Raises:
        ValueError: If no valid crop pairs can be produced.
    """

    cache_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = cache_dir / "manifest.csv"
    input_channel = input_channel.upper()
    target_channel = target_channel.upper()

    if crop_size <= 0:
        raise ValueError(f"crop_size must be positive, got {crop_size}")
    if (input_resolution is None) != (target_resolution is None):
        raise ValueError(
            "input_resolution and target_resolution must either both be set or both be None"
        )

    is_valid, existing_rows = _validate_manifest(
        manifest_path=manifest_path,
        input_resolution=input_resolution,
        target_resolution=target_resolution,
    )
    if is_valid:
        # Fast path: manifest already points to valid, existing cached crops.
        return CropCacheResult(
            manifest_path=manifest_path,
            image_specs=_infer_image_specs(rows=existing_rows),
        )

    scdf = _build_filtered_profiles(
        parquet_path=parquet_path,
        metadata_column_map=metadata_column_map,
    )

    target_width = int(crop_size)
    target_height = int(crop_size)

    rows: list[dict[str, str]] = []

    image_dir = image_dir.resolve(strict=True)

    image_index = _build_image_index(image_dir=image_dir)

    for image_key, keyed_images in image_index.items():
        plate_name, well_name, site_name = image_key

        dapi_img_path = keyed_images.get(input_channel)
        gold_img_path = keyed_images.get(target_channel)

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
        if dapi_img.shape != gold_img.shape:
            raise ValueError(
                "Expected DAPI and Gold full images to share the same field-of-view shape, "
                f"got dapi_img.shape={dapi_img.shape}, gold_img.shape={gold_img.shape}"
            )

        geometry = None
        if input_resolution is not None and target_resolution is not None:
            geometry = _build_resampling_geometry(
                input_shape=tuple(int(dim) for dim in dapi_img.shape),
                template_shape=tuple(int(dim) for dim in gold_img.shape),
                input_resolution=input_resolution,
                target_resolution=target_resolution,
            )
            dapi_img = _register_and_resample_full_image(dapi_img, geometry=geometry)
            gold_img = _register_and_resample_full_image(gold_img, geometry=geometry)

        image_df = image_df.copy()
        bbox_cols = [
            "Metadata_Nuclei_AreaShape_BoundingBoxMinimum_X",
            "Metadata_Nuclei_AreaShape_BoundingBoxMaximum_X",
            "Metadata_Nuclei_AreaShape_BoundingBoxMinimum_Y",
            "Metadata_Nuclei_AreaShape_BoundingBoxMaximum_Y",
        ]
        for col in bbox_cols:
            image_df[col] = image_df[col].astype(int)

        if geometry is not None:
            transformed_bboxes = image_df.apply(
                lambda nucleus: _transform_bbox_coordinates(
                    x0=int(nucleus["Metadata_Nuclei_AreaShape_BoundingBoxMinimum_X"]),
                    y0=int(nucleus["Metadata_Nuclei_AreaShape_BoundingBoxMinimum_Y"]),
                    x1=int(nucleus["Metadata_Nuclei_AreaShape_BoundingBoxMaximum_X"]),
                    y1=int(nucleus["Metadata_Nuclei_AreaShape_BoundingBoxMaximum_Y"]),
                    geometry=geometry,
                ),
                axis=1,
                result_type="expand",
            )
            transformed_bboxes.columns = bbox_cols
            image_df[bbox_cols] = transformed_bboxes.astype(int)

        for _, nucleus in image_df.iterrows():
            x0 = int(nucleus["Metadata_Nuclei_AreaShape_BoundingBoxMinimum_X"])
            y0 = int(nucleus["Metadata_Nuclei_AreaShape_BoundingBoxMinimum_Y"])
            x1 = int(nucleus["Metadata_Nuclei_AreaShape_BoundingBoxMaximum_X"])
            y1 = int(nucleus["Metadata_Nuclei_AreaShape_BoundingBoxMaximum_Y"])

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
            cropped_gold = gold_img[crop_y0:crop_y1, crop_x0:crop_x1].copy()

            if cropped_dapi.size == 0 or cropped_gold.size == 0:
                continue

            # Skip empty-signal crops.
            if cropped_dapi.max() == 0 or cropped_gold.max() == 0:
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
                    "input_channel": input_channel,
                    "target_channel": target_channel,
                    "input_path": str(input_path.resolve()),
                    "target_path": str(target_path.resolve()),
                    "input_resolution": "" if input_resolution is None else f"{input_resolution:.6f}",
                    "target_resolution": "" if target_resolution is None else f"{target_resolution:.6f}",
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
                "input_resolution",
                "target_resolution",
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
