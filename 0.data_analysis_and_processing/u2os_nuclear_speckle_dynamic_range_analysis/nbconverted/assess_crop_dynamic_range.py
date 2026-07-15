#!/usr/bin/env python
# coding: utf-8

"""Assess U2OS cached crop dynamic range for DAPI and GOLD separately.

This analysis targets the U2OS nuclear speckle dataset crop cache used by the
training workflow. Metrics are computed independently for the DAPI input crops
and GOLD target crops so apparent detector ceilings and usable dynamic range can
be interpreted per channel.
"""

from __future__ import annotations

import pathlib
from typing import Iterable

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile
from matplotlib import ticker
from scipy.stats import gaussian_kde

matplotlib.use("Agg")


def find_repo_root() -> pathlib.Path:
    """Find the git repository root from the current working directory.

    Returns:
        Absolute path to the repository root.

    Raises:
        FileNotFoundError: If no parent directory contains ``.git``.
    """

    cwd = pathlib.Path.cwd().resolve()
    if (cwd / ".git").is_dir():
        return cwd

    for parent in cwd.parents:
        if (parent / ".git").is_dir():
            return parent

    raise FileNotFoundError("No Git root directory found.")


def infer_bit_depth(dtype_name: str) -> int:
    """Infer integer bit depth from a numpy dtype name.

    Args:
        dtype_name: String dtype representation such as ``"uint16"``.

    Returns:
        Bit depth implied by the dtype.
    """

    return int(np.dtype(dtype_name).itemsize * 8)


def build_channel_title(channel_name: str, dtype_name: str, metric_label: str) -> str:
    """Build a plot title that includes dtype-derived bit depth.

    Args:
        channel_name: Human-readable channel label.
        dtype_name: String dtype representation for that channel.
        metric_label: Plot-specific metric description.

    Returns:
        Title string used in saved figures.
    """

    bit_depth = infer_bit_depth(dtype_name)
    return f"{channel_name} {metric_label} ({bit_depth}-bit {dtype_name})"


def compute_image_metrics(
    image_path: pathlib.Path,
    channel_name: str,
    channel_column_name: str,
    ceiling_values: Iterable[int],
    metadata: dict[str, object],
) -> dict[str, object]:
    """Compute per-image intensity summary metrics for one cached crop.

    Args:
        image_path: Absolute path to one cached TIFF crop.
        channel_name: Human-readable channel label.
        channel_column_name: Channel token from the cache manifest.
        ceiling_values: Intensity values to test as potential detector ceilings.
        metadata: Sample-level manifest metadata copied into the output row.

    Returns:
        Dictionary containing one image's summary statistics.
    """

    image = tifffile.imread(image_path)
    image_float = image.astype(np.float32, copy=False)
    percentiles = np.percentile(image_float, [1.0, 99.0])

    metrics: dict[str, object] = {
        "channel_name": channel_name,
        "channel_column_name": channel_column_name,
        "image_path": str(image_path),
        "dtype": str(image.dtype),
        "bit_depth": infer_bit_depth(str(image.dtype)),
        "dtype_max": int(np.iinfo(image.dtype).max),
        "height": int(image.shape[0]),
        "width": int(image.shape[1]),
        "min": float(image_float.min()),
        "max": float(image_float.max()),
        "mean": float(image_float.mean()),
        "std": float(image_float.std()),
        "p01": float(percentiles[0]),
        "p99": float(percentiles[1]),
        "dynamic_range": float(image_float.max() - image_float.min()),
        "effective_dynamic_range": float(percentiles[1] - percentiles[0]),
        "zero_fraction": float((image == 0).sum() / image.size),
    }
    metrics.update(metadata)

    for ceiling_value in ceiling_values:
        ceiling_hits = int((image == ceiling_value).sum())
        metrics[f"pixel_count_at_{ceiling_value}"] = ceiling_hits
        metrics[f"fraction_at_{ceiling_value}"] = float(ceiling_hits / image.size)
        metrics[f"has_any_{ceiling_value}"] = bool(ceiling_hits > 0)

    return metrics


def analyze_channel(
    manifest_df: pd.DataFrame,
    path_column: str,
    channel_label: str,
    channel_column_name: str,
    ceiling_values: Iterable[int],
) -> pd.DataFrame:
    """Compute per-image metrics for one cache channel.

    Args:
        manifest_df: Cache manifest with one row per paired crop.
        path_column: Manifest column containing TIFF paths for this channel.
        channel_label: Human-readable channel label.
        channel_column_name: Channel token from the manifest.
        ceiling_values: Intensity values to test as potential detector ceilings.

    Returns:
        DataFrame with one output row per image.
    """

    rows: list[dict[str, object]] = []
    total_images = len(manifest_df)

    for row_idx, row in enumerate(manifest_df.itertuples(index=False), start=1):
        if row_idx == 1 or row_idx % 1000 == 0 or row_idx == total_images:
            print(f"[{channel_label}] processed {row_idx}/{total_images} images")

        image_path = pathlib.Path(getattr(row, path_column)).resolve(strict=True)
        metadata = {
            "sample_id": row.sample_id,
            "group_id": row.group_id,
            "plate": row.plate,
            "well": row.well,
            "site": row.site,
            "input_channel": row.input_channel,
            "target_channel": row.target_channel,
        }
        rows.append(
            compute_image_metrics(
                image_path=image_path,
                channel_name=channel_label,
                channel_column_name=channel_column_name,
                ceiling_values=ceiling_values,
                metadata=metadata,
            )
        )

    return pd.DataFrame(rows)


def build_channel_summary(
    per_image_df: pd.DataFrame,
    ceiling_values: Iterable[int],
) -> pd.DataFrame:
    """Build one summary row for a channel.

    Args:
        per_image_df: Per-image metric table for one channel.
        ceiling_values: Intensity values to summarize as ceiling hits.

    Returns:
        Single-row channel summary DataFrame.
    """

    summary: dict[str, object] = {
        "channel_name": per_image_df["channel_name"].iloc[0],
        "channel_column_name": per_image_df["channel_column_name"].iloc[0],
        "n_images": int(len(per_image_df)),
        "dtype": str(per_image_df["dtype"].mode().iloc[0]),
        "bit_depth": int(per_image_df["bit_depth"].mode().iloc[0]),
        "dtype_max": int(per_image_df["dtype_max"].mode().iloc[0]),
        "height": int(per_image_df["height"].mode().iloc[0]),
        "width": int(per_image_df["width"].mode().iloc[0]),
        "global_min": float(per_image_df["min"].min()),
        "global_max": float(per_image_df["max"].max()),
        "median_image_min": float(per_image_df["min"].median()),
        "median_image_max": float(per_image_df["max"].median()),
        "p99_image_max": float(per_image_df["max"].quantile(0.99)),
        "median_image_mean": float(per_image_df["mean"].median()),
        "median_image_std": float(per_image_df["std"].median()),
        "median_dynamic_range": float(per_image_df["dynamic_range"].median()),
        "median_effective_dynamic_range": float(
            per_image_df["effective_dynamic_range"].median()
        ),
        "median_zero_fraction": float(per_image_df["zero_fraction"].median()),
    }

    total_pixels = int((per_image_df["height"] * per_image_df["width"]).sum())
    summary["total_pixels"] = total_pixels

    for ceiling_value in ceiling_values:
        summary[f"images_with_any_{ceiling_value}"] = int(
            per_image_df[f"has_any_{ceiling_value}"].sum()
        )
        summary[f"pixels_at_{ceiling_value}"] = int(
            per_image_df[f"pixel_count_at_{ceiling_value}"].sum()
        )
        summary[f"fraction_of_all_pixels_at_{ceiling_value}"] = float(
            per_image_df[f"pixel_count_at_{ceiling_value}"].sum() / total_pixels
        )

    return pd.DataFrame([summary])


def summarize_by_group(per_image_df: pd.DataFrame) -> pd.DataFrame:
    """Summarize dynamic range statistics across plate, well, and site levels.

    Args:
        per_image_df: Per-image metric table for one channel.

    Returns:
        Group-level summary DataFrame across plate, well, and site resolutions.
    """

    group_specs = {
        "plate": ["plate"],
        "well": ["plate", "well"],
        "site": ["plate", "well", "site"],
    }
    summary_frames: list[pd.DataFrame] = []

    for group_level, group_columns in group_specs.items():
        aggregated = (
            per_image_df.groupby(group_columns, dropna=False)
            .agg(
                n_images=("sample_id", "size"),
                median_min=("min", "median"),
                median_max=("max", "median"),
                mean_of_max=("max", "mean"),
                median_mean=("mean", "median"),
                median_std=("std", "median"),
                median_dynamic_range=("dynamic_range", "median"),
                median_effective_dynamic_range=(
                    "effective_dynamic_range",
                    "median",
                ),
                mean_zero_fraction=("zero_fraction", "mean"),
            )
            .reset_index()
        )
        aggregated.insert(0, "group_level", group_level)
        aggregated.insert(1, "channel_name", per_image_df["channel_name"].iloc[0])
        summary_frames.append(aggregated)

    return pd.concat(summary_frames, ignore_index=True)


def save_histogram(series: pd.Series, title: str, xlabel: str, output_path: pathlib.Path) -> None:
    """Save a histogram for one per-image metric.

    Args:
        series: Numeric values to histogram.
        title: Figure title.
        xlabel: X-axis label.
        output_path: Destination PNG path.
    """

    plt.figure(figsize=(9, 6))
    plt.hist(series.to_numpy(), bins=50, color="#4477AA", edgecolor="white")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Image Count")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_scatter(
    x: pd.Series,
    y: pd.Series,
    title: str,
    xlabel: str,
    ylabel: str,
    output_path: pathlib.Path,
) -> None:
    """Save a scatter plot for one channel.

    Args:
        x: X-axis values.
        y: Y-axis values.
        title: Figure title.
        xlabel: X-axis label.
        ylabel: Y-axis label.
        output_path: Destination PNG path.
    """

    x_values = x.to_numpy(dtype=float)
    y_values = y.to_numpy(dtype=float)

    fig = plt.figure(figsize=(9, 9), constrained_layout=True)
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=(4, 1.2),
        height_ratios=(1.2, 4),
        wspace=0.05,
        hspace=0.05,
    )
    ax_top = fig.add_subplot(grid[0, 0])
    ax_scatter = fig.add_subplot(grid[1, 0])
    ax_right = fig.add_subplot(grid[1, 1])

    ax_scatter.scatter(x_values, y_values, s=10, alpha=0.35, color="#228833")
    ax_scatter.set_title(title)
    ax_scatter.set_xlabel(xlabel)
    ax_scatter.set_ylabel(ylabel)

    if len(x_values) > 1 and np.unique(x_values).size > 1:
        x_grid = np.linspace(x_values.min(), x_values.max(), 256)
        x_kde = gaussian_kde(x_values)
        ax_top.plot(x_grid, x_kde(x_grid), color="#4477AA", linewidth=2)
        ax_top.fill_between(x_grid, x_kde(x_grid), color="#4477AA", alpha=0.2)
    if len(y_values) > 1 and np.unique(y_values).size > 1:
        y_grid = np.linspace(y_values.min(), y_values.max(), 256)
        y_kde = gaussian_kde(y_values)
        ax_right.plot(y_kde(y_grid), y_grid, color="#CC6677", linewidth=2)
        ax_right.fill_betweenx(y_grid, 0, y_kde(y_grid), color="#CC6677", alpha=0.2)

    ax_top.set_xlim(ax_scatter.get_xlim())
    ax_right.set_ylim(ax_scatter.get_ylim())
    ax_top.tick_params(axis="x", labelbottom=False)
    ax_top.set_ylabel("KDE")
    ax_right.tick_params(axis="y", labelleft=False)
    ax_right.set_xlabel("KDE")

    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def configure_bar_ticks(ax: plt.Axes, labels: list[str], values: list[int]) -> None:
    """Increase axis tick density for bar charts with more than two categories.

    Args:
        ax: Matplotlib axis containing the bar chart.
        labels: Bar category labels in plotted order.
        values: Bar heights in plotted order.
    """

    if len(labels) > 2:
        ax.set_xticks(range(len(labels)), labels)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=min(8, max(values) + 1)))
    else:
        ax.set_xticks(range(len(labels)), labels)
        ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))


def save_ceiling_barplot(
    per_image_df: pd.DataFrame,
    ceiling_values: Iterable[int],
    output_path: pathlib.Path,
) -> None:
    """Save a bar plot summarizing image counts with ceiling hits.

    Args:
        per_image_df: Per-image metric table for one channel.
        ceiling_values: Intensity values treated as potential ceilings.
        output_path: Destination PNG path.
    """

    labels: list[str] = []
    values: list[int] = []
    for ceiling_value in ceiling_values:
        labels.append(str(ceiling_value))
        values.append(int(per_image_df[f"has_any_{ceiling_value}"].sum()))

    plt.figure(figsize=(7, 5))
    plt.bar(range(len(labels)), values, color=["#CC6677", "#AA4499"][: len(values)])
    ax = plt.gca()
    configure_bar_ticks(ax=ax, labels=labels, values=values)
    plt.title(
        build_channel_title(
            channel_name=per_image_df["channel_name"].iloc[0],
            dtype_name=per_image_df["dtype"].iloc[0],
            metric_label="ceiling-hit images",
        )
    )
    plt.xlabel("Ceiling Value")
    plt.ylabel("Images With Any Ceiling Hit")
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_comparison_histograms(
    dapi_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    metric_column: str,
    xlabel: str,
    output_path: pathlib.Path,
) -> None:
    """Save side-by-side channel comparison histograms without pooling stats.

    Args:
        dapi_df: Per-image metric table for DAPI crops.
        gold_df: Per-image metric table for GOLD crops.
        metric_column: Column to visualize in both panels.
        xlabel: Shared x-axis label.
        output_path: Destination PNG path.
    """

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    channel_specs = [
        (dapi_df, "DAPI", "#4477AA"),
        (gold_df, "GOLD", "#CCBB44"),
    ]

    for ax, (channel_df, title, color) in zip(axes, channel_specs, strict=True):
        ax.hist(channel_df[metric_column].to_numpy(), bins=50, color=color, edgecolor="white")
        ax.set_title(
            build_channel_title(
                channel_name=title,
                dtype_name=channel_df["dtype"].iloc[0],
                metric_label=metric_column.replace("_", " "),
            )
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Image Count")

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def make_figures(
    per_image_df: pd.DataFrame,
    figures_dir: pathlib.Path,
    ceiling_values: Iterable[int],
) -> None:
    """Generate one set of per-channel diagnostic figures.

    Args:
        per_image_df: Per-image metric table for one channel.
        figures_dir: Output directory for PNG figures.
        ceiling_values: Intensity values treated as potential ceilings.
    """

    channel_prefix = per_image_df["channel_name"].iloc[0].lower()
    channel_name = per_image_df["channel_name"].iloc[0]
    dtype_name = per_image_df["dtype"].iloc[0]

    save_histogram(
        series=per_image_df["max"],
        title=build_channel_title(channel_name, dtype_name, "per-image maxima"),
        xlabel="Per-image max intensity",
        output_path=figures_dir / f"{channel_prefix}_image_max_histogram.png",
    )
    save_histogram(
        series=per_image_df["effective_dynamic_range"],
        title=build_channel_title(channel_name, dtype_name, "effective dynamic range"),
        xlabel="p99 - p01",
        output_path=figures_dir / f"{channel_prefix}_effective_dynamic_range_histogram.png",
    )
    save_histogram(
        series=per_image_df["mean"],
        title=build_channel_title(channel_name, dtype_name, "per-image means"),
        xlabel="Per-image mean intensity",
        output_path=figures_dir / f"{channel_prefix}_image_mean_histogram.png",
    )
    save_scatter(
        x=per_image_df["mean"],
        y=per_image_df["max"],
        title=build_channel_title(channel_name, dtype_name, "mean vs max"),
        xlabel="Per-image mean intensity",
        ylabel="Per-image max intensity",
        output_path=figures_dir / f"{channel_prefix}_mean_vs_max_scatter.png",
    )
    save_ceiling_barplot(
        per_image_df=per_image_df,
        ceiling_values=ceiling_values,
        output_path=figures_dir / f"{channel_prefix}_ceiling_check_barplot.png",
    )


def main() -> None:
    """Run the U2OS cache dynamic-range analysis and save tables and figures.

    The analysis reads the U2OS DAPI-to-GOLD crop cache, computes per-image and
    grouped summaries separately for both channels, and writes cache-specific
    outputs under this analysis directory.
    """

    repo_root = find_repo_root()
    # Expected format:
    # /mnt/big_drive/nuclear_speckle_data/<dataset>/<cache_parent>/<cache_name>
    # Update this path if your local cache lives elsewhere.
    cache_dir = pathlib.Path(
        "/mnt/big_drive/nuclear_speckle_data/u20s_dataset_jan_15_2026/model_cache/dapi_to_gold_crop_cache"
    ).resolve(strict=True)
    manifest_path = cache_dir / "manifest.csv"
    analysis_root = (
        repo_root / "0.data_analysis_and_processing/u2os_nuclear_speckle_dynamic_range_analysis"
    )
    run_name = cache_dir.name
    figures_dir = analysis_root / "figures" / run_name
    summary_dir = analysis_root / "summary_data" / run_name
    figures_dir.mkdir(parents=True, exist_ok=True)
    summary_dir.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(manifest_path)
    ceiling_values = (16383, 65535)

    dapi_df = analyze_channel(
        manifest_df=manifest_df,
        path_column="input_path",
        channel_label="DAPI",
        channel_column_name=str(manifest_df["input_channel"].iloc[0]),
        ceiling_values=ceiling_values,
    )
    gold_df = analyze_channel(
        manifest_df=manifest_df,
        path_column="target_path",
        channel_label="GOLD",
        channel_column_name=str(manifest_df["target_channel"].iloc[0]),
        ceiling_values=ceiling_values,
    )

    dapi_df.to_parquet(summary_dir / "dapi_per_image_dynamic_range.parquet", index=False)
    gold_df.to_parquet(summary_dir / "gold_per_image_dynamic_range.parquet", index=False)

    channel_summary_df = pd.concat(
        [
            build_channel_summary(dapi_df, ceiling_values=ceiling_values),
            build_channel_summary(gold_df, ceiling_values=ceiling_values),
        ],
        ignore_index=True,
    )
    channel_summary_df.to_csv(summary_dir / "channel_summary.csv", index=False)

    summarize_by_group(dapi_df).to_parquet(
        summary_dir / "dapi_group_summaries.parquet", index=False
    )
    summarize_by_group(gold_df).to_parquet(
        summary_dir / "gold_group_summaries.parquet", index=False
    )

    make_figures(dapi_df, figures_dir=figures_dir, ceiling_values=ceiling_values)
    make_figures(gold_df, figures_dir=figures_dir, ceiling_values=ceiling_values)
    save_comparison_histograms(
        dapi_df=dapi_df,
        gold_df=gold_df,
        metric_column="max",
        xlabel="Per-image max intensity",
        output_path=figures_dir / "channel_image_max_comparison.png",
    )
    save_comparison_histograms(
        dapi_df=dapi_df,
        gold_df=gold_df,
        metric_column="effective_dynamic_range",
        xlabel="p99 - p01",
        output_path=figures_dir / "channel_effective_dynamic_range_comparison.png",
    )

    print(f"Saved summaries to {summary_dir}")
    print(f"Saved figures to {figures_dir}")
    print(channel_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
