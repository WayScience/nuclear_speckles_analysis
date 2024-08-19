r"""°°°
# Filter Single-Cell Profiles
Single-cell profiles are filtered by nuclei bounding box dimensions across all plates.
Filtered is accomplished with robust z-score thresholding of nuclei bounding box dimensions.
°°°"""
#|%%--%%| <diZkzamaCr|JlHVCFc6jg>

import pathlib
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

#|%%--%%| <JlHVCFc6jg|09DzTtojev>
r"""°°°
## Find the root of the git repo on the host system
°°°"""
#|%%--%%| <09DzTtojev|ZSBTA9IqxD>

# Get the current working directory
cwd = pathlib.Path.cwd()

if (cwd / ".git").is_dir():
    root_dir = cwd

else:
    root_dir = None
    for parent in cwd.parents:
        if (parent / ".git").is_dir():
            root_dir = parent
            break

# Check if a Git root directory was found
if root_dir is None:
    raise FileNotFoundError("No Git root directory found.")

# Remove categorical warning in seaborn
warnings.filterwarnings("ignore", category=FutureWarning)

#|%%--%%| <ZSBTA9IqxD|fIxGUGaKkR>
r"""°°°
# Inputs
°°°"""
#|%%--%%| <fIxGUGaKkR|dlU1p2c9cE>

data_dir = root_dir / "nuclear_speckles_data"
nuclear_mask_dir = (data_dir / "Nuclear_masks").resolve(strict=True)
sc_profiles_path = list((data_dir / "Preprocessed_data/single_cell_profiles").resolve(strict=True).glob("*annotated*.parquet"))

# Load single-cell profile data
scdfs = [pd.read_parquet(sc_path) for sc_path in sc_profiles_path if sc_path.is_file()]

#|%%--%%| <dlU1p2c9cE|2nCHpCWHPX>
r"""°°°
# Outputs
°°°"""
#|%%--%%| <2nCHpCWHPX|1pXhOXPv3B>

filtered_sc_path = pathlib.Path("filtered_single_cells")
filtered_sc_figure_path = filtered_sc_path / "filtered_sc_figures"

filtered_sc_figure_path.mkdir(parents=True, exist_ok=True)

#|%%--%%| <1pXhOXPv3B|gsS8lw52My>

def filter_bounding_box_size(_scdf, _bounding_box_col):
    """
    Filter nuclei if the robust z score of a bounding box dimension is less than three.

    Parameters
    ----------
    _scdf: Pandas Dataframe
        The unfiltered single-cell data containing bounding box dimensions.

    _bounding_box_col: String
        Column specifying the bounding box dimension.

    Returns
    -------
    Filtered single-cell pandas dataframe.
    """

    median = scdfs[_bounding_box_col].median()

    # Calculate the absolute deviations from the median
    _scdf[f"{_bounding_box_col}_absolute_deviation"] = np.abs(_scdf[_bounding_box_col] - median)

    # Calculate the MAD
    mad = _scdf[f"{_bounding_box_col}_absolute_deviation"].median()

    # Calculate the robust Z-score
    _scdf[f"{_bounding_box_col}_robust_z_score"] = (_scdf[_bounding_box_col] - median) / mad

    # Returns the single cells with a robust z score less than three
    return _scdf.loc[_scdf[f"{_bounding_box_col}_robust_z_score"] < 3]

#|%%--%%| <gsS8lw52My|cox2pfXzaY>
r"""°°°
## Retain only Common Metadata and Bounding Box Columns
°°°"""
#|%%--%%| <cox2pfXzaY|By6mVuPf30>

bounding_box_cols = ["Nuclei_AreaShape_BoundingBoxMaximum_X", "Nuclei_AreaShape_BoundingBoxMinimum_X", "Nuclei_AreaShape_BoundingBoxMaximum_Y", "Nuclei_AreaShape_BoundingBoxMinimum_Y"]

common_cols = set(scdfs[0].columns[scdfs[0].columns.str.contains('Metadata')])

for scdf in scdfs[1:]:
    common_cols = set(scdf.columns[scdf.columns.str.contains('Metadata')]) & set(scdf.columns)

scdfs = pd.concat(scdfs, axis=0)[list(common_cols) + bounding_box_cols]

#|%%--%%| <By6mVuPf30|C2BsxepdYo>
r"""°°°
## Calculate Bounding Box Dimensions
°°°"""
#|%%--%%| <C2BsxepdYo|FUd55NGTk5>

scdfs["Nuclei_AreaShape_BoundingBoxDelta_X"] = scdfs["Nuclei_AreaShape_BoundingBoxMaximum_X"] - scdfs["Nuclei_AreaShape_BoundingBoxMinimum_X"]
scdfs["Nuclei_AreaShape_BoundingBoxDelta_Y"] = scdfs["Nuclei_AreaShape_BoundingBoxMaximum_Y"] - scdfs["Nuclei_AreaShape_BoundingBoxMinimum_Y"]

#|%%--%%| <FUd55NGTk5|KlhSF15FjX>

scdfs.head()

#|%%--%%| <KlhSF15FjX|CupudYb1Qk>
r"""°°°
# Bounding Box Dimension Distributions
Visualizes bounding box distributions before and after filtering and prints out cell counts information.
°°°"""
#|%%--%%| <CupudYb1Qk|yPLSjbmKgz>

pre_filter_sc_count = scdfs.shape[0]

g = sns.jointplot(data=scdfs, x="Nuclei_AreaShape_BoundingBoxDelta_X", y="Nuclei_AreaShape_BoundingBoxDelta_Y")
g.fig.set_size_inches(18, 10)
g.set_axis_labels("Bounding Box Height (Pixels)", "Bounding Box Length (Pixels)")
g.fig.suptitle("Distribution of Bounding Box Sizes before Filtering")
plt.savefig(filtered_sc_figure_path / "nuclei_dimension_distributions_before_filtering.png")
plt.show()

#|%%--%%| <yPLSjbmKgz|m5nV707xMI>
r"""°°°
## Filter Nuclei by Height and Length
°°°"""
#|%%--%%| <m5nV707xMI|VkK7mjxzYa>

scdfs = filter_bounding_box_size(_scdf=scdfs, _bounding_box_col="Nuclei_AreaShape_BoundingBoxDelta_X")
scdfs = filter_bounding_box_size(_scdf=scdfs, _bounding_box_col="Nuclei_AreaShape_BoundingBoxDelta_Y")

scdfs.filter(like="Metadata").to_parquet(filtered_sc_path / "filtered_single_cell_profiles.parquet")

#|%%--%%| <VkK7mjxzYa|EeGXVeEf3C>

scdfs.head()

#|%%--%%| <EeGXVeEf3C|v7yse5QbtU>

print(f"\nNumber of single cells removed after filtering by bounding box size:\n{pre_filter_sc_count - scdfs.shape[0]}")

#|%%--%%| <v7yse5QbtU|gIoEho5A8z>

g = sns.jointplot(data=scdfs, x="Nuclei_AreaShape_BoundingBoxDelta_X", y="Nuclei_AreaShape_BoundingBoxDelta_Y")
g.fig.set_size_inches(18, 10)
g.set_axis_labels("Bounding Box Height (Pixels)", "Bounding Box Length (Pixels)")
g.fig.suptitle("Distribution of Bounding Box Sizes after Filtering")
plt.savefig(filtered_sc_figure_path / "nuclei_dimension_distributions_after_filtering.png")
plt.show()
