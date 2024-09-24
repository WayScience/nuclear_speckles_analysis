#!/usr/bin/env python
# coding: utf-8

# # Evaluate Well Stain Relationship Strength
# This analysis compares Maximal Information Coefficient (MIC) scores between the same DAPI and nuclear speckle stain (A647 or GOLD) aggregated well features per well.
# Distributions of these MIC scores are visualized, between zero and one, where one indicates a perfect relationship and zero indicates no relationship.
# From this analysis we can underastand the relationships between the DAPI and nuclear speckle stains per cell population (Treatment, Well Position, Plate, etc...)
# With these insights we may further improve or stain feature regression and stain translation models.
# We save additional experimental details in a manifest for further evaluation.

# In[1]:


import pathlib
import re
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ## Find the root of the git repo on the host system

# In[2]:


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


# # Custom Imports

# In[3]:


sys.path.append(f"{root_dir}/0.data_analysis_and_processing/utils")

from MIC import MIC
from PairwiseCompare import PairwiseCompare
from ShuffledMIC import ShuffledMIC


# # Inputs

# In[4]:


# Paths to original nuclear speckle data
data_dir = root_dir / "nuclear_speckles_data"
nuclear_mask_dir = (data_dir / "Nuclear_masks").resolve(strict=True)
sc_profiles_path = list((data_dir / "Preprocessed_data/single_cell_profiles").resolve(strict=True).glob("*feature_selected*.parquet"))

# Load single-cell profile data
scdfs = [pd.read_parquet(sc_path) for sc_path in sc_profiles_path if sc_path.is_file()]


# # Outputs

# In[5]:


distribution_figures_path = pathlib.Path("well_sirna_mic_distribution_figures")
distribution_figures_path.mkdir(parents=True, exist_ok=True)

mic_comparisons_path = pathlib.Path("mic_comparisons_data")
mic_comparisons_path.mkdir(parents=True, exist_ok=True)


# # Processing

# ## Combine Common Data
# Column names are used to combine common single-cell data.

# In[6]:


common_columns = scdfs[0].columns
for scdf in scdfs[1:]:
    common_columns = common_columns.intersection(scdf.columns)

scdfs = pd.concat(scdfs, axis=0)[common_columns]


# In[7]:


scdfs.dropna(inplace=True)

print(scdfs)


# ## Seperate Gold, A647, and Dapi Features

# In[8]:


gold_scdfs = scdfs.loc[:, scdfs.columns.str.contains("GOLD|Metadata", regex=True)]
a647_scdfs = scdfs.loc[:, scdfs.columns.str.contains("A647|Metadata", regex=True)]
dapi_scdfs = scdfs.loc[:, scdfs.columns.str.contains("DAPI|Metadata", regex=True)]

gold_scdfs.columns = gold_scdfs.columns.str.replace('_GOLD', '', regex=False)
a647_scdfs.columns = gold_scdfs.columns.str.replace('_A647', '', regex=False)
dapi_scdfs.columns = dapi_scdfs.columns.str.replace('_DAPI', '', regex=False)


# ## Combine Seperated Stain Features

# In[9]:


gold_scdfs = gold_scdfs.assign(Metadata_Stain="GOLD")
dapi_scdfs = dapi_scdfs.assign(Metadata_Stain="DAPI")
a647_scdfs = a647_scdfs.assign(Metadata_Stain="A647")

common_cols = gold_scdfs.columns.intersection(dapi_scdfs.columns).intersection(a647_scdfs.columns)
scdfs = pd.concat([gold_scdfs[common_cols], dapi_scdfs[common_cols], a647_scdfs[common_cols]], axis=0)


# ## Specify Feature Metadata Columns

# In[10]:


feat_cols = scdfs.columns[~scdfs.columns.str.contains("Metadata")]


# ## Mean Aggregation to the Well Level

# In[11]:


# Define Placeholder Column
scdfs["Metadata_Cell_Count"] = scdfs["Metadata_Condition"].values

# Metadata to retain
agg_funcs = {
    "Metadata_Condition": "first",
    "Metadata_Cell_Count": "size"
}

agg_funcs |= {feat_col: "mean" for feat_col in feat_cols}
staindf = scdfs.groupby(["Metadata_Plate", "Metadata_Well", "Metadata_Stain"]).agg(agg_funcs).reset_index()


# In[12]:


print(staindf)


# # MIC Comparisons
# Compares MIC scores between DAPI and nuclear speckle stains of the same well.

# In[13]:


micdfs = []
speckle_stains = {"GOLD", "A647"}

for stain in speckle_stains:
    for feature_order in ("mic", "shuffled_mic"):

        if feature_order == "mic":
            mic_comparator = MIC()

        else:
            # Shuffles the samples/features depending on your perspective
            mic_comparator = ShuffledMIC()

        speckle_staindf = staindf.loc[staindf["Metadata_Stain"] != stain]

        comparer = PairwiseCompare(
            _df=speckle_staindf,
            _comparator=mic_comparator,
            _antehoc_group_cols=["Metadata_Plate", "Metadata_Well", "Metadata_Condition"],
            _posthoc_group_cols=["Metadata_Stain"],
            _feat_cols=feat_cols,
        )

        comparer.intra_comparisons()

        micdf = pd.DataFrame(mic_comparator.comparisons)
        micdf = micdf.assign(Metadata_Comparison_Type=feature_order)
        micdfs.append(micdf)

micdfs = pd.concat(micdfs, axis=0)


# In[14]:


micdfs = micdfs.loc[:, ~micdfs.columns.str.contains("__antehoc_group1")]
micdfs.columns = micdfs.columns.str.replace("__antehoc_group0", "", regex=False)


# In[15]:


agg_merge_cols = ["Metadata_Plate", "Metadata_Well"]

micdfs = pd.merge(
    left=micdfs,
    right=staindf[agg_merge_cols + ["Metadata_Cell_Count"]],
    how="inner",
    on=agg_merge_cols
)


# In[16]:


print(micdfs)


# # Save Results

# In[17]:


for stain in speckle_stains:

    pattern = f"DAPI|{re.escape(stain)}"
    stain_micdfs = micdfs.loc[micdfs["Metadata_Stain__posthoc_group0"].str.contains(pattern, regex=True) & micdfs["Metadata_Stain__posthoc_group1"].str.contains(pattern, regex=True)]
    print(stain_micdfs)

    sns.histplot(data=stain_micdfs, x="mic_e", hue="Metadata_Comparison_Type",
    palette={"shuffled_mic": 'blue', "mic": 'red'}, bins=10, kde=False)

    plt.gcf().set_size_inches(18, 10)

    plt.xlabel("MIC Score", fontsize=13)
    plt.ylabel("Number of Wells", fontsize=13)

    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    plt.title(f"Distributions of Maximal Information Coeficient (MIC) Scores\nBetween DAPI and {stain} Features per Well", fontsize=16)

    plt.xlim(0, 1)
    plt.ylim(0.0, 80.0)

    plt.savefig(distribution_figures_path / f"mic_distributions_dapi_{stain.lower()}_wells.png")
    plt.close()


# In[18]:


micdfs.to_parquet(mic_comparisons_path / "well_sirna_mic_comparisons.parquet")

