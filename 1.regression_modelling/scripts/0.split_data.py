#!/usr/bin/env python
# coding: utf-8

# ## Split data into training, testing, and holdout sets
#
# In this notebook, we use the normalized profiles to train regression models to predict each individual feature from each nuclear speckle stain (A647 and GOLD).
# We first have to remove any columns with NaN values and remove single-cells that are not included in the [filtered single cells file](../0.data_analysis_and_processing/filtered_single_cells/filtered_single_cell_profiles.parquet).
# The filtering was performed in the previous module where any single-cell crop had a height and width outside of a designated range was removed.
# We remove those single-cells in our method to ensure that both modelling methods use the same data.

# ## Import libraries

# In[1]:


import pathlib

import pandas as pd
from pycytominer import feature_select
from sklearn.model_selection import train_test_split

# ## Load in normalized profiles and concat into one dataframe

# In[2]:


# path to big drive where data is located (large)
drive_path = pathlib.Path("../../../../media/18tbdrive/")

# set path to folder in image profiling repo with the normalized profiles
norm_path = pathlib.Path(
    drive_path / "Github_Repositories/nuclear_speckle_image_profiling/4.preprocess_features/data/single_cell_profiles"
).resolve(strict=True)

# load all normalized parquet files
files = norm_path.glob("*_sc_normalized.parquet")
dfs = [pd.read_parquet(file) for file in files]

# concatenate them into one data frame
combined_df = pd.concat(dfs, ignore_index=True)

# perform feature selection to drop any columns that have NaN (avoid downstream issues)
combined_df = feature_select(
    combined_df,
    operation="drop_na_columns",
    na_cutoff=0
)

# print df
print(combined_df.shape)
combined_df.head()


# ## Load in filtered single-cell data from first module

# In[3]:


# load in filtered tuple dataframe
filtering_df = pd.read_parquet(
    pathlib.Path(
        "../0.data_analysis_and_processing/filtered_single_cells/filtered_single_cell_profiles.parquet"
    )
)

# print df
print(filtering_df.shape)
filtering_df.head()


# ## Filter out any cells that do not matched the filtered data tuple

# In[4]:


# define the columns to match that are important for filtering
columns_to_match = [
    'Metadata_Plate', 'Metadata_Well', 'Metadata_Site',
    'Metadata_Nuclei_Location_Center_X', 'Metadata_Nuclei_Location_Center_Y'
]

# convert rows to tuples
filtering_tuples = set(tuple(row) for row in filtering_df[columns_to_match].itertuples(index=False))
combined_df_tuples = combined_df[columns_to_match].apply(tuple, axis=1)

# create a boolean mask for rows in combined_df that match any rows in filtering_df
mask = combined_df_tuples.isin(filtering_tuples)

# filter combined_df to keep only rows that match any row in filtering_df
filtered_combined_df = combined_df[mask]

# reset index
filtered_combined_df.reset_index(drop=True, inplace=True)

print(filtered_combined_df.shape)
filtered_combined_df.head()


# ## Hold out all untreated single-cells from the `293T` cell line

# In[5]:


# set path for training, testing, and holdout datasets
data_dir = pathlib.Path("./data")
data_dir.mkdir(exist_ok=True)

# holdout all cells from the 293T cell line as CSV
holdout_data = filtered_combined_df[filtered_combined_df['Metadata_CellLine'] == '293T']

# Save the holdout data to a parquet file
holdout_data.to_parquet(f"{data_dir}/holdout_data_293T.parquet", index=False)

# Print the shape of holdout_data to verify
print(holdout_data.shape)


# ## Split data 70% training and 30% testing

# In[6]:


# Remove rows with Metadata_CellLine as 293T
remaining_data = filtered_combined_df[filtered_combined_df['Metadata_CellLine'] != '293T']

# Split the data into training and testing sets
train_data, test_data = train_test_split(
    remaining_data,
    test_size=0.3,
    random_state=0,  # For reproducibility
    shuffle=True     # Ensure data is shuffled before splitting
)

# Save the training and testing data to parquet files
train_data.to_parquet(f"{data_dir}/training_data.parquet", index=False)
test_data.to_parquet(f"{data_dir}/testing_data.parquet", index=False)

# Print the shapes of the splits to verify
print(f"Training data shape: {train_data.shape}")
print(f"Testing data shape: {test_data.shape}")

