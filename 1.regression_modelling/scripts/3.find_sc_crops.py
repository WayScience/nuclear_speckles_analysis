#!/usr/bin/env python
# coding: utf-8

# # Generate single-cell crops for min/max montages of the top feature (Zernike phase 2,2) for both stains

# ## Import libraries

# In[1]:


import pathlib
from pprint import pprint
from typing import List

import cv2
import pandas as pd

# ## Set functions for processing and generating single-cell crops

# In[2]:


# Function for formatting min/max row data frames into dictionaries
def create_sc_dict(dfs: List[pd.DataFrame], names: List[str]) -> dict:
    """Format lists of data frames and names into a dictionary with all relevant metadata to find single-cell images.

    Args:
        dfs (List[pd.DataFrame]): List of data frames each containing a single cell and relevant metadata.
        names (List[str]): List of names corresponding to the data frames.

    Returns:
        dict: Dictionary containing info relevant for finding single-cell crops.
    """
    sc_dict = {}
    for df, name in zip(dfs, names):
        for i, (_, row) in enumerate(df.iterrows()):
            if i >= 5:  # Limit to 5 rows
                break
            key = f"{name}_{i+1}"  # Create key with incrementing number
            sc_dict[key] = {
                "plate": row["Metadata_Plate"],
                "well": row["Metadata_Well"],
                "site": row["Metadata_Site"],
                "location_center_x": row["Metadata_Nuclei_Location_Center_X"],
                "location_center_y": row["Metadata_Nuclei_Location_Center_Y"],
            }
    return sc_dict


# In[3]:


# Function for generating and saving single-cell crops per channel as PNGs
def generate_sc_crops(
    sc_dict: dict,
    images_dir: pathlib.Path,
    output_img_dir: pathlib.Path,
    crop_size: int,
) -> None:
    """Using a dictionary with single-cell metadata info per image set, single-cell crops per channel are generated
    and saved as PNGs in an image set folder.

    Args:
        sc_dict (dict): Dictionary containing info relevant for finding single-cell crops.
        images_dir (pathlib.Path): Directory where illumination corrected images are found.
        output_img_dir (pathlib.Path): Main directory to save each image set single-cell crops
        crop_size (int): Size of the box in pixels (example: setting crop_size as 250 will make a 250x250 pixel crop around the single-cell center coordinates)
    """
    for key, info in sc_dict.items():
        # Initialize a list to store file paths for every image set
        file_paths = []

        # Create file paths with well, site, and channel
        for i in range(3):  # Update the range to start from 0 and end at 2
            filename = f"{images_dir}/{info['plate']}/{info['plate']}_{info['well']}_{info['site']}_CH{i}_Z09_illumcorrect.tiff"
            file_paths.append(filename)

            # Read the image
            channel_image = cv2.imread(filename, cv2.IMREAD_UNCHANGED)

            # Use the location_center_x and location_center_y to create a crop
            center_x = info.get("location_center_x")
            center_y = info.get("location_center_y")

            # Crop dimensions (including crop_size)
            half_crop = crop_size // 2

            # Ensure the center coordinates are valid
            if center_x is not None and center_y is not None:
                # Calculate crop boundaries
                top_left_x = max(int(center_x - half_crop), 0)
                top_left_y = max(int(center_y - half_crop), 0)
                bottom_right_x = min(int(center_x + half_crop), channel_image.shape[1])
                bottom_right_y = min(int(center_y + half_crop), channel_image.shape[0])

                # Perform cropping
                cropped_channel = channel_image[
                    top_left_y:bottom_right_y, top_left_x:bottom_right_x
                ]

                # Ensure the cropped image is of size 250x250
                cropped_channel = cv2.resize(cropped_channel, (crop_size, crop_size))

                # Make directory for the key to keep all channels for an image in one folder
                key_dir = pathlib.Path(f"{output_img_dir}/{key}")
                key_dir.mkdir(exist_ok=True, parents=True)

                # Save the cropped image with single_cell and channel information
                output_filename = str(
                    pathlib.Path(f"{key_dir}/{key}_CH{i}_cropped.png")
                )
                cv2.imwrite(output_filename, cropped_channel)


# ## Set paths and variables

# In[4]:


# Images directory for
images_dir = pathlib.Path(
    "/media/18tbdrive/Github_Repositories/nuclear_speckle_image_profiling/2.illumination_correction/IC_corrected_images"
).resolve(strict=True)

# Output dir for cropped images
output_img_dir = pathlib.Path("./sc_crops")
output_img_dir.mkdir(exist_ok=True)

# Define the size of the cropping box (250x250 pixels)
crop_size = 250

# Create open list for one row data frames for each top feature per channel per cell type
list_of_dfs = []

# Create open list of names to assign each data frame in a list relating to the feature, channel, and cell type
list_of_names = []


# ## Load in training data to use to find example crops

# In[5]:


# Path to the folder with models
model_dir = pathlib.Path("./models").resolve(strict=True)

# paths to the training data split
training_data_path = pathlib.Path("./data/training_data.parquet").resolve(strict=True)

# Load your datasets
training_df = pd.read_parquet(training_data_path)

print(training_df.shape)
training_df.head()


# ## Find min/max representative single-cells per stain

# In[6]:


# Get data frame with the top single-cells
top_A647 = training_df.nlargest(5, "Nuclei_RadialDistribution_ZernikePhase_A647_2_2")[
    [
        "Nuclei_RadialDistribution_ZernikePhase_A647_2_2",
        "Metadata_Well",
        "Metadata_Plate",
        "Metadata_Site",
        "Metadata_Nuclei_Location_Center_X",
        "Metadata_Nuclei_Location_Center_Y",
        "Metadata_Condition",
    ]
]

# Append the DataFrame and its name to the lists
list_of_dfs.append(top_A647)
list_of_names.append("top_A647")

print(top_A647.shape)
top_A647


# In[7]:


# Get data frame with the bottom single-cells
bottom_A647 = training_df.nsmallest(
    5, "Nuclei_RadialDistribution_ZernikePhase_A647_2_2"
)[
    [
        "Nuclei_RadialDistribution_ZernikePhase_A647_2_2",
        "Metadata_Well",
        "Metadata_Plate",
        "Metadata_Site",
        "Metadata_Nuclei_Location_Center_X",
        "Metadata_Nuclei_Location_Center_Y",
        "Metadata_Condition",
    ]
]

# Append the DataFrame and its name to the lists
list_of_dfs.append(bottom_A647)
list_of_names.append("bottom_A647")

print(bottom_A647.shape)
bottom_A647


# In[8]:


# Get data frame with the top single-cells
top_GOLD = training_df.nlargest(5, "Nuclei_RadialDistribution_ZernikePhase_GOLD_2_2")[
    [
        "Nuclei_RadialDistribution_ZernikePhase_GOLD_2_2",
        "Metadata_Well",
        "Metadata_Plate",
        "Metadata_Site",
        "Metadata_Nuclei_Location_Center_X",
        "Metadata_Nuclei_Location_Center_Y",
        "Metadata_Condition",
    ]
]

# Append the DataFrame and its name to the lists
list_of_dfs.append(top_GOLD)
list_of_names.append("top_GOLD")

print(top_GOLD.shape)
top_GOLD


# In[9]:


# Get data frame with the bottom single-cells
bottom_GOLD = training_df.nsmallest(
    5, "Nuclei_RadialDistribution_ZernikePhase_GOLD_2_2"
)[
    [
        "Nuclei_RadialDistribution_ZernikePhase_GOLD_2_2",
        "Metadata_Well",
        "Metadata_Plate",
        "Metadata_Site",
        "Metadata_Nuclei_Location_Center_X",
        "Metadata_Nuclei_Location_Center_Y",
        "Metadata_Condition",
    ]
]

# Append the DataFrame and its name to the lists
list_of_dfs.append(bottom_GOLD)
list_of_names.append("bottom_GOLD")

print(bottom_GOLD.shape)
bottom_GOLD


# ## Generate dictionary with all representative single-cells

# In[10]:


sc_dict = create_sc_dict(dfs=list_of_dfs, names=list_of_names)

# Check the created dictionary for the first two items
pprint(list(sc_dict.items())[:2], indent=4)


# ## Generate single-cell crops and save

# In[11]:


generate_sc_crops(
    sc_dict=sc_dict,
    images_dir=images_dir,
    output_img_dir=output_img_dir,
    crop_size=crop_size,
)

