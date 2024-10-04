#!/usr/bin/env python
# coding: utf-8

# # Visualize Well Cell Count Distribution

# In[1]:


import pathlib

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


# # Inputs

# In[3]:


mic_comparisons_path = root_dir / "0.data_analysis_and_processing/mic_analysis/mic_comparisons_data/well_sirna_mic_comparisons.parquet"
micdf = pd.read_parquet(mic_comparisons_path)


# # Outputs

# In[4]:


cell_count_path = pathlib.Path("cell_count_distribution_figures")
cell_count_path.mkdir(parents=True, exist_ok=True)


# In[5]:


micdf = micdf.drop_duplicates(subset=["Metadata_Plate", "Metadata_Well"])


# In[6]:


print(micdf)


# In[7]:


plt.figure(figsize=(18, 10))
sns.histplot(data=micdf, x="Metadata_Cell_Count", kde=False)
plt.xlabel("Well Cell Count Distribution")
plt.ylabel("Number of Wells")
plt.title("Well Cell Counts")

plt.savefig(cell_count_path / "well_cell_counts.png")

