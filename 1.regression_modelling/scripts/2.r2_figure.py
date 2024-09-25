#!/usr/bin/env python
# coding: utf-8

# # Calculate and generate R2 score figures to assess model performance across stains

# ## Import libraries

# In[1]:


import pathlib

import joblib
import yaml
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score


# ## Set paths

# In[2]:


# Output for figures
figures_dir = pathlib.Path("./figures")
figures_dir.mkdir(parents=True, exist_ok=True)


# ## Define features per stain to use for filtering columns

# In[3]:


# Define the path to the YAML file with all the features per stain
yaml_file_path = pathlib.Path("./features_dict.yml")

# Load the column names dictionary from the YAML file
with open(yaml_file_path, 'r') as yaml_file:
    features_dict = yaml.safe_load(yaml_file)

# Access the lists of features
nucleus_features = features_dict['nucleus_features']
a647_features = features_dict['a647_features']
gold_features = features_dict['gold_features']

# Now you can use these lists for further operations
print(f"Nucleus Features: {len(nucleus_features)}")
print(f"A647 Features: {len(a647_features)}")
print(f"Gold Features: {len(gold_features)}")


# ## Load in data splits

# In[4]:


model_dir = pathlib.Path('./models')  # Path to the folder with models

# paths to the data splits
training_data_path = pathlib.Path('./data/training_data.parquet')
testing_data_path = pathlib.Path('./data/testing_data.parquet')
holdout_data_path = pathlib.Path('./data/holdout_data_293T.parquet')

# Load your datasets
training_df = pd.read_parquet(training_data_path)
testing_df = pd.read_parquet(testing_data_path)
holdout_df = pd.read_parquet(holdout_data_path)


# ## Generate list with all models to access

# In[5]:


# List all model files in both 'final' and 'shuffled_baseline' folders
model_files = list(model_dir.rglob('*_tuned_model.joblib'))

len(model_files)


# ## Generate and store R2 results across data splits

# In[6]:


# Create an empty list to store the results
results = []

for model_file in model_files:
    # Get the parent folder (either 'final' or 'shuffled_baseline')
    folder_name = model_file.parent.name
    
    # Set the model type based on the folder name
    if folder_name == 'shuffled_baseline':
        model_name = model_file.stem.replace('_shuffled_tuned_model', '')
        model_type = 'shuffled'
    else:
        model_name = model_file.stem.replace('_tuned_model', '')
        model_type = 'final'

    # Check if the model name is in the A647 or GOLD features list
    if model_name in a647_features:
        stain = 'A647'
    elif model_name in gold_features:
        stain = 'GOLD'
    else:
        # Skip if the model name doesn't match any feature group
        print(f"Skipping {model_name} as it does not match any feature group.")
        continue

    # Load the model
    model = joblib.load(model_file)

    # X will contain nucleus features for all splits
    X_train = training_df[nucleus_features]
    X_test = testing_df[nucleus_features]
    X_holdout = holdout_df[nucleus_features]

    # y will be the column matching the model name
    y_train = training_df[model_name]
    y_test = testing_df[model_name]
    y_holdout = holdout_df[model_name]

    # Predict on training, testing, and holdout data
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    y_holdout_pred = model.predict(X_holdout)

    # Calculate R2 scores
    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)
    r2_holdout = r2_score(y_holdout, y_holdout_pred)

    # Append results to the list, including 'Model type'
    results.append({'Feature': model_name, 'Stain': stain, 'Data split': 'Training', 'R2 score': r2_train, 'Model type': model_type})
    results.append({'Feature': model_name, 'Stain': stain, 'Data split': 'Testing', 'R2 score': r2_test, 'Model type': model_type})
    results.append({'Feature': model_name, 'Stain': stain, 'Data split': 'Holdout', 'R2 score': r2_holdout, 'Model type': model_type})

# Convert the results into a dataframe
results_df = pd.DataFrame(results)

# Remove rows where 'Feature' starts with 'Nuclei_Location' (not an actual feature)
results_df = results_df[~results_df['Feature'].str.startswith('Nuclei_Location')]

print(results_df.shape)
results_df.head()


# ## Generate box plot comparing testing data across models and splits

# ### Set function

# In[7]:


# Function to create the plot for each stain
def plot_r2(results_df, palette_dict, datasplit):
    # Set the figure size
    plt.figure(figsize=(14, 8))

    # Create the horizontal box plot (one per Model type and Stain combination)
    sns.boxplot(
        data=results_df,
        x='R2 score',
        y='model_stain',  # model_stain on y-axis for horizontal comparison
        dodge=True,  # Adjust box plots for multiple hues
        fliersize=0,  # Hide the default outliers as we add scatter later
        width=0.5  # Adjust the width to increase separation
    )

    # Add scatter plot of individual points within each box, colored by 'Measurement'
    sns.stripplot(
        data=results_df,
        x='R2 score',
        y='model_stain',  # Same axis for horizontal separation
        hue='Measurement',  # Color by 'Measurement'
        palette=palette_dict,  # Customize colors with your palette
        jitter=True,  # Jitter the points slightly for better visibility
        dodge=True,
        size=8,  # Size of the scatter points
        edgecolor='black',
        linewidth=1,
        alpha=0.8  # Transparency of the points
    )

    # Customize the plot
    plt.title(f'R2 Scores by Model Type and Stain: {datasplit} data')
    plt.xlabel('R2 Score')
    plt.ylabel('Model Stain')

    # Move legend outside of the plot
    plt.legend(title='Measurement', bbox_to_anchor=(1.05, 1), loc='upper left')

    # Improve layout
    plt.tight_layout()

    # Save
    plt.savefig(f"./{figures_dir}/shuffled_versus_final_r2.png", dpi=600)

    # Show plot
    plt.show()


# ### Filter only training data and format columns

# In[8]:


# Only include rows where there is training data
filtered_results_df = results_df[results_df['Data split'] == 'Testing'].copy()

# Create the new 'model_stain' column
filtered_results_df['model_stain'] = filtered_results_df['Stain'] + ' (' + filtered_results_df['Model type'] + ')'

# Create a column to extract the first word from 'Feature'
filtered_results_df['Measurement'] = filtered_results_df['Feature'].apply(lambda x: x.split('_')[1])

print(filtered_results_df.shape)
filtered_results_df.head()


# ### Create color palette for measurements to use for all plots

# In[9]:


#  Get unique Measurement values to create a consistent color palette
unique_measurements = filtered_results_df['Measurement'].unique()

# Create a fixed color palette based on the unique Measurement values
palette = sns.color_palette('Dark2', len(unique_measurements))
palette_dict = dict(zip(unique_measurements, palette))


# ### Plot R2 scores between shuffled and final models

# In[10]:


plot_r2(filtered_results_df, palette_dict, "Testing")


# ## Split features to get an R2 score per stain and plot R2 results

# ### Function to split features based on stain to have one R2 score per stain

# In[11]:


def process_subset(df, split_type, model_type):
    # Create a copy of df_subset to avoid modifying the original slice
    df_subset = df.copy()
    
    # Step 1: Remove rows where 'Feature' contains both 'A647' and 'GOLD'
    df_subset = df_subset[~(df_subset['Feature'].str.contains('A647') & df_subset['Feature'].str.contains('GOLD'))]
    
    # Step 2: Add 'Feature_clean' without modifying 'Feature'
    df_subset['Feature_clean'] = df_subset['Feature'].str.replace('A647', '').str.replace('GOLD', '').str.strip('_')
    
    # Step 3: Find duplicates by 'Feature_clean' and keep R2 score for both 'A647' and 'GOLD'
    a647_scores = df_subset[df_subset['Stain'] == 'A647'][['Feature', 'Feature_clean', 'R2 score']].rename(columns={'Feature': 'Feature_A647', 'R2 score': 'A647 R2 score'})
    gold_scores = df_subset[df_subset['Stain'] == 'GOLD'][['Feature', 'Feature_clean', 'R2 score']].rename(columns={'Feature': 'Feature_GOLD', 'R2 score': 'GOLD R2 score'})
    
    # Step 4: Merge the dataframes based on 'Feature_clean'
    merged_results = pd.merge(a647_scores, gold_scores, on='Feature_clean', how='outer')
    
    # Add columns to indicate the data split and model type
    merged_results['Data split'] = split_type
    merged_results['Model type'] = model_type
    
    return merged_results


# ### Split out the data based on model type and data split to make it easier to process each feature without duplicates

# In[12]:


# Split the dataframe into 'Training', 'Testing', and 'Holdout' for both 'final' and 'shuffled'
df_training_final = results_df[(results_df['Data split'] == 'Training') & (results_df['Model type'] == 'final')]
df_testing_final = results_df[(results_df['Data split'] == 'Testing') & (results_df['Model type'] == 'final')]
df_holdout_final = results_df[(results_df['Data split'] == 'Holdout') & (results_df['Model type'] == 'final')]

df_training_shuffled = results_df[(results_df['Data split'] == 'Training') & (results_df['Model type'] == 'shuffled')]
df_testing_shuffled = results_df[(results_df['Data split'] == 'Testing') & (results_df['Model type'] == 'shuffled')]
df_holdout_shuffled = results_df[(results_df['Data split'] == 'Holdout') & (results_df['Model type'] == 'shuffled')]


# ### Format the feature and R2 score columns and concat all data back together

# In[13]:


# Process each subset with designation
training_final_results = process_subset(df_training_final, 'Training', 'final')
testing_final_results = process_subset(df_testing_final, 'Testing', 'final')
holdout_final_results = process_subset(df_holdout_final, 'Holdout', 'final')

training_shuffled_results = process_subset(df_training_shuffled, 'Training', 'shuffled')
testing_shuffled_results = process_subset(df_testing_shuffled, 'Testing', 'shuffled')
holdout_shuffled_results = process_subset(df_holdout_shuffled, 'Holdout', 'shuffled')

# Concatenate the results
final_results_df = pd.concat([training_final_results, testing_final_results, holdout_final_results, 
                              training_shuffled_results, testing_shuffled_results, holdout_shuffled_results])

# Extract 'Measurement' from 'Feature_clean' after merging
final_results_df['Measurement'] = final_results_df['Feature_clean'].str.split('_').str[1]

# Drop rows with any NaN values
final_results_df = final_results_df.dropna()

# Display the final dataframe
print(final_results_df.shape)
final_results_df.head(10)


# ### Create scatterplot comparing training and testing results

# In[14]:


# Filter the dataframe for 'Training' and 'Testing' data splits only
filtered_final_results_df = final_results_df[final_results_df['Data split'].isin(['Training', 'Testing'])]

# Create the figure and axes
fig, ax = plt.subplots(figsize=(8.5, 6))

# Plot with different shapes based on 'Data split'
sns.scatterplot(data=filtered_final_results_df,
                x='GOLD R2 score', y='A647 R2 score', 
                hue='Measurement', style='Data split', 
                palette=palette_dict, ax=ax)

# Add y = x line as reference
lims = [0, 1]  # For R² scores, which range from 0 to 1
ax.plot(lims, lims, '--', color='gray')

# Set plot labels
ax.set_xlabel('GOLD R2 Score')
ax.set_ylabel('A647 R2 Score')

# Add legend outside the plot
ax.legend(bbox_to_anchor=(1.05, 0.7), loc='upper left')

# Adjust layout to make space for the legend
plt.tight_layout()

# Save
plt.savefig(f"./{figures_dir}/training_versus_testing_scatter_r2.png", dpi=600)

# Show plot
plt.show()


# ### Create scatterplot comparing holdout and testing results

# In[15]:


# Filter the dataframe for 'Holdout' and 'Testing' data splits only
filtered_final_results_df = final_results_df[final_results_df['Data split'].isin(['Holdout', 'Testing'])]

# Set any R² score below 0 to 0
filtered_final_results_df.loc[:, 'GOLD R2 score'] = filtered_final_results_df['GOLD R2 score'].clip(lower=0)
filtered_final_results_df.loc[:, 'A647 R2 score'] = filtered_final_results_df['A647 R2 score'].clip(lower=0)

# Create the figure and axes
fig, ax = plt.subplots(figsize=(8.5, 6))

# Custom markers for 'Testing' (X) and 'Holdout' (Square)
markers = {'Testing': 'X', 'Holdout': 's'}

# Plot with different shapes based on 'Data split'
sns.scatterplot(data=filtered_final_results_df,
                x='GOLD R2 score', y='A647 R2 score', 
                hue='Measurement', style='Data split', 
                markers=markers, palette=palette_dict, ax=ax)

# Add y = x line as reference
lims = [0, 1]  # For R² scores, which range from 0 to 1
ax.plot(lims, lims, '--', color='gray')

# Set plot labels
ax.set_xlabel('GOLD R2 Score')
ax.set_ylabel('A647 R2 Score')

# Add legend outside the plot
ax.legend(bbox_to_anchor=(1.05, 0.7), loc='upper left')

# Adjust layout to make space for the legend
plt.tight_layout()

# Save
plt.savefig(f"./{figures_dir}/holdout_versus_testing_scatter_r2.png", dpi=600)

# Show plot
plt.show()


# ## Holdout data boxplot comparing between stains

# ### Filter only holdout data from the final model
# 
# Note: We know the shuffled models are under performing in the previous plot so we don't need to include it here.

# In[16]:


# Only include rows where 'Data split' contains 'Holdout'
holdout_df = results_df[results_df['Data split'] == 'Holdout']

# Remove rows where 'Model type' starts with 'shuffled'
holdout_df = holdout_df[~holdout_df['Model type'].str.startswith('shuffled')]

print(holdout_df.shape)
holdout_df.sort_values(by='R2 score', ascending=False).head()


# ### Generate boxplot

# In[17]:


# Replace R2 scores less than -1 with -1
holdout_df.loc[holdout_df['R2 score'] < -1, 'R2 score'] = -1

# Create a column to extract the first word from 'Feature'
holdout_df['Measurement'] = holdout_df['Feature'].apply(lambda x: x.split('_')[1])

# Set the figure size
plt.figure(figsize=(14, 8))

# Create the horizontal box plot (one per Model type and Stain combination)
sns.boxplot(
    data=holdout_df,
    x='R2 score',
    y='Stain',  
    dodge=True,  # Adjust box plots for multiple hues
    fliersize=0,  # Hide the default outliers as we add scatter later
    width=0.5  # Adjust the width to increase separation
)

# Add scatter plot of individual points within each box, colored by 'Measurement'
sns.stripplot(
    data=holdout_df,
    x='R2 score',
    y='Stain',  # Same axis for horizontal separation
    hue='Measurement',  # Color by 'Measurement'
    palette=palette_dict,  # Customize colors with your palette
    jitter=True,  # Jitter the points slightly for better visibility
    dodge=True,
    size=8,  # Size of the scatter points
    edgecolor='black',
    linewidth=1,
    alpha=0.8  # Transparency of the points
)

# Add a vertical dashed line at R2 score = 0.5
plt.axvline(x=0.5, color='red', linestyle='--', linewidth=1)

# Customize the plot
plt.xlabel('R2 Score')
plt.ylabel('Model Stain')

# Move legend outside of the plot
plt.legend(title='Measurement', bbox_to_anchor=(1.05, 1), loc='upper left')

# Improve layout
plt.tight_layout()

# Save
plt.savefig(f"./{figures_dir}/holdout_a647_versus_gold_r2.png", dpi=600)

# Show plot
plt.show()


# ## Generate boxplot figure comparing the top 2 model feature distributions across stains using the training data
# 
# Note: We do not save these figures as the results from them do not tell a clear story to use elsewhere.

# ### Identify top two features per stain
# 
# These will be the same two RadialDistribution features (FracatD and Zernike 2,2).

# In[18]:


# Create a list to store top features
top_features_list = []

# Iterate over each stain
for stain in ['A647', 'GOLD']:
    # Filter features for the current stain and sort by R2 score
    stain_features_df = final_results_df[
        ~final_results_df[f'{stain} R2 score'].isna()
    ].sort_values(by=f'{stain} R2 score', ascending=False)
    
    # Get the top 2 features for the each stain
    top_stain_features = stain_features_df.head(4)[f'Feature_{stain}'].tolist()
    
    # Add to the top features list
    top_features_list.extend(top_stain_features)

# Remove duplicates if any
top_features_list = list(set(top_features_list))

# Print the top features list
print(len(top_features_list))
print(top_features_list)


# ### Filter training data to only include metadata and top feature columns

# In[19]:


# Get all columns with the 'Metadata' prefix
metadata_columns = [col for col in training_df.columns if col.startswith('Metadata')]

# Combine the lists of top features and metadata columns
selected_columns = metadata_columns + top_features_list

# Filter the training_df dataframe to include only the selected columns
filtered_training_df = training_df[selected_columns]

print(filtered_training_df.shape)
filtered_training_df.head()


# ### Plot distributions across features per feature

# In[20]:


# Set the figure size
plt.figure(figsize=(12, 8))

# Create the vertical box plot
sns.boxplot(
    data=filtered_training_df,
    x='Metadata_Condition',  # Metadata_condition on x-axis
    y='Nuclei_RadialDistribution_FracAtD_GOLD_4of4',  # Nuclei_RadialDistribution_FracAtD_GOLD_4of4 on y-axis
    palette='viridis'  # You can choose a different palette if needed
)

# Customize the plot
plt.title('Distribution of Nuclei Radial Distribution by Metadata Condition')
plt.xlabel('Metadata Condition')
plt.ylabel('Nuclei Radial Distribution FracAtD (GOLD)')

# Rotate x-axis labels if needed for better readability
plt.xticks(rotation=45)

# Improve layout
plt.tight_layout()

# Show plot
plt.show()


# In[21]:


# Set the figure size
plt.figure(figsize=(12, 8))

# Create the vertical box plot
sns.boxplot(
    data=filtered_training_df,
    x='Metadata_Condition',  # Metadata_condition on x-axis
    y='Nuclei_RadialDistribution_FracAtD_A647_4of4',  
    palette='viridis'  # You can choose a different palette if needed
)

# Customize the plot
plt.title('Distribution of Nuclei Radial Distribution by Metadata Condition')
plt.xlabel('Metadata Condition')
plt.ylabel('Nuclei Radial Distribution FracAtD (A647)')

# Rotate x-axis labels if needed for better readability
plt.xticks(rotation=45)

# Improve layout
plt.tight_layout()

# Show plot
plt.show()


# In[22]:


# Set the figure size
plt.figure(figsize=(12, 8))

# Create the vertical box plot
sns.boxplot(
    data=filtered_training_df,
    x='Metadata_Condition',  # Metadata_condition on x-axis
    y='Nuclei_RadialDistribution_ZernikePhase_GOLD_2_2',  
    palette='viridis'  # You can choose a different palette if needed
)

# Customize the plot
plt.title('Distribution of Nuclei Radial Distribution by Metadata Condition')
plt.xlabel('Metadata Condition')
plt.ylabel('Nuclei Radial Distribution ZernikePhase (GOLD)')

# Rotate x-axis labels if needed for better readability
plt.xticks(rotation=45)

# Improve layout
plt.tight_layout()

# Show plot
plt.show()


# In[23]:


# Set the figure size
plt.figure(figsize=(12, 8))

# Create the vertical box plot
sns.boxplot(
    data=filtered_training_df,
    x='Metadata_Condition',  # Metadata_condition on x-axis
    y='Nuclei_RadialDistribution_ZernikePhase_A647_2_2',  
    palette='viridis'  # You can choose a different palette if needed
)

# Customize the plot
plt.title('Distribution of Nuclei Radial Distribution by Metadata Condition')
plt.xlabel('Metadata Condition')
plt.ylabel('Nuclei Radial Distribution ZernikePhase (A647)')

# Rotate x-axis labels if needed for better readability
plt.xticks(rotation=45)

# Improve layout
plt.tight_layout()

# Show plot
plt.show()


# In[ ]:




