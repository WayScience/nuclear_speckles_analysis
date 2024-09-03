#!/usr/bin/env python
# coding: utf-8

# # Train ElasticNet model to predict each A647 and GOLD feature using the nuclear features
# 
# In this notebook, we split the features into each group for nuclear speckle or nucleus and then train a model per nuclear speckle feature using the nuclear features to predict it.
# 
# We are looking to find the best nuclear speckle feature that can be predicted using nucleus features.

# ## Import libraries

# In[1]:


import pandas as pd
import pathlib
import numpy as np
import joblib
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold, RandomizedSearchCV
import warnings
import sys
import os
# Ignore the ConvergenceWarnings (only thing that will work 🙃)
if not sys.warnoptions:
    warnings.simplefilter("ignore")
    os.environ["PYTHONWARNINGS"] = "ignore"


# ## Set paths and random seed

# In[2]:


# set numpy seed to make sure any random operations performs are reproducible
np.random.seed(0)

# Directory for models to be outputted
model_dir = pathlib.Path("./models")
model_dir.mkdir(exist_ok=True, parents=True)

# Make specific folders in model dir for each model type
final_dir = model_dir / "final"
final_dir.mkdir(exist_ok=True, parents=True)
shuffled_dir = model_dir / "shuffled_baseline"
shuffled_dir.mkdir(exist_ok=True, parents=True)


# ## Load in training data and categorize the features as nuclear speckle (A647 or GOLD) or nucleus (DAPI) to use for model

# In[3]:


# load in training data
training_df = pd.read_csv(pathlib.Path("./data/training_data.csv"))

# Initialize lists to store column names for each feature group
nucleus_features = []
a647_features = []
gold_features = []

# Iterate over column names to categorize them
for column in training_df.columns:
    if not column.startswith("Metadata_"):  # Only look at feature columns
        parts = column.split("_")

        if "Correlation" in parts[1]:  # Check if it's a correlation feature
            if "DAPI" in column:  # If DAPI is present in a correlation feature
                if len(parts) > 4 and ("A647" in parts[3] or "A647" in parts[4]):
                    a647_features.append(column)
                elif len(parts) > 4 and ("GOLD" in parts[3] or "GOLD" in parts[4]):
                    gold_features.append(column)
            else:  # No DAPI in correlation feature, check only 4th part
                if len(parts) > 3 and "A647" in parts[3]:
                    a647_features.append(column)
                elif len(parts) > 3 and "GOLD" in parts[3]:
                    gold_features.append(column)
        else:  # Non-correlation features
            if len(parts) > 4 and "Location" in parts[1]:  # If it's a Location feature
                if parts[4] == "DAPI":
                    nucleus_features.append(column)
                elif parts[4] == "A647":
                    a647_features.append(column)
                elif parts[4] == "GOLD":
                    gold_features.append(column)
            elif len(parts) > 3 and "DAPI" in parts[3]:
                nucleus_features.append(column)
            elif len(parts) > 3 and "A647" in parts[3]:
                a647_features.append(column)
            elif len(parts) > 3 and "GOLD" in parts[3]:
                gold_features.append(column)
            else:
                nucleus_features.append(column)  # Default to nucleus_features


# Prepare X data for with all nucleus features
X = training_df[nucleus_features]

# Generate shuffled data for the shuffled models to use (only do this once)
X_shuffled = X.copy()
for col in X_shuffled.columns:
    np.random.shuffle(X_shuffled[col].values)  # Shuffle values in place, independently

# Print the lists to verify
print(f"Nucleus Features: {len(nucleus_features)}")
print(f"A647 Features: {len(a647_features)}")
print(f"Gold Features: {len(gold_features)}")


# ## Set hyperparameter parameters and search space

# In[4]:


# Set folds for k-fold cross-validation
k_folds = KFold(n_splits=5, shuffle=True, random_state=0)

# Set ElasticNet regression model parameters
elasticnet_params = {
    "alpha": 1.0,  # Equivalent to 'C' in LogisticRegression, but in reverse
    "l1_ratio": 0.5,  # Mixture of L1 and L2 regularization
    "max_iter": 10,
    "random_state": 0,
}

# Define the hyperparameter search space for RandomizedSearchCV
param_dist = {
    "alpha": np.logspace(-3, 3, 7),  # Regularization strength
    "l1_ratio": np.linspace(0, 1, 11),  # Mix of L1 and L2 regularization
}

# Set the random search hyperparameterization method parameters
random_search_params = {
    "param_distributions": param_dist,
    "scoring": "neg_mean_squared_error",  # Suitable for regression
    "random_state": 0,
    "n_jobs": -1,
    "cv": k_folds,
}


# ## Train A647 models

# In[5]:


# Suppress all warnings
warnings.filterwarnings("ignore")

for a647_feature in a647_features:
    y = training_df[a647_feature]

    # Train regular model with hyperparameter tuning
    logreg = ElasticNet(**elasticnet_params)

    # Initialize random search and fit model
    random_search = RandomizedSearchCV(logreg, **random_search_params)
    random_search.fit(X, y)

    # Save the tuned model
    model_filename = final_dir / f"{a647_feature}_tuned_model.joblib"
    joblib.dump(random_search.best_estimator_, model_filename)

    random_search_shuffled = RandomizedSearchCV(logreg, **random_search_params)
    random_search_shuffled.fit(X_shuffled, y)

    # Save the shuffled tuned model
    shuffled_model_filename = (
        shuffled_dir / f"{a647_feature}_shuffled_tuned_model.joblib"
    )
    joblib.dump(random_search_shuffled.best_estimator_, shuffled_model_filename)

    # Print confirmation
    print(f"Trained and saved tuned model for {a647_feature}")
    print(f"Trained and saved shuffled tuned model for {a647_feature}")

print("All A647 models have been trained and tuned!")


# # Train GOLD models

# In[6]:


for gold_feature in gold_features:
    y = training_df[gold_feature]

    # Train regular model with hyperparameter tuning
    logreg = ElasticNet(**elasticnet_params)

    # Initialize random search
    random_search = RandomizedSearchCV(logreg, **random_search_params)
    random_search.fit(X, y)

    # Save the tuned model
    model_filename = final_dir / f"{gold_feature}_tuned_model.joblib"
    joblib.dump(random_search.best_estimator_, model_filename)

    random_search_shuffled = RandomizedSearchCV(logreg, **random_search_params)
    random_search_shuffled.fit(X_shuffled, y)

    # Save the shuffled tuned model
    shuffled_model_filename = (
        shuffled_dir / f"{gold_feature}_shuffled_tuned_model.joblib"
    )
    joblib.dump(random_search_shuffled.best_estimator_, shuffled_model_filename)

    # Print confirmation
    print(f"Trained and saved tuned model for {gold_feature}")
    print(f"Trained and saved shuffled tuned model for {gold_feature}")

print("All GOLD models have been trained and tuned!")

