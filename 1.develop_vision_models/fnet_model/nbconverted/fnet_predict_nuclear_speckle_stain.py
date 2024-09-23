#!/usr/bin/env python
# coding: utf-8

# # Predict GOLD Stain from DAPI (Fnet)
# The 2d fnet architecture (https://doi.org/10.1038/s41592-018-0111-2) is trained to predict the GOLD stain from cropped DAPI Nuclei Images.
# This model was trained on a similar task, therefore, we are interested in model's performance when trained on our task.

# In[1]:


import pathlib
import random
import sys
from collections import defaultdict
from typing import Tuple

import albumentations as A
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim


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


# ## Custom Imports

# In[3]:


sys.path.append(str((root_dir / "1.develop_vision_models").resolve(strict=True)))

from ImageDataset import ImageDataset
from models.fnet_nn_2d import Net
from ModelTrainer import ModelTrainer
from transforms.CropNPixels import CropNPixels
from transforms.StandardScaler import StandardScaler


# ## Set random seeds

# In[4]:


random.seed(0)
np.random.seed(0)

mlflow.log_param("random_seed", 0)


# # Inputs

# In[5]:


# Nuclei crops path of treated nuclei in the Dapi channel with all original pixel values
treated_dapi_crops = (root_dir / "vision_nuclear_speckle_prediction/treated_nuclei_dapi_crops").resolve(strict=True)

# Nuclei crops path of nuclei in the Gold channel with all original pixel values
gold_crops = (root_dir / "vision_nuclear_speckle_prediction/gold_cropped_nuclei").resolve(strict=True)

# Paths to original nuclear speckle data
data_dir = (root_dir / "nuclear_speckles_data").resolve(strict=True)
nuclear_mask_dir = (data_dir / "Nuclear_masks").resolve(strict=True)
sc_profiles_path = list((data_dir / "Preprocessed_data/single_cell_profiles").resolve(strict=True).glob("*feature_selected*.parquet"))

# Load single-cell profile data
scdfs = [pd.read_parquet(sc_path) for sc_path in sc_profiles_path if sc_path.is_file()]
scdfs = pd.concat(scdfs, axis=0).reset_index(drop=True)


# # Outputs

# In[6]:


figure_path = pathlib.Path("fnet_validation_images")
figure_path.mkdir(parents=True, exist_ok=True)

metrics_path = pathlib.Path("metrics")
metrics_path.mkdir(parents=True, exist_ok=True)

model_path = pathlib.Path("model")
model_path.mkdir(parents=True, exist_ok=True)


# In[7]:


description = "Here we leverage the 2d fnet architecture in https://doi.org/10.1038/s41592-018-0111-2 to predict the GOLD stain from cropped DAPI Nuclei Images. We retain all pixel values in the cropped images"
mlflow.set_tag("mlflow.note.content", description)


# # Image Generation Functions

# In[8]:


def format_img(_tensor_img: torch.Tensor) -> np.ndarray:
    """Reshapes an image and rescales pixel values from the StandardScaler transform."""

    mean = trainer.val_dataset.dataset.input_transform[0].mean
    std = trainer.val_dataset.dataset.input_transform[0].std

    return (torch.squeeze(_tensor_img) * std + mean).to(torch.uint16).cpu().numpy()


# In[9]:


def evaluate_and_format_imgs(_input: torch.Tensor, _target: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

    single_input = input.unsqueeze(1).to(device)
    single_target = target.to(device)

    model.eval()
    with torch.no_grad():
        # Forward Pass
        output = model(single_input)

    return format_img(single_input), format_img(single_target), format_img(output)


# # Initialize and Train Model

# In[10]:


transforms = A.Compose([
    StandardScaler(_always_apply=True),
    CropNPixels(_pixel_count=1, _always_apply=True)
])

img_dataset = ImageDataset(
    _input_dir=treated_dapi_crops,
    _target_dir=gold_crops,
    _input_transform=transforms,
    _target_transform=transforms
)


# In[11]:


model = Net()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model.to(device)


# In[12]:


optim_params = {
    "lr": 1e-3,
    "betas": (0.5, 0.999)
}

optimizer = optim.Adam(
    model.parameters(),
    **optim_params
)

mlflow.log_param("Optimizer", "ADAM")
mlflow.log_params(optim_params)


# In[13]:


# These keys will be in the names of the logged losses
tracked_losses = {
    "mse loss": nn.MSELoss(),
    "mae loss": nn.L1Loss()
}

backprop_loss_name = "mae loss"

mlflow.log_param("Training Loss", backprop_loss_name)


# In[14]:


trainer_params = {
    "_batch_size": 32,
    "_epochs": 2_000,
    "_patience": 15
}


# In[15]:


trainer = ModelTrainer(
    _model=model,
    _image_dataset=img_dataset,
    _optimizer=optimizer,
    _tracked_losses=tracked_losses,
    _backprop_loss_name=backprop_loss_name,
    **trainer_params
)


# In[16]:


trainer.train()


# # Generate Images
# Evaluate the model by generating the same number of example images for each siRNA.

# In[17]:


example_images_per_sirna = 10
max_pixel_val = 2**16 - 1

sirna_img_counts = {sirna: 0 for sirna in scdfs.loc[scdfs["Metadata_Condition"] != "untreated"]["Metadata_Condition"].unique()}
i = 0

for input, target in iter(trainer.val_dataset):

    img_name = trainer.val_dataset.dataset.input_name
    img_name = img_name.replace("_illumcorrect.tiff", "")

    cell_id = img_name.split("_")[0]
    sirna = scdfs.loc[int(cell_id)]["Metadata_Condition"]

    if sirna not in list(sirna_img_counts.keys()):
        continue

    input, target, output = evaluate_and_format_imgs(input, target)

    titles = ['DAPI Image', 'Predicted GOLD Image', 'Target GOLD Image']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    fig.suptitle(f"{img_name}", fontsize=16)

    plt.subplots_adjust(wspace=0.3, hspace=0)  # Adjust `wspace` if titles overlap

    axes[0].imshow(input, cmap="grey", vmin=0, vmax=max_pixel_val)
    axes[0].axis('off')
    axes[0].set_title(titles[0], fontsize=14)

    axes[1].imshow(output, cmap="grey", vmin=0, vmax=max_pixel_val)
    axes[1].axis('off')
    axes[1].set_title(titles[1], fontsize=14)

    axes[2].imshow(target, cmap="grey", vmin=0, vmax=max_pixel_val)
    axes[2].axis('off')
    axes[2].set_title(titles[2], fontsize=14)

    plt.savefig(figure_path / f"{img_name}.png", bbox_inches='tight', pad_inches=0)
    plt.close()

    sirna_img_counts[sirna] += 1

    if sirna_img_counts[sirna] >= example_images_per_sirna:
        sirna_img_counts.pop(sirna)

        if not sirna_img_counts:
            break


# # Log Metrics and Model

# In[18]:


client = mlflow.MlflowClient()

run_id = mlflow.active_run().info.run_id
run = client.get_run(run_id)

metrics_per_epoch = defaultdict(list)

for metric_name in run.data.metrics.keys():
    metric_history = client.get_metric_history(run_id=run_id, key=metric_name)
    for metric in metric_history:
        metrics_per_epoch[metric_name].append(metric.value)

metricsdf = pd.DataFrame(metrics_per_epoch)
metricsdf["epoch"] = np.arange(metricsdf.shape[0])
metricsdf.to_csv(metrics_path / "fnet_metrics_per_epoch.csv", index=False)


# In[19]:


mlflow.pytorch.log_model(pytorch_model=model.cpu(), artifact_path="model", conda_env=str(root_dir / "1.develop_vision_models" / "environment.yml"))

# Save model for github
torch.save(model.state_dict(), model_path / "fnet_model_states.pth")

