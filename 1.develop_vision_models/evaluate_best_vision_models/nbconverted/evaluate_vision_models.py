#!/usr/bin/env python
# coding: utf-8

# # Evaluate Vision Models
# Here, the best vision models, thus far, are evaluated according to four metrics on both the training and validation datasets:
# L1 Loss, L2 Loss, PSNR, and SSIM

# In[1]:


import copy
import pathlib
import random
import sys
from collections import defaultdict

import albumentations as A
import mlflow
import numpy as np
import pandas as pd
import torch
from torch.utils.data import random_split


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
sys.path.append(str((root_dir / "1.develop_vision_models/losses").resolve(strict=True)))
sys.path.append(str((root_dir / "1.develop_vision_models/models").resolve(strict=True)))

from ImageDataset import ImageDataset
from L1Loss import L1Loss
from L2Loss import L2Loss
from PSNR import PSNR
from SSIM import SSIM
from transforms.CropNPixels import CropNPixels
from transforms.MinMaxNormalize import MinMaxNormalize


# ## Set random seeds

# In[4]:


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

mlflow.log_param("random_seed", 0)


# # Inputs

# In[5]:


# Nuclei crops path of treated nuclei in the Dapi channel with all original pixel values
treated_dapi_crops = (root_dir / "vision_nuclear_speckle_prediction/treated_nuclei_dapi_crops_same_background").resolve(strict=True)

# Nuclei crops path of nuclei in the Gold channel with all original pixel values
gold_crops = (root_dir / "vision_nuclear_speckle_prediction/gold_cropped_nuclei_same_background").resolve(strict=True)

# Contains model metadata
model_manifestdf = pd.read_csv("model_manifest.csv")


# # Outputs

# In[6]:


metrics_path = pathlib.Path("model_metrics")
metrics_path.mkdir(parents=True, exist_ok=True)


# # Evaluate Models

# In[7]:


loss_funcs = {
    "l1_loss": L1Loss(_metric_name="l1_loss"),
    "l2_loss": L2Loss(_metric_name="l2_loss"),
    "psnr": PSNR(_metric_name="psnr"),
    "ssim": SSIM(_metric_name="ssim")
}

losses = defaultdict(list)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Iterate through best models
for _, model_metadata in model_manifestdf.iterrows():

    if "fnet" in model_metadata["model_name"]:

        input_transforms = A.Compose([
            MinMaxNormalize(_normalization_factor=(2 ** 16) - 1, _always_apply=True),
            CropNPixels(_pixel_count=1, _always_apply=True)
        ])

    else:

        input_transforms = A.Compose([
            MinMaxNormalize(_normalization_factor=(2 ** 16) - 1, _always_apply=True),
        ])

    target_transforms = copy.deepcopy(input_transforms)

    img_dataset = ImageDataset(
        _input_dir=treated_dapi_crops,
        _target_dir=gold_crops,
        _input_transform=input_transforms,
        _target_transform=target_transforms
    )

    # Same splitting procedure as in model trainers
    train_size = int(0.7 * len(img_dataset))
    val_size = int(0.15 * len(img_dataset))
    test_size = len(img_dataset) - train_size - val_size
    train_dataset, val_dataset, _ = random_split(img_dataset, [train_size, val_size, test_size])

    with torch.no_grad():

        generator_model = mlflow.pytorch.load_model(model_metadata["model_path"]).eval().to(device)
        val_metric_counts = defaultdict(float)
        train_metric_counts = defaultdict(float)

        for input, target in val_dataset:
            target = target.unsqueeze(0).to(device)
            output = generator_model(input.unsqueeze(0).to(device))

            for loss_name, loss_func in loss_funcs.items():
                val_metric_counts[loss_name] += loss_func(_generated_outputs=output, _targets=target)

        for input, target in train_dataset:
            target = target.unsqueeze(0).to(device)
            output = generator_model(input.unsqueeze(0).to(device))

            for loss_name, loss_func in loss_funcs.items():
                train_metric_counts[loss_name] += loss_func(_generated_outputs=output, _targets=target)

        losses["model_name"].append(model_metadata["model_name"])
        losses["model_name"].append(model_metadata["model_name"])
        losses["datasplit"].append("training")
        losses["datasplit"].append("validation")

        for loss_name, loss_func in loss_funcs.items():
            losses[loss_name].append(train_metric_counts[loss_name].item() / len(train_dataset))
            losses[loss_name].append(val_metric_counts[loss_name].item() / len(val_dataset))


# In[8]:


lossdf = pd.DataFrame(losses)
lossdf.to_csv(metrics_path / "best_model_metrics.csv")


# In[9]:


lossdf.head()

