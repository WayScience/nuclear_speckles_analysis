#!/usr/bin/env python
# coding: utf-8

# # Predict GOLD Stain from DAPI
# The pretrained pix2pix architecture is further trained using the WGAN-GP Cross-Zamirski model to predict the GOLD stain from cropped DAPI Nuclei Images. The discriminator is a CNN without any normalization.
# Here, we optimize this model.

# In[ ]:


import copy
import pathlib
import random
import shutil
import sys

import albumentations as A
import joblib
import mlflow
import numpy as np
import optuna
import pandas as pd
import torch


# ## Find the root of the git repo on the host system

# In[ ]:


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

# In[ ]:


sys.path.append(str((root_dir / "1.develop_vision_models").resolve(strict=True)))
sys.path.append(str((root_dir / "1.develop_vision_models/losses").resolve(strict=True)))
sys.path.append(str((root_dir / "1.develop_vision_models/models").resolve(strict=True)))
sys.path.append(str((root_dir / "1.develop_vision_models/utils").resolve(strict=True)))

from CPTextureVarianceLoss import CPTextureVarianceLoss
from datasets.ImageMetaDataset import ImageMetaDataset
from Pix2PixDiscriminatorNoNorm import Pix2PixDiscriminator
from trainers.WGANGPCPPix2PixMetaTrainer import WGANGPCPPix2PixMetaTrainer
from transforms.MinMaxNormalize import MinMaxNormalize
from unet_model import UNet
from WassersteinCPTextureVarianceGeneratorLoss import \
    WassersteinCPTextureVarianceGeneratorLoss
from WassersteinGradientPenaltyLoss import WassersteinGradientPenaltyLoss


# ## Set random seeds

# In[1]:


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

mlflow.log_param("random_seed", 0)


# # Inputs

# In[ ]:


conda_env = str(root_dir / "1.develop_vision_models" / "environment.yml")

# Nuclei crops path of treated nuclei in the Dapi channel with all original pixel values
treated_dapi_crops = (
    root_dir
    / "vision_nuclear_speckle_prediction/treated_nuclei_dapi_crops_same_background"
).resolve(strict=True)

# Nuclei crops path of nuclei in the Gold channel with all original pixel values
gold_crops = (
    root_dir / "vision_nuclear_speckle_prediction/gold_cropped_nuclei_same_background"
).resolve(strict=True)

# Paths to original nuclear speckle data
data_dir = (root_dir / "nuclear_speckles_data").resolve(strict=True)
nuclear_mask_dir = (data_dir / "Nuclear_masks").resolve(strict=True)
sc_profiles_path = list(
    (data_dir / "Preprocessed_data/single_cell_profiles")
    .resolve(strict=True)
    .glob("*feature_selected*.parquet")
)

# Load single-cell profile data
scdfs = [pd.read_parquet(sc_path) for sc_path in sc_profiles_path if sc_path.is_file()]
scdfs = pd.concat(scdfs, axis=0).reset_index(drop=True)

pipeline_path = root_dir / "1.develop_vision_models/utils/GOLD_crop_analysis.cppipe"

# Also hard-coded in WGANGPPix2PixMetaTrainer
img_montage_dir = pathlib.Path("generated_image_epoch_montage")


# # Outputs

# In[ ]:


figure_path = pathlib.Path("pix2pix_validation_images")
figure_path.mkdir(parents=True, exist_ok=True)

metrics_path = pathlib.Path("metrics")
metrics_path.mkdir(parents=True, exist_ok=True)

model_path = pathlib.Path("model")
model_path.mkdir(parents=True, exist_ok=True)


# In[ ]:


description = """
- Architecture: Cross-Zamirski CP (Pix2pix wgangp CP) without conditional component
- Image Modification: Cropped to nuclei using CP bounding box and min-max normalized
- No normalization layers in the discriminator model
- Batch normalization is present in the generator model
- Pretrained Generator model (from best Cross-Zamirski model)
- Additional Loss Modification: Includes the Texture_Variance_GOLD_3_00_256 CP loss component weighted by a hyperparameter importance factor (+lambda*CP)
- Smaller CP Loss
"""

mlflow.set_tag("mlflow.note.content", description)


# # Initialize and Train Model

# In[ ]:


input_transforms = A.Compose(
    [
        MinMaxNormalize(_normalization_factor=(2**16) - 1, _always_apply=True),
    ]
)

target_transforms = copy.deepcopy(input_transforms)

img_dataset = ImageMetaDataset(
    _input_dir=treated_dapi_crops,
    _target_dir=gold_crops,
    _input_transform=input_transforms,
    _target_transform=target_transforms,
)


# In[ ]:


model_data = {
    "best_loss": float("inf"),
}

mlflow.set_tag("model_architecture", "pix2pix")

mlflow.log_param("optimizer_generator", "adam")
mlflow.log_param("optimizer_discriminator", "adam")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def objective(trial):
    global trainer
    global trainer_params

    trainer_params = {"epochs": 35, "patience": 15, "example_images_per_epoch": 10}

    optimizer_params = {}
    model_hyperparams = {}

    with mlflow.start_run(nested=True, run_name=f"trial {trial.number}"):
        discriminator_model = Pix2PixDiscriminator(
            _number_input_channels=1, _number_output_channels=32, _conv_depth=4
        )
        generator_model = mlflow.pytorch.load_model(
            "file:///home/camo/projects/nuclear_speckles_analysis/mlruns/598485576074396081/8c78e96b9e374078bcbea98fe259e048/artifacts/generator_model"
        )

        discriminator_model.to(device)
        generator_model.to(device)

        trainer_params["batch_size"] = trial.suggest_int("batch_size", 1, 20)
        trainer_params["discriminator_update_frequency"] = 1
        optimizer_params["lr_generator"] = trial.suggest_float(
            "lr_generator", 1e-5, 1e-2, log=True
        )
        optimizer_params["lr_discriminator"] = trial.suggest_float(
            "lr_discriminator", 1e-5, 1e-2, log=True
        )
        model_hyperparams["wasserstein_gp_importance"] = trial.suggest_int(
            "wasserstein_gp_importance", 1, 20
        )
        model_hyperparams["reconstruction_importance"] = trial.suggest_int(
            "reconstruction_importance", 5, 200
        )
        model_hyperparams["cp_loss_importance"] = trial.suggest_float(
            "cp_loss_importance", 0, 7
        )
        mlflow.log_params(trainer_params)
        trainer_params = {
            f"_{param_name}": param_value
            for param_name, param_value in trainer_params.items()
        }
        trainer_params["_save_pretrained_generated_imgs"] = True

        optimizers = {
            "generator_optimizer": torch.optim.Adam(
                generator_model.parameters(),
                lr=optimizer_params["lr_generator"],
                betas=(0.5, 0.999),
            ),
            "discriminator_optimizer": torch.optim.Adam(
                discriminator_model.parameters(),
                lr=optimizer_params["lr_discriminator"],
                betas=(0.5, 0.999),
            ),
        }

        for opt_name, opt in optimizers.items():
            opt_params = opt.param_groups[0].copy()

            del opt_params["params"]
            opt_params = {
                f"{opt_name}_{opt_param_name}": opt_param
                for opt_param_name, opt_param in opt_params.items()
            }
            opt_params[opt_name] = opt.__class__.__name__.lower()

            mlflow.log_params(opt_params)

        mlflow.log_params(trainer_params)
        mlflow.log_params(model_hyperparams)

        cp_texture_var_loss = CPTextureVarianceLoss(
            _metric_name="cp_texture_variance_loss",
            _pipeline_path=pipeline_path,
            _targets_path=gold_crops,
        )

        trainer = WGANGPCPPix2PixMetaTrainer(
            _generator_model=generator_model,
            _discriminator_model=discriminator_model,
            _image_dataset=img_dataset,
            _generator_optimizer=optimizers["generator_optimizer"],
            _discriminator_optimizer=optimizers["discriminator_optimizer"],
            _discriminator_loss=WassersteinGradientPenaltyLoss(
                "discriminator_wgan_gp_cross_zamirski_classification_average",
                _gradient_penalty_importance=model_hyperparams[
                    "wasserstein_gp_importance"
                ],
            ),
            _generator_loss=WassersteinCPTextureVarianceGeneratorLoss(
                _metric_name="generator_wgan_gp_cp_texture_variance_average",
                _cp_loss=cp_texture_var_loss,
                _cp_loss_importance=model_hyperparams["cp_loss_importance"],
                _reconstruction_importance=model_hyperparams[
                    "reconstruction_importance"
                ],
            ),
            **trainer_params,
        )

        loss, best_generator, best_discriminator = trainer.train()

        mlflow.pytorch.log_model(
            pytorch_model=best_generator.cpu(),
            artifact_path="generator_model",
            conda_env=conda_env,
        )
        mlflow.pytorch.log_model(
            pytorch_model=best_discriminator.cpu(),
            artifact_path="discriminator_model",
            conda_env=conda_env,
        )

        for img_set in img_montage_dir.iterdir():
            if img_set.is_dir():
                mlflow.log_artifacts(img_set, artifact_path=img_set.name)

        shutil.rmtree(img_montage_dir)

        return loss


# In[2]:


study = optuna.create_study(
    direction="minimize", sampler=optuna.samplers.RandomSampler(seed=0)
)
study.optimize(objective, n_trials=10)


# In[ ]:


joblib.dump(study, "pix2pix_optuna_study.joblib")
mlflow.log_artifact("pix2pix_optuna_study.joblib")

