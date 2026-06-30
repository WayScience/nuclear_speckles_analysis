#!/usr/bin/env bash
set +e

source ".venv/bin/activate"
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///mlflow.db}"

OPTUNA_STORAGE="${OPTUNA_STORAGE:-sqlite:///optuna_study.db}"
CHECKPOINT_ROOT="${CHECKPOINT_ROOT:-trial_checkpoints}"

status=0

mlflow run . -e train_model --env-manager local --experiment-name "initial_nuclear_speckle_prediction_dapi_gold" -P dataset=initial -P epochs=50 -P n_trials=30 -P enable_image_savers=1 -P crop_size=256 -P study_name=initial_wgan_gp -P optuna_storage="$OPTUNA_STORAGE" -P checkpoint_root="$CHECKPOINT_ROOT" -P resume=1 || status=1
mlflow run . -e train_model --env-manager local --experiment-name "u2os_nuclear_speckle_prediction_dapi_gold" -P dataset=u2os -P epochs=50 -P n_trials=30 -P enable_image_savers=1 -P crop_size=256 -P study_name=u2os_wgan_gp -P optuna_storage="$OPTUNA_STORAGE" -P checkpoint_root="$CHECKPOINT_ROOT" -P resume=1 || status=1

exit "$status"
