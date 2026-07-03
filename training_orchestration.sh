#!/usr/bin/env bash
set +e

source ".venv/bin/activate"
export MLFLOW_TRACKING_URI="sqlite:///mlflow.db"

status=0

mlflow run . -e train_model \
  --env-manager local \
  --experiment-name "initial_nuclear_speckle_prediction_dapi_gold" \
  -P dataset=initial \
  -P epochs=20 \
  -P n_trials=15 \
  -P max_train_batches=-1 \
  -P max_eval_batches=-1 \
  -P eval_batch_size=10 \
  -P eval_use_amp=0 \
  -P train_use_amp=1 \
  -P enable_image_savers=1 \
  -P batch_metric_log_every_n=1 \
  -P crop_size=256 \
  -P optuna_storage=sqlite:///optuna_study.db \
  -P checkpoint_root=trial_checkpoints \
  -P resume=0 \
  -P parent_run_id= \
  || status=1

mlflow run . -e train_model \
  --env-manager local \
  --experiment-name "u2os_nuclear_speckle_prediction_dapi_gold" \
  -P dataset=u2os \
  -P epochs=20 \
  -P n_trials=15 \
  -P max_train_batches=-1 \
  -P max_eval_batches=-1 \
  -P eval_batch_size=10 \
  -P eval_use_amp=0 \
  -P train_use_amp=1 \
  -P enable_image_savers=1 \
  -P batch_metric_log_every_n=1 \
  -P crop_size=256 \
  -P optuna_storage=sqlite:///optuna_study.db \
  -P checkpoint_root=trial_checkpoints \
  -P resume=0 \
  -P parent_run_id= \
  || status=1

exit "$status"
