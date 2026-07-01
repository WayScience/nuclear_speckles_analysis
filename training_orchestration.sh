#!/usr/bin/env bash

# Exit immediately if a command fails (-e), treat unset variables as errors (-u),
# and propagate failures through pipelines (-o pipefail).
set -euo pipefail

# ---------------------------------------------------------------------------
# MLflow configuration
# ---------------------------------------------------------------------------

# Use the existing MLflow tracking URI if one is already set; otherwise,
# default to the shared SQLite database used for experiment tracking.
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:////pl/active/koala/nuclear_speckle_data/training_results/mlflow.db}"

# ---------------------------------------------------------------------------
# Load the uv module on Alpine
# ---------------------------------------------------------------------------

# Add the shared modulefiles directory to the module search path.
module use --append /pl/active/koala/software/lmod-files

# Load the uv package manager.
module load uv

# Store all uv-managed files in the project's shared filesystem instead of
# the default location in the user's home directory.
export UV_BASE=/projects/$USER/uv
export UV_CACHE_DIR="$UV_BASE/cache"
export UV_PYTHON_INSTALL_DIR="$UV_BASE/python"
export UV_TOOL_DIR="$UV_BASE/tools"

# Print the installed uv version for debugging and reproducibility.
uv -v

# ---------------------------------------------------------------------------
# Activate the Python environment
# ---------------------------------------------------------------------------

# Activate the project's virtual environment. This assumes the environment
# has already been created.
source ".venv/bin/activate"

# Ensure the directory used for MLflow artifacts and the Optuna database
# exists before starting training.
mkdir -p /pl/active/koala/nuclear_speckle_data/training_results

# ---------------------------------------------------------------------------
# Launch the MLflow project
# ---------------------------------------------------------------------------

# Run the MLflow project using the local Python environment rather than
# creating a new environment.
mlflow run . -e train_model --env-manager local \
  --experiment-name "test_smoke" \
  -P dataset=u2os \
  -P epochs=2 \
  -P n_trials=2 \
  -P max_train_batches=2 \
  -P max_eval_batches=2 \
  -P enable_image_savers=1 \
  -P crop_size=256 \
  -P study_name=test_smoke \
  -P optuna_storage="sqlite:////pl/active/koala/nuclear_speckle_data/training_results/optuna_study.db" \
  -P checkpoint_root="/pl/active/koala/nuclear_speckle_data/training_results" \
  -P resume=0
