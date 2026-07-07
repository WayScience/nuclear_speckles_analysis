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
mkdir -p /pl/active/koala/nuclear_speckle_data/training_results/mlartifacts

# Create the experiment with an explicit artifact location before mlflow run.
# Otherwise a SQLite-only tracking URI falls back to a repo-local ./mlruns path.
EXPERIMENT_NAME="test_smoke"
python - "$EXPERIMENT_NAME" <<'PY'
import re
import sys
from pathlib import Path

import mlflow

experiment_name = sys.argv[1]
artifact_root = Path("/pl/active/koala/nuclear_speckle_data/training_results/mlartifacts")
artifact_component = re.sub(r"[^A-Za-z0-9._-]+", "_", experiment_name).strip("._-") or "experiment"
artifact_location = artifact_root / artifact_component

client = mlflow.MlflowClient()
existing = client.get_experiment_by_name(experiment_name)
if existing is None:
    artifact_location.mkdir(parents=True, exist_ok=True)
    experiment_id = client.create_experiment(
        experiment_name,
        artifact_location=str(artifact_location),
    )
    print(
        f"Created MLflow experiment {experiment_name!r} "
        f"with artifact_location={str(artifact_location)!r} (id={experiment_id})"
    )
else:
    print(
        f"Using existing MLflow experiment {experiment_name!r} "
        f"with artifact_location={existing.artifact_location!r} (id={existing.experiment_id})"
    )
    if existing.artifact_location != str(artifact_location):
        print(
            "Warning: existing experiment artifact location differs from the expected "
            f"dedicated artifact root: {existing.artifact_location!r}",
            file=sys.stderr,
        )
PY

# ---------------------------------------------------------------------------
# Launch the MLflow project
# ---------------------------------------------------------------------------

# Run the MLflow project using the local Python environment rather than
# creating a new environment.
mlflow run . -e train_model --env-manager local \
  --experiment-name "$EXPERIMENT_NAME" \
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
