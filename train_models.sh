#!/usr/bin/env bash

set -u -o pipefail

if ! git rev-parse --show-toplevel >/dev/null 2>&1; then
  echo "Run this script from inside the git repository."
  exit 1
fi

BASE_EXPERIMENT_NAME="nuclear_speckle_prediction_dapi_gold"

# Format: "ref|dataset"
# `ref` can be a branch, tag, or commit SHA.
RUNS=(
  "main|initial"
  # "abc1234|initial"
  # "feature-branch|other_dataset"
)

COMMON_ARGS=(
  -e train_model
  --env-manager local
  -P epochs=2
  -P n_trials=2
  -P enable_image_savers=1
)

failures=()
original_ref="$(git symbolic-ref --quiet --short HEAD || git rev-parse HEAD)"

for run in "${RUNS[@]}"; do
  IFS='|' read -r ref dataset <<< "$run"
  experiment_name="${BASE_EXPERIMENT_NAME}_${dataset}"

  echo
  echo "=== checking out $ref ==="
  if ! git checkout "$ref"; then
    echo "Checkout failed for $ref"
    failures+=("$ref|checkout|$dataset")
    continue
  fi

  echo "=== training ref=$ref dataset=$dataset experiment=$experiment_name ==="
  if ! uv run mlflow run . \
    "${COMMON_ARGS[@]}" \
    --experiment-name "$experiment_name" \
    -P dataset="$dataset"; then
    echo "Training failed for $ref"
    failures+=("$ref|train|$dataset")
    continue
  fi

  echo "=== completed $ref successfully ==="
done

echo
echo "=== restoring original ref: $original_ref ==="
git checkout "$original_ref" >/dev/null 2>&1 || true

echo
if [ "${#failures[@]}" -eq 0 ]; then
  echo "All runs completed successfully."
  exit 0
fi

echo "Some runs failed:"
for failure in "${failures[@]}"; do
  IFS='|' read -r ref stage dataset <<< "$failure"
  echo "  - $ref (dataset=$dataset): $stage failed"
done

exit 1
