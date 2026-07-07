#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: inspect_mlflow_store.sh --db-path /path/to/mlflow.db

Prints experiment metadata, object counts, and distinct artifact roots for a
SQLite-backed MLflow tracking store.
EOF
}

DB_PATH=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --db-path)
      DB_PATH="${2:-}"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      printf 'Unknown argument: %s\n' "$1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "$DB_PATH" ]]; then
  usage >&2
  exit 1
fi

if [[ ! -f "$DB_PATH" ]]; then
  printf 'MLflow DB not found: %s\n' "$DB_PATH" >&2
  exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
  printf 'python3 is required\n' >&2
  exit 1
fi

python3 - "$DB_PATH" <<'PY'
import os
import sqlite3
import sys
from urllib.parse import urlparse

db_path = sys.argv[1]
conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row


def table_exists(name: str) -> bool:
    row = conn.execute(
        "select 1 from sqlite_master where type='table' and name=?", (name,)
    ).fetchone()
    return row is not None


def artifact_root(location: str) -> str:
    if not location:
        return "<missing>"
    if location.startswith("file://"):
        parsed = urlparse(location)
        return os.path.dirname(parsed.path.rstrip("/")) or "/"
    if location.startswith("/"):
        return os.path.dirname(location.rstrip("/")) or "/"
    if location.startswith("mlflow-artifacts:/"):
        return "mlflow-artifacts:/"
    if "://" in location:
        return f"<unsupported-scheme:{location.split('://', 1)[0]}>"
    return f"<relative:{os.path.dirname(location.rstrip('/')) or '.'}>"


print(f"DB: {db_path}")
print()
print("Experiments")
print("-----------")
experiments = conn.execute(
    """
    select experiment_id, name, artifact_location, lifecycle_stage
    from experiments
    order by cast(experiment_id as integer)
    """
).fetchall()
for row in experiments:
    print(
        f"- id={row['experiment_id']}"
        f" name={row['name']!r}"
        f" lifecycle_stage={row['lifecycle_stage']}"
        f" artifact_location={row['artifact_location']!r}"
    )

print()
print("Object Counts")
print("-------------")
for table in [
    "experiments",
    "runs",
    "logged_models",
    "registered_models",
    "trace_info",
    "evaluation_datasets",
]:
    if table_exists(table):
        count = conn.execute(f"select count(*) from {table}").fetchone()[0]
        print(f"- {table}: {count}")
    else:
        print(f"- {table}: <table missing>")

print()
print("Artifact Roots")
print("--------------")
roots = sorted({artifact_root(row["artifact_location"]) for row in experiments})
for root in roots:
    print(f"- {root}")

print()
print("Migration Notes")
print("---------------")
unsupported = [root for root in roots if root.startswith("<unsupported") or root.startswith("<relative")]
if "mlflow-artifacts:/" in roots:
    unsupported.append("mlflow-artifacts:/")
if unsupported:
    print("- Some artifact locations use unsupported, server-managed, or relative schemes. The migration script will stop until those paths are handled explicitly.")
else:
    print("- Copy each absolute artifact root above from the source host before starting the temporary source MLflow server.")
    print("- If you see repo-root mlruns paths here, those runs wrote artifacts directly into a checkout instead of a dedicated artifact root.")
PY
