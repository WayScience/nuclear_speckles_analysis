#!/usr/bin/env bash

set -euo pipefail

usage() {
  cat <<'EOF'
Usage: migrate_mlflow_from_alpine.sh --source-host alpine [options]

Copies a SQLite-backed MLflow store from a remote host, rewrites copied
artifact paths into a local staging area, starts temporary source/target
MLflow servers, and migrates all MLflow data with mlflow-export-import.

Options:
  --source-host HOST         Remote SSH host. Default: alpine
  --source-db PATH           Remote MLflow SQLite DB.
                             Default: /pl/active/koala/nuclear_speckle_data/training_results/mlflow.db
  --target-repo PATH         Local target repo root.
                             Default: /home/camo/projects/nuclear_speckles_analysis
  --staging-root PATH        Local staging root.
                             Default: /tmp/opencode/mlflow_migration
  --source-port PORT         Temporary local source MLflow server port. Default: 6001
  --target-port PORT         Temporary local target MLflow server port. Default: 6002
  --help                     Show this message.
EOF
}

SOURCE_HOST="alpine"
SOURCE_DB="/pl/active/koala/nuclear_speckle_data/training_results/mlflow.db"
TARGET_REPO="/home/camo/projects/nuclear_speckles_analysis"
STAGING_ROOT="/tmp/opencode/mlflow_migration"
SOURCE_PORT="6001"
TARGET_PORT="6002"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-host)
      SOURCE_HOST="${2:-}"
      shift 2
      ;;
    --source-db)
      SOURCE_DB="${2:-}"
      shift 2
      ;;
    --target-repo)
      TARGET_REPO="${2:-}"
      shift 2
      ;;
    --staging-root)
      STAGING_ROOT="${2:-}"
      shift 2
      ;;
    --source-port)
      SOURCE_PORT="${2:-}"
      shift 2
      ;;
    --target-port)
      TARGET_PORT="${2:-}"
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

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    printf 'Required command not found: %s\n' "$1" >&2
    exit 1
  fi
}

sqlite_uri() {
  local path="$1"
  printf 'sqlite:////%s' "${path#/}"
}

wait_for_http() {
  local url="$1"
  local label="$2"
  local attempts=60
  local sleep_seconds=1

  for ((i = 1; i <= attempts; i++)); do
    if curl -fsS "$url" >/dev/null 2>&1; then
      return 0
    fi
    sleep "$sleep_seconds"
  done

  printf 'Timed out waiting for %s at %s\n' "$label" "$url" >&2
  return 1
}

cleanup() {
  if [[ -n "${TARGET_SERVER_PID:-}" ]]; then
    kill "$TARGET_SERVER_PID" >/dev/null 2>&1 || true
  fi
  if [[ -n "${SOURCE_SERVER_PID:-}" ]]; then
    kill "$SOURCE_SERVER_PID" >/dev/null 2>&1 || true
  fi
}

trap cleanup EXIT

require_cmd rsync
require_cmd sqlite3
require_cmd python3
require_cmd curl
require_cmd uv

if [[ ! -d "$TARGET_REPO" ]]; then
  printf 'Target repo not found: %s\n' "$TARGET_REPO" >&2
  exit 1
fi

TARGET_DB="$TARGET_REPO/mlflow.db"
TARGET_ARTIFACT_ROOT="$TARGET_REPO/mlartifacts"

if [[ ! -f "$TARGET_DB" ]]; then
  printf 'Target MLflow DB not found: %s\n' "$TARGET_DB" >&2
  exit 1
fi

RUN_STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="$STAGING_ROOT/$RUN_STAMP"
SOURCE_STAGE_DIR="$RUN_ROOT/source"
ARTIFACT_STAGE_DIR="$RUN_ROOT/source_artifacts"
EXPORT_DIR="$RUN_ROOT/export"
LOG_DIR="$RUN_ROOT/logs"
mkdir -p "$SOURCE_STAGE_DIR" "$ARTIFACT_STAGE_DIR" "$EXPORT_DIR" "$LOG_DIR" "$TARGET_ARTIFACT_ROOT"

SOURCE_DB_LOCAL="$SOURCE_STAGE_DIR/mlflow.db"
ARTIFACT_MAP_FILE="$RUN_ROOT/artifact_root_map.tsv"
TARGET_RUNS_BEFORE="$RUN_ROOT/target_runs_before.txt"
TARGET_RUNS_AFTER="$RUN_ROOT/target_runs_after.txt"
IMPORTED_RUNS_REPORT="$RUN_ROOT/imported_runs.tsv"
SOURCE_SERVER_LOG="$LOG_DIR/source_mlflow_server.log"
TARGET_SERVER_LOG="$LOG_DIR/target_mlflow_server.log"

printf 'Copying source MLflow DB from %s:%s\n' "$SOURCE_HOST" "$SOURCE_DB"
rsync -avz --progress "$SOURCE_HOST:$SOURCE_DB" "$SOURCE_DB_LOCAL"

printf '\nInspecting copied source DB before path rewrite\n'
"$(dirname "$0")/inspect_mlflow_store.sh" --db-path "$SOURCE_DB_LOCAL"

python3 - "$SOURCE_DB_LOCAL" "$ARTIFACT_STAGE_DIR" "$ARTIFACT_MAP_FILE" <<'PY'
import os
import sqlite3
import sys
from urllib.parse import urlparse

db_path, artifact_stage_dir, out_path = sys.argv[1:4]
conn = sqlite3.connect(db_path)
rows = conn.execute(
    "select distinct artifact_location from experiments where artifact_location is not null order by artifact_location"
).fetchall()

roots = []
seen = set()
for (location,) in rows:
    if location.startswith("file://"):
        root = os.path.dirname(urlparse(location).path.rstrip("/")) or "/"
    elif location.startswith("/"):
        root = os.path.dirname(location.rstrip("/")) or "/"
    elif location.startswith("mlflow-artifacts:/"):
        raise SystemExit(
            "Unsupported source artifact scheme 'mlflow-artifacts:/'. "
            "This script expects file-backed experiment artifact locations."
        )
    else:
        raise SystemExit(
            f"Unsupported source artifact location: {location!r}. "
            "Only absolute file paths and file:// URIs are supported."
        )

    if root not in seen:
        seen.add(root)
        roots.append(root)

with open(out_path, "w", encoding="utf-8") as handle:
    for index, root in enumerate(roots, start=1):
        local_root = os.path.join(artifact_stage_dir, f"root_{index:02d}")
        handle.write(f"{root}\t{local_root}\n")
PY

printf '\nRsync commands the script is about to run\n'
while IFS=$'\t' read -r source_root local_root; do
  printf 'rsync -avz --progress %q:%q/ %q/\n' "$SOURCE_HOST" "$source_root" "$local_root"
done < "$ARTIFACT_MAP_FILE"

while IFS=$'\t' read -r source_root local_root; do
  mkdir -p "$local_root"
  rsync -avz --progress "$SOURCE_HOST:$source_root/" "$local_root/"
done < "$ARTIFACT_MAP_FILE"

python3 - "$SOURCE_DB_LOCAL" "$ARTIFACT_MAP_FILE" <<'PY'
import os
import sqlite3
import sys
from urllib.parse import urlparse

db_path, mapping_path = sys.argv[1:3]

with open(mapping_path, "r", encoding="utf-8") as handle:
    mappings = [tuple(line.rstrip("\n").split("\t", 1)) for line in handle if line.strip()]

conn = sqlite3.connect(db_path)


def rewrite_location(location: str) -> str:
    if location is None:
        return location
    for source_root, local_root in mappings:
        if location.startswith("file://"):
            parsed = urlparse(location)
            path = parsed.path
            if path == source_root or path.startswith(source_root + "/"):
                suffix = path[len(source_root):]
                return f"file://{local_root}{suffix}"
        else:
            if location == source_root or location.startswith(source_root + "/"):
                suffix = location[len(source_root):]
                return f"{local_root}{suffix}"
    return location


experiments = conn.execute("select experiment_id, artifact_location from experiments").fetchall()
for experiment_id, artifact_location in experiments:
    rewritten = rewrite_location(artifact_location)
    if rewritten != artifact_location:
        conn.execute(
            "update experiments set artifact_location = ? where experiment_id = ?",
            (rewritten, experiment_id),
        )

runs = conn.execute("select run_uuid, artifact_uri from runs").fetchall()
for run_uuid, artifact_uri in runs:
    rewritten = rewrite_location(artifact_uri)
    if rewritten != artifact_uri:
        conn.execute(
            "update runs set artifact_uri = ? where run_uuid = ?",
            (rewritten, run_uuid),
        )

conn.commit()
PY

printf '\nInspecting copied source DB after path rewrite\n'
"$(dirname "$0")/inspect_mlflow_store.sh" --db-path "$SOURCE_DB_LOCAL"

sqlite3 "$TARGET_DB" "select run_uuid from runs order by run_uuid;" > "$TARGET_RUNS_BEFORE"

SOURCE_TRACKING_URI="http://127.0.0.1:$SOURCE_PORT"
TARGET_TRACKING_URI="http://127.0.0.1:$TARGET_PORT"

printf '\nStarting temporary source MLflow server on %s\n' "$SOURCE_TRACKING_URI"
(
  cd "$RUN_ROOT"
  uv run mlflow server \
    --host 127.0.0.1 \
    --port "$SOURCE_PORT" \
    --backend-store-uri "$(sqlite_uri "$SOURCE_DB_LOCAL")"
) > "$SOURCE_SERVER_LOG" 2>&1 &
SOURCE_SERVER_PID=$!
wait_for_http "$SOURCE_TRACKING_URI/" "source MLflow server"

printf 'Starting temporary target MLflow server on %s\n' "$TARGET_TRACKING_URI"
(
  cd "$TARGET_REPO"
  uv run mlflow server \
    --host 127.0.0.1 \
    --port "$TARGET_PORT" \
    --backend-store-uri "$(sqlite_uri "$TARGET_DB")" \
    --serve-artifacts \
    --artifacts-destination "file://$TARGET_ARTIFACT_ROOT"
) > "$TARGET_SERVER_LOG" 2>&1 &
TARGET_SERVER_PID=$!
wait_for_http "$TARGET_TRACKING_URI/" "target MLflow server"

printf '\nExporting all MLflow objects from copied alpine store\n'
(
  cd "$TARGET_REPO"
  MLFLOW_TRACKING_URI="$SOURCE_TRACKING_URI" \
    uv run --group migration export-all \
    --output-dir "$EXPORT_DIR" \
    --export-deleted-runs True
)

printf '\nImporting all MLflow objects into target repo store\n'
(
  cd "$TARGET_REPO"
  MLFLOW_TRACKING_URI="$TARGET_TRACKING_URI" \
    uv run --group migration import-all \
    --input-dir "$EXPORT_DIR" \
    --import-source-tags True
)

sqlite3 "$TARGET_DB" "select run_uuid from runs order by run_uuid;" > "$TARGET_RUNS_AFTER"

python3 - "$TARGET_DB" "$TARGET_RUNS_BEFORE" "$TARGET_RUNS_AFTER" "$IMPORTED_RUNS_REPORT" <<'PY'
import sqlite3
import sys

db_path, before_path, after_path, out_path = sys.argv[1:5]

with open(before_path, "r", encoding="utf-8") as handle:
    before = {line.strip() for line in handle if line.strip()}
with open(after_path, "r", encoding="utf-8") as handle:
    after = [line.strip() for line in handle if line.strip()]

new_run_ids = [run_id for run_id in after if run_id not in before]

conn = sqlite3.connect(db_path)
conn.row_factory = sqlite3.Row

with open(out_path, "w", encoding="utf-8") as handle:
    handle.write("destination_run_id\tsource_run_id\texperiment_name\tstart_time\n")
    for run_id in new_run_ids:
        row = conn.execute(
            """
            select r.run_uuid as destination_run_id,
                   src.value as source_run_id,
                   e.name as experiment_name,
                   r.start_time as start_time
            from runs r
            join experiments e on e.experiment_id = r.experiment_id
            left join tags src
              on src.run_uuid = r.run_uuid
             and src.key = 'mlflow_exim.run_info.run_id'
            where r.run_uuid = ?
            """,
            (run_id,),
        ).fetchone()
        handle.write(
            f"{row['destination_run_id']}\t{row['source_run_id'] or ''}\t{row['experiment_name']}\t{row['start_time'] or ''}\n"
        )
PY

printf '\nImported runs\n'
if command -v column >/dev/null 2>&1; then
  column -t -s $'\t' "$IMPORTED_RUNS_REPORT"
else
  cat "$IMPORTED_RUNS_REPORT"
fi

printf '\nMigration complete\n'
printf -- '- staging root: %s\n' "$RUN_ROOT"
printf -- '- source server log: %s\n' "$SOURCE_SERVER_LOG"
printf -- '- target server log: %s\n' "$TARGET_SERVER_LOG"
