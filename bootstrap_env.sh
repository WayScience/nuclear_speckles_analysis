#!/usr/bin/env bash

set -euo pipefail

module use --append /pl/active/koala/software/lmod-files
module load uv

export UV_BASE=/projects/$USER/uv
export UV_CACHE_DIR="$UV_BASE/cache"
export UV_PYTHON_INSTALL_DIR="$UV_BASE/python"
export UV_TOOL_DIR="$UV_BASE/tools"

uv sync
