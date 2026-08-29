#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_BIN="$REPO_DIR/.venv/Scripts/python.exe"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

echo "Historical odds will be stored outside Git under ~/.ufc-data-lab."
echo "The run is resumable; rerun this same command after any clean pause."

exec "$PYTHON_BIN" "$REPO_DIR/src/backfill_bestfightodds_history.py" \
  --acknowledge-source-policy \
  "$@"
