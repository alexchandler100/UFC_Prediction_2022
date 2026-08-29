#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$REPO_DIR/.venv/Scripts/python.exe"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python"
fi

echo "Method odds will be stored separately under ~/.ufc-data-lab."
echo "Wait for the winner-price backfill to finish or pause before starting this job."
echo "The run is resumable; rerun the same command after any clean pause."

exec "$PYTHON_BIN" "$REPO_DIR/src/backfill_bestfightodds_method_history.py" \
  --acknowledge-source-policy \
  "$@"
