#!/usr/bin/env bash

# Run a broad, causal posterior-predictive study without writing large artifacts
# into the Git repository. Safe to stop with Ctrl-C and run again: completed
# fight/seed pairs are checkpointed and --resume restores them.

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Usage: bash scripts/run_full_simulation_study.sh [--dry-run | --prepare-only]

Defaults are chosen for a useful population study that remains bounded on the
machine used for this repository:
  * newest 100 complete UFC events
  * both fighters must have at least 3 strictly prior UFC fights
  * 64 event-card bootstrap parameter replicas
  * 1,024 Monte Carlo paths per fight for each of 2 independent seeds
  * at most 22 hours in one terminal session
  * at most 4 GiB beneath the external study directory

Large results default to:
  $HOME/.ufc-data-lab/simulation-studies/all-eligible-v1

Optional environment overrides:
  SIM_STUDY_ROOT       Exact external study directory (Git Bash form, e.g. /d/ufc-study)
  SIM_STUDY_NAME       Directory name below the default parent
  SIM_WORKERS          Worker processes (default: 6, maximum: 16)
  SIM_MAX_WALL_HOURS   Per-session limit (default: 22, maximum: 22)
  SIM_MAX_STORAGE_GB   Study plus fit-cache limit (default: 4)
  SIM_MIN_FREE_GB      Stop before launching if free space is lower (default: 8)

Use a new SIM_STUDY_NAME to start over with different scientific settings.
Do not edit the settings for an existing study; the simulator will reject a
resume whose contract differs.
EOF
}

DRY_RUN=0
PREPARE_ONLY=0
case "${1:-}" in
  "") ;;
  --dry-run) DRY_RUN=1 ;;
  --prepare-only) PREPARE_ONLY=1 ;;
  -h|--help) usage; exit 0 ;;
  *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
esac

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/.." && pwd -P)"

if [[ -x "$REPO_ROOT/.venv/Scripts/python.exe" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/Scripts/python.exe"
elif [[ -x "$REPO_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
else
  PYTHON_BIN="${PYTHON_BIN:-python}"
fi

STUDY_NAME="${SIM_STUDY_NAME:-all-eligible-v1}"
DEFAULT_STUDY_PARENT="$HOME/.ufc-data-lab/simulation-studies"
STUDY_ROOT="${SIM_STUDY_ROOT:-$DEFAULT_STUDY_PARENT/$STUDY_NAME}"
WORKERS="${SIM_WORKERS:-6}"
MAX_WALL_HOURS="${SIM_MAX_WALL_HOURS:-22}"
MAX_STORAGE_GB="${SIM_MAX_STORAGE_GB:-4}"
MIN_FREE_GB="${SIM_MIN_FREE_GB:-8}"

for pair in \
  "SIM_WORKERS:$WORKERS:1:16" \
  "SIM_MAX_WALL_HOURS:$MAX_WALL_HOURS:2:22" \
  "SIM_MAX_STORAGE_GB:$MAX_STORAGE_GB:1:64" \
  "SIM_MIN_FREE_GB:$MIN_FREE_GB:1:1024"; do
  IFS=: read -r label value minimum maximum <<<"$pair"
  if [[ ! "$value" =~ ^[0-9]+$ ]] || (( value < minimum || value > maximum )); then
    echo "$label must be an integer from $minimum through $maximum; received '$value'." >&2
    exit 2
  fi
done

STUDY_ROOT_NATIVE="$($PYTHON_BIN -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).expanduser().resolve())' "$STUDY_ROOT")"
REPO_ROOT_NATIVE="$($PYTHON_BIN -c 'import pathlib,sys; print(pathlib.Path(sys.argv[1]).resolve())' "$REPO_ROOT")"
if "$PYTHON_BIN" -c 'import pathlib,sys; repo=pathlib.Path(sys.argv[1]); out=pathlib.Path(sys.argv[2]); sys.exit(0 if out == repo or repo in out.parents else 1)' "$REPO_ROOT_NATIVE" "$STUDY_ROOT_NATIVE"; then
  echo "Refusing to store the study inside the Git repository: $STUDY_ROOT" >&2
  echo "Choose an external SIM_STUDY_ROOT." >&2
  exit 2
fi

INPUT_DIR="$STUDY_ROOT/input-snapshot"
CODE_DIR="$STUDY_ROOT/code-snapshot"
RESULT_DIR="$STUDY_ROOT/results"
CACHE_DIR="$STUDY_ROOT/causal-fit-cache"
LOG_DIR="$STUDY_ROOT/logs"
LOG_FILE="$LOG_DIR/runner.log"
SETTINGS_FILE="$STUDY_ROOT/study-settings.txt"
SNAPSHOT_MARKER="$STUDY_ROOT/.snapshot-complete"
SNAPSHOT_VERSION=2

RAW_SOURCE="$REPO_ROOT/src/content/data/processed/ufc_fights_reported_doubled.csv"
PROFILES_SOURCE="$REPO_ROOT/src/content/data/processed/fighter_stats.csv"
ROUNDS_SOURCE="$REPO_ROOT/src/content/data/processed/ufc_fight_round_stats_doubled.csv"
CONFIG_SOURCE="$REPO_ROOT/SIMULATION_MECHANICS_BASELINE_V1.json"
RAW_SNAPSHOT="$INPUT_DIR/ufc_fights_reported_doubled.csv"
PROFILES_SNAPSHOT="$INPUT_DIR/fighter_stats.csv"
ROUNDS_SNAPSHOT="$INPUT_DIR/ufc_fight_round_stats_doubled.csv"
CONFIG_SNAPSHOT="$INPUT_DIR/SIMULATION_MECHANICS_BASELINE_V1.json"

LAST_EVENTS=100
MIN_PRIOR_FIGHTS=3
BOOTSTRAP_MEMBERS=64
PATHS_PER_MATCHUP=1024
SEED_REPEATS=2
CHUNK_SIZE=64
SLICE_SECONDS=3300
COMMAND_HARD_LIMIT_SECONDS=3900
FINAL_RESERVE_SECONDS=300
MAX_WALL_SECONDS=$((MAX_WALL_HOURS * 3600))

echo "Study directory: $STUDY_ROOT"
echo "Repository remains free of generated study artifacts."
PATHS_PER_MEMBER=$((PATHS_PER_MATCHUP / BOOTSTRAP_MEMBERS))
TOTAL_PATHS_PER_FIGHT=$((PATHS_PER_MATCHUP * SEED_REPEATS))
echo "Precision: $TOTAL_PATHS_PER_FIGHT total paths/fight ($SEED_REPEATS independent seeds; $PATHS_PER_MEMBER paths across each of $BOOTSTRAP_MEMBERS parameter replicas per seed)."
echo "Scope: newest $LAST_EVENTS complete events; both fighters require $MIN_PRIOR_FIGHTS prior UFC fights."
echo "Limits: ${MAX_WALL_HOURS}h session, ${MAX_STORAGE_GB} GiB study cap, ${MIN_FREE_GB} GiB minimum free space."

if (( DRY_RUN )); then
  echo "Dry run only; no directories or files were created."
  exit 0
fi

if ! command -v timeout >/dev/null 2>&1 || ! timeout --version 2>/dev/null | grep -q 'GNU coreutils'; then
  echo "GNU timeout is required for the hard runtime bound. It is included with Git Bash." >&2
  exit 2
fi

for source in "$RAW_SOURCE" "$PROFILES_SOURCE" "$ROUNDS_SOURCE" "$CONFIG_SOURCE"; do
  if [[ ! -f "$source" ]]; then
    echo "Required study input is missing: $source" >&2
    exit 2
  fi
done

if [[ -d "$STUDY_ROOT" && ! -f "$SNAPSHOT_MARKER" ]] && [[ -n "$(find "$STUDY_ROOT" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
  echo "The study directory exists but its input/code snapshot is incomplete: $STUDY_ROOT" >&2
  echo "Choose a new SIM_STUDY_NAME; this script will not delete or overwrite it." >&2
  exit 2
fi

snapshot_dependencies_present() {
  [[ -f "$CODE_DIR/fight_semantics.py" ]] &&
    [[ -f "$CODE_DIR/ufc_round_data.py" ]] &&
    [[ -f "$CODE_DIR/ufcstats_client.py" ]]
}

# Version 1 omitted shared top-level modules. It could not start the CLI, so it
# is safe to repair that specific pre-run snapshot in place. Never alter a code
# snapshot after a simulator run manifest exists.
if [[ -f "$SNAPSHOT_MARKER" ]] && ! snapshot_dependencies_present; then
  if [[ -f "$RESULT_DIR/run-manifest.json" ]]; then
    echo "The saved code snapshot is incomplete but simulation results already exist." >&2
    echo "Choose a new SIM_STUDY_NAME so the existing study remains immutable." >&2
    exit 2
  fi
  cp -p "$REPO_ROOT"/src/*.py "$CODE_DIR/"
  printf 'version=%s\n' "$SNAPSHOT_VERSION" >"$SNAPSHOT_MARKER"
  if ! grep -q '^snapshot_version=' "$SETTINGS_FILE" 2>/dev/null; then
    printf 'snapshot_version=%s\n' "$SNAPSHOT_VERSION" >>"$SETTINGS_FILE"
  fi
  echo "Repaired the pre-run code snapshot with its shared Python modules."
fi

if [[ ! -f "$SNAPSHOT_MARKER" ]]; then
  mkdir -p "$INPUT_DIR" "$CODE_DIR" "$CACHE_DIR" "$LOG_DIR"
  cp -p "$RAW_SOURCE" "$RAW_SNAPSHOT"
  cp -p "$PROFILES_SOURCE" "$PROFILES_SNAPSHOT"
  cp -p "$ROUNDS_SOURCE" "$ROUNDS_SNAPSHOT"
  cp -p "$CONFIG_SOURCE" "$CONFIG_SNAPSHOT"
  cp -R "$REPO_ROOT/src/fight_sim" "$CODE_DIR/fight_sim"
  cp -R "$REPO_ROOT/src/market_tracker" "$CODE_DIR/market_tracker"
  cp -p "$REPO_ROOT"/src/*.py "$CODE_DIR/"
  printf '%s\n' \
    "created_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "source_repository=$REPO_ROOT_NATIVE" \
    "study_directory=$STUDY_ROOT_NATIVE" \
    "python=$PYTHON_BIN" \
    "events=$LAST_EVENTS" \
    "min_prior_ufc_fights=$MIN_PRIOR_FIGHTS" \
    "bootstrap_members=$BOOTSTRAP_MEMBERS" \
    "paths_per_matchup=$PATHS_PER_MATCHUP" \
    "seed_repeats=$SEED_REPEATS" \
    "workers=$WORKERS" \
    "snapshot_version=$SNAPSHOT_VERSION" \
    "simulator_config=SIMULATION_MECHANICS_BASELINE_V1.json" >"$SETTINGS_FILE"
  printf 'version=%s\n' "$SNAPSHOT_VERSION" >"$SNAPSHOT_MARKER"
  echo "Created immutable input and simulator-code snapshots outside the repository."
fi

mkdir -p "$CACHE_DIR" "$LOG_DIR"
export PYTHONPATH="$CODE_DIR"

"$PYTHON_BIN" -m fight_sim posterior-backtest --help >/dev/null
"$PYTHON_BIN" -c 'import fight_sim, numpy, pandas; print("Simulator environment ready.")'

if (( PREPARE_ONLY )); then
  echo "Preparation and full CLI import check passed; no simulations were run."
  exit 0
fi

storage_check() {
  local status used_bytes free_bytes limit_bytes minimum_free_bytes
  status="$($PYTHON_BIN -c 'import pathlib,shutil,sys; root=pathlib.Path(sys.argv[1]); used=sum(p.stat().st_size for p in root.rglob("*") if p.is_file()); free=shutil.disk_usage(root).free; print(used,free)' "$STUDY_ROOT")"
  read -r used_bytes free_bytes <<<"$status"
  limit_bytes=$((MAX_STORAGE_GB * 1024 * 1024 * 1024))
  minimum_free_bytes=$((MIN_FREE_GB * 1024 * 1024 * 1024))
  echo "Storage used: $($PYTHON_BIN -c 'import sys; print(f"{int(sys.argv[1])/1024**3:.2f} GiB")' "$used_bytes"); free: $($PYTHON_BIN -c 'import sys; print(f"{int(sys.argv[1])/1024**3:.1f} GiB")' "$free_bytes")."
  if (( used_bytes >= limit_bytes )); then
    echo "Stopping before the study exceeds its ${MAX_STORAGE_GB} GiB storage cap." >&2
    return 1
  fi
  if (( free_bytes < minimum_free_bytes )); then
    echo "Stopping because less than ${MIN_FREE_GB} GiB remains free." >&2
    return 1
  fi
}

study_complete() {
  [[ -f "$RESULT_DIR/population-summary.json" ]] || return 1
  "$PYTHON_BIN" -c 'import json,pathlib,sys; d=json.loads(pathlib.Path(sys.argv[1]).read_text()); r=d["runtime"]; s=d["selection"]; done=(r["completed_fight_seed_pairs"] >= r["planned_fight_seed_pairs"] and s["completed_fights"] >= s["eligible_fights"] and not r["stopped_by_time_limit"]); sys.exit(0 if done else 1)' "$RESULT_DIR/population-summary.json"
}

print_progress() {
  [[ -f "$RESULT_DIR/population-summary.json" ]] || return 0
  "$PYTHON_BIN" "$SCRIPT_DIR/simulation_study_progress.py" \
    "$RESULT_DIR/population-summary.json"
}

if study_complete; then
  echo "This study is already complete."
  print_progress
  echo "Open: $RESULT_DIR/population-report.html"
  exit 0
fi

START_EPOCH="$(date +%s)"
SLICE=0
while true; do
  NOW_EPOCH="$(date +%s)"
  ELAPSED_SECONDS=$((NOW_EPOCH - START_EPOCH))
  REMAINING_SECONDS=$((MAX_WALL_SECONDS - ELAPSED_SECONDS))
  REQUIRED_SECONDS=$((COMMAND_HARD_LIMIT_SECONDS + FINAL_RESERVE_SECONDS))
  if (( REMAINING_SECONDS < REQUIRED_SECONDS )); then
    echo "Stopping cleanly before the ${MAX_WALL_HOURS}-hour session limit. Run the same command later to resume."
    break
  fi
  if ! storage_check; then
    echo "Completed checkpoints and reports remain resumable at $STUDY_ROOT" >&2
    break
  fi

  SLICE=$((SLICE + 1))
  RESUME_ARGS=()
  if [[ -f "$RESULT_DIR/run-manifest.json" ]]; then
    RESUME_ARGS=(--resume)
  fi
  echo "Starting bounded slice $SLICE at $(date -u +%Y-%m-%dT%H:%M:%SZ)."
  set +e
  timeout --foreground --signal=INT --kill-after=120 "$COMMAND_HARD_LIMIT_SECONDS" \
    "$PYTHON_BIN" -m fight_sim posterior-backtest \
      --raw "$RAW_SNAPSHOT" \
      --profiles "$PROFILES_SNAPSHOT" \
      --round-stats "$ROUNDS_SNAPSHOT" \
      --simulator-config "$CONFIG_SNAPSHOT" \
      --last-events "$LAST_EVENTS" \
      --min-prior-ufc-fights "$MIN_PRIOR_FIGHTS" \
      --bootstrap-members "$BOOTSTRAP_MEMBERS" \
      --paths-per-matchup "$PATHS_PER_MATCHUP" \
      --seed-repeats "$SEED_REPEATS" \
      --snapshot-parameter-mode full \
      --max-runtime-seconds "$SLICE_SECONDS" \
      --workers "$WORKERS" \
      --chunk-size "$CHUNK_SIZE" \
      --fit-cache-dir "$CACHE_DIR" \
      --output-dir "$RESULT_DIR" \
      "${RESUME_ARGS[@]}" 2>&1 | tee -a "$LOG_FILE"
  COMMAND_STATUS=${PIPESTATUS[0]}
  set -e

  if (( COMMAND_STATUS != 0 )); then
    if (( COMMAND_STATUS == 124 || COMMAND_STATUS == 130 || COMMAND_STATUS == 137 )); then
      echo "A hard slice timeout stopped the current fight. Previously completed fights remain checkpointed." >&2
    else
      echo "The simulator exited with status $COMMAND_STATUS. See $LOG_FILE" >&2
    fi
    exit "$COMMAND_STATUS"
  fi

  print_progress
  if study_complete; then
    echo "Study complete."
    break
  fi
done

storage_check || true
print_progress
if study_complete; then
  echo "HTML analysis: $RESULT_DIR/population-report.html"
  echo "Machine-readable summary: $RESULT_DIR/population-summary.json"
  echo "Compressed fight forecasts: $RESULT_DIR/forecast-ledger.jsonl.gz"
else
  echo "Partial analysis: $RESULT_DIR/population-report.html"
  echo "Resume with: bash scripts/run_full_simulation_study.sh"
fi
