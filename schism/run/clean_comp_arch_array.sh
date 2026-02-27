#!/usr/bin/env bash
#SBATCH -J schism_clean_comp_array
#SBATCH -o schism_clean_comp_array.%A_%a.out
#SBATCH -e schism_clean_comp_array.%A_%a.err
#SBATCH -p cpu-small
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -t 2-00:00:00

set -euo pipefail

# Wrapper to distribute clean_comp_arch.sh work across Slurm array tasks.
#
# Typical usage:
#   bash clean_comp_arch_array.sh RUN22e \
#     --part-start 1 --num-parts 107 \
#     --array-count 12 --array-max-parallel 4 \
#     -- --extract-root ./extracted --stage-root ./stage --repack-dir ./repacked \
#        --workers 4 --mode interactive
#
# Notes:
# - Array tasks process contiguous part-index chunks.
# - The wrapper invokes clean_comp_arch.sh once per part (with --part-glob exact match).
# - Use '--' to separate wrapper args from clean_comp_arch.sh args.

# -------------------------
# Config (edit here for no-flag runs; CLI still overrides)
# -------------------------
# How this wrapper splits work:
# - You choose a global part range using PART_START + (PART_END or NUM_PARTS).
# - The wrapper divides that range into ARRAY_COUNT contiguous chunks.
# - Each Slurm array task processes one chunk, and calls clean_comp_arch.sh once per part.
# - Inside each clean_comp_arch.sh call, compression parallelism is controlled by
#   CLEAN_ARGS_DEFAULT (typically "--workers N").
#
# Practical concurrency:
# - Total concurrent nccopy processes is approximately:
#     ARRAY_MAX_PARALLEL_DEFAULT * <clean_comp_arch.sh --workers>
#   (or ARRAY_COUNT * workers if ARRAY_MAX_PARALLEL_DEFAULT is empty).
# - Start conservative on shared storage, then increase after testing I/O behavior.
#
# CLI override rule:
# - Positional PREFIX overrides PREFIX_DEFAULT.
# - Wrapper flags (e.g. --array-count) override *_DEFAULT values below.
# - Args passed after '--' are appended to CLEAN_ARGS_DEFAULT, so later duplicates win.

# Wrapper defaults
# Run prefix (e.g., RUN22e). This becomes the positional PREFIX for clean_comp_arch.sh.
PREFIX_DEFAULT="RUN22e"
# First part index in the global range (1-based). Example: 1 -> part0001.
PART_START_DEFAULT=1
# Set ONE of these global range controls:
# - PART_END_DEFAULT: inclusive last part index (e.g., 107 -> part0107)
# - NUM_PARTS_DEFAULT: number of parts starting at PART_START_DEFAULT
# If PART_END_DEFAULT is non-empty, it is used; otherwise NUM_PARTS_DEFAULT is used.
PART_END_DEFAULT=""
NUM_PARTS_DEFAULT=117
# Zero-padding width for part index (4 => part0001, part0107).
PART_WIDTH_DEFAULT=4
# Number of Slurm array tasks to create. More tasks = smaller chunks per task.
ARRAY_COUNT_DEFAULT=12
# Optional Slurm array throttle (%N). Limits how many array tasks run at once.
# Set empty ("") to allow all array tasks to run concurrently.
ARRAY_MAX_PARALLEL_DEFAULT=6
# Wrapper mode:
# - submit: submit an sbatch array job (normal use)
# - run: execute one array-task chunk directly (testing / inside array task)
MODE_DEFAULT="submit"   # submit | run
# 1 = print commands only, 0 = execute
DRY_RUN_DEFAULT=0

# Extra sbatch args for the wrapper submission (optional).
# Examples:
#   "--account=ABC123"
#   "--qos=normal"
#   "--partition=cpu-small"
#   "--time=3-00:00:00"
SBATCH_EXTRA_ARGS_DEFAULT=(
  # "--account=your_account"
)

# Default args forwarded to clean_comp_arch.sh.
# CLI args after '--' are appended and therefore override duplicates.
# Put your common workflow here (full pass vs repack-only, workers, directories).
# Examples:
# - Full pass: leave as stage+clean+compress+repack (script defaults)
# - Repack-only: add "--no-stage" "--no-clean" "--no-compress" "--repack"
# - Preserve stage/source during retries: set env when launching wrapper:
#     CLEAN_STAGE_AFTER=0 DELETE_SOURCE_AFTER_REPACK=0 bash clean_comp_arch_array.sh
CLEAN_ARGS_DEFAULT=(
)

usage() {
  cat <<'USAGE'
Usage:
  clean_comp_arch_array.sh [PREFIX] [wrapper-options] -- [clean_comp_arch.sh options]

  PREFIX is optional if set in the script Config section.

Wrapper options:
  --clean-script PATH         Path to clean_comp_arch.sh (default: same directory)
  --part-start N              First part number (1-based, default: 1)
  --part-end N                Last part number (inclusive)
  --num-parts N               Number of parts (alternative to --part-end)
  --part-width N              Zero-pad width for part index (default: 4)
  --array-count N             Number of array tasks (default: 8)
  --array-max-parallel N      Add Slurm array throttle: --array=...%N
  --mode MODE                 submit|run (default: submit; run is internal/testing)
  --submit                    Alias of --mode submit
  --run                       Alias of --mode run
  --sbatch-arg ARG            Extra sbatch arg (repeatable)
  --dry-run                   Print commands only
  -h, --help

clean_comp_arch.sh options:
  Pass after '--'. The wrapper will append:
    <PREFIX> --part-glob <exact-part> --mode interactive
  so wrapper-controlled part selection always wins.

Examples:
  1) Submit 12-array job over parts 1..107, run 4 array tasks concurrently:
     bash clean_comp_arch_array.sh RUN22e \
       --part-start 1 --num-parts 107 --array-count 12 --array-max-parallel 4 \
       -- --extract-root ./extracted --stage-root ./stage --repack-dir ./repacked \
          --workers 4

  2) Local test of one array task split logic (no Slurm):
     SLURM_ARRAY_TASK_ID=0 bash clean_comp_arch_array.sh RUN22e \
       --part-start 1 --num-parts 10 --array-count 3 --run --dry-run -- \
       --extract-root ./extracted --stage-root ./stage --repack-dir ./repacked
USAGE
}

log() { echo "$*"; }
warn() { echo "[WARN] $*" >&2; }
err() { echo "[ERROR] $*" >&2; }

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    err "Missing required command: $1"
    exit 1
  }
}

run_cmd() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY] $*"
  else
    "$@"
  fi
}

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PREFIX="$PREFIX_DEFAULT"
CLEAN_SCRIPT="$script_dir/clean_comp_arch.sh"
PART_START="$PART_START_DEFAULT"
PART_END="$PART_END_DEFAULT"
NUM_PARTS="$NUM_PARTS_DEFAULT"
PART_WIDTH="$PART_WIDTH_DEFAULT"
ARRAY_COUNT="$ARRAY_COUNT_DEFAULT"
ARRAY_MAX_PARALLEL="$ARRAY_MAX_PARALLEL_DEFAULT"
MODE="$MODE_DEFAULT"  # submit | run
DRY_RUN="$DRY_RUN_DEFAULT"
SBATCH_EXTRA_ARGS=()
CLEAN_ARGS=()
# Bash 3.2 + set -u can error on empty-array expansion. Copy defaults with nounset
# temporarily disabled so empty config arrays remain valid.
set +u
SBATCH_EXTRA_ARGS=("${SBATCH_EXTRA_ARGS_DEFAULT[@]}")
CLEAN_ARGS=("${CLEAN_ARGS_DEFAULT[@]}")
set -u

if [[ $# -gt 0 && "${1:-}" != -* ]]; then
  PREFIX="$1"
  shift
fi

while [[ $# -gt 0 ]]; do
  case "$1" in
    --clean-script) CLEAN_SCRIPT="$2"; shift 2 ;;
    --part-start) PART_START="$2"; shift 2 ;;
    --part-end) PART_END="$2"; shift 2 ;;
    --num-parts) NUM_PARTS="$2"; shift 2 ;;
    --part-width) PART_WIDTH="$2"; shift 2 ;;
    --array-count) ARRAY_COUNT="$2"; shift 2 ;;
    --array-max-parallel) ARRAY_MAX_PARALLEL="$2"; shift 2 ;;
    --mode) MODE="$2"; shift 2 ;;
    --submit) MODE="submit"; shift ;;
    --run) MODE="run"; shift ;;
    --sbatch-arg) SBATCH_EXTRA_ARGS+=("$2"); shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --) shift; CLEAN_ARGS+=("$@"); break ;;
    -h|--help) usage; exit 0 ;;
    *) err "Unknown wrapper arg: $1"; usage; exit 2 ;;
  esac
done

[[ -n "$PREFIX" ]] || { err "PREFIX is required (set config or pass positional PREFIX)"; usage; exit 2; }
[[ "$MODE" == "submit" || "$MODE" == "run" ]] || { err "--mode must be submit|run"; exit 2; }
[[ "$PART_START" =~ ^[0-9]+$ && "$PART_START" -ge 1 ]] || { err "--part-start must be >=1"; exit 2; }
[[ "$PART_WIDTH" =~ ^[0-9]+$ && "$PART_WIDTH" -ge 1 ]] || { err "--part-width must be >=1"; exit 2; }
[[ "$ARRAY_COUNT" =~ ^[0-9]+$ && "$ARRAY_COUNT" -ge 1 ]] || { err "--array-count must be >=1"; exit 2; }

if [[ -n "$PART_END" && -n "$NUM_PARTS" ]]; then
  err "Use only one of --part-end or --num-parts"
  exit 2
fi
if [[ -z "$PART_END" && -z "$NUM_PARTS" ]]; then
  err "Provide one of --part-end or --num-parts"
  exit 2
fi
if [[ -n "$PART_END" ]]; then
  [[ "$PART_END" =~ ^[0-9]+$ && "$PART_END" -ge "$PART_START" ]] || { err "--part-end must be >= --part-start"; exit 2; }
  TOTAL_PARTS=$((PART_END - PART_START + 1))
else
  [[ "$NUM_PARTS" =~ ^[0-9]+$ && "$NUM_PARTS" -ge 1 ]] || { err "--num-parts must be >=1"; exit 2; }
  TOTAL_PARTS="$NUM_PARTS"
  PART_END=$((PART_START + NUM_PARTS - 1))
fi

if [[ -n "$ARRAY_MAX_PARALLEL" ]]; then
  [[ "$ARRAY_MAX_PARALLEL" =~ ^[0-9]+$ && "$ARRAY_MAX_PARALLEL" -ge 1 ]] || { err "--array-max-parallel must be >=1"; exit 2; }
fi

[[ -f "$CLEAN_SCRIPT" ]] || { err "clean script not found: $CLEAN_SCRIPT"; exit 1; }

calc_task_chunk() {
  # Sets TASK_CHUNK_SIZE and TASK_OFFSET for given task id.
  local task_id="$1"
  local base rem
  base=$((TOTAL_PARTS / ARRAY_COUNT))
  rem=$((TOTAL_PARTS % ARRAY_COUNT))
  if [[ "$task_id" -lt "$rem" ]]; then
    TASK_CHUNK_SIZE=$((base + 1))
    TASK_OFFSET=$((task_id * (base + 1)))
  else
    TASK_CHUNK_SIZE=$base
    TASK_OFFSET=$((rem * (base + 1) + (task_id - rem) * base))
  fi
}

run_one_part() {
  local idx="$1"
  local part_name part_glob cmd
  printf -v part_name "%s.part%0*d" "$PREFIX" "$PART_WIDTH" "$idx"
  part_glob="$part_name"

  cmd=(bash "$CLEAN_SCRIPT" "$PREFIX")
  if [[ "${#CLEAN_ARGS[@]}" -gt 0 ]]; then
    cmd+=("${CLEAN_ARGS[@]}")
  fi
  # Force exact one-part selection and interactive mode inside wrapper tasks.
  cmd+=(--part-glob "$part_glob" --mode interactive)

  log "  -> $part_name"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY] ${cmd[*]}"
    return 0
  fi
  "${cmd[@]}"
}

run_mode() {
  local task_id="${SLURM_ARRAY_TASK_ID:-0}"
  [[ "$task_id" =~ ^[0-9]+$ ]] || { err "SLURM_ARRAY_TASK_ID must be integer (got: $task_id)"; exit 2; }
  if [[ "$task_id" -ge "$ARRAY_COUNT" ]]; then
    warn "task id $task_id >= array-count $ARRAY_COUNT; nothing to do"
    return 0
  fi

  calc_task_chunk "$task_id"
  if [[ "$TASK_CHUNK_SIZE" -le 0 ]]; then
    log "Task $task_id: empty assignment (array-count > total-parts); skip"
    return 0
  fi

  local first_idx last_idx idx rc=0
  first_idx=$((PART_START + TASK_OFFSET))
  last_idx=$((first_idx + TASK_CHUNK_SIZE - 1))

  log "Array task id         : $task_id / $((ARRAY_COUNT - 1))"
  log "Prefix                : $PREFIX"
  log "Part range (global)   : ${PREFIX}.part$(printf "%0*d" "$PART_WIDTH" "$PART_START") .. ${PREFIX}.part$(printf "%0*d" "$PART_WIDTH" "$PART_END")"
  log "Assigned part idx     : $first_idx .. $last_idx (count=$TASK_CHUNK_SIZE)"
  log "clean_comp_arch.sh    : $CLEAN_SCRIPT"
  if [[ "${#CLEAN_ARGS[@]}" -gt 0 ]]; then
    log "Forwarded args        : ${CLEAN_ARGS[*]}"
  fi
  echo

  for ((idx=first_idx; idx<=last_idx; idx++)); do
    if ! run_one_part "$idx"; then
      rc=1
      warn "failed part index: $idx"
    fi
  done
  return "$rc"
}

submit_mode() {
  need_cmd sbatch

  local arr_spec="0-$((ARRAY_COUNT - 1))"
  if [[ -n "$ARRAY_MAX_PARALLEL" ]]; then
    arr_spec="${arr_spec}%${ARRAY_MAX_PARALLEL}"
  fi

  local cmd=(sbatch --array "$arr_spec")
  if [[ "${#SBATCH_EXTRA_ARGS[@]}" -gt 0 ]]; then
    cmd+=("${SBATCH_EXTRA_ARGS[@]}")
  fi
  cmd+=("$0" "$PREFIX"
        --clean-script "$CLEAN_SCRIPT"
        --part-start "$PART_START"
        --part-end "$PART_END"
        --part-width "$PART_WIDTH"
        --array-count "$ARRAY_COUNT"
        --mode run)
  if [[ "$DRY_RUN" -eq 1 ]]; then
    cmd+=(--dry-run)
  fi
  if [[ "${#CLEAN_ARGS[@]}" -gt 0 ]]; then
    cmd+=(-- "${CLEAN_ARGS[@]}")
  fi

  log "Submit mode"
  log "Prefix                : $PREFIX"
  log "Part count            : $TOTAL_PARTS ($PART_START..$PART_END)"
  log "Array tasks           : $ARRAY_COUNT"
  log "Array spec            : $arr_spec"
  if [[ "${#CLEAN_ARGS[@]}" -gt 0 ]]; then
    log "Forwarded clean args  : ${CLEAN_ARGS[*]}"
  fi
  echo

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY] ${cmd[*]}"
  else
    "${cmd[@]}"
  fi
}

main() {
  if [[ "$MODE" == "submit" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" && -n "${SLURM_ARRAY_TASK_ID:-}" ]]; then
      # Already inside array allocation; execute directly.
      run_mode
    elif [[ -n "${SLURM_JOB_ID:-}" ]]; then
      warn "Already in non-array Slurm allocation; running directly."
      run_mode
    else
      submit_mode
    fi
  else
    run_mode
  fi
}

main "$@"
