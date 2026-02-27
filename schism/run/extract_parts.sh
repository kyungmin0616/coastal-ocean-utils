#!/usr/bin/env bash
#SBATCH -J schism_extract_parts
#SBATCH -o schism_extract_parts.%j.out
#SBATCH -e schism_extract_parts.%j.err
#SBATCH -p cpu-small
#SBATCH -N 1
#SBATCH -n 8
#SBATCH -t 2-00:00:00

# Extract SCHISM part archives into the layout expected by clean_comp_arch.sh.
#
# Expected archive layout:
#   <ARCHIVES_ROOT>/<PREFIX>/<PREFIX>.partXXXX.tar.gz
#   <ARCHIVES_ROOT>/<PREFIX>/<PREFIX>.partXXXX.tar.gz.sha256
#
# Output layout:
#   <EXTRACT_ROOT>/<PREFIX>.partXXXX/<RUN_SUBDIR>/...
#
# Examples:
#   bash extract_parts.sh RUN22e --archives-root ./archives_resume --extract-root ./extracted
#   bash extract_parts.sh --prefix-glob 'RUN22*' --archives-root ./archives_resume
#   bash extract_parts.sh RUN22e --dry-run

set -euo pipefail

PREFIX=""
PREFIX_GLOB="${PREFIX_GLOB:-RUN*}"
ARCHIVES_ROOT="${ARCHIVES_ROOT:-./archives_resume}"
EXTRACT_ROOT="${EXTRACT_ROOT:-./extracted}"
RUN_SUBDIR=""
PART_GLOB=""

VERIFY_SHA=1
REQUIRE_SHA=1
SKIP_EXISTING=1
OVERWRITE_EXTRACT=0
NWORKERS="${NWORKERS:-1}"
MODE="${MODE:-interactive}"   # interactive | submit
DRY_RUN=0
VERBOSE=1
SBATCH_EXTRA_ARGS=()

TOTAL_TARS=0
TOTAL_EXTRACTED=0
TOTAL_SKIPPED=0
TOTAL_FAILED=0
TOTAL_SHA_FAILED=0
TOTAL_SHA_MISSING=0

usage() {
  cat <<'USAGE'
Usage:
  extract_parts.sh [PREFIX] [options]

Positional:
  PREFIX                       One run prefix (e.g. RUN22e). If omitted, all matching --prefix-glob are processed.

Options:
  --archives-root DIR          Root containing run folders with part tar files (default: ./archives_resume)
  --extract-root DIR           Destination root for extracted parts (default: ./extracted)
  --run-subdir NAME            Expected run subdir in tar (default: same as run folder name)
  --part-glob GLOB             Tar glob inside each run folder (default: <PREFIX>.part*.tar.gz)
  --prefix-glob GLOB           Run folder glob when PREFIX is omitted (default: RUN*)
  --workers N                  Parallel extractions (default: 1)
  --mode MODE                  interactive|submit (default: interactive)
  --submit                     Alias of --mode submit
  --sbatch-arg ARG             Extra argument passed to sbatch (repeatable)
  --dry-run

Checksum:
  --verify-sha                 Verify <tar>.sha256 before extraction (default: on)
  --no-verify-sha              Disable checksum verification
  --require-sha                Fail extraction if sha file is missing (default: on)
  --no-require-sha             Allow missing sha file

Extract policy:
  --skip-existing              Skip if destination part dir already has files (default: on)
  --no-skip-existing           Do not skip existing extraction output
  --overwrite-extract          Remove destination part dir before extraction

Other:
  --quiet
  -h, --help
USAGE
}

log() {
  if [[ "$VERBOSE" -eq 1 ]]; then
    echo "$*"
  fi
}

warn() {
  echo "[WARN] $*" >&2
}

err() {
  echo "[ERROR] $*" >&2
}

run_cmd() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY] $*"
  else
    "$@"
  fi
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    err "Missing required command: $1"
    exit 1
  }
}

have_sha_tool() {
  command -v sha256sum >/dev/null 2>&1 || command -v shasum >/dev/null 2>&1
}

verify_sha_for_tar() {
  local tar_f="$1"
  local sha_f="${tar_f}.sha256"
  local tar_dir tar_base sha_base
  tar_dir="$(dirname "$tar_f")"
  tar_base="$(basename "$tar_f")"
  sha_base="$(basename "$sha_f")"

  if [[ "$VERIFY_SHA" -ne 1 ]]; then
    return 0
  fi

  if [[ ! -f "$sha_f" ]]; then
    TOTAL_SHA_MISSING=$((TOTAL_SHA_MISSING + 1))
    if [[ "$REQUIRE_SHA" -eq 1 ]]; then
      warn "missing checksum file: $sha_f"
      return 2
    fi
    warn "checksum file missing (allowed): $sha_f"
    return 0
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY] verify checksum: $sha_f"
    return 0
  fi

  if command -v sha256sum >/dev/null 2>&1; then
    if (cd "$tar_dir" && sha256sum -c "$sha_base" >/dev/null); then
      return 0
    fi
    warn "checksum verify failed: $tar_f"
    TOTAL_SHA_FAILED=$((TOTAL_SHA_FAILED + 1))
    return 1
  fi

  if command -v shasum >/dev/null 2>&1; then
    local expected actual
    expected="$(awk 'NR==1{print $1}' "$sha_f")"
    actual="$(shasum -a 256 "$tar_f" | awk '{print $1}')"
    if [[ -n "$expected" && "$expected" == "$actual" ]]; then
      return 0
    fi
    warn "checksum verify failed: $tar_f"
    TOTAL_SHA_FAILED=$((TOTAL_SHA_FAILED + 1))
    return 1
  fi

  warn "no sha tool available; cannot verify checksum"
  return 1
}

extract_one_tar() {
  # Return codes:
  #   0 extracted
  #  10 skipped existing
  #   1 failed
  local tar_f="$1"
  local run_name="$2"
  local partbase dst_part
  partbase="$(basename "$tar_f" .tar.gz)"
  dst_part="$EXTRACT_ROOT/$partbase"

  if [[ "$OVERWRITE_EXTRACT" -eq 1 && -d "$dst_part" ]]; then
    run_cmd rm -rf "$dst_part"
  fi

  if [[ "$SKIP_EXISTING" -eq 1 && -d "$dst_part" ]]; then
    if find "$dst_part" -mindepth 1 -print -quit | grep -q .; then
      log "  [skip] already extracted: $dst_part"
      return 10
    fi
  fi

  if ! verify_sha_for_tar "$tar_f"; then
    return 1
  fi

  run_cmd mkdir -p "$dst_part"

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY] tar -xzf \"$tar_f\" -C \"$dst_part\""
    return 0
  fi

  if ! tar -xzf "$tar_f" -C "$dst_part"; then
    warn "extract failed: $tar_f"
    return 1
  fi

  local expected_subdir="$RUN_SUBDIR"
  if [[ -z "$expected_subdir" ]]; then
    expected_subdir="$run_name"
  fi
  if [[ ! -d "$dst_part/$expected_subdir" ]]; then
    warn "extracted but expected run dir not found: $dst_part/$expected_subdir"
  fi

  return 0
}

process_run_dir() {
  local run_dir="$1"
  local run_name run_subdir_local part_glob_local
  run_name="$(basename "$run_dir")"
  run_subdir_local="$RUN_SUBDIR"
  part_glob_local="$PART_GLOB"

  if [[ -z "$run_subdir_local" ]]; then
    run_subdir_local="$run_name"
  fi
  if [[ -z "$part_glob_local" ]]; then
    part_glob_local="${run_name}.part*.tar.gz"
  fi

  echo "========================================"
  echo "Run: $run_name"
  echo "========================================"
  log "  source dir : $run_dir"
  log "  part glob  : $part_glob_local"
  log "  run subdir : $run_subdir_local"

  local tar_files=()
  local t=""
  while IFS= read -r t; do
    tar_files+=("$t")
  done < <(find "$run_dir" -maxdepth 1 -type f -name "$part_glob_local" | sort)

  if [[ "${#tar_files[@]}" -eq 0 ]]; then
    warn "no tar files found in $run_dir with pattern $part_glob_local"
    return 0
  fi

  local i rc=0 one_rc
  TOTAL_TARS=$((TOTAL_TARS + ${#tar_files[@]}))

  if [[ "$NWORKERS" -le 1 ]]; then
    for ((i=0; i<${#tar_files[@]}; i++)); do
      extract_one_tar "${tar_files[$i]}" "$run_name"
      one_rc=$?
      case "$one_rc" in
        0) TOTAL_EXTRACTED=$((TOTAL_EXTRACTED + 1)) ;;
        10) TOTAL_SKIPPED=$((TOTAL_SKIPPED + 1)) ;;
        *) TOTAL_FAILED=$((TOTAL_FAILED + 1)); rc=1 ;;
      esac
    done
  else
    local pids=()
    local files=()
    local idx
    for ((i=0; i<${#tar_files[@]}; i++)); do
      extract_one_tar "${tar_files[$i]}" "$run_name" &
      pids+=("$!")
      files+=("${tar_files[$i]}")
      if [[ "${#pids[@]}" -ge "$NWORKERS" ]]; then
        if wait "${pids[0]}"; then
          TOTAL_EXTRACTED=$((TOTAL_EXTRACTED + 1))
        else
          one_rc=$?
          if [[ "$one_rc" -eq 10 ]]; then
            TOTAL_SKIPPED=$((TOTAL_SKIPPED + 1))
          else
            TOTAL_FAILED=$((TOTAL_FAILED + 1))
            rc=1
            warn "worker failed: ${files[0]}"
          fi
        fi
        pids=("${pids[@]:1}")
        files=("${files[@]:1}")
      fi
    done
    for idx in "${!pids[@]}"; do
      if wait "${pids[$idx]}"; then
        TOTAL_EXTRACTED=$((TOTAL_EXTRACTED + 1))
      else
        one_rc=$?
        if [[ "$one_rc" -eq 10 ]]; then
          TOTAL_SKIPPED=$((TOTAL_SKIPPED + 1))
        else
          TOTAL_FAILED=$((TOTAL_FAILED + 1))
          rc=1
          warn "worker failed: ${files[$idx]}"
        fi
      fi
    done
  fi

  return "$rc"
}

parse_args() {
  if [[ $# -gt 0 && "${1:-}" != -* ]]; then
    PREFIX="$1"
    shift
  fi

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --archives-root) ARCHIVES_ROOT="$2"; shift 2 ;;
      --extract-root) EXTRACT_ROOT="$2"; shift 2 ;;
      --run-subdir) RUN_SUBDIR="$2"; shift 2 ;;
      --part-glob) PART_GLOB="$2"; shift 2 ;;
      --prefix-glob) PREFIX_GLOB="$2"; shift 2 ;;
      --workers) NWORKERS="$2"; shift 2 ;;
      --mode) MODE="$2"; shift 2 ;;
      --submit) MODE="submit"; shift ;;
      --sbatch-arg) SBATCH_EXTRA_ARGS+=("$2"); shift 2 ;;
      --dry-run) DRY_RUN=1; shift ;;

      --verify-sha) VERIFY_SHA=1; shift ;;
      --no-verify-sha) VERIFY_SHA=0; shift ;;
      --require-sha) REQUIRE_SHA=1; shift ;;
      --no-require-sha) REQUIRE_SHA=0; shift ;;

      --skip-existing) SKIP_EXISTING=1; shift ;;
      --no-skip-existing) SKIP_EXISTING=0; shift ;;
      --overwrite-extract) OVERWRITE_EXTRACT=1; shift ;;

      --quiet) VERBOSE=0; shift ;;
      -h|--help) usage; exit 0 ;;
      *) err "Unknown arg: $1"; usage; exit 2 ;;
    esac
  done

  [[ "$MODE" == "interactive" || "$MODE" == "submit" ]] || {
    err "--mode must be interactive or submit"
    exit 2
  }
}

validate() {
  [[ -d "$ARCHIVES_ROOT" ]] || { err "ARCHIVES_ROOT not found: $ARCHIVES_ROOT"; exit 1; }
  [[ "$NWORKERS" =~ ^[0-9]+$ && "$NWORKERS" -ge 1 ]] || { err "--workers must be >=1"; exit 1; }

  need_cmd find
  need_cmd tar

  if [[ "$VERIFY_SHA" -eq 1 && "$DRY_RUN" -ne 1 ]]; then
    have_sha_tool || { err "No sha tool found (need sha256sum or shasum)."; exit 1; }
  fi

  run_cmd mkdir -p "$EXTRACT_ROOT"
}

main() {
  local raw_args=("$@")
  parse_args "$@"

  if [[ "$MODE" == "submit" ]]; then
    if [[ -n "${SLURM_JOB_ID:-}" ]]; then
      warn "Already in Slurm allocation (SLURM_JOB_ID=${SLURM_JOB_ID}); switching to interactive mode."
      MODE="interactive"
    else
      if [[ "$DRY_RUN" -ne 1 ]]; then
        need_cmd sbatch
      fi
      local forwarded=()
      local i=0
      local n="${#raw_args[@]}"
      while [[ "$i" -lt "$n" ]]; do
        case "${raw_args[$i]}" in
          --mode)
            i=$((i + 2))
            continue
            ;;
          --submit)
            i=$((i + 1))
            continue
            ;;
          --sbatch-arg)
            i=$((i + 2))
            continue
            ;;
          *)
            forwarded+=("${raw_args[$i]}")
            i=$((i + 1))
            ;;
        esac
      done
      forwarded+=(--mode interactive)

      local cmd=(sbatch)
      if [[ "${#SBATCH_EXTRA_ARGS[@]}" -gt 0 ]]; then
        cmd+=("${SBATCH_EXTRA_ARGS[@]}")
      fi
      cmd+=("$0")
      cmd+=("${forwarded[@]}")

      if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[DRY] ${cmd[*]}"
      else
        echo "Submitting job:"
        echo "  ${cmd[*]}"
        "${cmd[@]}"
      fi
      return 0
    fi
  fi

  validate

  echo "PREFIX                : ${PREFIX:-<all by prefix-glob>}"
  echo "PREFIX_GLOB           : $PREFIX_GLOB"
  echo "ARCHIVES_ROOT         : $ARCHIVES_ROOT"
  echo "EXTRACT_ROOT          : $EXTRACT_ROOT"
  echo "RUN_SUBDIR            : ${RUN_SUBDIR:-<auto from run folder>}"
  echo "PART_GLOB             : ${PART_GLOB:-<auto from run folder>}"
  echo "MODE                  : $MODE"
  echo "workers               : $NWORKERS"
  echo "verify sha            : $VERIFY_SHA (require=$REQUIRE_SHA)"
  echo "skip existing         : $SKIP_EXISTING"
  echo "overwrite extract     : $OVERWRITE_EXTRACT"
  echo "dry run               : $DRY_RUN"
  echo

  local run_dirs=()
  local d
  if [[ -n "$PREFIX" ]]; then
    d="$ARCHIVES_ROOT/$PREFIX"
    [[ -d "$d" ]] || { err "Run dir not found for PREFIX: $d"; exit 1; }
    run_dirs+=("$d")
  else
    while IFS= read -r d; do
      run_dirs+=("$d")
    done < <(find "$ARCHIVES_ROOT" -maxdepth 1 -mindepth 1 -type d -name "$PREFIX_GLOB" | sort)
    [[ "${#run_dirs[@]}" -gt 0 ]] || {
      err "No run dirs matched: $ARCHIVES_ROOT/$PREFIX_GLOB"
      exit 1
    }
  fi

  local rc=0
  for d in "${run_dirs[@]}"; do
    process_run_dir "$d" || rc=1
  done

  echo
  echo "=== Summary ==="
  echo "tars found            : $TOTAL_TARS"
  echo "extracted             : $TOTAL_EXTRACTED"
  echo "skipped existing      : $TOTAL_SKIPPED"
  echo "failed                : $TOTAL_FAILED"
  echo "sha missing           : $TOTAL_SHA_MISSING"
  echo "sha failed            : $TOTAL_SHA_FAILED"
  echo "DONE."

  return "$rc"
}

main "$@"

