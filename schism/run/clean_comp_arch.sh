#!/usr/bin/env bash
#SBATCH -J schism_clean_comp_repack
#SBATCH -o schism_clean_comp_repack.%j.out
#SBATCH -e schism_clean_comp_repack.%j.err
#SBATCH -p cpu-small
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 5-00:00:00

# Clean, compress, and optionally re-archive extracted SCHISM run-part folders.
#
# Expected extracted layout:
#   <EXTRACT_ROOT>/<PREFIX>.partXXXX/<RUN_SUBDIR>/
#
# Typical sbatch usage:
#   sbatch clean_comp_arch.sh RUN22e
#
# Dry-run first:
#   bash clean_comp_arch.sh RUN22e --dry-run
#
# Key safety defaults:
#   - Compression workers default to 1 to reduce temp-space pressure.
#   - Stage + repack are separated from source extraction tree.
#   - Stage is built selectively (no full-tree rsync required).
#   - Already-deflated NetCDF files are skipped by default.

set -euo pipefail

# -------------------------
# Defaults (override by env or CLI)
# -------------------------
PREFIX_DEFAULT="RUN22a"
PREFIX="${PREFIX_DEFAULT}"

EXTRACT_ROOT="${EXTRACT_ROOT:-./extracted}"
STAGE_ROOT="${STAGE_ROOT:-./stage}"
REPACK_DIR="${REPACK_DIR:-./repacked}"

RUN_SUBDIR=""
PART_GLOB=""

DO_STAGE=1
DO_CLEAN=1
DO_COMPRESS=1
DO_REPACK=1
DRY_RUN=0
MODE="${MODE:-submit}"   # interactive | submit
SBATCH_EXTRA_ARGS=()

# Compression settings
COMPRESS_SCOPE="all"        # outputs | all
COMPRESS_NAME_GLOB="*.nc"       # e.g. horizontalVelX_*.nc
DEFLATE_LEVEL="${DEFLATE_LEVEL:-9}"
USE_SHUFFLE=1
NWORKERS="${NWORKERS:-4}"      # keep small by default to avoid disk pressure
SKIP_ALREADY_COMPRESSED=1
VERIFY_DEFLATE=1

# Space guard (MB): skip compression if available scratch space in target dir
# is less than (input_file_size + MIN_FREE_MB).
MIN_FREE_MB="${MIN_FREE_MB:-0}"

CLEAN_STAGE_AFTER="${CLEAN_STAGE_AFTER:-1}"
DELETE_SOURCE_AFTER_REPACK="${DELETE_SOURCE_AFTER_REPACK:-1}"
OVERWRITE_ARCHIVE=0
VERBOSE=1

# Cleanup pattern config (edit in this section)
# - Top-level patterns apply under: <RUN_SUBDIR>/
# - Outputs patterns apply under:   <RUN_SUBDIR>/outputs/
DELETE_TOP_PATTERNS=(
  "er.*"
  "ot.*"
  "core.*"
)
DELETE_OUTPUTS_PATTERNS=(
  "hotstart*.nc"
)
DELETE_PARENT_HOTSTARTS=1
KEEP_PARENT_HOTSTART="hotstart.nc"

# Compression skip config (edit in this section)
# - If COMPRESS_SKIP_CLEAN_MATCHES=1, any file matched by clean rules is not compressed.
# - COMPRESS_EXCLUDE_BASENAME_PATTERNS apply by basename in both parent and outputs trees.
# - If nccopy fails and COMPRESS_FAIL_COPY_ON_ERROR=1, source file is copied as-is.
COMPRESS_SKIP_CLEAN_MATCHES=1
COMPRESS_EXCLUDE_BASENAME_PATTERNS=(
)
COMPRESS_FAIL_COPY_ON_ERROR=1

# Summary counters
TOTAL_PARTS=0
TOTAL_PARTS_OK=0
TOTAL_FILES_REMOVED=0
TOTAL_NC_FOUND=0
TOTAL_NC_COMPRESSED=0
TOTAL_NC_SKIPPED=0
TOTAL_NC_FAILED=0
TOTAL_NC_FALLBACK_COPIED=0
TOTAL_ARCHIVES=0
TOTAL_SOURCE_DELETED=0

# Last repack state (set by repack_part)
REPACK_LAST_TAR=""
REPACK_LAST_SHA=""
REPACK_LAST_WRITTEN=0

usage() {
  cat <<'USAGE'
Usage:
  clean_comp_arch.sh [PREFIX] [options]

Positional:
  PREFIX                       Run name prefix (default: RUN22e)

Options:
  --extract-root DIR           Source extracted root (default: ./extracted)
  --stage-root DIR             Stage root (default: ./stage)
  --repack-dir DIR             Output tar root (default: ./repacked)
  --run-subdir NAME            Run folder inside each part (default: PREFIX)
  --part-glob GLOB             Part directory glob (default: <PREFIX>.part*)
  --mode MODE                  interactive|submit (default: interactive)
  --submit                     Alias of --mode submit
  --sbatch-arg ARG             Extra argument passed to sbatch (repeatable)

  --stage / --no-stage
  --clean / --no-clean
  --compress / --no-compress
  --repack / --no-repack
  --dry-run

Compression:
  --compress-scope MODE        outputs|all (default: all)
  --compress-name-glob GLOB    default: *.nc
  --deflate N                  nccopy deflate 0..9 (default: 9)
  --no-shuffle                 disable nccopy -s
  --workers N                  parallel compress workers per part (default: 4)
  --no-skip-compressed         recompress even if already deflated
  --no-verify-deflate          skip metadata verification step
  --min-free-mb N              free-space guard buffer (default: 1024)

Archive:
  --overwrite-archive          overwrite existing tar.gz output
  --clean-stage-after          remove staged part after successful repack
  --delete-source-after-repack remove extracted part only after verification

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

fsize_bytes() {
  local f="$1"
  stat -c%s "$f" 2>/dev/null || stat -f%z "$f"
}

avail_kb() {
  local d="$1"
  df -Pk "$d" | awk 'NR==2{print $4}'
}

strip_wrapping_quotes() {
  # Normalize basename for matching when some tools display or preserve
  # wrapped quote chars, e.g. 'hotstart_it=0.nc' or "file.nc".
  local s="$1"
  if [[ ${#s} -ge 2 ]]; then
    if [[ "$s" == \'*\' ]]; then
      s="${s:1:${#s}-2}"
    elif [[ "$s" == \"*\" ]]; then
      s="${s:1:${#s}-2}"
    fi
  fi
  printf "%s" "$s"
}

sha_write() {
  local f="$1"
  local out="$2"
  if command -v sha256sum >/dev/null 2>&1; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "[DRY] (cd \"$(dirname "$f")\" && sha256sum \"$(basename "$f")\" > \"$out\")"
    else
      (cd "$(dirname "$f")" && sha256sum "$(basename "$f")" > "$out")
    fi
  elif command -v shasum >/dev/null 2>&1; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "[DRY] (cd \"$(dirname "$f")\" && shasum -a 256 \"$(basename "$f")\" > \"$out\")"
    else
      (cd "$(dirname "$f")" && shasum -a 256 "$(basename "$f")" > "$out")
    fi
  else
    warn "No sha256 tool found. Skip checksum for: $f"
  fi
}

sha_verify() {
  local tar_f="$1"
  local sha_f="$2"

  [[ -f "$sha_f" ]] || {
    warn "checksum file missing: $sha_f"
    return 1
  }

  if command -v sha256sum >/dev/null 2>&1; then
    (cd "$(dirname "$tar_f")" && sha256sum -c "$(basename "$sha_f")" >/dev/null)
    return $?
  fi

  if command -v shasum >/dev/null 2>&1; then
    local expected actual
    expected="$(awk 'NR==1{print $1}' "$sha_f")"
    actual="$(shasum -a 256 "$tar_f" | awk '{print $1}')"
    [[ -n "$expected" && "$expected" == "$actual" ]]
    return $?
  fi

  warn "No sha verification tool found."
  return 1
}

remove_files() {
  local desc="$1"
  shift
  local files=("$@")
  local n=0
  local f
  for f in "${files[@]}"; do
    [[ -e "$f" ]] || continue
    n=$((n + 1))
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "[DRY] rm -f \"$f\""
    else
      rm -f -- "$f"
    fi
  done
  if [[ "$n" -gt 0 ]]; then
    log "    removed $n file(s): $desc"
  fi
  TOTAL_FILES_REMOVED=$((TOTAL_FILES_REMOVED + n))
}

cleanup_orphan_tmps() {
  local part_stage="$1"
  # common leftovers from interrupted nccopy runs
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY] find \"$part_stage\" -type f \\( -name '*.nccopy_tmp.*' -o -name '.*.nccopy.tmp.*' -o -name '.horizontalVelX_*.nc.*' \\) -delete"
    return
  fi
  find "$part_stage" -type f \( \
    -name '*.nccopy_tmp.*' -o \
    -name '.*.nccopy.tmp.*' -o \
    -name '.horizontalVelX_*.nc.*' \
  \) -delete || true
}

cleanup_hotstarts_and_logs() {
  local rundir="$1"
  local outdir="$rundir/outputs"

  log "  [clean] $rundir"

  local remove_list=()
  local f rel

  # Top-level files under RUN_SUBDIR
  while IFS= read -r f; do
    rel="$RUN_SUBDIR/$(basename "$f")"
    if should_skip_clean_rel "$rel"; then
      remove_list+=("$f")
    fi
  done < <(find "$rundir" -maxdepth 1 -type f -print)

  # Files under outputs/
  if [[ -d "$outdir" ]]; then
    while IFS= read -r f; do
      rel="$RUN_SUBDIR/outputs/${f#$outdir/}"
      if should_skip_clean_rel "$rel"; then
        remove_list+=("$f")
      fi
    done < <(find "$outdir" -type f -print)
  fi

  remove_files "config clean rules" "${remove_list[@]:-}"

  log "    remaining parent hotstart_it files:"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY] (cd \"$rundir\" && ls -1 hotstart_it=*.nc 2>/dev/null || true)"
  else
    (cd "$rundir" && ls -1 hotstart_it=*.nc 2>/dev/null || true)
  fi
}

is_deflated_nc() {
  local nc="$1"
  # If _DeflateLevel value is >0, treat as already compressed.
  ncdump -hs "$nc" 2>/dev/null | awk '
    /_DeflateLevel[[:space:]]*=/ {
      v=$0
      sub(/.*_DeflateLevel[[:space:]]*=[[:space:]]*/, "", v)
      sub(/[[:space:]]*;.*/, "", v)
      gsub(/[^0-9]/, "", v)
      if (v != "" && v+0 > 0) ok=1
    }
    END { exit(ok?0:1) }
  '
}

should_compress_rel() {
  # Returns 0 if relative path should be part of compression target set.
  local rel="$1"
  local base_raw="${rel##*/}"
  local base_norm
  base_norm="$(strip_wrapping_quotes "$base_raw")"

  if [[ "$COMPRESS_SCOPE" == "outputs" ]]; then
    [[ "$rel" == "$RUN_SUBDIR"/outputs/* ]] || return 1
  else
    [[ "$rel" == "$RUN_SUBDIR"/* ]] || return 1
  fi
  [[ "$base_raw" == $COMPRESS_NAME_GLOB || "$base_norm" == $COMPRESS_NAME_GLOB ]]
}

should_skip_clean_rel() {
  # Returns 0 if relative path should be removed/skipped by cleaning rules.
  local rel="$1"
  local base_raw="${rel##*/}"
  local base_norm
  base_norm="$(strip_wrapping_quotes "$base_raw")"
  local pat

  # Top-level files only: <RUN_SUBDIR>/<file>
  if [[ "$rel" == "$RUN_SUBDIR"/* && "$rel" != "$RUN_SUBDIR"/*/* ]]; then
    if [[ "${#DELETE_TOP_PATTERNS[@]}" -gt 0 ]]; then
      for pat in "${DELETE_TOP_PATTERNS[@]}"; do
        [[ "$base_raw" == $pat || "$base_norm" == $pat ]] && return 0
      done
    fi
    if [[ "$DELETE_PARENT_HOTSTARTS" -eq 1 ]] && [[ "$base_raw" == hotstart_it=*.nc || "$base_norm" == hotstart_it=*.nc ]] && [[ "$base_raw" != "$KEEP_PARENT_HOTSTART" && "$base_norm" != "$KEEP_PARENT_HOTSTART" ]]; then
      return 0
    fi
  fi

  # outputs subtree: <RUN_SUBDIR>/outputs/...
  if [[ "$rel" == "$RUN_SUBDIR"/outputs/* ]]; then
    if [[ "${#DELETE_OUTPUTS_PATTERNS[@]}" -gt 0 ]]; then
      for pat in "${DELETE_OUTPUTS_PATTERNS[@]}"; do
        [[ "$base_raw" == $pat || "$base_norm" == $pat ]] && return 0
      done
    fi
  fi

  return 1
}

should_skip_compress_rel() {
  # Returns 0 if relative path should be skipped from compression.
  local rel="$1"
  local base_raw="${rel##*/}"
  local base_norm
  base_norm="$(strip_wrapping_quotes "$base_raw")"
  local pat

  if [[ "$COMPRESS_SKIP_CLEAN_MATCHES" -eq 1 && "$DO_CLEAN" -eq 1 ]]; then
    if should_skip_clean_rel "$rel"; then
      return 0
    fi
  fi

  if [[ "${#COMPRESS_EXCLUDE_BASENAME_PATTERNS[@]}" -gt 0 ]]; then
    for pat in "${COMPRESS_EXCLUDE_BASENAME_PATTERNS[@]}"; do
      [[ "$base_raw" == $pat || "$base_norm" == $pat ]] && return 0
    done
  fi
  return 1
}

compress_one_to_stage() {
  # Return codes:
  #   0 compressed/copied
  #  10 skipped (already compressed)
  #   1 failure
  local src_nc="$1"
  local dst_nc="$2"
  local dst_dir dst_base tmp
  dst_dir="$(dirname "$dst_nc")"
  dst_base="$(basename "$dst_nc")"
  tmp="$dst_dir/.${dst_base}.nccopy.tmp.$$"

  run_cmd mkdir -p "$dst_dir"

  if [[ "$SKIP_ALREADY_COMPRESSED" -eq 1 ]] && is_deflated_nc "$src_nc"; then
    log "    [skip] already deflated at source: $src_nc"
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "[DRY] cp -p \"$src_nc\" \"$dst_nc\""
    else
      cp -p "$src_nc" "$dst_nc"
    fi
    return 10
  fi

  local in_bytes avail need_kb min_buffer_kb
  in_bytes="$(fsize_bytes "$src_nc")"
  avail="$(avail_kb "$dst_dir")"
  min_buffer_kb=$((MIN_FREE_MB * 1024))
  need_kb=$(( (in_bytes + 1023) / 1024 + min_buffer_kb ))

  if [[ "$avail" -lt "$need_kb" ]]; then
    warn "low space, skip compression: $src_nc (need_kb=$need_kb avail_kb=$avail)"
    if [[ "$COMPRESS_FAIL_COPY_ON_ERROR" -eq 1 ]]; then
      warn "low space fallback copy as-is: $src_nc"
      if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[DRY] cp -p \"$src_nc\" \"$dst_nc\""
        return 11
      fi
      if cp -p "$src_nc" "$dst_nc"; then
        return 11
      fi
    fi
    return 1
  fi

  local args_primary=(-k 4 -d "$DEFLATE_LEVEL")
  local args_retry=(-k nc4 -d "$DEFLATE_LEVEL")
  if [[ "$USE_SHUFFLE" -eq 1 ]]; then
    args_primary+=(-s)
    args_retry+=(-s)
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY] nccopy ${args_primary[*]} \"$src_nc\" \"$tmp\" && mv -f \"$tmp\" \"$dst_nc\""
    echo "[DRY] (retry on failure) nccopy ${args_retry[*]} \"$src_nc\" \"$tmp\" && mv -f \"$tmp\" \"$dst_nc\""
    return 0
  fi

  rm -f -- "$tmp" 2>/dev/null || true
  if ! nccopy "${args_primary[@]}" "$src_nc" "$tmp"; then
    warn "primary nccopy failed, retry with -k nc4: $src_nc"
    rm -f -- "$tmp" 2>/dev/null || true
    if ! nccopy "${args_retry[@]}" "$src_nc" "$tmp"; then
      rm -f -- "$tmp" 2>/dev/null || true
      if [[ "$COMPRESS_FAIL_COPY_ON_ERROR" -eq 1 ]]; then
        warn "retry failed, fallback copy as-is: $src_nc"
        if cp -p "$src_nc" "$dst_nc"; then
          return 11
        fi
      fi
      warn "nccopy failed after retry: $src_nc"
      return 1
    fi
  fi

  touch -r "$src_nc" "$tmp" 2>/dev/null || true
  mv -f -- "$tmp" "$dst_nc"

  if [[ "$VERIFY_DEFLATE" -eq 1 ]]; then
    if ! is_deflated_nc "$dst_nc"; then
      warn "compressed but deflate metadata not found: $dst_nc"
    fi
  fi
  return 0
}

collect_source_nc_files() {
  local src_rundir="$1"
  local search_root="$src_rundir"
  if [[ "$COMPRESS_SCOPE" == "outputs" ]]; then
    search_root="$src_rundir/outputs"
  fi
  if [[ ! -d "$search_root" ]]; then
    return 0
  fi
  local f rel
  while IFS= read -r -d '' f; do
    rel="$RUN_SUBDIR/${f#$src_rundir/}"
    should_compress_rel "$rel" || continue
    should_skip_compress_rel "$rel" && continue
    echo "$f"
  done < <(find "$search_root" -type f -print0) | sort
}

compress_source_files_to_stage() {
  local src_part="$1"
  local dst_part="$2"
  shift 2
  local src_files=("$@")
  local n="${#src_files[@]}"
  TOTAL_NC_FOUND=$((TOTAL_NC_FOUND + n))

  if [[ "$n" -eq 0 ]]; then
    log "    no NetCDF files matched for compression"
    return 0
  fi

  log "  [compress] count=$n workers=$NWORKERS deflate=$DEFLATE_LEVEL shuffle=$USE_SHUFFLE"
  local i rc=0

  if [[ "$NWORKERS" -le 1 ]]; then
    local s rel d one_rc
    for ((i=0; i<n; i++)); do
      s="${src_files[$i]}"
      rel="${s#$src_part/}"
      d="$dst_part/$rel"
      compress_one_to_stage "$s" "$d"
      one_rc=$?
      case "$one_rc" in
        0) TOTAL_NC_COMPRESSED=$((TOTAL_NC_COMPRESSED + 1)) ;;
        10) TOTAL_NC_SKIPPED=$((TOTAL_NC_SKIPPED + 1)) ;;
        11) TOTAL_NC_FALLBACK_COPIED=$((TOTAL_NC_FALLBACK_COPIED + 1)) ;;
        *) TOTAL_NC_FAILED=$((TOTAL_NC_FAILED + 1)); rc=1 ;;
      esac
    done
  else
    # bounded parallel queue; aggregate exit codes in parent
    local pids=()
    local srcs=()
    local dsts=()
    local pid idx one_rc rel d

    for ((i=0; i<n; i++)); do
      rel="${src_files[$i]#$src_part/}"
      d="$dst_part/$rel"
      compress_one_to_stage "${src_files[$i]}" "$d" &
      pid=$!
      pids+=("$pid")
      srcs+=("${src_files[$i]}")
      dsts+=("$d")

      if [[ "${#pids[@]}" -ge "$NWORKERS" ]]; then
        if wait "${pids[0]}"; then
          TOTAL_NC_COMPRESSED=$((TOTAL_NC_COMPRESSED + 1))
        else
          one_rc=$?
          if [[ "$one_rc" -eq 10 ]]; then
            TOTAL_NC_SKIPPED=$((TOTAL_NC_SKIPPED + 1))
          elif [[ "$one_rc" -eq 11 ]]; then
            TOTAL_NC_FALLBACK_COPIED=$((TOTAL_NC_FALLBACK_COPIED + 1))
          else
            TOTAL_NC_FAILED=$((TOTAL_NC_FAILED + 1))
            rc=1
            warn "worker failed: ${srcs[0]}"
          fi
        fi
        pids=("${pids[@]:1}")
        srcs=("${srcs[@]:1}")
        dsts=("${dsts[@]:1}")
      fi
    done

    for idx in "${!pids[@]}"; do
      if wait "${pids[$idx]}"; then
        TOTAL_NC_COMPRESSED=$((TOTAL_NC_COMPRESSED + 1))
      else
        one_rc=$?
        if [[ "$one_rc" -eq 10 ]]; then
          TOTAL_NC_SKIPPED=$((TOTAL_NC_SKIPPED + 1))
        elif [[ "$one_rc" -eq 11 ]]; then
          TOTAL_NC_FALLBACK_COPIED=$((TOTAL_NC_FALLBACK_COPIED + 1))
        else
          TOTAL_NC_FAILED=$((TOTAL_NC_FAILED + 1))
          rc=1
          warn "worker failed: ${srcs[$idx]}"
        fi
      fi
    done
  fi
  return "$rc"
}

stage_part() {
  local src_part="$1"
  local dst_part="$2"

  run_cmd mkdir -p "$dst_part"
  run_cmd mkdir -p "$dst_part/$RUN_SUBDIR"
  cleanup_orphan_tmps "$dst_part"

  log "  [stage] selective copy $src_part -> $dst_part"
  local f rel dst_f
  while IFS= read -r -d '' f; do
    rel="${f#$src_part/}"

    # Apply cleaning exclusions at stage-copy time only if clean is enabled
    if [[ "$DO_CLEAN" -eq 1 ]]; then
      if should_skip_clean_rel "$rel"; then
        continue
      fi
    fi

    # If compression is enabled, do not copy target .nc now.
    # They will be written directly as compressed files from source -> stage.
    if [[ "$DO_COMPRESS" -eq 1 ]] && should_compress_rel "$rel"; then
      # If this path is excluded from compression, keep normal copy behavior.
      if ! should_skip_compress_rel "$rel"; then
        continue
      fi
    fi

    dst_f="$dst_part/$rel"
    run_cmd mkdir -p "$(dirname "$dst_f")"
    run_cmd cp -p "$f" "$dst_f"
  done < <(find "$src_part" -type f -print0)

  # Ensure outputs dir exists for compression-only staging
  if [[ "$COMPRESS_SCOPE" == "outputs" ]]; then
    run_cmd mkdir -p "$dst_part/$RUN_SUBDIR/outputs"
  fi
}

verify_repacked_part() {
  local partbase="$1"
  local staged_part="$2"
  local out_tar="$3"
  local out_sha="$4"
  local rc=0

  log "  [verify] $partbase"

  if [[ ! -d "$staged_part/$RUN_SUBDIR" ]]; then
    warn "verify failed: missing staged run directory: $staged_part/$RUN_SUBDIR"
    return 1
  fi

  if [[ ! -s "$out_tar" ]]; then
    warn "verify failed: archive missing or empty: $out_tar"
    return 1
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY] tar -tzf \"$out_tar\" >/dev/null"
    echo "[DRY] verify checksum \"$out_sha\""
    return 0
  fi

  if ! tar -tzf "$out_tar" >/dev/null; then
    warn "verify failed: tar test failed: $out_tar"
    rc=1
  fi

  # Use full-scan awk instead of grep -q to avoid pipeline edge cases with
  # pipefail and to allow repeated "./" prefixes in tar listings.
  if ! tar -tzf "$out_tar" | awk -v run="$RUN_SUBDIR" '
      $0 ~ "^((\\./)*)" run "/" { found=1 }
      END { exit(found ? 0 : 1) }
    '; then
    warn "verify failed: archive does not include expected run subdir: $RUN_SUBDIR"
    rc=1
  fi

  local stage_files tar_files
  stage_files="$(find "$staged_part" -type f | wc -l | awk '{print $1}')"
  tar_files="$(tar -tzf "$out_tar" | awk 'substr($0,length($0),1)!="/" {c++} END{print c+0}')"
  if [[ "$tar_files" -lt "$stage_files" ]]; then
    warn "verify failed: tar file count ($tar_files) < stage file count ($stage_files)"
    rc=1
  fi

  if ! sha_verify "$out_tar" "$out_sha"; then
    warn "verify failed: checksum mismatch: $out_tar"
    rc=1
  fi

  return "$rc"
}

remove_source_part() {
  local src_part="$1"
  local partbase="$2"

  if [[ ! -d "$src_part" ]]; then
    warn "source part already missing, skip delete: $src_part"
    return 0
  fi

  case "$src_part" in
    "$EXTRACT_ROOT"/*) ;;
    *)
      warn "refuse delete outside EXTRACT_ROOT: $src_part"
      return 1
      ;;
  esac

  if [[ "$(basename "$src_part")" != "$partbase" ]]; then
    warn "refuse delete with basename mismatch: $src_part"
    return 1
  fi

  log "  [delete-source] $src_part"
  run_cmd rm -rf "$src_part"
  TOTAL_SOURCE_DELETED=$((TOTAL_SOURCE_DELETED + 1))
  return 0
}

repack_part() {
  local partbase="$1"
  local staged_part="$2"

  run_cmd mkdir -p "$REPACK_DIR"
  local out_tar="$REPACK_DIR/${partbase}.tar.gz"
  # Repack runs tar from inside staged_part; keep output path absolute so
  # relative REPACK_DIR works regardless of current working directory.
  if [[ "$out_tar" != /* ]]; then
    out_tar="$PWD/$out_tar"
  fi
  local out_sha="${out_tar}.sha256"
  REPACK_LAST_TAR="$out_tar"
  REPACK_LAST_SHA="$out_sha"
  REPACK_LAST_WRITTEN=0

  if [[ -e "$out_tar" && "$OVERWRITE_ARCHIVE" -ne 1 ]]; then
    log "  [repack] archive exists, verifying before skip: $out_tar"
    if verify_repacked_part "$partbase" "$staged_part" "$out_tar" "$out_sha"; then
      log "  [repack] existing archive verified; skip rebuild: $out_tar"
      return 3
    fi
    warn "existing archive failed verification, rebuilding: $out_tar"
    run_cmd rm -f "$out_tar" "$out_sha"
  fi

  log "  [repack] $out_tar"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY] (cd \"$staged_part\" && tar -czf \"$out_tar\" .)"
  else
    (cd "$staged_part" && tar -czf "$out_tar" .)
  fi
  sha_write "$out_tar" "$out_sha"
  TOTAL_ARCHIVES=$((TOTAL_ARCHIVES + 1))
  REPACK_LAST_WRITTEN=1
  return 0
}

process_part() {
  local src_part="$1"
  local partbase staged_part rundir
  partbase="$(basename "$src_part")"
  staged_part="$STAGE_ROOT/$partbase"
  rundir="$staged_part/$RUN_SUBDIR"

  echo "========================================"
  echo "Processing $partbase"
  echo "========================================"

  if [[ "$DO_STAGE" -eq 1 ]]; then
    stage_part "$src_part" "$staged_part"
  fi

  if [[ ! -d "$rundir" ]]; then
    warn "run directory not found in stage: $rundir"
    return 1
  fi

  if [[ "$DO_CLEAN" -eq 1 ]]; then
    cleanup_hotstarts_and_logs "$rundir"
  fi

  if [[ "$DO_COMPRESS" -eq 1 ]]; then
    local src_rundir="$src_part/$RUN_SUBDIR"
    local nc_files=()
    local f
    while IFS= read -r f; do nc_files+=("$f"); done < <(collect_source_nc_files "$src_rundir")
    if [[ "${#nc_files[@]}" -gt 0 ]]; then
      compress_source_files_to_stage "$src_part" "$staged_part" "${nc_files[@]}" || true
    else
      compress_source_files_to_stage "$src_part" "$staged_part" || true
    fi
    cleanup_orphan_tmps "$staged_part"
  fi

  if [[ "$DO_REPACK" -eq 1 ]]; then
    local repack_rc=0
    repack_part "$partbase" "$staged_part" || repack_rc=$?
    if [[ "$repack_rc" -ne 0 && "$repack_rc" -ne 3 ]]; then
      warn "repack failed for part: $partbase"
      return 1
    fi
  fi

  if [[ "$DELETE_SOURCE_AFTER_REPACK" -eq 1 ]]; then
    if [[ "$DO_REPACK" -ne 1 ]]; then
      warn "delete-source-after-repack requested but repack disabled; source kept: $src_part"
    elif [[ "$REPACK_LAST_WRITTEN" -ne 1 ]]; then
      warn "source delete skipped (archive was not newly written): $src_part"
    else
      if verify_repacked_part "$partbase" "$staged_part" "$REPACK_LAST_TAR" "$REPACK_LAST_SHA"; then
        remove_source_part "$src_part" "$partbase" || return 1
      else
        warn "verification failed; source kept: $src_part"
        return 1
      fi
    fi
  fi

  if [[ "$CLEAN_STAGE_AFTER" -eq 1 && "$DO_REPACK" -eq 1 ]]; then
    log "  [clean-stage-after] remove $staged_part"
    run_cmd rm -rf "$staged_part"
  fi

  return 0
}

parse_args() {
  if [[ $# -gt 0 && "${1:-}" != -* ]]; then
    PREFIX="$1"
    shift
  else
    PREFIX="${PREFIX:-$PREFIX_DEFAULT}"
  fi

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --extract-root) EXTRACT_ROOT="$2"; shift 2 ;;
      --stage-root) STAGE_ROOT="$2"; shift 2 ;;
      --repack-dir) REPACK_DIR="$2"; shift 2 ;;
      --run-subdir) RUN_SUBDIR="$2"; shift 2 ;;
      --part-glob) PART_GLOB="$2"; shift 2 ;;
      --mode) MODE="$2"; shift 2 ;;
      --submit) MODE="submit"; shift ;;
      --sbatch-arg) SBATCH_EXTRA_ARGS+=("$2"); shift 2 ;;

      --stage) DO_STAGE=1; shift ;;
      --no-stage) DO_STAGE=0; shift ;;
      --clean) DO_CLEAN=1; shift ;;
      --no-clean) DO_CLEAN=0; shift ;;
      --compress) DO_COMPRESS=1; shift ;;
      --no-compress) DO_COMPRESS=0; shift ;;
      --repack) DO_REPACK=1; shift ;;
      --no-repack) DO_REPACK=0; shift ;;
      --dry-run) DRY_RUN=1; shift ;;

      --compress-scope) COMPRESS_SCOPE="$2"; shift 2 ;;
      --compress-name-glob) COMPRESS_NAME_GLOB="$2"; shift 2 ;;
      --deflate) DEFLATE_LEVEL="$2"; shift 2 ;;
      --no-shuffle) USE_SHUFFLE=0; shift ;;
      --workers) NWORKERS="$2"; shift 2 ;;
      --no-skip-compressed) SKIP_ALREADY_COMPRESSED=0; shift ;;
      --no-verify-deflate) VERIFY_DEFLATE=0; shift ;;
      --min-free-mb) MIN_FREE_MB="$2"; shift 2 ;;

      --overwrite-archive) OVERWRITE_ARCHIVE=1; shift ;;
      --clean-stage-after) CLEAN_STAGE_AFTER=1; shift ;;
      --delete-source-after-repack) DELETE_SOURCE_AFTER_REPACK=1; shift ;;
      --no-delete-source-after-repack) DELETE_SOURCE_AFTER_REPACK=0; shift ;;

      --quiet) VERBOSE=0; shift ;;
      -h|--help) usage; exit 0 ;;
      *) err "Unknown arg: $1"; usage; exit 2 ;;
    esac
  done

  [[ -n "$RUN_SUBDIR" ]] || RUN_SUBDIR="$PREFIX"
  [[ -n "$PART_GLOB" ]] || PART_GLOB="${PREFIX}.part*"
  [[ "$MODE" == "interactive" || "$MODE" == "submit" ]] || {
    err "--mode must be interactive or submit"
    exit 2
  }
}

validate() {
  [[ -d "$EXTRACT_ROOT" ]] || { err "EXTRACT_ROOT not found: $EXTRACT_ROOT"; exit 1; }
  case "$DEFLATE_LEVEL" in 0|1|2|3|4|5|6|7|8|9) ;; *) err "--deflate must be 0..9"; exit 1 ;; esac
  [[ "$COMPRESS_SCOPE" == "outputs" || "$COMPRESS_SCOPE" == "all" ]] || {
    err "--compress-scope must be outputs|all"
    exit 1
  }
  [[ "$NWORKERS" =~ ^[0-9]+$ && "$NWORKERS" -ge 1 ]] || { err "--workers must be >=1"; exit 1; }
  [[ "$MIN_FREE_MB" =~ ^[0-9]+$ ]] || { err "--min-free-mb must be integer"; exit 1; }

  need_cmd find
  need_cmd cp
  need_cmd tar
  if [[ "$DO_COMPRESS" -eq 1 ]]; then
    need_cmd nccopy
    need_cmd ncdump
  fi

  run_cmd mkdir -p "$STAGE_ROOT" "$REPACK_DIR"
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

  echo "PREFIX                : $PREFIX"
  echo "EXTRACT_ROOT          : $EXTRACT_ROOT"
  echo "STAGE_ROOT            : $STAGE_ROOT"
  echo "REPACK_DIR            : $REPACK_DIR"
  echo "RUN_SUBDIR            : $RUN_SUBDIR"
  echo "PART_GLOB             : $PART_GLOB"
  echo "MODE                  : $MODE"
  echo "Actions               : stage=$DO_STAGE clean=$DO_CLEAN compress=$DO_COMPRESS repack=$DO_REPACK dry_run=$DRY_RUN"
  echo "Compression           : scope=$COMPRESS_SCOPE name_glob=$COMPRESS_NAME_GLOB deflate=$DEFLATE_LEVEL shuffle=$USE_SHUFFLE workers=$NWORKERS"
  echo "Compress skip clean   : $COMPRESS_SKIP_CLEAN_MATCHES"
  echo "Compress fail->copy   : $COMPRESS_FAIL_COPY_ON_ERROR"
  echo "Skip compressed       : $SKIP_ALREADY_COMPRESSED"
  echo "Min free MB buffer    : $MIN_FREE_MB"
  echo "Clean stage after     : $CLEAN_STAGE_AFTER"
  echo "Delete source after   : $DELETE_SOURCE_AFTER_REPACK"
  echo

  local partdirs=()
  local pdir=""
  while IFS= read -r pdir; do
    partdirs+=("$pdir")
  done < <(find "$EXTRACT_ROOT" -maxdepth 1 -mindepth 1 -type d -name "$PART_GLOB" | sort)
  [[ "${#partdirs[@]}" -gt 0 ]] || { err "No part dirs matched: $EXTRACT_ROOT/$PART_GLOB"; exit 1; }

  local p
  for p in "${partdirs[@]}"; do
    TOTAL_PARTS=$((TOTAL_PARTS + 1))
    if process_part "$p"; then
      TOTAL_PARTS_OK=$((TOTAL_PARTS_OK + 1))
    else
      warn "part failed: $p"
    fi
  done

  echo
  echo "=== Summary ==="
  echo "parts processed       : $TOTAL_PARTS_OK / $TOTAL_PARTS"
  echo "files removed         : $TOTAL_FILES_REMOVED"
  echo "nc found              : $TOTAL_NC_FOUND"
  echo "nc compressed         : $TOTAL_NC_COMPRESSED"
  echo "nc skipped            : $TOTAL_NC_SKIPPED"
  echo "nc fallback copied    : $TOTAL_NC_FALLBACK_COPIED"
  echo "nc failed             : $TOTAL_NC_FAILED"
  echo "archives created      : $TOTAL_ARCHIVES"
  echo "source parts deleted  : $TOTAL_SOURCE_DELETED"
  echo "DONE."
}

main "$@"
