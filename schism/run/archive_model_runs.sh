#!/usr/bin/env bash
#SBATCH -J schism_archive_runs
#SBATCH -o schism_archive_runs.%j.out
#SBATCH -e schism_archive_runs.%j.err
#SBATCH -p cpu-small
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 5-00:00:00

# Clean, compress, and archive SCHISM run directories.
#
# Typical project layout:
#   <PROJECT_ROOT>/
#     pre-proc/
#     post-proc/
#     run/
#       RUN01a/
#       RUN05b/
#
# This script processes selected run folders under run/ and writes:
#   <ARCHIVE_ROOT>/<RUN>.tar.gz
#   <ARCHIVE_ROOT>/<RUN>.tar.gz.sha256
#
# Workflow per run (in-place mode):
#   1) Clean source run folder by exclusions
#   2) Compress target NetCDF files in-place using nccopy
#   3) Archive to tar.gz (optionally split by size) + checksums
#   4) Verify archive/checksum before optional source deletion

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
RUN_ROOT="${RUN_ROOT:-$PROJECT_ROOT/run}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-$PROJECT_ROOT/archive_store}"

RUN_GLOB="${RUN_GLOB:-RUN*}"
RUN_NAMES=()

DO_CLEAN=1
DO_COMPRESS=1
DO_ARCHIVE=1

DRY_RUN=0
MODE="${MODE:-interactive}"   # interactive | submit
SBATCH_EXTRA_ARGS=()
VERBOSE=1

COMPRESS_SCOPE="all"          # outputs | all
COMPRESS_NAME_GLOB="*.nc"
DEFLATE_LEVEL="${DEFLATE_LEVEL:-9}"
USE_SHUFFLE=1
NWORKERS="${NWORKERS:-1}"
COMPRESS_BATCH_SIZE="${COMPRESS_BATCH_SIZE:-2000}"
MIN_FREE_MB="${MIN_FREE_MB:-1024}"
SKIP_ALREADY_COMPRESSED=1
VERIFY_DEFLATE=1
COMPRESS_FAIL_COPY_ON_ERROR=1
COMPRESS_SKIP_CLEAN_MATCHES=1
DEREFERENCE_NC_SYMLINKS=1

OVERWRITE_ARCHIVE=0
DELETE_SOURCE_AFTER_VERIFY="${DELETE_SOURCE_AFTER_VERIFY:-1}"
ARCHIVE_MAX_GB="${ARCHIVE_MAX_GB:-1000}"   # 0 => no split, >0 => split size limit in GB

# Cleanup rules (edit here)
DELETE_TOP_PATTERNS=(
  "er.*"
  "ot.*"
  "core.*"
  "runmdl"
  "runmdl_short"
  "pschism_*"
  "schismview"
)
DELETE_OUTPUTS_PATTERNS=(
  "hotstart*.nc"
)
# Relative-path cleanup rules under run root (recursive, path-based).
# Examples for typical SCHISM run dirs:
#   schismview/*      -> viewer cache/output folder
#   outputs/hotstart* -> handled above by DELETE_OUTPUTS_PATTERNS
DELETE_REL_PATH_PATTERNS=(
)
DELETE_PARENT_HOTSTARTS=0
KEEP_PARENT_HOTSTART="hotstart_it=0.nc"

# Compression exclude rules (edit here)
COMPRESS_EXCLUDE_BASENAME_PATTERNS=(
)

TOTAL_RUNS=0
TOTAL_RUNS_OK=0
TOTAL_FILES_REMOVED=0
TOTAL_NC_FOUND=0
TOTAL_NC_COMPRESSED=0
TOTAL_NC_SKIPPED=0
TOTAL_NC_FAILED=0
TOTAL_NC_FALLBACK_COPIED=0
TOTAL_ARCHIVES=0
TOTAL_SOURCE_DELETED=0

ARCHIVE_LAST_TAR=""
ARCHIVE_LAST_SHA=""
ARCHIVE_LAST_WRITTEN=0
ARCHIVE_LAST_VERIFIED_OK=0
ARCHIVE_LAST_SPLIT=0
ARCHIVE_LAST_PART_LIST=""
ARCHIVE_LAST_PART_SHA=""

usage() {
  cat <<'USAGE'
Usage:
  archive_model_runs.sh [RUN_NAME ...] [options]

Positional:
  RUN_NAME                     Specific run folder names under RUN_ROOT (e.g., RUN01a RUN05b).
                               If omitted, process all matching --run-glob.

Options:
  --project-root DIR           Project root containing run/ (default: current directory)
  --run-root DIR               Run root path (default: <project-root>/run)
  --archive-root DIR           Archive output path (default: <project-root>/archive_store)
  --run-glob GLOB              Run directory glob when no RUN_NAME provided (default: RUN*)

Actions:
  --clean / --no-clean
  --compress / --no-compress
  --archive / --no-archive
  --dry-run

Compression:
  --compress-scope MODE        outputs|all (default: all)
  --compress-name-glob GLOB    default: *.nc
  --deflate N                  nccopy deflate 0..9 (default: 9)
  --no-shuffle                 disable nccopy -s
  --workers N                  parallel workers (default: 4)
  --compress-batch-size N      compression input batch size (default: 2000)
  --min-free-mb N              free-space guard (default: 1024)
  --no-skip-compressed         recompress already deflated source files
  --no-verify-deflate          skip post-compress metadata check
  --no-fail-copy               fail instead of fallback copy when nccopy fails
  --compress-include-clean-matches
                               include clean-matched files in compression
  --no-dereference-nc-symlinks keep .nc symlinks instead of dereferencing for compression

Archive:
  --archive-max-gb N           max size per archive piece in GB (default: 1000, 0=single tar.gz)
  --overwrite-archive          overwrite existing archive output
  --delete-source-after-verify remove source run folder after archive verification

Submit:
  --mode MODE                  interactive|submit (default: interactive)
  --submit                     alias of --mode submit
  --sbatch-arg ARG             extra argument passed to sbatch (repeatable)

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

strip_wrapping_quotes() {
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

fsize_bytes() {
  local f="$1"
  stat -c%s "$f" 2>/dev/null || stat -f%z "$f"
}

avail_kb() {
  local d="$1"
  df -Pk "$d" | awk 'NR==2{print $4}'
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

write_parts_sha() {
  local list_f="$1"
  local out_f="$2"
  local p
  : > "$out_f"
  if command -v sha256sum >/dev/null 2>&1; then
    while IFS= read -r p; do
      sha256sum "$p" >> "$out_f"
    done < "$list_f"
    return 0
  fi
  if command -v shasum >/dev/null 2>&1; then
    while IFS= read -r p; do
      shasum -a 256 "$p" >> "$out_f"
    done < "$list_f"
    return 0
  fi
  warn "No sha tool found. Skip part checksums."
  return 1
}

cat_split_parts_stream() {
  local list_f="$1"
  local p
  while IFS= read -r p; do
    [[ -n "$p" ]] || continue
    cat "$p"
  done < "$list_f"
}

tar_list_has_run_folder() {
  local run_name="$1"
  awk -v run="$run_name" '
    {
      if (index($0, run "/") == 1 || index($0, "./" run "/") == 1) {
        found=1
      }
    }
    END { exit(found ? 0 : 1) }
  '
}

rename_split_parts_tar_gz() {
  local prefix="$1"
  local p
  local renamed=0
  while IFS= read -r p; do
    [[ -n "$p" ]] || continue
    [[ "$p" == *.tar.gz ]] && continue
    mv -f -- "$p" "${p}.tar.gz"
    renamed=1
  done < <(find "$(dirname "$prefix")" -maxdepth 1 -type f -name "$(basename "$prefix")*" | sort)
  return 0
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
  local path="$1"
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY] find \"$path\" -type f \\( -name '*.nccopy_tmp.*' -o -name '.*.nccopy.tmp.*' -o -name '.*.nc.nccopy.tmp.*' \\) -delete"
    return
  fi
  find "$path" -type f \( \
    -name '*.nccopy_tmp.*' -o \
    -name '.*.nccopy.tmp.*' -o \
    -name '.*.nc.nccopy.tmp.*' \
  \) -delete || true
}

is_deflated_nc() {
  local nc="$1"
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

is_compression_sufficient_nc() {
  # Return success only when source appears to already meet requested
  # compression settings (deflate >= DEFLATE_LEVEL and shuffle when requested).
  local nc="$1"
  local want_d="$DEFLATE_LEVEL"
  local want_s="$USE_SHUFFLE"
  ncdump -hs "$nc" 2>/dev/null | awk -v want_d="$want_d" -v want_s="$want_s" '
    /_DeflateLevel[[:space:]]*=/ {
      seen_d=1
      v=$0
      sub(/.*_DeflateLevel[[:space:]]*=[[:space:]]*/, "", v)
      sub(/[[:space:]]*;.*/, "", v)
      gsub(/[^0-9]/, "", v)
      if (v == "" || v+0 < want_d) bad_d=1
    }
    /_Shuffle[[:space:]]*=/ {
      seen_s=1
      s=$0
      sub(/.*_Shuffle[[:space:]]*=[[:space:]]*/, "", s)
      sub(/[[:space:]]*;.*/, "", s)
      gsub(/[^0-9]/, "", s)
      if (s == "" || s+0 == 0) bad_s=1
    }
    END {
      if (!seen_d) exit 1
      if (bad_d) exit 1
      if (want_s+0 == 1) {
        if (!seen_s) exit 1
        if (bad_s) exit 1
      }
      exit 0
    }
  '
}

should_skip_clean_rel() {
  local rel="$1"
  local base_raw="${rel##*/}"
  local base_norm
  base_norm="$(strip_wrapping_quotes "$base_raw")"
  local pat

  # Full relative-path patterns (recursive).
  if [[ "${#DELETE_REL_PATH_PATTERNS[@]}" -gt 0 ]]; then
    for pat in "${DELETE_REL_PATH_PATTERNS[@]}"; do
      [[ "$rel" == $pat ]] && return 0
    done
  fi

  if [[ "$rel" != */* ]]; then
    if [[ "${#DELETE_TOP_PATTERNS[@]}" -gt 0 ]]; then
      for pat in "${DELETE_TOP_PATTERNS[@]}"; do
        [[ "$base_raw" == $pat || "$base_norm" == $pat ]] && return 0
      done
    fi
    if [[ "$DELETE_PARENT_HOTSTARTS" -eq 1 ]] && [[ "$base_raw" == hotstart_it=*.nc || "$base_norm" == hotstart_it=*.nc ]] && [[ "$base_raw" != "$KEEP_PARENT_HOTSTART" && "$base_norm" != "$KEEP_PARENT_HOTSTART" ]]; then
      return 0
    fi
  fi

  if [[ "$rel" == outputs/* ]]; then
    if [[ "${#DELETE_OUTPUTS_PATTERNS[@]}" -gt 0 ]]; then
      for pat in "${DELETE_OUTPUTS_PATTERNS[@]}"; do
        [[ "$base_raw" == $pat || "$base_norm" == $pat ]] && return 0
      done
    fi
  fi
  return 1
}

should_compress_rel() {
  local rel="$1"
  local base_raw="${rel##*/}"
  local base_norm
  base_norm="$(strip_wrapping_quotes "$base_raw")"

  if [[ "$COMPRESS_SCOPE" == "outputs" ]]; then
    [[ "$rel" == outputs/* ]] || return 1
  fi
  [[ "$base_raw" == $COMPRESS_NAME_GLOB || "$base_norm" == $COMPRESS_NAME_GLOB ]]
}

should_skip_compress_rel() {
  local rel="$1"
  local base_raw="${rel##*/}"
  local base_norm
  base_norm="$(strip_wrapping_quotes "$base_raw")"
  local pat

  if [[ "$COMPRESS_SKIP_CLEAN_MATCHES" -eq 1 && "$DO_CLEAN" -eq 1 ]]; then
    should_skip_clean_rel "$rel" && return 0
  fi

  if [[ "${#COMPRESS_EXCLUDE_BASENAME_PATTERNS[@]}" -gt 0 ]]; then
    for pat in "${COMPRESS_EXCLUDE_BASENAME_PATTERNS[@]}"; do
      [[ "$base_raw" == $pat || "$base_norm" == $pat ]] && return 0
    done
  fi
  return 1
}

copy_entry() {
  local src="$1"
  local dst="$2"
  run_cmd mkdir -p "$(dirname "$dst")"
  run_cmd cp -a "$src" "$dst"
}

compress_one_to_stage() {
  # Return codes:
  #   0 compressed
  #  10 skipped (already compressed)
  #  11 fallback-copied
  #   1 failure
  local src_nc="$1"
  local dst_nc="$2"
  local dst_dir dst_base tmp
  dst_dir="$(dirname "$dst_nc")"
  dst_base="$(basename "$dst_nc")"
  tmp="$dst_dir/.${dst_base}.nccopy.tmp.$$"

  run_cmd mkdir -p "$dst_dir"

  if [[ "$SKIP_ALREADY_COMPRESSED" -eq 1 ]] && is_compression_sufficient_nc "$src_nc"; then
    log "    [skip] already at/above target compression: $src_nc"
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "[DRY] cp -p \"$src_nc\" \"$dst_nc\""
    else
      # In-place mode can map src and dst to the same file; no copy needed.
      if [[ "$src_nc" == "$dst_nc" ]] || [[ -e "$dst_nc" && "$src_nc" -ef "$dst_nc" ]]; then
        return 10
      fi
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
        if [[ "$src_nc" == "$dst_nc" ]] || cp -p "$src_nc" "$dst_nc"; then
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

stage_run() {
  local src_run="$1"
  local dst_run="$2"
  run_cmd mkdir -p "$dst_run"
  cleanup_orphan_tmps "$dst_run"

  log "  [stage] selective copy $src_run -> $dst_run"
  local f rel dst_f
  while IFS= read -r -d '' f; do
    rel="${f#$src_run/}"

    if [[ "$DO_CLEAN" -eq 1 ]]; then
      should_skip_clean_rel "$rel" && continue
    fi

    if [[ "$DO_COMPRESS" -eq 1 ]] && should_compress_rel "$rel"; then
      if ! should_skip_compress_rel "$rel"; then
        continue
      fi
    fi

    dst_f="$dst_run/$rel"
    copy_entry "$f" "$dst_f"
  done < <(find "$src_run" -mindepth 1 \( -type f -o -type l \) -print0)

  if [[ "$COMPRESS_SCOPE" == "outputs" ]]; then
    run_cmd mkdir -p "$dst_run/outputs"
  fi
}

collect_source_nc_files() {
  local src_run="$1"
  local search_root="$src_run"
  if [[ "$COMPRESS_SCOPE" == "outputs" ]]; then
    search_root="$src_run/outputs"
  fi
  [[ -d "$search_root" ]] || return 0

  local f rel
  while IFS= read -r -d '' f; do
    rel="${f#$src_run/}"
    should_compress_rel "$rel" || continue
    should_skip_compress_rel "$rel" && continue
    echo "$f"
  done < <(find "$search_root" \( -type f -o -type l \) -name "$COMPRESS_NAME_GLOB" -print0) | sort
}

resolve_nc_source_path() {
  local src="$1"
  if [[ -L "$src" && "$DEREFERENCE_NC_SYMLINKS" -eq 1 ]]; then
    local target
    target="$(readlink -f "$src" 2>/dev/null || true)"
    if [[ -n "$target" && -f "$target" ]]; then
      printf "%s" "$target"
      return 0
    fi
    warn "nc symlink target missing/unreadable, use link itself: $src"
  fi
  printf "%s" "$src"
}

compress_source_files_to_stage() {
  local src_run="$1"
  local dst_run="$2"
  shift 2
  local src_files=("$@")
  local n="${#src_files[@]}"
  TOTAL_NC_FOUND=$((TOTAL_NC_FOUND + n))

  if [[ "$n" -eq 0 ]]; then
    log "    no NetCDF files matched for compression"
    return 0
  fi

  log "  [compress] count=$n workers=$NWORKERS deflate=$DEFLATE_LEVEL shuffle=$USE_SHUFFLE"
  local i rc=0 one_rc

  if [[ "$NWORKERS" -le 1 ]]; then
    local s rel d src_nc
    for ((i=0; i<n; i++)); do
      s="${src_files[$i]}"
      rel="${s#$src_run/}"
      d="$dst_run/$rel"
      src_nc="$(resolve_nc_source_path "$s")"
      compress_one_to_stage "$src_nc" "$d"
      one_rc=$?
      case "$one_rc" in
        0) TOTAL_NC_COMPRESSED=$((TOTAL_NC_COMPRESSED + 1)) ;;
        10) TOTAL_NC_SKIPPED=$((TOTAL_NC_SKIPPED + 1)) ;;
        11) TOTAL_NC_FALLBACK_COPIED=$((TOTAL_NC_FALLBACK_COPIED + 1)) ;;
        *) TOTAL_NC_FAILED=$((TOTAL_NC_FAILED + 1)); rc=1 ;;
      esac
    done
  else
    local pids=()
    local files=()
    local rels=()
    local idx s rel d src_nc
    for ((i=0; i<n; i++)); do
      s="${src_files[$i]}"
      rel="${s#$src_run/}"
      d="$dst_run/$rel"
      src_nc="$(resolve_nc_source_path "$s")"
      compress_one_to_stage "$src_nc" "$d" &
      pids+=("$!")
      files+=("$s")
      rels+=("$rel")

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
            warn "worker failed: ${files[0]}"
          fi
        fi
        pids=("${pids[@]:1}")
        files=("${files[@]:1}")
        rels=("${rels[@]:1}")
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
          warn "worker failed: ${files[$idx]}"
        fi
      fi
    done
  fi

  return "$rc"
}

archive_run() {
  local run_name="$1"
  local src_run_dir="$2"
  local out_tar="$ARCHIVE_ROOT/${run_name}.tar.gz"
  local out_sha="${out_tar}.sha256"
  local split_base="$ARCHIVE_ROOT/${run_name}"
  local part_prefix="${split_base}.part."
  local part_list="${split_base}.parts.list"
  local part_sha="${split_base}.parts.sha256"

  ARCHIVE_LAST_TAR="$out_tar"
  ARCHIVE_LAST_SHA="$out_sha"
  ARCHIVE_LAST_WRITTEN=0
  ARCHIVE_LAST_VERIFIED_OK=0
  ARCHIVE_LAST_SPLIT=0
  ARCHIVE_LAST_PART_LIST=""
  ARCHIVE_LAST_PART_SHA=""

  run_cmd mkdir -p "$ARCHIVE_ROOT"

  if [[ "$ARCHIVE_MAX_GB" -gt 0 ]]; then
    if [[ -e "$part_list" && "$OVERWRITE_ARCHIVE" -ne 1 ]]; then
      ARCHIVE_LAST_SPLIT=1
      ARCHIVE_LAST_PART_LIST="$part_list"
      ARCHIVE_LAST_PART_SHA="$part_sha"
      log "  [archive] split archive exists, verifying before skip: $part_list"
      if verify_archived_run "$run_name" "$src_run_dir" "$out_tar" "$out_sha"; then
        ARCHIVE_LAST_VERIFIED_OK=1
        log "  [archive] existing split archive verified; skip rebuild: $part_list"
        return 3
      fi
      warn "existing split archive failed verification, rebuilding: $part_list"
      run_cmd rm -f "$out_tar" "$out_sha" "$part_list" "$part_sha"
      if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[DRY] rm -f \"${part_prefix}\"*"
      else
        rm -f "${part_prefix}"*
      fi
    fi
    if [[ "$OVERWRITE_ARCHIVE" -eq 1 ]]; then
      run_cmd rm -f "$out_tar" "$out_sha" "$part_list" "$part_sha"
      if [[ "$DRY_RUN" -eq 1 ]]; then
        echo "[DRY] rm -f \"${part_prefix}\"*"
      else
        rm -f "${part_prefix}"*
      fi
    fi
  else
    if [[ -e "$out_tar" && "$OVERWRITE_ARCHIVE" -ne 1 ]]; then
      log "  [archive] archive exists, verifying before skip: $out_tar"
      if verify_archived_run "$run_name" "$src_run_dir" "$out_tar" "$out_sha"; then
        ARCHIVE_LAST_VERIFIED_OK=1
        log "  [archive] existing archive verified; skip rebuild: $out_tar"
        return 3
      fi
      warn "existing archive failed verification, rebuilding: $out_tar"
      run_cmd rm -f "$out_tar" "$out_sha"
    fi
  fi

  log "  [archive] $run_name -> $ARCHIVE_ROOT (split_gb=$ARCHIVE_MAX_GB)"
  if [[ "$ARCHIVE_MAX_GB" -gt 0 ]]; then
    if [[ "$DRY_RUN" -eq 1 ]]; then
      echo "[DRY] tar -czf - -C \"$RUN_ROOT\" \"$run_name\" | split -d -a 4 -b \"${ARCHIVE_MAX_GB}G\" - \"${part_prefix}\""
      echo "[DRY] rename split parts to \"${run_name}.part.XXXX.tar.gz\""
      echo "[DRY] find \"$ARCHIVE_ROOT\" -maxdepth 1 -type f -name \"$(basename "$part_prefix")*.tar.gz\" | sort > \"$part_list\""
      echo "[DRY] write part checksums -> \"$part_sha\""
    else
      tar -czf - -C "$RUN_ROOT" "$run_name" | split -d -a 4 -b "${ARCHIVE_MAX_GB}G" - "$part_prefix"
      rename_split_parts_tar_gz "$part_prefix"
      find "$ARCHIVE_ROOT" -maxdepth 1 -type f -name "$(basename "$part_prefix")*.tar.gz" | sort > "$part_list"
      if [[ ! -s "$part_list" ]]; then
        warn "split archive failed: no part files created for $run_name"
        return 1
      fi
      write_parts_sha "$part_list" "$part_sha" || true
    fi
    TOTAL_ARCHIVES=$((TOTAL_ARCHIVES + 1))
    ARCHIVE_LAST_WRITTEN=1
    ARCHIVE_LAST_SPLIT=1
    ARCHIVE_LAST_PART_LIST="$part_list"
    ARCHIVE_LAST_PART_SHA="$part_sha"
    return 0
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY] tar -czf \"$out_tar\" -C \"$RUN_ROOT\" \"$run_name\""
  else
    tar -czf "$out_tar" -C "$RUN_ROOT" "$run_name"
  fi
  sha_write "$out_tar" "$out_sha"
  TOTAL_ARCHIVES=$((TOTAL_ARCHIVES + 1))
  ARCHIVE_LAST_WRITTEN=1
  return 0
}

verify_archived_run() {
  local run_name="$1"
  local src_run_dir="$2"
  local out_tar="$3"
  local out_sha="$4"
  local rc=0

  log "  [verify] $run_name"
  [[ -d "$src_run_dir" ]] || {
    warn "verify failed: source run dir missing: $src_run_dir"
    return 1
  }

  if [[ "$DRY_RUN" -eq 1 ]]; then
    if [[ "$ARCHIVE_LAST_SPLIT" -eq 1 ]]; then
      echo "[DRY] cat \$(cat \"$ARCHIVE_LAST_PART_LIST\") | tar -tzf - >/dev/null"
      echo "[DRY] verify part checksums \"$ARCHIVE_LAST_PART_SHA\""
    else
      echo "[DRY] tar -tzf \"$out_tar\" >/dev/null"
      echo "[DRY] verify checksum \"$out_sha\""
    fi
    return 0
  fi

  local source_entries tar_entries
  source_entries="$(find "$src_run_dir" \( -type f -o -type l \) | wc -l | awk '{print $1}')"

  if [[ "$ARCHIVE_LAST_SPLIT" -eq 1 ]]; then
    [[ -f "$ARCHIVE_LAST_PART_LIST" ]] || {
      warn "verify failed: part list missing: $ARCHIVE_LAST_PART_LIST"
      return 1
    }
    if [[ -f "$ARCHIVE_LAST_PART_SHA" ]]; then
      if command -v sha256sum >/dev/null 2>&1; then
        if ! sha256sum -c "$ARCHIVE_LAST_PART_SHA" >/dev/null; then
          warn "verify failed: split part checksum mismatch"
          rc=1
        fi
      fi
    fi

    if ! cat_split_parts_stream "$ARCHIVE_LAST_PART_LIST" | tar -tzf - >/dev/null; then
      warn "verify failed: split tar stream test failed"
      rc=1
    fi
    if ! cat_split_parts_stream "$ARCHIVE_LAST_PART_LIST" | tar -tzf - | tar_list_has_run_folder "$run_name"; then
      warn "verify failed: split archive missing run folder: $run_name"
      rc=1
    fi
    tar_entries="$(cat_split_parts_stream "$ARCHIVE_LAST_PART_LIST" | tar -tzf - | awk 'substr($0,length($0),1)!="/" {c++} END{print c+0}')"
  else
    [[ -s "$out_tar" ]] || {
      warn "verify failed: archive missing/empty: $out_tar"
      return 1
    }
    if ! tar -tzf "$out_tar" >/dev/null; then
      warn "verify failed: tar test failed: $out_tar"
      rc=1
    fi
    if ! tar -tzf "$out_tar" | tar_list_has_run_folder "$run_name"; then
      warn "verify failed: archive missing run folder: $run_name"
      rc=1
    fi
    if ! sha_verify "$out_tar" "$out_sha"; then
      warn "verify failed: checksum mismatch: $out_tar"
      rc=1
    fi
    tar_entries="$(tar -tzf "$out_tar" | awk 'substr($0,length($0),1)!="/" {c++} END{print c+0}')"
  fi

  if [[ "$tar_entries" -lt "$source_entries" ]]; then
    warn "verify failed: tar entries ($tar_entries) < source entries ($source_entries)"
    rc=1
  fi
  return "$rc"
}

delete_source_run() {
  local src_run="$1"
  local run_name="$2"
  [[ -d "$src_run" ]] || {
    warn "source run already missing, skip delete: $src_run"
    return 0
  }

  case "$src_run" in
    "$RUN_ROOT"/*) ;;
    *)
      warn "refuse delete outside RUN_ROOT: $src_run"
      return 1
      ;;
  esac

  if [[ "$(basename "$src_run")" != "$run_name" ]]; then
    warn "refuse delete with basename mismatch: $src_run"
    return 1
  fi

  log "  [delete-source] $src_run"
  run_cmd rm -rf "$src_run"
  TOTAL_SOURCE_DELETED=$((TOTAL_SOURCE_DELETED + 1))
  return 0
}

process_one_run() {
  local src_run="$1"
  local run_name
  run_name="$(basename "$src_run")"

  echo "========================================"
  echo "Processing run: $run_name"
  echo "========================================"

  if [[ ! -d "$src_run" ]]; then
    warn "run dir missing: $src_run"
    return 1
  fi

  if [[ "$DO_CLEAN" -eq 1 ]]; then
    local remove_list=()
    local f rel
    while IFS= read -r -d '' f; do
      rel="${f#$src_run/}"
      should_skip_clean_rel "$rel" && remove_list+=("$f")
    done < <(find "$src_run" -mindepth 1 -type f -print0)
    remove_files "config clean rules" "${remove_list[@]:-}"
  fi

  if [[ "$DO_COMPRESS" -eq 1 ]]; then
    local nc_batch=()
    local batch_count=0
    local saw_nc=0
    local f
    while IFS= read -r f; do
      saw_nc=1
      nc_batch+=("$f")
      batch_count=$((batch_count + 1))
      if [[ "$batch_count" -ge "$COMPRESS_BATCH_SIZE" ]]; then
        compress_source_files_to_stage "$src_run" "$src_run" "${nc_batch[@]}" || true
        nc_batch=()
        batch_count=0
      fi
    done < <(collect_source_nc_files "$src_run")
    if [[ "$batch_count" -gt 0 ]]; then
      compress_source_files_to_stage "$src_run" "$src_run" "${nc_batch[@]}" || true
    elif [[ "$saw_nc" -eq 0 ]]; then
      compress_source_files_to_stage "$src_run" "$src_run" || true
    fi
    cleanup_orphan_tmps "$src_run"
  fi

  if [[ "$DO_ARCHIVE" -eq 1 ]]; then
    local arch_rc=0
    archive_run "$run_name" "$src_run" || arch_rc=$?
    if [[ "$arch_rc" -ne 0 && "$arch_rc" -ne 3 ]]; then
      warn "archive failed for run: $run_name"
      return 1
    fi
  fi

  if [[ "$DELETE_SOURCE_AFTER_VERIFY" -eq 1 ]]; then
    if [[ "$DO_ARCHIVE" -ne 1 ]]; then
      warn "delete-source requested but archive disabled; source kept: $src_run"
    elif [[ "$ARCHIVE_LAST_WRITTEN" -ne 1 ]]; then
      if [[ "$ARCHIVE_LAST_VERIFIED_OK" -ne 1 ]]; then
        warn "source delete skipped (archive not newly written/verified): $src_run"
      else
        if verify_archived_run "$run_name" "$src_run" "$ARCHIVE_LAST_TAR" "$ARCHIVE_LAST_SHA"; then
          delete_source_run "$src_run" "$run_name" || return 1
        else
          warn "verification failed; source kept: $src_run"
          return 1
        fi
      fi
    else
      if verify_archived_run "$run_name" "$src_run" "$ARCHIVE_LAST_TAR" "$ARCHIVE_LAST_SHA"; then
        delete_source_run "$src_run" "$run_name" || return 1
      else
        warn "verification failed; source kept: $src_run"
        return 1
      fi
    fi
  fi

  return 0
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --project-root) PROJECT_ROOT="$2"; shift 2 ;;
      --run-root) RUN_ROOT="$2"; shift 2 ;;
      --archive-root) ARCHIVE_ROOT="$2"; shift 2 ;;
      --run-glob) RUN_GLOB="$2"; shift 2 ;;

      --clean) DO_CLEAN=1; shift ;;
      --no-clean) DO_CLEAN=0; shift ;;
      --compress) DO_COMPRESS=1; shift ;;
      --no-compress) DO_COMPRESS=0; shift ;;
      --archive) DO_ARCHIVE=1; shift ;;
      --no-archive) DO_ARCHIVE=0; shift ;;
      --dry-run) DRY_RUN=1; shift ;;

      --compress-scope) COMPRESS_SCOPE="$2"; shift 2 ;;
      --compress-name-glob) COMPRESS_NAME_GLOB="$2"; shift 2 ;;
      --deflate) DEFLATE_LEVEL="$2"; shift 2 ;;
      --no-shuffle) USE_SHUFFLE=0; shift ;;
      --workers) NWORKERS="$2"; shift 2 ;;
      --compress-batch-size) COMPRESS_BATCH_SIZE="$2"; shift 2 ;;
      --min-free-mb) MIN_FREE_MB="$2"; shift 2 ;;
      --no-skip-compressed) SKIP_ALREADY_COMPRESSED=0; shift ;;
      --no-verify-deflate) VERIFY_DEFLATE=0; shift ;;
      --no-fail-copy) COMPRESS_FAIL_COPY_ON_ERROR=0; shift ;;
      --compress-include-clean-matches) COMPRESS_SKIP_CLEAN_MATCHES=0; shift ;;
      --no-dereference-nc-symlinks) DEREFERENCE_NC_SYMLINKS=0; shift ;;

      --archive-max-gb) ARCHIVE_MAX_GB="$2"; shift 2 ;;
      --overwrite-archive) OVERWRITE_ARCHIVE=1; shift ;;
      --delete-source-after-verify) DELETE_SOURCE_AFTER_VERIFY=1; shift ;;
      --no-delete-source-after-verify) DELETE_SOURCE_AFTER_VERIFY=0; shift ;;

      --mode) MODE="$2"; shift 2 ;;
      --submit) MODE="submit"; shift ;;
      --sbatch-arg) SBATCH_EXTRA_ARGS+=("$2"); shift 2 ;;

      --quiet) VERBOSE=0; shift ;;
      -h|--help) usage; exit 0 ;;
      --*) err "Unknown arg: $1"; usage; exit 2 ;;
      *) RUN_NAMES+=("$1"); shift ;;
    esac
  done

  [[ "$MODE" == "interactive" || "$MODE" == "submit" ]] || {
    err "--mode must be interactive or submit"
    exit 2
  }
}

validate() {
  [[ -d "$RUN_ROOT" ]] || { err "RUN_ROOT not found: $RUN_ROOT"; exit 1; }
  case "$DEFLATE_LEVEL" in 0|1|2|3|4|5|6|7|8|9) ;; *) err "--deflate must be 0..9"; exit 1 ;; esac
  [[ "$COMPRESS_SCOPE" == "outputs" || "$COMPRESS_SCOPE" == "all" ]] || {
    err "--compress-scope must be outputs|all"
    exit 1
  }
  [[ "$ARCHIVE_MAX_GB" =~ ^[0-9]+$ ]] || { err "--archive-max-gb must be integer >=0"; exit 1; }
  [[ "$NWORKERS" =~ ^[0-9]+$ && "$NWORKERS" -ge 1 ]] || { err "--workers must be >=1"; exit 1; }
  [[ "$COMPRESS_BATCH_SIZE" =~ ^[0-9]+$ && "$COMPRESS_BATCH_SIZE" -ge 1 ]] || { err "--compress-batch-size must be >=1"; exit 1; }
  [[ "$MIN_FREE_MB" =~ ^[0-9]+$ ]] || { err "--min-free-mb must be integer"; exit 1; }

  need_cmd find
  need_cmd cp
  need_cmd tar
  if [[ "$DO_COMPRESS" -eq 1 ]]; then
    need_cmd nccopy
    need_cmd ncdump
    need_cmd readlink
  fi

  run_cmd mkdir -p "$ARCHIVE_ROOT"
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

  echo "RUN_ROOT               : $RUN_ROOT"
  echo "ARCHIVE_ROOT           : $ARCHIVE_ROOT"
  echo "MODE                   : $MODE"
  echo "Actions                : clean=$DO_CLEAN compress=$DO_COMPRESS archive=$DO_ARCHIVE dry_run=$DRY_RUN"
  echo "Compression            : scope=$COMPRESS_SCOPE name_glob=$COMPRESS_NAME_GLOB deflate=$DEFLATE_LEVEL shuffle=$USE_SHUFFLE workers=$NWORKERS"
  echo "Compression batch size : $COMPRESS_BATCH_SIZE"
  echo "Compress skip clean    : $COMPRESS_SKIP_CLEAN_MATCHES"
  echo "Dereference nc symlink : $DEREFERENCE_NC_SYMLINKS"
  echo "Skip compressed        : $SKIP_ALREADY_COMPRESSED"
  echo "Min free MB buffer     : $MIN_FREE_MB"
  echo "Archive max GB/part    : $ARCHIVE_MAX_GB (0=single tar.gz)"
  echo "Overwrite archive      : $OVERWRITE_ARCHIVE"
  echo "Delete source after    : $DELETE_SOURCE_AFTER_VERIFY"
  if [[ "${#RUN_NAMES[@]}" -gt 0 ]]; then
    echo "Run selection          : ${RUN_NAMES[*]}"
  else
    echo "Run selection          : glob=$RUN_GLOB"
  fi
  echo

  local runs=()
  local r
  if [[ "${#RUN_NAMES[@]}" -gt 0 ]]; then
    for r in "${RUN_NAMES[@]}"; do
      if [[ -d "$RUN_ROOT/$r" ]]; then
        runs+=("$RUN_ROOT/$r")
      else
        warn "run dir not found, skip: $RUN_ROOT/$r"
      fi
    done
  else
    while IFS= read -r r; do
      runs+=("$r")
    done < <(find "$RUN_ROOT" -maxdepth 1 -mindepth 1 -type d -name "$RUN_GLOB" | sort)
  fi

  [[ "${#runs[@]}" -gt 0 ]] || { err "No run dirs selected under $RUN_ROOT"; exit 1; }

  local run_path
  for run_path in "${runs[@]}"; do
    TOTAL_RUNS=$((TOTAL_RUNS + 1))
    if process_one_run "$run_path"; then
      TOTAL_RUNS_OK=$((TOTAL_RUNS_OK + 1))
    else
      warn "run failed: $run_path"
    fi
  done

  echo
  echo "=== Summary ==="
  echo "runs processed         : $TOTAL_RUNS_OK / $TOTAL_RUNS"
  echo "files removed          : $TOTAL_FILES_REMOVED"
  echo "nc found               : $TOTAL_NC_FOUND"
  echo "nc compressed          : $TOTAL_NC_COMPRESSED"
  echo "nc skipped             : $TOTAL_NC_SKIPPED"
  echo "nc fallback copied     : $TOTAL_NC_FALLBACK_COPIED"
  echo "nc failed              : $TOTAL_NC_FAILED"
  echo "archives created       : $TOTAL_ARCHIVES"
  echo "source runs deleted    : $TOTAL_SOURCE_DELETED"
  echo "DONE."
}

main "$@"
