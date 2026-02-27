#!/bin/bash
#SBATCH -J unzip_RUN22e_parts
#SBATCH -o unzip.%j.out
#SBATCH -e unzip.%j.err
#SBATCH -p cpu-small
#SBATCH -N 1
#SBATCH -n 16
#SBATCH -t 3-00:00:00
#SBATCH --mail-type=all
#SBATCH -A gts-ed70
#SBATCH --mail-user=kmpark19900616@gmail.com

SCRIPT_NAME="$(basename "$0")"

usage() {
  cat <<EOF
Usage:
  ${SCRIPT_NAME} [PREFIX] [START_PART] [END_PART]
  ${SCRIPT_NAME} interactive [PREFIX] [START_PART] [END_PART]
  ${SCRIPT_NAME} submit [PREFIX] [START_PART] [END_PART]

Examples:
  ${SCRIPT_NAME} RUN22f 34          # run now (local/interactive shell), part 34 -> end
  ${SCRIPT_NAME} submit RUN22f 34   # submit to Slurm, part 34 -> end
  sbatch ${SCRIPT_NAME} RUN22f 34   # direct Slurm submit (still supported)
EOF
}

case "${1:-}" in
  -h|--help)
    usage
    exit 0
    ;;
  submit)
    shift
    if ! command -v sbatch >/dev/null 2>&1; then
      echo "ERROR: sbatch not found in PATH" >&2
      exit 1
    fi
    echo "Submitting to Slurm: sbatch $0 $*"
    exec sbatch "$0" "$@"
    ;;
  interactive|run)
    shift
    ;;
esac

module list
pwd
date

set -euo pipefail

# -----------------------
# User settings
# -----------------------
PREFIX="${1:-RUN22e}"          # usage: unzip.sh [submit|interactive] [PREFIX] [START_PART] [END_PART]
START_PART_ARG="${2:-}"        # examples: 34, 0034, part0034, RUN22f.part0034.tar.gz
END_PART_ARG="${3:-}"          # optional; same accepted formats as START_PART_ARG
OUT_DIR="${OUT_DIR:-./extracted}"

mkdir -p "$OUT_DIR"

# -----------------------
# Choose checksum tool
# -----------------------
if command -v sha256sum >/dev/null 2>&1; then
  SHA_CHECK=(sha256sum -c)
elif command -v shasum >/dev/null 2>&1; then
  SHA_CHECK=(shasum -a 256 -c)
else
  echo "ERROR: sha256sum or shasum not found in PATH" >&2
  exit 1
fi

# -----------------------
# Find parts
# -----------------------
mapfile -t PARTS < <(ls -1 "${PREFIX}.part"*.tar.gz 2>/dev/null | sort)

if [[ ${#PARTS[@]} -eq 0 ]]; then
  echo "ERROR: No ${PREFIX}.partXXXX.tar.gz files found in $(pwd)" >&2
  exit 1
fi

extract_part_num() {
  local value="$1"
  local digits
  if [[ "$value" =~ part([0-9]+) ]]; then
    digits="${BASH_REMATCH[1]}"
  elif [[ "$value" =~ ^[0-9]+$ ]]; then
    digits="$value"
  else
    return 1
  fi
  printf '%d\n' "$((10#$digits))"
}

START_PART_NUM=""
END_PART_NUM=""

if [[ -n "$START_PART_ARG" ]]; then
  if ! START_PART_NUM="$(extract_part_num "$START_PART_ARG")"; then
    echo "ERROR: Invalid START_PART: $START_PART_ARG" >&2
    exit 1
  fi
fi

if [[ -n "$END_PART_ARG" ]]; then
  if ! END_PART_NUM="$(extract_part_num "$END_PART_ARG")"; then
    echo "ERROR: Invalid END_PART: $END_PART_ARG" >&2
    exit 1
  fi
fi

if [[ -n "$START_PART_NUM" && -n "$END_PART_NUM" && "$START_PART_NUM" -gt "$END_PART_NUM" ]]; then
  echo "ERROR: START_PART ($START_PART_NUM) is greater than END_PART ($END_PART_NUM)" >&2
  exit 1
fi

if [[ -n "$START_PART_NUM" || -n "$END_PART_NUM" ]]; then
  FILTERED_PARTS=()
  for TAR in "${PARTS[@]}"; do
    if [[ ! "$TAR" =~ \.part([0-9]+)\.tar\.gz$ ]]; then
      continue
    fi
    PART_NUM=$((10#${BASH_REMATCH[1]}))
    if [[ -n "$START_PART_NUM" && "$PART_NUM" -lt "$START_PART_NUM" ]]; then
      continue
    fi
    if [[ -n "$END_PART_NUM" && "$PART_NUM" -gt "$END_PART_NUM" ]]; then
      continue
    fi
    FILTERED_PARTS+=("$TAR")
  done
  PARTS=("${FILTERED_PARTS[@]}")
fi

if [[ ${#PARTS[@]} -eq 0 ]]; then
  echo "ERROR: No parts matched the requested range for prefix ${PREFIX}" >&2
  exit 1
fi

echo "Found ${#PARTS[@]} tar.gz parts for prefix: ${PREFIX}"
echo "Extraction directory: ${OUT_DIR}"
if [[ -n "$START_PART_NUM" || -n "$END_PART_NUM" ]]; then
  echo "Selected range: ${START_PART_NUM:-start} to ${END_PART_NUM:-end}"
fi
echo

# -----------------------
# Extract each part
# -----------------------
for TAR in "${PARTS[@]}"; do
  BASE="$(basename "$TAR" .tar.gz)"
  DEST="${OUT_DIR}/${BASE}"

  echo "==> Processing: $TAR"

  # Verify checksum if present
  if [[ -f "${TAR}.sha256" ]]; then
    echo "  - verifying checksum: ${TAR}.sha256"
    ( cd "$(dirname "$TAR")" && "${SHA_CHECK[@]}" "$(basename "$TAR").sha256" )
  else
    echo "  - no checksum file found (skipping verify)"
  fi

  # Extract
  echo "  - extracting to: $DEST"
  mkdir -p "$DEST"
  tar -xzf "$TAR" -C "$DEST"

  echo "  - done"
  echo
done

date
echo "All parts extracted successfully."
