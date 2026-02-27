#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  gsi_merge_tifs.sh --in-dir DIR [options]

Options:
  --in-dir DIR         Directory containing GeoTIFF tiles (required)
  --out-dir DIR        Output directory (default: same as --in-dir)
  --pattern GLOB       TIFF filename glob (default: FG-GML-5741*.tif)
  --name NAME          Output base name without extension (default: merged_5741)
  --crs EPSG:XXXX      Optional output CRS for warp (default: keep source CRS)
  --overwrite          Overwrite existing outputs
  --dry-run            Print commands without executing
  -h, --help

Outputs:
  <out-dir>/<name>.vrt
  <out-dir>/<name>.tif               (when --crs is omitted)
  <out-dir>/<name>_EPSG_XXXX.tif     (when --crs is provided)
USAGE
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || {
    echo "[ERROR] Missing required command: $1" >&2
    exit 2
  }
}

run_cmd() {
  if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "[DRY] $*"
  else
    "$@"
  fi
}

IN_DIR=""
OUT_DIR=""
PATTERN="FG-GML-5741*.tif"
NAME="merged_5741"
CRS=""
OVERWRITE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --in-dir) IN_DIR="$2"; shift 2 ;;
    --out-dir) OUT_DIR="$2"; shift 2 ;;
    --pattern) PATTERN="$2"; shift 2 ;;
    --name) NAME="$2"; shift 2 ;;
    --crs) CRS="$2"; shift 2 ;;
    --overwrite) OVERWRITE=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[ERROR] Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

[[ -n "$IN_DIR" ]] || {
  echo "[ERROR] --in-dir is required" >&2
  usage
  exit 2
}

[[ -d "$IN_DIR" ]] || {
  echo "[ERROR] Input directory not found: $IN_DIR" >&2
  exit 1
}

if [[ -z "$OUT_DIR" ]]; then
  OUT_DIR="$IN_DIR"
fi
mkdir -p "$OUT_DIR"

IN_DIR="$(cd "$IN_DIR" && pwd)"
OUT_DIR="$(cd "$OUT_DIR" && pwd)"

need_cmd find
if [[ "$DRY_RUN" -ne 1 ]]; then
  need_cmd gdalbuildvrt
  if [[ -n "$CRS" ]]; then
    need_cmd gdalwarp
  else
    need_cmd gdal_translate
  fi
else
  if ! command -v gdalbuildvrt >/dev/null 2>&1; then
    echo "[WARN] gdalbuildvrt not found (dry-run mode: command preview only)"
  fi
  if [[ -n "$CRS" ]] && ! command -v gdalwarp >/dev/null 2>&1; then
    echo "[WARN] gdalwarp not found (dry-run mode: command preview only)"
  fi
  if [[ -z "$CRS" ]] && ! command -v gdal_translate >/dev/null 2>&1; then
    echo "[WARN] gdal_translate not found (dry-run mode: command preview only)"
  fi
fi

tmp_list="$(mktemp)"
trap 'rm -f "$tmp_list"' EXIT

find "$IN_DIR" -maxdepth 1 -type f -name "$PATTERN" ! -name "*.aux.xml" | sort > "$tmp_list"
n_tiles="$(wc -l < "$tmp_list" | awk '{print $1}')"

if [[ "$n_tiles" -eq 0 ]]; then
  echo "[ERROR] No files matched pattern '$PATTERN' in $IN_DIR" >&2
  exit 1
fi

VRT="$OUT_DIR/${NAME}.vrt"
OUT_TIF="$OUT_DIR/${NAME}.tif"
if [[ -n "$CRS" ]]; then
  OUT_TIF="$OUT_DIR/${NAME}_${CRS//:/_}.tif"
fi

if [[ "$OVERWRITE" -ne 1 ]]; then
  [[ ! -e "$VRT" ]] || { echo "[ERROR] Exists: $VRT (use --overwrite)"; exit 1; }
  [[ ! -e "$OUT_TIF" ]] || { echo "[ERROR] Exists: $OUT_TIF (use --overwrite)"; exit 1; }
else
  run_cmd rm -f "$VRT" "$OUT_TIF"
fi

echo "Input dir    : $IN_DIR"
echo "Pattern      : $PATTERN"
echo "Tile count   : $n_tiles"
echo "Output dir   : $OUT_DIR"
echo "VRT          : $VRT"
echo "Output TIFF  : $OUT_TIF"
if [[ -n "$CRS" ]]; then
  echo "Target CRS   : $CRS"
fi
echo

run_cmd gdalbuildvrt -q -input_file_list "$tmp_list" "$VRT"

if [[ -n "$CRS" ]]; then
  run_cmd gdalwarp -q -t_srs "$CRS" -r bilinear -multi -wo NUM_THREADS=ALL_CPUS \
    -of GTiff -co TILED=YES -co COMPRESS=LZW -co BIGTIFF=IF_SAFER \
    "$VRT" "$OUT_TIF"
else
  run_cmd gdal_translate -q -of GTiff \
    -co TILED=YES -co COMPRESS=LZW -co BIGTIFF=IF_SAFER \
    "$VRT" "$OUT_TIF"
fi

echo "Done."
