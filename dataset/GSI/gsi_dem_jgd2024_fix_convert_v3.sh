#!/usr/bin/env bash
set -euo pipefail

# GSI JPGIS DEM (JGD2024-tag) -> GeoTIFF (multi-zip, macOS Bash 3.2 safe)
# - Accepts: a single .zip, a directory of .zip, or a directory of XML/GML
# - For each ZIP: unpack -> patch jgd2024 -> re-zip -> jpgis-dem -> <name>.tif
# - For XML/GML: patch -> jpgis-dem -> <name>.tif
# - Builds a VRT of all produced TIFFs and (optionally) warps to a target CRS
#
# Requires: jpgis-dem, unzip, zip, sed, gdalbuildvrt, gdalwarp

usage(){ echo "Usage: $0 --in <zip_or_dir> --out <out_dir> [--name mosaic_name] [--crs EPSG:XXXX]"; exit 1; }

IN=""
OUT=""
NAME="dem_mosaic"
CRS=""

while [ $# -gt 0 ]; do
  case "$1" in
    --in) IN="$2"; shift 2;;
    --out) OUT="$2"; shift 2;;
    --name) NAME="$2"; shift 2;;
    --crs) CRS="$2"; shift 2;;
    *) usage;;
  esac
done

[ -z "${IN}" ] && usage
[ -z "${OUT}" ] && usage

for req in jpgis-dem unzip zip gdalbuildvrt gdalwarp; do
  command -v "$req" >/dev/null 2>&1 || { echo "Missing: $req"; exit 2; }
done

# Normalize paths and create dirs
mkdir -p "${OUT}"
OUT="$(cd "${OUT}" && pwd)"
# resolve IN to absolute
IN_DIRNAME="$(cd "$(dirname "${IN}")" && pwd)"
IN_BASENAME="$(basename "${IN}")"
IN_ABS="${IN_DIRNAME}/${IN_BASENAME}"

TIF_DIR="${OUT}/_tif"
mkdir -p "${TIF_DIR}"

# detect GNU vs BSD sed
if sed --version >/dev/null 2>&1; then
  SED_INPLACE=(sed -i)
else
  SED_INPLACE=(sed -i '')
fi

patch_srs_in_tree() {
  # $1: folder
  # replace fguuid:jgd2024.bl -> fguuid:jgd2011.bl
  find "$1" -type f \( -iname "*.xml" -o -iname "*.gml" \) -print0 | \
  xargs -0 "${SED_INPLACE[@]}" 's/fguuid:jgd2024\.bl/fguuid:jgd2011.bl/g'
}

process_one_zip() {
  # $1: abs path to zip
  local zip_in="$1"
  local base="$(basename "${zip_in}")"
  base="${base%.*}"
  local out_tif="${TIF_DIR}/${base}.tif"
  if [ -s "${out_tif}" ]; then
    echo "[skip] ${base}.tif exists"
    return 0
  fi

  local TMP
  TMP="$(mktemp -d)"
  trap 'rm -rf "${TMP}"' RETURN

  unzip -oq "${zip_in}" -d "${TMP}/work"
  patch_srs_in_tree "${TMP}/work"
  (cd "${TMP}/work" && zip -qr "${TMP}/patched.zip" .)
  echo "[convert] ${base}.zip -> ${base}.tif"
  jpgis-dem xml2tif "${TMP}/patched.zip" "${out_tif}" >/dev/null
}

process_one_xml() {
  # $1: abs path to xml/gml
  local xml_in="$1"
  local base="$(basename "${xml_in}")"
  base="${base%.*}"
  local out_tif="${TIF_DIR}/${base}.tif"
  if [ -s "${out_tif}" ]; then
    echo "[skip] ${base}.tif exists"
    return 0
  fi

  local TMP
  TMP="$(mktemp -d)"
  trap 'rm -rf "${TMP}"' RETURN

  mkdir -p "${TMP}/work"
  cp "${xml_in}" "${TMP}/work/$(basename "${xml_in}")"
  patch_srs_in_tree "${TMP}/work"
  echo "[convert] ${base}.xml -> ${base}.tif"
  jpgis-dem xml2tif "${TMP}/work/$(basename "${xml_in}")" "${out_tif}" >/dev/null
}

# --- Decide what to process ---
zip_list_file="$(mktemp)"; : > "${zip_list_file}"
xml_list_file="$(mktemp)"; : > "${xml_list_file}"
trap 'rm -f "${zip_list_file}" "${xml_list_file}"' EXIT

if [ -f "${IN_ABS}" ]; then
  case "${IN_ABS##*.}" in
    zip|ZIP) echo "${IN_ABS}" >> "${zip_list_file}";;
    xml|XML|gml|GML) echo "${IN_ABS}" >> "${xml_list_file}";;
    *) echo "Input file must be .zip/.xml/.gml"; exit 3;;
  esac
elif [ -d "${IN_ABS}" ]; then
  # collect zips at top level
  find "${IN_ABS}" -maxdepth 1 -type f \( -iname "*.zip" \) -print > "${zip_list_file}"
  # collect xml/gml recursively
  find "${IN_ABS}" -type f \( -iname "*.xml" -o -iname "*.gml" \) -print > "${xml_list_file}"
else
  echo "Input must be a .zip or a directory."; exit 3
fi

# --- Convert ZIPs ---
if [ -s "${zip_list_file}" ]; then
  echo "== Converting ZIP(s)"
  while IFS= read -r z; do
    [ -z "${z}" ] && continue
    process_one_zip "${z}"
  done < "${zip_list_file}"
fi

# --- Convert XML/GML ---
if [ -s "${xml_list_file}" ]; then
  echo "== Converting XML/GML file(s)"
  while IFS= read -r x; do
    [ -z "${x}" ] && continue
    process_one_xml "${x}"
  done < "${xml_list_file}"
fi

# --- Build VRT from all produced TIFFs ---
tif_list_file="$(mktemp)"; : > "${tif_list_file}"
find "${TIF_DIR}" -maxdepth 1 -type f -iname "*.tif" -print | sort > "${tif_list_file}"
if ! [ -s "${tif_list_file}" ]; then
  echo "No TIFFs produced."; exit 4
fi

VRT="${OUT}/${NAME}.vrt"
echo "[vrt] ${VRT}"
gdalbuildvrt -q -input_file_list "${tif_list_file}" "${VRT}"

# --- Optional warp ---
if [ -n "${CRS}" ]; then
  MOS="${OUT}/${NAME}_${CRS//:/_}.tif"
  echo "[warp] -> ${MOS} (${CRS})"
  # Add -tr 5 5 -tap and -te/-te_srs here if you want fixed grid & clip
  gdalwarp -q -t_srs "${CRS}" -r bilinear -multi -wo NUM_THREADS=ALL_CPUS \
    -of GTiff -co TILED=YES -co COMPRESS=LZW -co BIGTIFF=IF_SAFER \
    "${VRT}" "${MOS}"
  echo "Final: ${MOS}"
else
  echo "Final VRT: ${VRT}"
fi
