#!/usr/bin/env python3
"""
Analyze trusted-source JODC current NC observations with boundary-based filtering.

Core workflow:
1) Select trusted sources from nc_trusted_sources.csv (or explicit --sources).
2) Select files via nc_filename_classification.csv.
3) Parse current vectors from NC files.
4) Build track-level points (time, lon, lat) and obs-level vectors (time, lon, lat, depth, u, v, ...).
5) Read boundary polyline from SOB.shp, extract closed ring(s), and filter inside points using
   pylib.inside_polygon.
6) Write summary tables, NPZ bundles, and quicklook diagnostics.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import struct
import warnings
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from pylib import ReadNC, date2num, inside_polygon, num2date, savez, zdata


KNOTS_TO_MS = 0.514444
FILL_CUTOFF = 9.0e4


USER_CONFIG: Dict[str, Any] = {
    "enable": True,
    "nc_dir": "/Users/kpark/Documents/Projects/Active/AIMEC_TohokuCoast/01_Data/JODC/Current/vector-140E-145E35N-40N_NCFiles",
    "trusted_csv": "/Users/kpark/Documents/Codes/coastal-ocean-utils/dataset/JODC/nc_trusted_sources.csv",
    "classification_csv": "/Users/kpark/Documents/Codes/coastal-ocean-utils/dataset/JODC/nc_filename_classification.csv",
    "boundary_shp": "/Users/kpark/Documents/Projects/Active/AIMEC_TohokuCoast/01_Data/Grid/shp/SOB.shp",
    "trusted_tier": "core",
    "sources": None,  # Optional explicit source list; overrides tier if set
    "data_types": None,  # Optional filter list, e.g., ["CA","CU"]
    "start": "2012-02-01",  # YYYY-MM-DD or YYYY-MM-DD HH:MM[:SS]
    "end": "2014-12-31",
    "min_record_length_hours": 6.0,  # segment temporal span threshold; <=0 disables
    "min_depth": None,
    "max_depth": None,
    "inside_mode": "largest_closed_ring",  # or "all_closed_rings"
    "ring_close_tol": 1.0e-6,
    "outdir": "./trusted_current_analysis",
    "save_inside_npz": True,
    "plot": True,
    "max_track_plots_per_source": 3,
    "max_segment_plots_per_type": 4,
    "max_segment_plots_per_source": 0,  # <=0 means all segments for each source
    "plot_each_segment": True,
    "max_individual_segment_plots_per_source": 0,  # <=0 means all segments per source
    "plot_sample_max": 180000,
    "qc_keep_codes": [0],  # Official QF: 0=good,1=unknown,4=questionable,8=bad
    "qc_mode": "strict_both",  # or "either"
    "report_all_qc_codes": True,
    "depth_bins": [0.0, 15.0, 50.0, 20000.0],
    "depth_bin_labels": ["surface", "mid", "deep"],
    "export_collocation_csv": True,
    "export_collocation_npz": True,
}


DATA_TYPE_NAME = {
    "CA": "ADCP",
    "CD": "Drift by Vessels",
    "CU": "GEK",
    "CV": "ARGOS Drifting Buoy",
    "BE": "Vessel Cooling Water",
    "IT": "Intake Method",
}

# Track segmentation thresholds by observation type.
# A new segment starts when either condition is exceeded between consecutive points.
SEGMENT_THRESHOLDS = {
    "CA": {"gap_hours": 6.0, "jump_km": 35.0},
    "CU": {"gap_hours": 24.0, "jump_km": 120.0},
    "CD": {"gap_hours": 24.0, "jump_km": 180.0},
    "CV": {"gap_hours": 48.0, "jump_km": 250.0},
    "DEFAULT": {"gap_hours": 24.0, "jump_km": 120.0},
}


def _resolve_path(path_like: str) -> Path:
    p = Path(os.path.expanduser(str(path_like)))
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()


def _parse_bound(value: Optional[str], is_end: bool) -> Optional[datetime]:
    if value is None:
        return None
    text = str(value).strip()
    fmts = ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%Y-%m-%d")
    parsed = None
    used_fmt = None
    for fmt in fmts:
        try:
            parsed = datetime.strptime(text, fmt)
            used_fmt = fmt
            break
        except ValueError:
            continue
    if parsed is None:
        raise ValueError(f"Invalid datetime: {value}")
    if is_end and used_fmt == "%Y-%m-%d":
        parsed = parsed + timedelta(days=1) - timedelta(seconds=1)
    return parsed


def _safe_nc_attr(nc: Any, name: str, default: str = "") -> str:
    # netCDF4-like accessor
    try:
        return str(nc.getncattr(name)).strip()
    except Exception:
        pass

    # pylib.ReadNC(fmt=0) stores global attrs directly on the returned object
    try:
        if hasattr(nc, name):
            v = getattr(nc, name)
            return str(v).strip()
    except Exception:
        pass
    try:
        if isinstance(getattr(nc, "__dict__", None), dict) and name in nc.__dict__:
            return str(nc.__dict__[name]).strip()
    except Exception:
        pass
    return default


def _read_nc_quiet(path: Path) -> Any:
    """
    Read NC via pylib.ReadNC while silencing known non-fatal netCDF attribute-cast
    warnings (`valid_min`/`valid_max`) that are noisy for this dataset.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r".*valid_min not used.*",
            category=UserWarning,
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*valid_max not used.*",
            category=UserWarning,
        )
        return ReadNC(str(path))


def _get_nc_var(nc: Any, name: str) -> Optional[np.ndarray]:
    # netCDF4-like / ncfile(fmt=1) path
    try:
        if hasattr(nc, "variables") and name in nc.variables:
            v = nc.variables[name]
            # pylib wrapper variable may store values in .val
            if hasattr(v, "val"):
                return np.asarray(v.val)
            return np.asarray(v[:])
    except Exception:
        pass

    # pylib.ReadNC(fmt=0): variables are direct attrs; each var is zdata with .val
    try:
        if hasattr(nc, name):
            v = getattr(nc, name)
            if hasattr(v, "val"):
                return np.asarray(v.val)
            return np.asarray(v)
    except Exception:
        pass

    # Final fallback through __dict__ (defensive for pylib variants)
    try:
        d = getattr(nc, "__dict__", {})
        if isinstance(d, dict) and name in d:
            v = d[name]
            if hasattr(v, "val"):
                return np.asarray(v.val)
            return np.asarray(v)
    except Exception:
        pass

    return None


def _parse_obs_datetime(obs_date: np.ndarray, obs_time: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    n = len(obs_date)
    out = np.full(n, np.nan, dtype=float)
    idx: List[int] = []
    dts: List[datetime] = []
    c = Counter()

    for i in range(n):
        try:
            d = int(obs_date[i])
        except Exception:
            c["bad_obs_date_parse"] += 1
            continue
        if d < 18000101 or d > 21001231:
            c["bad_obs_date_range"] += 1
            continue

        t = 0
        try:
            t = int(obs_time[i])
        except Exception:
            c["bad_obs_time_parse"] += 1
            t = 0
        hh = t // 10000
        mm = (t % 10000) // 100
        ss = t % 100
        if hh > 23 or mm > 59 or ss > 59:
            c["bad_obs_time_range"] += 1
            continue

        try:
            dt = datetime.strptime(str(d), "%Y%m%d").replace(hour=hh, minute=mm, second=ss)
        except Exception:
            c["bad_obs_datetime"] += 1
            continue

        idx.append(i)
        dts.append(dt)

    if len(dts) > 0:
        nums = np.asarray(date2num(dts), dtype=float)
        out[np.asarray(idx, dtype=int)] = nums
    c["valid_obs_datetime"] = int(np.count_nonzero(np.isfinite(out)))
    return out, dict(c)


def _load_trusted_sources(trusted_csv: Path, tier: str) -> List[str]:
    out: List[str] = []
    with trusted_csv.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            src = str(row.get("source", "")).strip()
            src_tier = str(row.get("tier", "")).strip().lower()
            if src == "":
                continue
            if tier == "all" or src_tier == tier:
                out.append(src)
    # Preserve deterministic order while removing duplicates.
    seen = set()
    uniq = []
    for s in out:
        if s in seen:
            continue
        seen.add(s)
        uniq.append(s)
    return uniq


def _load_filename_source_map(classification_csv: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with classification_csv.open("r", encoding="utf-8", newline="") as f:
        rd = csv.DictReader(f)
        for row in rd:
            fn = str(row.get("filename", "")).strip()
            src = str(row.get("source_token", "")).strip()
            if fn == "":
                continue
            out[fn] = src
    return out


def _read_polyline_parts(shp_path: Path) -> Dict[str, Any]:
    data = shp_path.read_bytes()
    if len(data) < 100:
        raise RuntimeError(f"Shapefile too short: {shp_path}")

    header = data[:100]
    file_code = struct.unpack(">i", header[0:4])[0]
    if file_code != 9994:
        raise RuntimeError(f"Unexpected shapefile file_code={file_code} for {shp_path}")

    shape_type = struct.unpack("<i", header[32:36])[0]
    if shape_type not in (3, 13, 23):
        raise RuntimeError(
            f"Expected PolyLine shapefile (type 3/13/23) but got type={shape_type} in {shp_path}"
        )

    xmin, ymin, xmax, ymax = struct.unpack("<4d", header[36:68])
    parts: List[np.ndarray] = []
    rec_types = Counter()
    rec_count = 0

    pos = 100
    nbytes = len(data)
    while pos + 8 <= nbytes:
        rec_count += 1
        _, content_len_words = struct.unpack(">2i", data[pos : pos + 8])
        pos += 8
        content_len = content_len_words * 2
        if content_len <= 0 or pos + content_len > nbytes:
            break
        content = data[pos : pos + content_len]
        pos += content_len
        if len(content) < 4:
            continue

        rec_shape_type = struct.unpack("<i", content[:4])[0]
        rec_types[rec_shape_type] += 1
        if rec_shape_type == 0:
            continue
        if rec_shape_type not in (3, 13, 23):
            continue
        if len(content) < 44:
            continue

        num_parts, num_points = struct.unpack("<2i", content[36:44])
        if num_parts <= 0 or num_points <= 0:
            continue

        off = 44
        need = off + 4 * num_parts + 16 * num_points
        if len(content) < need:
            continue

        idx = np.frombuffer(content, dtype="<i4", count=num_parts, offset=off).astype(int)
        off += 4 * num_parts
        pts = np.frombuffer(content, dtype="<f8", count=2 * num_points, offset=off).reshape(num_points, 2).copy()

        for ip, s in enumerate(idx):
            e = int(idx[ip + 1]) if ip + 1 < len(idx) else int(num_points)
            s = int(s)
            if e - s < 2:
                continue
            parts.append(pts[s:e, :])

    if len(parts) == 0:
        raise RuntimeError(f"No polyline parts found in {shp_path}")

    return {
        "parts": parts,
        "bbox": (float(xmin), float(ymin), float(xmax), float(ymax)),
        "shape_type": int(shape_type),
        "record_count": int(rec_count),
        "record_types": dict(rec_types),
    }


def _is_closed_ring(xy: np.ndarray, tol: float) -> bool:
    if xy.ndim != 2 or xy.shape[0] < 4:
        return False
    dx = float(xy[0, 0] - xy[-1, 0])
    dy = float(xy[0, 1] - xy[-1, 1])
    return math.hypot(dx, dy) <= float(tol)


def _polygon_area_abs(xy: np.ndarray) -> float:
    if xy.ndim != 2 or xy.shape[0] < 4:
        return 0.0
    x = xy[:, 0]
    y = xy[:, 1]
    # Safe even when first=last.
    return 0.5 * float(abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])))


def _extract_closed_rings(parts: List[np.ndarray], tol: float) -> List[Dict[str, Any]]:
    rings: List[Dict[str, Any]] = []
    for i, seg in enumerate(parts):
        if not _is_closed_ring(seg, tol):
            continue
        area = _polygon_area_abs(seg)
        rings.append({"part_index": i, "xy": seg, "area": float(area), "npt": int(seg.shape[0])})
    rings.sort(key=lambda r: r["area"], reverse=True)
    return rings


def _inside_from_rings(
    points_xy: np.ndarray,
    rings: List[Dict[str, Any]],
    inside_mode: str,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    if len(rings) == 0:
        raise RuntimeError(
            "No closed polygon ring detected from boundary polyline. "
            "Try --inside-mode all_closed_rings or use a polygon boundary source."
        )

    n = points_xy.shape[0]
    mask_inside = np.zeros(n, dtype=bool)
    meta: Dict[str, Any] = {
        "inside_mode": inside_mode,
        "rings_used": 0,
        "selected_ring_part_index": None,
        "selected_ring_area": None,
    }

    if inside_mode == "largest_closed_ring":
        ring = rings[0]
        px = ring["xy"][:, 0]
        py = ring["xy"][:, 1]
        raw = np.asarray(inside_polygon(points_xy, px, py)).reshape(-1)
        mask_inside = raw == 1
        meta["rings_used"] = 1
        meta["selected_ring_part_index"] = int(ring["part_index"])
        meta["selected_ring_area"] = float(ring["area"])
    elif inside_mode == "all_closed_rings":
        for ring in rings:
            px = ring["xy"][:, 0]
            py = ring["xy"][:, 1]
            raw = np.asarray(inside_polygon(points_xy, px, py)).reshape(-1)
            mask_inside |= raw == 1
        meta["rings_used"] = len(rings)
        meta["selected_ring_part_index"] = -1
        meta["selected_ring_area"] = float(rings[0]["area"])
    else:
        raise ValueError(f"Unsupported inside mode: {inside_mode}")

    return mask_inside, meta


def _inside_from_all_parts_path(points_xy: np.ndarray, parts: List[np.ndarray]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Build a NaN-separated multi-part boundary path and evaluate inside mask.

    This uses `inside_polygon` intentionally. `inside_polygon` (method=0 default)
    delegates to matplotlib Path.contains_points, which can handle NaN-separated
    multipart paths in this workflow.
    """
    blocks: List[np.ndarray] = []
    for seg in parts:
        if seg.ndim != 2 or seg.shape[0] < 2:
            continue
        blocks.append(np.asarray(seg, dtype=float))
        blocks.append(np.array([[np.nan, np.nan]], dtype=float))
    if len(blocks) == 0:
        return np.zeros(points_xy.shape[0], dtype=bool), {"path_vertices": 0}

    path_xy = np.vstack(blocks[:-1])  # drop trailing separator
    raw = np.asarray(inside_polygon(points_xy, path_xy[:, 0], path_xy[:, 1])).reshape(-1)
    mask_inside = raw == 1
    meta = {
        "path_vertices": int(path_xy.shape[0]),
        "path_parts": int(len(parts)),
    }
    return mask_inside, meta


def _fmt_time(num: float) -> str:
    if not np.isfinite(num):
        return ""
    return num2date(float(num)).strftime("%Y-%m-%d %H:%M:%S")


def _parse_code_list(values: Any, default: Sequence[int]) -> List[int]:
    if values is None:
        return [int(x) for x in default]
    items: List[str] = []
    if isinstance(values, str):
        items = [values]
    elif isinstance(values, (list, tuple, np.ndarray)):
        items = [str(v) for v in values]
    else:
        items = [str(values)]

    out: List[int] = []
    seen = set()
    for token in items:
        for x in str(token).replace(",", " ").split():
            v = int(x)
            if v in seen:
                continue
            seen.add(v)
            out.append(v)
    if len(out) == 0:
        return [int(x) for x in default]
    return out


def _parse_depth_bins_and_labels(
    bins_cfg: Any,
    labels_cfg: Any,
) -> Tuple[np.ndarray, List[str]]:
    if bins_cfg is None:
        bins = np.array([0.0, 15.0, 50.0, 20000.0], dtype=float)
    else:
        toks: List[str]
        if isinstance(bins_cfg, str):
            toks = [x for x in bins_cfg.replace(",", " ").split() if x != ""]
        else:
            toks = []
            for v in bins_cfg:
                toks.extend([x for x in str(v).replace(",", " ").split() if x != ""])
        bins = np.asarray([float(x) for x in toks], dtype=float)

    if bins.size < 2:
        raise ValueError("depth_bins requires at least 2 edges.")
    if not np.all(np.isfinite(bins)):
        raise ValueError("depth_bins has non-finite values.")
    if not np.all(np.diff(bins) > 0.0):
        raise ValueError("depth_bins must be strictly increasing.")

    nbin = int(bins.size - 1)
    if labels_cfg is None:
        labels = [f"{bins[i]:g}-{bins[i+1]:g}m" for i in range(nbin)]
    else:
        if isinstance(labels_cfg, str):
            labels = [x.strip() for x in labels_cfg.split(",") if x.strip() != ""]
            if len(labels) == 1 and " " in labels_cfg:
                labels = [x for x in labels_cfg.split() if x != ""]
        else:
            labels = [str(x).strip() for x in labels_cfg if str(x).strip() != ""]
    if len(labels) != nbin:
        raise ValueError(
            f"depth_bin_labels length ({len(labels)}) must match n_bins ({nbin})."
        )
    return bins, labels


def _qc_pass_mask(qc_u: np.ndarray, qc_v: np.ndarray, keep_codes: Sequence[int], mode: str) -> np.ndarray:
    keep = np.asarray(list(keep_codes), dtype=int)
    mu = np.isin(qc_u, keep)
    mv = np.isin(qc_v, keep)
    if str(mode) == "either":
        return mu | mv
    return mu & mv


def _sample_indices(n: int, max_n: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    if max_n <= 0 or n <= max_n:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, max_n, dtype=int)


def _to_ym(year: int, month: int) -> str:
    return f"{int(year):04d}-{int(month):02d}"


def _compute_dt_stats(times_num: np.ndarray) -> Dict[str, float]:
    arr = np.asarray(times_num, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size < 2:
        return {
            "dt_count": 0,
            "dt_median_min": np.nan,
            "dt_p90_min": np.nan,
            "dt_mode_min": np.nan,
            "dt_mode_ratio": np.nan,
            "irregular": np.nan,
        }

    arr = np.sort(arr)
    dmin = np.diff(arr) * 24.0 * 60.0
    dmin = dmin[np.isfinite(dmin)]
    dmin = dmin[dmin > 0.0]
    if dmin.size == 0:
        return {
            "dt_count": 0,
            "dt_median_min": np.nan,
            "dt_p90_min": np.nan,
            "dt_mode_min": np.nan,
            "dt_mode_ratio": np.nan,
            "irregular": np.nan,
        }

    dt_median = float(np.nanmedian(dmin))
    dt_p90 = float(np.nanpercentile(dmin, 90.0))
    rounded = np.rint(dmin).astype(int)
    vals, counts = np.unique(rounded, return_counts=True)
    imax = int(np.argmax(counts))
    dt_mode = float(vals[imax])
    mode_ratio = float(counts[imax] / len(rounded))
    irregular = float(1 if mode_ratio < 0.7 else 0)

    return {
        "dt_count": int(len(dmin)),
        "dt_median_min": dt_median,
        "dt_p90_min": dt_p90,
        "dt_mode_min": dt_mode,
        "dt_mode_ratio": mode_ratio,
        "irregular": irregular,
    }


def _haversine_km(lon1: np.ndarray, lat1: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    r = 6371.0
    lam1 = np.deg2rad(lon1)
    lam2 = np.deg2rad(lon2)
    phi1 = np.deg2rad(lat1)
    phi2 = np.deg2rad(lat2)
    dlam = lam2 - lam1
    dphi = phi2 - phi1
    a = np.sin(dphi / 2.0) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlam / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(np.maximum(0.0, 1.0 - a)))
    return r * c


def _segment_track_points(
    times_num: np.ndarray,
    lon: np.ndarray,
    lat: np.ndarray,
    data_type: str,
) -> np.ndarray:
    n = len(times_num)
    seg_seq = np.zeros(n, dtype=int)
    if n == 0:
        return seg_seq

    th = SEGMENT_THRESHOLDS.get(str(data_type).upper(), SEGMENT_THRESHOLDS["DEFAULT"])
    gap_h = float(th["gap_hours"])
    jump_km = float(th["jump_km"])

    seg = 1
    seg_seq[0] = seg
    for i in range(1, n):
        split = False
        t0 = float(times_num[i - 1])
        t1 = float(times_num[i])
        x0 = float(lon[i - 1])
        x1 = float(lon[i])
        y0 = float(lat[i - 1])
        y1 = float(lat[i])

        if not (np.isfinite(t0) and np.isfinite(t1) and np.isfinite(x0) and np.isfinite(x1) and np.isfinite(y0) and np.isfinite(y1)):
            split = True
        else:
            dt_h = (t1 - t0) * 24.0
            if dt_h < -1.0e-9 or dt_h > gap_h:
                split = True
            else:
                d_km = float(_haversine_km(np.array([x0]), np.array([y0]), np.array([x1]), np.array([y1]))[0])
                if (not np.isfinite(d_km)) or d_km > jump_km:
                    split = True

        if split:
            seg += 1
        seg_seq[i] = seg
    return seg_seq


def _unique_mean(x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if len(x) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    f = np.isfinite(x) & np.isfinite(y)
    x = x[f]
    y = y[f]
    if len(x) == 0:
        return np.array([], dtype=float), np.array([], dtype=float)
    ux, inv = np.unique(x, return_inverse=True)
    s = np.bincount(inv, weights=y)
    c = np.bincount(inv)
    return ux, s / np.maximum(1, c)


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames))
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _ensure_plot_backend():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _plot_boundary(ax: Any, parts: Sequence[np.ndarray]) -> None:
    for seg in parts:
        ax.plot(seg[:, 0], seg[:, 1], "k-", lw=0.7, alpha=0.75)


def _source_color_map(plt: Any, sources: Sequence[str]) -> Dict[str, Any]:
    if len(sources) == 0:
        return {}
    cmap = plt.get_cmap("tab20", max(1, len(sources)))
    return {s: cmap(i) for i, s in enumerate(sources)}


def _plot_map_all(
    plt: Any,
    outpath: Path,
    parts: Sequence[np.ndarray],
    lon: np.ndarray,
    lat: np.ndarray,
    sample_max: int,
) -> None:
    idx = _sample_indices(len(lon), sample_max)
    fig, ax = plt.subplots(1, 1, figsize=(8.6, 7.2))
    _plot_boundary(ax, parts)
    if len(idx) > 0:
        ax.scatter(lon[idx], lat[idx], s=2, c="#2563eb", alpha=0.6, edgecolors="none")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"All Trusted Track Points (sampled={len(idx):,} / total={len(lon):,})")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def _plot_map_inside(
    plt: Any,
    outpath: Path,
    parts: Sequence[np.ndarray],
    lon: np.ndarray,
    lat: np.ndarray,
    source: np.ndarray,
    inside: np.ndarray,
    sources: Sequence[str],
    sample_max: int,
) -> None:
    idx_all = _sample_indices(len(lon), sample_max)
    idx_in = np.where(inside)[0]
    idx_in = idx_in[_sample_indices(len(idx_in), sample_max)]

    colors = _source_color_map(plt, sources)
    fig, ax = plt.subplots(1, 1, figsize=(8.8, 7.4))
    _plot_boundary(ax, parts)
    if len(idx_all) > 0:
        ax.scatter(lon[idx_all], lat[idx_all], s=2, c="0.80", alpha=0.5, edgecolors="none", label="All points")
    for s in sources:
        f = idx_in[source[idx_in] == s]
        if len(f) == 0:
            continue
        ax.scatter(lon[f], lat[f], s=5, color=colors[s], alpha=0.7, edgecolors="none", label=s)

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"Inside-Boundary Track Points (inside={int(np.count_nonzero(inside)):,})")
    ax.grid(alpha=0.25)
    if len(sources) <= 16:
        ax.legend(loc="best", fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def _plot_source_facets_inside(
    plt: Any,
    outpath: Path,
    parts: Sequence[np.ndarray],
    lon: np.ndarray,
    lat: np.ndarray,
    source: np.ndarray,
    inside: np.ndarray,
    sources: Sequence[str],
    sample_max: int,
) -> None:
    nsrc = len(sources)
    if nsrc == 0:
        fig, ax = plt.subplots(1, 1, figsize=(7, 5))
        _plot_boundary(ax, parts)
        ax.set_title("No selected sources")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        fig.tight_layout()
        fig.savefig(outpath, dpi=220)
        plt.close(fig)
        return

    ncol = min(4, nsrc)
    nrow = int(math.ceil(nsrc / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.0 * ncol, 3.6 * nrow), squeeze=False)
    for i, s in enumerate(sources):
        ax = axs[i // ncol, i % ncol]
        _plot_boundary(ax, parts)
        f = np.where((source == s) & inside)[0]
        if len(f) > 0:
            f = f[_sample_indices(len(f), sample_max)]
            ax.scatter(lon[f], lat[f], s=3, c="#dc2626", alpha=0.7, edgecolors="none")
        else:
            ax.text(0.5, 0.5, "No inside points", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(f"{s} (inside={len(f):,})")
        ax.grid(alpha=0.20)
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")

    for i in range(nsrc, nrow * ncol):
        axs[i // ncol, i % ncol].axis("off")

    fig.suptitle("Inside Points by Source", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def _plot_coverage_yearly(
    plt: Any,
    outpath: Path,
    years_arr: np.ndarray,
    source_arr: np.ndarray,
    inside_arr: np.ndarray,
    sources: Sequence[str],
) -> None:
    if len(years_arr) == 0:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))
        ax.text(0.5, 0.5, "No track points", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(outpath, dpi=220)
        plt.close(fig)
        return

    years_in = years_arr[inside_arr]
    src_in = source_arr[inside_arr]
    if years_in.size == 0:
        years_in = years_arr
        src_in = source_arr

    y0 = int(np.nanmin(years_in))
    y1 = int(np.nanmax(years_in))
    years = np.arange(y0, y1 + 1, dtype=int)
    y_to_i = {int(y): i for i, y in enumerate(years)}
    s_to_i = {s: i for i, s in enumerate(sources)}
    mat = np.zeros((len(sources), len(years)), dtype=int)
    for y, s in zip(years_in, src_in):
        if s not in s_to_i:
            continue
        j = y_to_i.get(int(y))
        if j is None:
            continue
        mat[s_to_i[s], j] += 1

    fig, ax = plt.subplots(1, 1, figsize=(max(9, 0.35 * len(years)), max(4.5, 0.35 * len(sources))))
    im = ax.imshow(mat, aspect="auto", interpolation="nearest", cmap="viridis")
    cbar = fig.colorbar(im, ax=ax, shrink=0.95)
    cbar.set_label("Inside track-point count")

    ax.set_yticks(np.arange(len(sources)))
    ax.set_yticklabels(sources)
    xt = np.arange(0, len(years), max(1, len(years) // 12))
    ax.set_xticks(xt)
    ax.set_xticklabels([str(years[k]) for k in xt], rotation=45, ha="right")
    ax.set_xlabel("Year")
    ax.set_ylabel("Source")
    ax.set_title("Yearly Coverage by Source (Inside Boundary)")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def _plot_track_samples_by_source(
    plt: Any,
    outpath: Path,
    parts: Sequence[np.ndarray],
    source_arr: np.ndarray,
    track_id_arr: np.ndarray,
    time_arr: np.ndarray,
    lon_arr: np.ndarray,
    lat_arr: np.ndarray,
    inside_arr: np.ndarray,
    sources: Sequence[str],
    max_track_plots_per_source: int,
) -> None:
    # Build inside counts per track.
    uniq_tid = np.unique(track_id_arr)
    track_inside_count: Dict[int, int] = {}
    for tid in uniq_tid:
        f = np.where(track_id_arr == tid)[0]
        track_inside_count[int(tid)] = int(np.count_nonzero(inside_arr[f]))

    nsrc = len(sources)
    ncol = min(4, max(1, nsrc))
    nrow = int(math.ceil(nsrc / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.9 * nrow), squeeze=False)

    for i, src in enumerate(sources):
        ax = axs[i // ncol, i % ncol]
        _plot_boundary(ax, parts)
        fsrc = np.where(source_arr == src)[0]
        tids = np.unique(track_id_arr[fsrc]).astype(int)
        tids = sorted(
            tids,
            key=lambda tid: (track_inside_count.get(int(tid), 0), np.count_nonzero(track_id_arr == tid)),
            reverse=True,
        )
        tids = tids[: max(1, int(max_track_plots_per_source))]

        n_drawn = 0
        for tid in tids:
            ff = np.where((track_id_arr == tid) & inside_arr)[0]
            if len(ff) == 0:
                continue
            tt = time_arr[ff]
            oo = np.argsort(tt)
            ff = ff[oo]
            tt = tt[oo]
            if len(ff) < 2:
                ax.plot(lon_arr[ff], lat_arr[ff], "o", ms=3, color="#0ea5e9", alpha=0.9)
                n_drawn += 1
                continue

            # Use normalized time for color in each track.
            tmin = float(np.nanmin(tt))
            tmax = float(np.nanmax(tt))
            denom = max(1.0e-12, tmax - tmin)
            cval = (tt - tmin) / denom
            ax.plot(lon_arr[ff], lat_arr[ff], "-", lw=0.8, color="0.5", alpha=0.5)
            ax.scatter(lon_arr[ff], lat_arr[ff], c=cval, s=8, cmap="viridis", alpha=0.85, edgecolors="none")
            n_drawn += 1

        if n_drawn == 0:
            ax.text(0.5, 0.5, "No inside tracks", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(f"{src}: sample tracks={n_drawn}")
        ax.set_xlabel("Lon")
        ax.set_ylabel("Lat")
        ax.grid(alpha=0.20)

    for i in range(nsrc, nrow * ncol):
        axs[i // ncol, i % ncol].axis("off")

    fig.suptitle("Representative Inside Tracks by Source (colored by relative time)", y=0.995)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def _plot_timestep_distribution(
    plt: Any,
    outpath: Path,
    track_rows: List[Dict[str, Any]],
    sources: Sequence[str],
) -> None:
    # Per-source arrays from per-track statistics.
    dt_med_by_source: Dict[str, List[float]] = defaultdict(list)
    irr_by_source: Dict[str, List[float]] = defaultdict(list)
    for r in track_rows:
        s = str(r["source"])
        dtm = float(r["dt_median_min"]) if r["dt_median_min"] not in ("", None) else np.nan
        irr = float(r["irregular"]) if r["irregular"] not in ("", None) else np.nan
        if np.isfinite(dtm):
            dt_med_by_source[s].append(dtm)
        if np.isfinite(irr):
            irr_by_source[s].append(irr)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.3))

    # Left: boxplot of median dt by source.
    data = []
    labels = []
    for s in sources:
        arr = np.asarray(dt_med_by_source.get(s, []), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        data.append(arr)
        labels.append(s)
    if len(data) > 0:
        ax1.boxplot(data, tick_labels=labels, showfliers=False)
        ax1.set_yscale("log")
        ax1.set_ylabel("Per-track median timestep (min, log scale)")
        ax1.set_xlabel("Source")
        ax1.tick_params(axis="x", rotation=45)
    else:
        ax1.text(0.5, 0.5, "No timestep data", ha="center", va="center", transform=ax1.transAxes)
    ax1.set_title("Track Timestep Distribution")
    ax1.grid(alpha=0.25)

    # Right: irregular track rate by source.
    vals = []
    for s in sources:
        arr = np.asarray(irr_by_source.get(s, []), dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            vals.append(np.nan)
        else:
            vals.append(float(np.nanmean(arr)))
    x = np.arange(len(sources), dtype=int)
    bar = np.array(vals, dtype=float)
    ax2.bar(x, np.nan_to_num(bar, nan=0.0), color="#ef4444", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(list(sources), rotation=45, ha="right")
    ax2.set_ylim(0.0, 1.0)
    ax2.set_ylabel("Irregular track fraction")
    ax2.set_xlabel("Source")
    ax2.set_title("Irregularity Rate (mode_ratio < 0.7)")
    ax2.grid(alpha=0.25, axis="y")

    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def _pick_top_segments(
    segment_id: np.ndarray,
    data_type: np.ndarray,
    inside: np.ndarray,
    dtype: str,
    nmax: int,
) -> List[int]:
    f = np.where((data_type == dtype) & inside & (segment_id > 0))[0]
    if len(f) == 0:
        return []
    segs = segment_id[f].astype(int)
    vals, counts = np.unique(segs, return_counts=True)
    order = np.argsort(counts)[::-1]
    return [int(vals[i]) for i in order[: max(1, int(nmax))]]


def _pick_segments_by_source(
    segment_id: np.ndarray,
    source: np.ndarray,
    inside: np.ndarray,
    source_token: str,
    nmax: int,
) -> List[int]:
    f = np.where((source == source_token) & inside & (segment_id > 0))[0]
    if len(f) == 0:
        return []
    segs = segment_id[f].astype(int)
    vals, counts = np.unique(segs, return_counts=True)
    order = np.argsort(counts)[::-1]
    seg_sorted = [int(vals[i]) for i in order]
    if int(nmax) <= 0:
        return seg_sorted
    return seg_sorted[: max(1, int(nmax))]


def _safe_plot_tag(text: str) -> str:
    out = "".join(ch if (ch.isalnum() or ch in ("-", "_", ".")) else "_" for ch in str(text))
    out = out.strip("_")
    return out if out != "" else "unknown"


def _plot_segment_speed_time_by_type(
    plt: Any,
    outpath: Path,
    obs: Dict[str, np.ndarray],
    obs_use: np.ndarray,
    data_types: Sequence[str],
    nmax: int,
) -> None:
    import matplotlib.dates as mdates

    ntype = len(data_types)
    if ntype == 0:
        fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.0))
        ax.text(0.5, 0.5, "No data types available", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(outpath, dpi=220)
        plt.close(fig)
        return
    ncol = min(2, max(1, ntype))
    nrow = int(math.ceil(ntype / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(6.2 * ncol, 3.8 * nrow), squeeze=False)

    for i, dt in enumerate(data_types):
        ax = axs[i // ncol, i % ncol]
        top_segments = _pick_top_segments(obs["segment_id"], obs["data_type"], obs_use, dt, nmax=nmax)
        if len(top_segments) == 0:
            ax.text(0.5, 0.5, f"No QC+inside segments ({dt})", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"{dt} ({DATA_TYPE_NAME.get(dt,'Unknown')})")
            ax.set_xlabel("UTC datetime")
            ax.set_ylabel("Speed (m/s)")
            ax.grid(alpha=0.2)
            continue

        for sid in top_segments:
            f = np.where((obs["segment_id"] == sid) & obs_use)[0]
            tt = obs["time"][f]
            sp = obs["spd"][f]
            tuniq, sp_mean = _unique_mean(tt, sp)
            if len(tuniq) == 0:
                continue
            tdt = [num2date(float(t)) for t in tuniq]
            ax.plot(tdt, sp_mean, lw=1.1, alpha=0.9, label=f"seg {sid}")

        ax.set_title(f"{dt} ({DATA_TYPE_NAME.get(dt,'Unknown')})")
        ax.set_xlabel("UTC datetime")
        ax.set_ylabel("Depth-mean speed (m/s)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
        ax.grid(alpha=0.25)
        if len(top_segments) <= 8:
            ax.legend(fontsize=7, loc="best")

    for i in range(ntype, nrow * ncol):
        axs[i // ncol, i % ncol].axis("off")
    fig.suptitle("Representative Segment Speed Time Histories (QC-filtered, Inside Boundary, UTC)", y=0.995)
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def _plot_segment_speed_time_by_source(
    plt: Any,
    outdir: Path,
    obs: Dict[str, np.ndarray],
    obs_use: np.ndarray,
    sources: Sequence[str],
    nmax: int,
) -> List[Path]:
    import matplotlib.dates as mdates

    outdir.mkdir(parents=True, exist_ok=True)
    outpaths: List[Path] = []
    for src in sources:
        outpath = outdir / f"segment_speed_time_source_{_safe_plot_tag(src)}.png"
        fig, ax = plt.subplots(1, 1, figsize=(10.0, 4.6))
        segs = _pick_segments_by_source(obs["segment_id"], obs["source"], obs_use, str(src), nmax=nmax)
        if len(segs) == 0:
            ax.text(0.5, 0.5, f"No QC+inside segments ({src})", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"{src} Segment Speed Time Histories")
            ax.set_xlabel("UTC datetime")
            ax.set_ylabel("Depth-mean speed (m/s)")
            ax.grid(alpha=0.2)
        else:
            cmap = plt.get_cmap("turbo", max(2, len(segs)))
            for i, sid in enumerate(segs):
                f = np.where((obs["source"] == src) & (obs["segment_id"] == sid) & obs_use)[0]
                tt = obs["time"][f]
                sp = obs["spd"][f]
                tuniq, sp_mean = _unique_mean(tt, sp)
                if len(tuniq) == 0:
                    continue
                tdt = [num2date(float(t)) for t in tuniq]
                lbl = f"seg {sid}" if len(segs) <= 20 else None
                ax.plot(tdt, sp_mean, lw=0.95, alpha=0.9, color=cmap(i), label=lbl)
            ax.set_title(f"{src} Segment Speed Time Histories (n_segments={len(segs)})")
            ax.set_xlabel("UTC datetime")
            ax.set_ylabel("Depth-mean speed (m/s)")
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
            ax.grid(alpha=0.25)
            if len(segs) <= 20:
                ax.legend(fontsize=7, loc="best", ncol=2)
        fig.autofmt_xdate(rotation=30, ha="right")
        fig.tight_layout()
        fig.savefig(outpath, dpi=220)
        plt.close(fig)
        outpaths.append(outpath)
    return outpaths


def _plot_segment_hodograph_by_type(
    plt: Any,
    outpath: Path,
    obs: Dict[str, np.ndarray],
    obs_use: np.ndarray,
    data_types: Sequence[str],
    nmax: int,
) -> None:
    ntype = len(data_types)
    if ntype == 0:
        fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.0))
        ax.text(0.5, 0.5, "No data types available", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(outpath, dpi=220)
        plt.close(fig)
        return
    ncol = min(2, max(1, ntype))
    nrow = int(math.ceil(ntype / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(6.0 * ncol, 4.2 * nrow), squeeze=False)

    for i, dt in enumerate(data_types):
        ax = axs[i // ncol, i % ncol]
        top_segments = _pick_top_segments(obs["segment_id"], obs["data_type"], obs_use, dt, nmax=nmax)
        if len(top_segments) == 0:
            ax.text(0.5, 0.5, f"No QC+inside segments ({dt})", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"{dt} Hodograph")
            ax.set_xlabel("u (m/s)")
            ax.set_ylabel("v (m/s)")
            ax.grid(alpha=0.2)
            continue

        umax = 0.0
        for sid in top_segments:
            f = np.where((obs["segment_id"] == sid) & obs_use)[0]
            tt = obs["time"][f]
            uu = obs["u"][f]
            vv = obs["v"][f]
            tuniq, umean = _unique_mean(tt, uu)
            _, vmean = _unique_mean(tt, vv)
            if len(tuniq) == 0:
                continue
            if len(tuniq) > 1:
                cn = (tuniq - float(np.nanmin(tuniq))) / max(1.0e-12, float(np.nanmax(tuniq) - np.nanmin(tuniq)))
            else:
                cn = np.array([0.0], dtype=float)
            ax.plot(umean, vmean, "-", lw=0.8, color="0.55", alpha=0.7)
            ax.scatter(umean, vmean, c=cn, s=10, cmap="viridis", alpha=0.9, edgecolors="none", label=f"seg {sid}")
            ax.plot(umean[0], vmean[0], "o", ms=3, color="black", alpha=0.9)
            umax = max(umax, float(np.nanmax(np.abs(umean))), float(np.nanmax(np.abs(vmean))))

        lim = max(0.3, umax * 1.15)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.axhline(0.0, color="0.5", lw=0.7)
        ax.axvline(0.0, color="0.5", lw=0.7)
        ax.set_aspect("equal", adjustable="box")
        ax.set_title(f"{dt} ({DATA_TYPE_NAME.get(dt,'Unknown')}) Hodograph (UTC progression color)")
        ax.set_xlabel("u (m/s)")
        ax.set_ylabel("v (m/s)")
        ax.grid(alpha=0.25)
        if len(top_segments) <= 8:
            ax.legend(fontsize=7, loc="best")

    for i in range(ntype, nrow * ncol):
        axs[i // ncol, i % ncol].axis("off")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def _plot_segment_hodograph_by_source(
    plt: Any,
    outdir: Path,
    obs: Dict[str, np.ndarray],
    obs_use: np.ndarray,
    sources: Sequence[str],
    nmax: int,
) -> List[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    outpaths: List[Path] = []
    for src in sources:
        outpath = outdir / f"segment_hodograph_source_{_safe_plot_tag(src)}.png"
        fig, ax = plt.subplots(1, 1, figsize=(7.2, 6.3))
        segs = _pick_segments_by_source(obs["segment_id"], obs["source"], obs_use, str(src), nmax=nmax)
        if len(segs) == 0:
            ax.text(0.5, 0.5, f"No QC+inside segments ({src})", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"{src} Segment Hodographs")
            ax.set_xlabel("u (m/s)")
            ax.set_ylabel("v (m/s)")
            ax.grid(alpha=0.2)
        else:
            umax = 0.0
            cmap = plt.get_cmap("turbo", max(2, len(segs)))
            for i, sid in enumerate(segs):
                f = np.where((obs["source"] == src) & (obs["segment_id"] == sid) & obs_use)[0]
                tt = obs["time"][f]
                uu = obs["u"][f]
                vv = obs["v"][f]
                tuniq, umean = _unique_mean(tt, uu)
                _, vmean = _unique_mean(tt, vv)
                if len(tuniq) == 0:
                    continue
                lbl = f"seg {sid}" if len(segs) <= 20 else None
                ax.plot(umean, vmean, "-", lw=0.9, color=cmap(i), alpha=0.9, label=lbl)
                if len(umean) > 0:
                    ax.plot(umean[0], vmean[0], "o", ms=2.6, color=cmap(i), alpha=0.9)
                umax = max(umax, float(np.nanmax(np.abs(umean))), float(np.nanmax(np.abs(vmean))))

            lim = max(0.3, umax * 1.15)
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)
            ax.axhline(0.0, color="0.5", lw=0.7)
            ax.axvline(0.0, color="0.5", lw=0.7)
            ax.set_aspect("equal", adjustable="box")
            ax.set_title(f"{src} Segment Hodographs (n_segments={len(segs)})")
            ax.set_xlabel("u (m/s)")
            ax.set_ylabel("v (m/s)")
            ax.grid(alpha=0.25)
            if len(segs) <= 20:
                ax.legend(fontsize=7, loc="best", ncol=2)
        fig.tight_layout()
        fig.savefig(outpath, dpi=220)
        plt.close(fig)
        outpaths.append(outpath)
    return outpaths


def _plot_individual_segments_by_source(
    plt: Any,
    outdir: Path,
    obs: Dict[str, np.ndarray],
    obs_use: np.ndarray,
    sources: Sequence[str],
    nmax: int,
) -> List[Path]:
    import matplotlib.dates as mdates

    outdir.mkdir(parents=True, exist_ok=True)
    outpaths: List[Path] = []
    for src in sources:
        segs = _pick_segments_by_source(obs["segment_id"], obs["source"], obs_use, str(src), nmax=nmax)
        if len(segs) == 0:
            continue
        src_dir = outdir / f"source_{_safe_plot_tag(src)}"
        src_dir.mkdir(parents=True, exist_ok=True)
        for sid in segs:
            f = np.where((obs["source"] == src) & (obs["segment_id"] == sid) & obs_use)[0]
            if len(f) == 0:
                continue

            tt = obs["time"][f]
            uu = obs["u"][f]
            vv = obs["v"][f]
            sp = obs["spd"][f]
            dd = obs["depth"][f]
            tuniq, sp_mean = _unique_mean(tt, sp)
            _, umean = _unique_mean(tt, uu)
            _, vmean = _unique_mean(tt, vv)
            if len(tuniq) == 0:
                continue

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.0, 4.5))

            tdt = [num2date(float(t)) for t in tuniq]
            ax1.plot(tdt, sp_mean, lw=1.2, color="tab:blue")
            ax1.set_title(f"{src} seg {sid}: Speed (depth-mean)")
            ax1.set_xlabel("UTC datetime")
            ax1.set_ylabel("Speed (m/s)")
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
            ax1.grid(alpha=0.25)

            if len(tuniq) > 1:
                cn = (tuniq - float(np.nanmin(tuniq))) / max(1.0e-12, float(np.nanmax(tuniq) - np.nanmin(tuniq)))
            else:
                cn = np.array([0.0], dtype=float)
            ax2.plot(umean, vmean, "-", lw=0.8, color="0.6", alpha=0.8)
            sc = ax2.scatter(umean, vmean, c=cn, s=16, cmap="viridis", alpha=0.95, edgecolors="none")
            if len(umean) > 0:
                ax2.plot(umean[0], vmean[0], "o", ms=3, color="black", alpha=0.9)
            umax = max(float(np.nanmax(np.abs(umean))), float(np.nanmax(np.abs(vmean))))
            lim = max(0.3, umax * 1.15)
            ax2.set_xlim(-lim, lim)
            ax2.set_ylim(-lim, lim)
            ax2.axhline(0.0, color="0.5", lw=0.7)
            ax2.axvline(0.0, color="0.5", lw=0.7)
            ax2.set_aspect("equal", adjustable="box")
            ax2.set_title(f"{src} seg {sid}: Hodograph")
            ax2.set_xlabel("u (m/s)")
            ax2.set_ylabel("v (m/s)")
            ax2.grid(alpha=0.25)
            cb = fig.colorbar(sc, ax=ax2, shrink=0.9)
            cb.set_label("UTC progression")

            depth_min = float(np.nanmin(dd)) if len(dd) > 0 else np.nan
            depth_max = float(np.nanmax(dd)) if len(dd) > 0 else np.nan
            fig.suptitle(
                f"Segment {sid} | source={src} | n_obs={len(f):,} | depth_range={depth_min:.1f}-{depth_max:.1f} m",
                y=0.995,
            )
            fig.autofmt_xdate(rotation=30, ha="right")
            fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.955))

            outpath = src_dir / f"segment_{sid}.png"
            fig.savefig(outpath, dpi=220)
            plt.close(fig)
            outpaths.append(outpath)

    return outpaths


def _plot_ca_depth_time_speed(
    plt: Any,
    outpath: Path,
    obs: Dict[str, np.ndarray],
    obs_use: np.ndarray,
) -> None:
    import matplotlib.dates as mdates

    fca = np.where((obs["data_type"] == "CA") & obs_use & (obs["segment_id"] > 0))[0]
    fig, ax = plt.subplots(1, 1, figsize=(9.0, 4.8))
    if len(fca) == 0:
        ax.text(0.5, 0.5, "No QC+inside CA segment", transform=ax.transAxes, ha="center", va="center")
        ax.set_title("CA Depth-Time Speed (QC-filtered, Inside Boundary)")
        ax.set_xlabel("UTC datetime")
        ax.set_ylabel("Depth (m)")
        fig.tight_layout()
        fig.savefig(outpath, dpi=220)
        plt.close(fig)
        return

    segs = obs["segment_id"][fca].astype(int)
    vals, counts = np.unique(segs, return_counts=True)
    sid = int(vals[np.argmax(counts)])
    ff = np.where((obs["segment_id"] == sid) & obs_use)[0]
    tt = obs["time"][ff]
    dd = obs["depth"][ff]
    sp = obs["spd"][ff]

    if len(ff) == 0:
        ax.text(0.5, 0.5, "No QC+inside CA segment", transform=ax.transAxes, ha="center", va="center")
    else:
        tdt = [num2date(float(t)) for t in tt]
        sc = ax.scatter(tdt, dd, c=sp, s=8, cmap="plasma", alpha=0.85, edgecolors="none")
        cb = fig.colorbar(sc, ax=ax, shrink=0.94)
        cb.set_label("Speed (m/s)")
        ax.invert_yaxis()
        ax.set_title(f"CA Depth-Time Speed (segment {sid}, QC-filtered inside)")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    ax.set_xlabel("UTC datetime")
    ax.set_ylabel("Depth (m)")
    ax.grid(alpha=0.25)
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def _plot_depthbin_speed_time_by_type(
    plt: Any,
    outpath: Path,
    obs: Dict[str, np.ndarray],
    obs_use: np.ndarray,
    data_types: Sequence[str],
    depth_bins: np.ndarray,
    depth_bin_labels: Sequence[str],
) -> None:
    import matplotlib.dates as mdates

    ntype = len(data_types)
    if ntype == 0:
        fig, ax = plt.subplots(1, 1, figsize=(7.0, 4.0))
        ax.text(0.5, 0.5, "No data types available", transform=ax.transAxes, ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(outpath, dpi=220)
        plt.close(fig)
        return

    ncol = min(2, max(1, ntype))
    nrow = int(math.ceil(ntype / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(6.3 * ncol, 3.9 * nrow), squeeze=False)
    cmap = plt.get_cmap("tab10", max(1, len(depth_bin_labels)))

    for i, dt in enumerate(data_types):
        ax = axs[i // ncol, i % ncol]
        fdt = np.where((obs["data_type"] == dt) & obs_use)[0]
        if len(fdt) == 0:
            ax.text(0.5, 0.5, f"No QC+inside data ({dt})", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"{dt} ({DATA_TYPE_NAME.get(dt,'Unknown')})")
            ax.set_xlabel("UTC datetime")
            ax.set_ylabel("Speed (m/s)")
            ax.grid(alpha=0.2)
            continue

        drew = 0
        for ib in range(len(depth_bins) - 1):
            z0 = float(depth_bins[ib])
            z1 = float(depth_bins[ib + 1])
            m = fdt[(obs["depth"][fdt] >= z0) & (obs["depth"][fdt] < z1)]
            if len(m) == 0:
                continue
            tuniq, sp_mean = _unique_mean(obs["time"][m], obs["spd"][m])
            if len(tuniq) == 0:
                continue
            tdt = [num2date(float(t)) for t in tuniq]
            ax.plot(
                tdt,
                sp_mean,
                lw=1.1,
                alpha=0.9,
                color=cmap(ib),
                label=f"{depth_bin_labels[ib]} ({z0:g}-{z1:g}m)",
            )
            drew += 1

        if drew == 0:
            ax.text(0.5, 0.5, "No data in configured bins", transform=ax.transAxes, ha="center", va="center")
        ax.set_title(f"{dt} ({DATA_TYPE_NAME.get(dt,'Unknown')})")
        ax.set_xlabel("UTC datetime")
        ax.set_ylabel("Depth-bin mean speed (m/s)")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
        ax.grid(alpha=0.25)
        if drew > 0:
            ax.legend(fontsize=7, loc="best")

    for i in range(ntype, nrow * ncol):
        axs[i // ncol, i % ncol].axis("off")
    fig.suptitle("Depth-Bin Speed Time Histories (QC-filtered, Inside Boundary, UTC)", y=0.995)
    fig.autofmt_xdate(rotation=30, ha="right")
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.965))
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Analyze trusted JODC current NC data with boundary overlay and inside filtering."
    )
    p.add_argument("--nc-dir", default=None, help="Directory containing NC files.")
    p.add_argument("--trusted-csv", default=None, help="Path to nc_trusted_sources.csv.")
    p.add_argument("--classification-csv", default=None, help="Path to nc_filename_classification.csv.")
    p.add_argument("--boundary-shp", default=None, help="Path to boundary shapefile (SOB.shp).")

    p.add_argument("--trusted-tier", choices=["core", "extended", "all"], default=None, help="Trusted source tier.")
    p.add_argument("--sources", nargs="+", default=None, help="Explicit source list (overrides tier).")
    p.add_argument(
        "--data-types",
        nargs="+",
        default=None,
        help="Optional DATA_TYPE filter (e.g., CA CU CD CV).",
    )

    p.add_argument("--start", default=None, help="Start time filter: YYYY-MM-DD[ HH:MM[:SS]]")
    p.add_argument("--end", default=None, help="End time filter: YYYY-MM-DD[ HH:MM[:SS]]")
    p.add_argument(
        "--min-record-length-hours",
        type=float,
        default=None,
        help="Minimum temporal record length (hours) per segment after time window filtering. <=0 disables.",
    )
    p.add_argument("--min-depth", type=float, default=None, help="Minimum depth [m].")
    p.add_argument("--max-depth", type=float, default=None, help="Maximum depth [m].")
    p.add_argument(
        "--qc-keep-codes",
        nargs="+",
        default=None,
        help="QC flag codes to keep (e.g., 0 or 0 1 or 0,1). Default: 0.",
    )
    p.add_argument(
        "--qc-mode",
        choices=["strict_both", "either"],
        default=None,
        help="QC pass mode: strict_both keeps rows where both u_QC and v_QC pass.",
    )
    p.add_argument(
        "--report-all-qc-codes",
        dest="report_all_qc_codes",
        action="store_true",
        help="Report counts for all observed QC codes in QC summary.",
    )
    p.add_argument(
        "--no-report-all-qc-codes",
        dest="report_all_qc_codes",
        action="store_false",
        help="Report only selected keep codes in QC summary.",
    )
    p.set_defaults(report_all_qc_codes=None)

    p.add_argument(
        "--inside-mode",
        choices=["largest_closed_ring", "all_closed_rings"],
        default=None,
        help="Inside-polygon mode from closed rings.",
    )
    p.add_argument("--ring-close-tol", type=float, default=None, help="Closure tolerance for ring detection.")

    p.add_argument("--outdir", default=None, help="Output directory.")
    p.add_argument("--save-inside-npz", dest="save_inside_npz", action="store_true", help="Save inside-only obs NPZ.")
    p.add_argument("--no-save-inside-npz", dest="save_inside_npz", action="store_false", help="Disable inside NPZ.")
    p.set_defaults(save_inside_npz=None)

    p.add_argument("--plot", dest="plot", action="store_true", help="Write plot outputs.")
    p.add_argument("--no-plot", dest="plot", action="store_false", help="Disable plots.")
    p.set_defaults(plot=None)

    p.add_argument(
        "--max-track-plots-per-source",
        type=int,
        default=None,
        help="Max sampled tracks per source for track_samples_by_source plot.",
    )
    p.add_argument(
        "--max-segment-plots-per-type",
        type=int,
        default=None,
        help="Max representative segments per data type for time-history plots.",
    )
    p.add_argument(
        "--max-segment-plots-per-source",
        type=int,
        default=None,
        help="Max segments per source for source-wise segment plots (<=0 means all).",
    )
    p.add_argument(
        "--plot-each-segment",
        dest="plot_each_segment",
        action="store_true",
        help="Write one separate figure for each segment (grouped by source).",
    )
    p.add_argument(
        "--no-plot-each-segment",
        dest="plot_each_segment",
        action="store_false",
        help="Disable separate per-segment figures.",
    )
    p.set_defaults(plot_each_segment=None)
    p.add_argument(
        "--max-individual-segment-plots-per-source",
        type=int,
        default=None,
        help="Max per-segment figures per source (<=0 means all segments).",
    )
    p.add_argument(
        "--plot-sample-max",
        type=int,
        default=None,
        help="Max sampled points per scatter-heavy plot.",
    )
    p.add_argument(
        "--depth-bins",
        nargs="+",
        default=None,
        help="Depth bin edges in meters (e.g., 0 15 50 20000 or 0,15,50,20000).",
    )
    p.add_argument(
        "--depth-bin-labels",
        nargs="+",
        default=None,
        help="Depth bin labels (must match number of bins), e.g., surface mid deep.",
    )
    p.add_argument(
        "--export-collocation-csv",
        dest="export_collocation_csv",
        action="store_true",
        help="Export QC-filtered inside observations as model-ready collocation CSV.",
    )
    p.add_argument(
        "--no-export-collocation-csv",
        dest="export_collocation_csv",
        action="store_false",
        help="Disable collocation CSV export.",
    )
    p.set_defaults(export_collocation_csv=None)
    p.add_argument(
        "--export-collocation-npz",
        dest="export_collocation_npz",
        action="store_true",
        help="Export QC-filtered inside observations as model-ready collocation NPZ.",
    )
    p.add_argument(
        "--no-export-collocation-npz",
        dest="export_collocation_npz",
        action="store_false",
        help="Disable collocation NPZ export.",
    )
    p.set_defaults(export_collocation_npz=None)

    return p.parse_args(argv)


def _merge_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(USER_CONFIG) if USER_CONFIG.get("enable", False) else {}
    keys = (
        "nc_dir",
        "trusted_csv",
        "classification_csv",
        "boundary_shp",
        "trusted_tier",
        "sources",
        "data_types",
        "start",
        "end",
        "min_record_length_hours",
        "min_depth",
        "max_depth",
        "qc_keep_codes",
        "qc_mode",
        "report_all_qc_codes",
        "inside_mode",
        "ring_close_tol",
        "outdir",
        "save_inside_npz",
        "plot",
        "max_track_plots_per_source",
        "max_segment_plots_per_type",
        "max_segment_plots_per_source",
        "plot_each_segment",
        "max_individual_segment_plots_per_source",
        "plot_sample_max",
        "depth_bins",
        "depth_bin_labels",
        "export_collocation_csv",
        "export_collocation_npz",
    )
    for k in keys:
        v = getattr(args, k, None)
        if v is not None:
            cfg[k] = v

    cfg.setdefault("nc_dir", None)
    cfg.setdefault("trusted_csv", None)
    cfg.setdefault("classification_csv", None)
    cfg.setdefault("boundary_shp", None)
    cfg.setdefault("trusted_tier", "core")
    cfg.setdefault("sources", None)
    cfg.setdefault("data_types", None)
    cfg.setdefault("start", None)
    cfg.setdefault("end", None)
    cfg.setdefault("min_record_length_hours", 0.0)
    cfg.setdefault("min_depth", None)
    cfg.setdefault("max_depth", None)
    cfg.setdefault("qc_keep_codes", [0])
    cfg.setdefault("qc_mode", "strict_both")
    cfg.setdefault("report_all_qc_codes", True)
    cfg.setdefault("inside_mode", "largest_closed_ring")
    cfg.setdefault("ring_close_tol", 1.0e-6)
    cfg.setdefault("outdir", "./trusted_current_analysis")
    cfg.setdefault("save_inside_npz", True)
    cfg.setdefault("plot", True)
    cfg.setdefault("max_track_plots_per_source", 3)
    cfg.setdefault("max_segment_plots_per_type", 4)
    cfg.setdefault("max_segment_plots_per_source", 0)
    cfg.setdefault("plot_each_segment", True)
    cfg.setdefault("max_individual_segment_plots_per_source", 0)
    cfg.setdefault("plot_sample_max", 180000)
    cfg.setdefault("depth_bins", [0.0, 15.0, 50.0, 20000.0])
    cfg.setdefault("depth_bin_labels", ["surface", "mid", "deep"])
    cfg.setdefault("export_collocation_csv", True)
    cfg.setdefault("export_collocation_npz", True)
    return cfg


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cfg = _merge_config(args)

    start_dt = _parse_bound(cfg.get("start"), is_end=False)
    end_dt = _parse_bound(cfg.get("end"), is_end=True)
    if start_dt is not None and end_dt is not None and end_dt < start_dt:
        raise ValueError("end must be >= start")

    start_num = float(date2num([start_dt])[0]) if start_dt is not None else None
    end_num = float(date2num([end_dt])[0]) if end_dt is not None else None
    min_record_length_hours = float(cfg.get("min_record_length_hours", 0.0) or 0.0)
    if min_record_length_hours < 0.0:
        raise ValueError("min_record_length_hours must be >= 0")

    nc_dir = _resolve_path(str(cfg["nc_dir"]))
    trusted_csv = _resolve_path(str(cfg["trusted_csv"]))
    classification_csv = _resolve_path(str(cfg["classification_csv"]))
    boundary_shp = _resolve_path(str(cfg["boundary_shp"]))
    outdir = _resolve_path(str(cfg["outdir"]))
    outdir.mkdir(parents=True, exist_ok=True)

    if not nc_dir.is_dir():
        raise FileNotFoundError(f"nc_dir not found: {nc_dir}")
    if not trusted_csv.is_file():
        raise FileNotFoundError(f"trusted_csv not found: {trusted_csv}")
    if not classification_csv.is_file():
        raise FileNotFoundError(f"classification_csv not found: {classification_csv}")
    if not boundary_shp.is_file():
        raise FileNotFoundError(f"boundary_shp not found: {boundary_shp}")

    trusted_tier = str(cfg["trusted_tier"]).strip().lower()
    explicit_sources = cfg.get("sources")
    if explicit_sources is not None and len(explicit_sources) > 0:
        selected_sources = [str(s).strip() for s in explicit_sources if str(s).strip() != ""]
        # Remove duplicates while preserving user order.
        seen = set()
        selected_sources = [s for s in selected_sources if not (s in seen or seen.add(s))]
        source_selection_mode = "explicit_sources"
    else:
        selected_sources = _load_trusted_sources(trusted_csv, trusted_tier)
        source_selection_mode = f"trusted_tier:{trusted_tier}"

    if len(selected_sources) == 0:
        raise RuntimeError("No source selected. Check --trusted-tier / --sources inputs.")

    data_types_cfg = cfg.get("data_types")
    if data_types_cfg is not None and len(data_types_cfg) > 0:
        selected_data_types = sorted({str(x).strip().upper() for x in data_types_cfg if str(x).strip() != ""})
    else:
        selected_data_types = []
    qc_keep_codes = _parse_code_list(cfg.get("qc_keep_codes"), default=[0])
    qc_mode = str(cfg.get("qc_mode", "strict_both")).strip().lower()
    if qc_mode not in ("strict_both", "either"):
        raise ValueError(f"Unsupported qc_mode: {qc_mode}")
    depth_bins, depth_bin_labels = _parse_depth_bins_and_labels(
        cfg.get("depth_bins"),
        cfg.get("depth_bin_labels"),
    )

    file_source = _load_filename_source_map(classification_csv)
    nc_files_all = sorted(nc_dir.glob("*.nc"))
    stats = Counter()
    selected_files: List[Tuple[Path, str]] = []
    for fp in nc_files_all:
        stats["nc_files_total"] += 1
        src = file_source.get(fp.name)
        if src is None:
            stats["nc_files_unmapped"] += 1
            continue
        if src not in selected_sources:
            stats["nc_files_untrusted"] += 1
            continue
        selected_files.append((fp, src))

    stats["nc_files_selected"] = len(selected_files)
    if len(selected_files) == 0:
        raise RuntimeError("No NC files selected after trusted-source filter.")

    dt_txt = ",".join(selected_data_types) if len(selected_data_types) > 0 else "ALL"
    start_txt = _fmt_time(float(start_num)) if start_num is not None else "None"
    end_txt = _fmt_time(float(end_num)) if end_num is not None else "None"
    print(f"[INFO] effective time window: start={start_txt}, end={end_txt}", flush=True)
    print(
        f"[INFO] segment record length filter: min_record_length_hours={min_record_length_hours:.3f}",
        flush=True,
    )
    print(
        f"[INFO] selected files: {len(selected_files):,}/{len(nc_files_all):,} "
        f"(mode={source_selection_mode}, n_sources={len(selected_sources)}, data_types={dt_txt}, "
        f"qc_keep={qc_keep_codes}, qc_mode={qc_mode})",
        flush=True,
    )

    # Parse NC files.
    track_sink: Dict[str, List[Any]] = {
        "time": [],
        "lon": [],
        "lat": [],
        "source": [],
        "track_file": [],
        "track_id": [],
        "segment_id": [],
        "segment_seq": [],
        "ship": [],
        "data_type": [],
        "time_index": [],
    }
    obs_sink: Dict[str, List[Any]] = {
        "track_ref": [],
        "time": [],
        "lon": [],
        "lat": [],
        "depth": [],
        "u": [],
        "v": [],
        "spd": [],
        "spd_kn": [],
        "dir": [],
        "qc_u": [],
        "qc_v": [],
        "source": [],
        "track_file": [],
        "track_id": [],
        "segment_id": [],
        "segment_seq": [],
        "ship": [],
        "data_type": [],
        "time_index": [],
        "depth_index": [],
    }
    c = Counter()

    track_id_next = 1
    segment_id_next = 1
    for ifile, (fp, source_token) in enumerate(selected_files, start=1):
        c["nc_file_seen"] += 1
        try:
            nc = _read_nc_quiet(fp)
        except Exception:
            c["nc_file_read_fail"] += 1
            continue

        depth = _get_nc_var(nc, "depth")
        lon = _get_nc_var(nc, "longitude")
        lat = _get_nc_var(nc, "latitude")
        u = _get_nc_var(nc, "u")
        v = _get_nc_var(nc, "v")
        obs_date = _get_nc_var(nc, "obs_date")
        obs_time = _get_nc_var(nc, "obs_time")
        u_qc = _get_nc_var(nc, "u_QC")
        v_qc = _get_nc_var(nc, "v_QC")

        if depth is None or lon is None or lat is None or u is None or v is None or obs_date is None:
            c["nc_file_missing_core"] += 1
            continue
        if obs_time is None:
            obs_time = np.zeros_like(obs_date)
            c["nc_file_missing_obs_time"] += 1

        depth = np.asarray(depth, dtype=float).reshape(-1)
        lon = np.asarray(lon, dtype=float).reshape(-1)
        lat = np.asarray(lat, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)
        obs_date = np.asarray(obs_date).reshape(-1)
        obs_time = np.asarray(obs_time).reshape(-1)

        if u.ndim == 1:
            u = u[:, None]
        if v.ndim == 1:
            v = v[:, None]
        if u.ndim != 2 or v.ndim != 2 or u.shape != v.shape:
            c["nc_file_uv_shape_bad"] += 1
            continue

        tlen, dlen = u.shape
        if depth.size != dlen:
            if depth.size == tlen and dlen != tlen:
                u = u.T
                v = v.T
                tlen, dlen = u.shape
            if depth.size != dlen:
                c["nc_file_depth_shape_bad"] += 1
                continue

        if lon.size == 1:
            lon = np.repeat(lon, tlen)
        if lat.size == 1:
            lat = np.repeat(lat, tlen)
        if obs_date.size == 1 and tlen > 1:
            obs_date = np.repeat(obs_date, tlen)
        if obs_time.size == 1 and tlen > 1:
            obs_time = np.repeat(obs_time, tlen)
        if lon.size != tlen or lat.size != tlen or obs_date.size != tlen or obs_time.size != tlen:
            c["nc_file_time_axis_shape_bad"] += 1
            continue

        tnum, dt_stats = _parse_obs_datetime(obs_date, obs_time)
        c.update(dt_stats)

        ship = _safe_nc_attr(nc, "SHIP_CODE", "")
        data_type = _safe_nc_attr(nc, "DATA_TYPE", "").upper()
        if data_type == "":
            data_type = "UNK"
        if len(selected_data_types) > 0 and data_type not in selected_data_types:
            c["nc_file_filtered_data_type"] += 1
            continue
        track_id = track_id_next
        track_id_next += 1

        tmask_base = np.isfinite(tnum)
        tmask_base &= np.isfinite(lon) & np.isfinite(lat)
        tmask_base &= (np.abs(lon) <= 180.0) & (np.abs(lat) <= 90.0)
        if np.any(tmask_base):
            c["nc_file_with_valid_datetime_pos"] += 1

        tmask = tmask_base.copy()
        if start_num is not None:
            tmask &= tnum >= float(start_num)
        if end_num is not None:
            tmask &= tnum <= float(end_num)
        if np.any(tmask):
            c["nc_file_with_track_points_in_time_window"] += 1
        if not np.any(tmask):
            c["nc_file_no_valid_track_points"] += 1
            if np.any(tmask_base):
                c["nc_file_outside_time_window"] += 1
            continue

        tids = np.where(tmask)[0]
        t_keep = tnum[tids]
        lon_keep = lon[tids]
        lat_keep = lat[tids]
        # Normalize to temporal order for trajectory segmentation.
        o = np.argsort(t_keep)
        tids = tids[o]
        t_keep = t_keep[o]
        lon_keep = lon_keep[o]
        lat_keep = lat_keep[o]
        seg_seq_local = _segment_track_points(t_keep, lon_keep, lat_keep, data_type)
        if min_record_length_hours > 0.0:
            keep_seg = np.zeros(len(t_keep), dtype=bool)
            for seg_seq in np.unique(seg_seq_local):
                fs = np.where(seg_seq_local == seg_seq)[0]
                if len(fs) == 0:
                    continue
                span_hours = float(t_keep[fs[-1]] - t_keep[fs[0]]) * 24.0 if len(fs) > 1 else 0.0
                if span_hours >= min_record_length_hours:
                    keep_seg[fs] = True
                    c["segment_kept_record_length"] += 1
                else:
                    c["segment_short_record_skipped"] += 1
                    c["track_rows_skipped_short_record"] += int(len(fs))

            if not np.any(keep_seg):
                c["nc_file_short_record_skipped"] += 1
                continue

            tids = tids[keep_seg]
            t_keep = t_keep[keep_seg]
            lon_keep = lon_keep[keep_seg]
            lat_keep = lat_keep[keep_seg]
            seg_seq_local = seg_seq_local[keep_seg]
            seg_old = np.unique(seg_seq_local)
            seg_map = {int(s): i for i, s in enumerate(seg_old.tolist())}
            seg_seq_local = np.asarray([seg_map[int(s)] for s in seg_seq_local], dtype=int)

        seg_gid_local = np.zeros(len(tids), dtype=int)
        for seg_seq in np.unique(seg_seq_local):
            gid = segment_id_next
            segment_id_next += 1
            seg_gid_local[seg_seq_local == seg_seq] = gid

        n_track_add = len(tids)
        track_ref_map = np.full(tlen, -1, dtype=int)
        seg_id_map = np.full(tlen, -1, dtype=int)
        seg_seq_map = np.full(tlen, -1, dtype=int)
        track_ref_start = len(track_sink["time"])
        track_ref_map[tids] = np.arange(track_ref_start, track_ref_start + n_track_add, dtype=int)
        seg_id_map[tids] = seg_gid_local
        seg_seq_map[tids] = seg_seq_local

        track_sink["time"].extend(t_keep.tolist())
        track_sink["lon"].extend(lon_keep.tolist())
        track_sink["lat"].extend(lat_keep.tolist())
        track_sink["source"].extend([source_token] * n_track_add)
        track_sink["track_file"].extend([fp.name] * n_track_add)
        track_sink["track_id"].extend([track_id] * n_track_add)
        track_sink["segment_id"].extend(seg_gid_local.astype(int).tolist())
        track_sink["segment_seq"].extend(seg_seq_local.astype(int).tolist())
        track_sink["ship"].extend([ship if ship else "UNK"] * n_track_add)
        track_sink["data_type"].extend([data_type] * n_track_add)
        track_sink["time_index"].extend(tids.astype(int).tolist())

        qu = None
        qv = None
        if u_qc is not None and v_qc is not None:
            qu = np.asarray(u_qc)
            qv = np.asarray(v_qc)
            if qu.ndim == 1:
                qu = qu[:, None]
            if qv.ndim == 1:
                qv = qv[:, None]
            if qu.shape != u.shape or qv.shape != v.shape:
                qu = None
                qv = None
                c["nc_file_qc_shape_bad"] += 1

        obs_rows_this_file = 0
        for j in range(dlen):
            dval = float(depth[j])
            if cfg.get("min_depth") is not None and dval < float(cfg["min_depth"]):
                c["depth_skip_min"] += 1
                continue
            if cfg.get("max_depth") is not None and dval > float(cfg["max_depth"]):
                c["depth_skip_max"] += 1
                continue

            ucol = u[:, j].astype(float)
            vcol = v[:, j].astype(float)
            mask = tmask.copy()
            mask &= np.isfinite(ucol) & np.isfinite(vcol)
            mask &= (np.abs(ucol) < FILL_CUTOFF) & (np.abs(vcol) < FILL_CUTOFF)
            if not np.any(mask):
                continue

            idx = np.where(mask)[0]
            tref = track_ref_map[idx]
            segid = seg_id_map[idx]
            segseq = seg_seq_map[idx]
            keep = (tref >= 0) & (segid > 0)
            if not np.any(keep):
                continue

            idx = idx[keep]
            tref = tref[keep]
            segid = segid[keep]
            segseq = segseq[keep]
            n = len(idx)

            u_kn = ucol[idx]
            v_kn = vcol[idx]
            spd_kn = np.sqrt(u_kn**2 + v_kn**2)
            spd = spd_kn * KNOTS_TO_MS
            dire = np.rad2deg(np.arctan2(u_kn, v_kn)) % 360.0
            q_u = np.full(n, -1, dtype=int)
            q_v = np.full(n, -1, dtype=int)
            if qu is not None and qv is not None:
                q_u = qu[idx, j].astype(int)
                q_v = qv[idx, j].astype(int)

            obs_sink["track_ref"].extend(tref.astype(int).tolist())
            obs_sink["time"].extend(tnum[idx].tolist())
            obs_sink["lon"].extend(lon[idx].tolist())
            obs_sink["lat"].extend(lat[idx].tolist())
            obs_sink["depth"].extend([dval] * n)
            obs_sink["u"].extend((u_kn * KNOTS_TO_MS).tolist())
            obs_sink["v"].extend((v_kn * KNOTS_TO_MS).tolist())
            obs_sink["spd"].extend(spd.tolist())
            obs_sink["spd_kn"].extend(spd_kn.tolist())
            obs_sink["dir"].extend(dire.tolist())
            obs_sink["qc_u"].extend(q_u.tolist())
            obs_sink["qc_v"].extend(q_v.tolist())
            obs_sink["source"].extend([source_token] * n)
            obs_sink["track_file"].extend([fp.name] * n)
            obs_sink["track_id"].extend([track_id] * n)
            obs_sink["segment_id"].extend(segid.astype(int).tolist())
            obs_sink["segment_seq"].extend(segseq.astype(int).tolist())
            obs_sink["ship"].extend([ship if ship else "UNK"] * n)
            obs_sink["data_type"].extend([data_type] * n)
            obs_sink["time_index"].extend(idx.astype(int).tolist())
            obs_sink["depth_index"].extend([j] * n)
            c["obs_rows_kept"] += n
            obs_rows_this_file += n

        if obs_rows_this_file > 0:
            c["nc_file_with_obs_rows_in_time_window"] += 1

        c["track_rows_kept"] += n_track_add

        if ifile % 250 == 0:
            print(
                f"[INFO] parsed {ifile}/{len(selected_files)} files, "
                f"track_rows={c['track_rows_kept']:,}, obs_rows={c['obs_rows_kept']:,}",
                flush=True,
            )

    if len(track_sink["time"]) == 0:
        fail_detail = {
            "nc_file_seen": int(c.get("nc_file_seen", 0)),
            "nc_file_read_fail": int(c.get("nc_file_read_fail", 0)),
            "nc_file_missing_core": int(c.get("nc_file_missing_core", 0)),
            "nc_file_uv_shape_bad": int(c.get("nc_file_uv_shape_bad", 0)),
            "nc_file_depth_shape_bad": int(c.get("nc_file_depth_shape_bad", 0)),
            "nc_file_time_axis_shape_bad": int(c.get("nc_file_time_axis_shape_bad", 0)),
            "nc_file_with_valid_datetime_pos": int(c.get("nc_file_with_valid_datetime_pos", 0)),
            "nc_file_with_track_points_in_time_window": int(c.get("nc_file_with_track_points_in_time_window", 0)),
            "nc_file_outside_time_window": int(c.get("nc_file_outside_time_window", 0)),
            "nc_file_short_record_skipped": int(c.get("nc_file_short_record_skipped", 0)),
            "segment_short_record_skipped": int(c.get("segment_short_record_skipped", 0)),
            "nc_file_no_valid_track_points": int(c.get("nc_file_no_valid_track_points", 0)),
        }
        raise RuntimeError(
            "No valid track points after NC parsing/filtering. "
            f"debug_counts={fail_detail}"
        )

    print(
        "[INFO] time-window effect:"
        f" files_with_valid_datetime_pos={int(c.get('nc_file_with_valid_datetime_pos', 0)):,},"
        f" files_with_track_points_in_window={int(c.get('nc_file_with_track_points_in_time_window', 0)):,},"
        f" files_outside_time_window={int(c.get('nc_file_outside_time_window', 0)):,},"
        f" files_short_record_skipped={int(c.get('nc_file_short_record_skipped', 0)):,},"
        f" segments_short_record_skipped={int(c.get('segment_short_record_skipped', 0)):,},"
        f" files_with_obs_rows_in_window={int(c.get('nc_file_with_obs_rows_in_time_window', 0)):,}",
        flush=True,
    )

    # Convert to arrays.
    track = {
        "time": np.asarray(track_sink["time"], dtype=float),
        "lon": np.asarray(track_sink["lon"], dtype=float),
        "lat": np.asarray(track_sink["lat"], dtype=float),
        "source": np.asarray(track_sink["source"], dtype="U16"),
        "track_file": np.asarray(track_sink["track_file"], dtype="U128"),
        "track_id": np.asarray(track_sink["track_id"], dtype=int),
        "segment_id": np.asarray(track_sink["segment_id"], dtype=int),
        "segment_seq": np.asarray(track_sink["segment_seq"], dtype=int),
        "ship": np.asarray(track_sink["ship"], dtype="U16"),
        "data_type": np.asarray(track_sink["data_type"], dtype="U8"),
        "time_index": np.asarray(track_sink["time_index"], dtype=int),
    }
    obs = {
        "track_ref": np.asarray(obs_sink["track_ref"], dtype=int),
        "time": np.asarray(obs_sink["time"], dtype=float),
        "lon": np.asarray(obs_sink["lon"], dtype=float),
        "lat": np.asarray(obs_sink["lat"], dtype=float),
        "depth": np.asarray(obs_sink["depth"], dtype=float),
        "u": np.asarray(obs_sink["u"], dtype=float),
        "v": np.asarray(obs_sink["v"], dtype=float),
        "spd": np.asarray(obs_sink["spd"], dtype=float),
        "spd_kn": np.asarray(obs_sink["spd_kn"], dtype=float),
        "dir": np.asarray(obs_sink["dir"], dtype=float),
        "qc_u": np.asarray(obs_sink["qc_u"], dtype=int),
        "qc_v": np.asarray(obs_sink["qc_v"], dtype=int),
        "source": np.asarray(obs_sink["source"], dtype="U16"),
        "track_file": np.asarray(obs_sink["track_file"], dtype="U128"),
        "track_id": np.asarray(obs_sink["track_id"], dtype=int),
        "segment_id": np.asarray(obs_sink["segment_id"], dtype=int),
        "segment_seq": np.asarray(obs_sink["segment_seq"], dtype=int),
        "ship": np.asarray(obs_sink["ship"], dtype="U16"),
        "data_type": np.asarray(obs_sink["data_type"], dtype="U8"),
        "time_index": np.asarray(obs_sink["time_index"], dtype=int),
        "depth_index": np.asarray(obs_sink["depth_index"], dtype=int),
    }

    # Boundary parsing and inside filtering.
    shp_info = _read_polyline_parts(boundary_shp)
    parts = shp_info["parts"]
    rings = _extract_closed_rings(parts, tol=float(cfg["ring_close_tol"]))
    if len(rings) == 0:
        raise RuntimeError(
            f"No closed ring was found in {boundary_shp}. "
            "Try --inside-mode all_closed_rings or provide another polygon boundary source."
        )
    points_xy = np.c_[track["lon"], track["lat"]]
    track_inside, inside_meta = _inside_from_rings(points_xy, rings, str(cfg["inside_mode"]))
    n_inside_ring = int(np.count_nonzero(track_inside))

    # Practical safeguard for polyline boundaries like SOB.shp:
    # closed rings can be small coastal/island loops, while the intended domain
    # is represented by a multipart path. If ring-based mask yields zero, fallback.
    if n_inside_ring == 0:
        fallback_inside, fallback_meta = _inside_from_all_parts_path(points_xy, parts)
        n_inside_fallback = int(np.count_nonzero(fallback_inside))
        if n_inside_fallback > 0:
            print(
                "[WARN] ring-based inside detection returned zero points; "
                f"fallback to multipart-path mode recovered inside={n_inside_fallback:,}.",
                flush=True,
            )
            track_inside = fallback_inside
            inside_meta["fallback_used"] = True
            inside_meta["fallback_mode"] = "all_parts_nan_path"
            inside_meta["fallback_inside_count"] = n_inside_fallback
            inside_meta["fallback"] = fallback_meta
        else:
            inside_meta["fallback_used"] = True
            inside_meta["fallback_mode"] = "all_parts_nan_path"
            inside_meta["fallback_inside_count"] = 0
            inside_meta["fallback"] = fallback_meta
    obs_inside = np.zeros(len(obs["time"]), dtype=bool)
    if len(obs_inside) > 0:
        tref = obs["track_ref"]
        ok = (tref >= 0) & (tref < len(track_inside))
        obs_inside[ok] = track_inside[tref[ok]]
        c["obs_track_ref_oob"] += int(np.count_nonzero(~ok))

    # QC pass mask (quality flags, not vector signs).
    qc_u = obs["qc_u"]
    qc_v = obs["qc_v"]
    qc_pass_obs = _qc_pass_mask(qc_u, qc_v, qc_keep_codes, qc_mode)
    obs_use = qc_pass_obs
    obs_use_inside = obs_use & obs_inside

    valid_tref = (obs["track_ref"] >= 0) & (obs["track_ref"] < len(track["time"]))
    track_has_qc = np.zeros(len(track["time"]), dtype=bool)
    if np.any(valid_tref & obs_use):
        bpass = np.bincount(obs["track_ref"][valid_tref & obs_use], minlength=len(track["time"]))
        track_has_qc = bpass > 0
    track_use = track_has_qc
    track_use_inside = track_use & track_inside

    c["obs_qc_pass"] = int(np.count_nonzero(obs_use))
    c["obs_qc_fail"] = int(len(obs_use) - np.count_nonzero(obs_use))
    c["obs_qc_inside"] = int(np.count_nonzero(obs_use_inside))
    c["track_qc_pass"] = int(np.count_nonzero(track_use))
    c["track_qc_inside"] = int(np.count_nonzero(track_use_inside))

    # QC flag summaries.
    qc_codes_u, qc_counts_u = np.unique(qc_u, return_counts=True) if len(qc_u) > 0 else (np.array([], dtype=int), np.array([], dtype=int))
    qc_codes_v, qc_counts_v = np.unique(qc_v, return_counts=True) if len(qc_v) > 0 else (np.array([], dtype=int), np.array([], dtype=int))
    qc_u_map = {int(k): int(v) for k, v in zip(qc_codes_u, qc_counts_u)}
    qc_v_map = {int(k): int(v) for k, v in zip(qc_codes_v, qc_counts_v)}
    if bool(cfg.get("report_all_qc_codes", True)):
        qc_report_codes = sorted(set(qc_u_map.keys()) | set(qc_v_map.keys()))
    else:
        qc_report_codes = sorted(set(int(x) for x in qc_keep_codes))
    qc_flag_rows: List[Dict[str, Any]] = []
    for code in qc_report_codes:
        qc_flag_rows.append(
            {
                "qc_code": int(code),
                "u_qc_count": int(qc_u_map.get(int(code), 0)),
                "v_qc_count": int(qc_v_map.get(int(code), 0)),
                "is_keep_code": int(int(code) in set(int(x) for x in qc_keep_codes)),
            }
        )

    # Track summary rows (QC-filtered primary).
    track_rows: List[Dict[str, Any]] = []
    obs_count_per_trackref = np.zeros(len(track["time"]), dtype=int)
    obs_inside_count_per_trackref = np.zeros(len(track["time"]), dtype=int)
    if len(obs["track_ref"]) > 0:
        m1 = valid_tref & obs_use
        m2 = valid_tref & obs_use_inside
        if np.any(m1):
            binc = np.bincount(obs["track_ref"][m1], minlength=len(track["time"]))
            obs_count_per_trackref[: len(binc)] = binc[: len(track["time"])]
        if np.any(m2):
            binc = np.bincount(obs["track_ref"][m2], minlength=len(track["time"]))
            obs_inside_count_per_trackref[: len(binc)] = binc[: len(track["time"])]

    uniq_track_ids = np.unique(track["track_id"]).astype(int)
    for tid in sorted(uniq_track_ids.tolist()):
        f_all = np.where(track["track_id"] == tid)[0]
        if len(f_all) == 0:
            continue
        f = f_all[track_use[f_all]]
        if len(f) == 0:
            continue
        src = str(track["source"][f[0]])
        tfile = str(track["track_file"][f[0]])
        ship = str(track["ship"][f[0]])
        dtyp = str(track["data_type"][f[0]])

        tvals = track["time"][f]
        inside = track_use_inside[f]
        n_total = int(len(f))
        n_inside = int(np.count_nonzero(inside))
        n_segments = int(len(np.unique(track["segment_id"][f])))
        obs_total = int(np.sum(obs_count_per_trackref[f]))
        obs_inside_n = int(np.sum(obs_inside_count_per_trackref[f])) if n_inside > 0 else 0

        tmin = float(np.nanmin(tvals)) if np.any(np.isfinite(tvals)) else np.nan
        tmax = float(np.nanmax(tvals)) if np.any(np.isfinite(tvals)) else np.nan
        span_h = float((tmax - tmin) * 24.0) if np.isfinite(tmin) and np.isfinite(tmax) else np.nan
        dt = _compute_dt_stats(tvals)
        track_rows.append(
            {
                "source": src,
                "track_id": tid,
                "track_file": tfile,
                "ship": ship,
                "data_type": dtyp,
                "instrument_name": DATA_TYPE_NAME.get(dtyp, "Unknown"),
                "n_segments": n_segments,
                "n_track_points": n_total,
                "n_track_points_inside": n_inside,
                "track_inside_ratio": (n_inside / n_total) if n_total > 0 else np.nan,
                "n_obs_points": obs_total,
                "n_obs_points_inside": obs_inside_n,
                "obs_inside_ratio": (obs_inside_n / obs_total) if obs_total > 0 else np.nan,
                "time_min": _fmt_time(tmin),
                "time_max": _fmt_time(tmax),
                "span_hours": span_h,
                "dt_count": dt["dt_count"],
                "dt_median_min": dt["dt_median_min"],
                "dt_p90_min": dt["dt_p90_min"],
                "dt_mode_min": dt["dt_mode_min"],
                "dt_mode_ratio": dt["dt_mode_ratio"],
                "irregular": dt["irregular"],
            }
        )

    # Segment summary rows (QC-filtered primary).
    segment_rows: List[Dict[str, Any]] = []
    if len(obs["segment_id"]) > 0:
        mseg_qc = obs_use & (obs["segment_id"] > 0)
        mseg_qc_in = obs_use_inside & (obs["segment_id"] > 0)
        seg_vals, seg_counts = np.unique(obs["segment_id"][mseg_qc], return_counts=True) if np.any(mseg_qc) else (np.array([], dtype=int), np.array([], dtype=int))
        seg_obs_total_map = {int(s): int(cn) for s, cn in zip(seg_vals, seg_counts)}
        seg_vals_in, seg_counts_in = np.unique(obs["segment_id"][mseg_qc_in], return_counts=True) if np.any(mseg_qc_in) else (np.array([], dtype=int), np.array([], dtype=int))
        seg_obs_inside_map = {int(s): int(cn) for s, cn in zip(seg_vals_in, seg_counts_in)}
    else:
        seg_obs_total_map = {}
        seg_obs_inside_map = {}

    uniq_seg_ids = np.unique(track["segment_id"][track_use]) if np.any(track_use) else np.array([], dtype=int)
    uniq_seg_ids = uniq_seg_ids[uniq_seg_ids > 0].astype(int)
    for sid in sorted(uniq_seg_ids.tolist()):
        f = np.where((track["segment_id"] == sid) & track_use)[0]
        if len(f) == 0:
            continue
        src = str(track["source"][f[0]])
        dtyp = str(track["data_type"][f[0]])
        tfile = str(track["track_file"][f[0]])
        ship = str(track["ship"][f[0]])
        tid = int(track["track_id"][f[0]])
        sseq = int(track["segment_seq"][f[0]])
        tvals = track["time"][f]
        inside = track_use_inside[f]
        n_total = int(len(f))
        n_inside = int(np.count_nonzero(inside))
        obs_total = int(seg_obs_total_map.get(sid, 0))
        obs_inside_n = int(seg_obs_inside_map.get(sid, 0))
        tmin = float(np.nanmin(tvals)) if np.any(np.isfinite(tvals)) else np.nan
        tmax = float(np.nanmax(tvals)) if np.any(np.isfinite(tvals)) else np.nan
        span_h = float((tmax - tmin) * 24.0) if np.isfinite(tmin) and np.isfinite(tmax) else np.nan
        dt = _compute_dt_stats(tvals)
        segment_rows.append(
            {
                "source": src,
                "data_type": dtyp,
                "instrument_name": DATA_TYPE_NAME.get(dtyp, "Unknown"),
                "track_id": tid,
                "segment_id": sid,
                "segment_seq": sseq,
                "track_file": tfile,
                "ship": ship,
                "n_track_points": n_total,
                "n_track_points_inside": n_inside,
                "track_inside_ratio": (n_inside / n_total) if n_total > 0 else np.nan,
                "n_obs_points": obs_total,
                "n_obs_points_inside": obs_inside_n,
                "obs_inside_ratio": (obs_inside_n / obs_total) if obs_total > 0 else np.nan,
                "time_min": _fmt_time(tmin),
                "time_max": _fmt_time(tmax),
                "span_hours": span_h,
                "dt_count": dt["dt_count"],
                "dt_median_min": dt["dt_median_min"],
                "dt_p90_min": dt["dt_p90_min"],
                "dt_mode_min": dt["dt_mode_min"],
                "dt_mode_ratio": dt["dt_mode_ratio"],
                "irregular": dt["irregular"],
            }
        )

    # Data-type summary rows (QC-filtered primary).
    type_rows: List[Dict[str, Any]] = []
    types_set = set()
    if np.any(track_use):
        types_set.update(np.unique(track["data_type"][track_use]).tolist())
    if np.any(obs_use):
        types_set.update(np.unique(obs["data_type"][obs_use]).tolist())
    types_present = sorted(str(x) for x in types_set)
    for dt in types_present:
        ft = np.where((track["data_type"] == dt) & track_use)[0]
        fo = np.where((obs["data_type"] == dt) & obs_use)[0]
        n_t = len(ft)
        n_o = len(fo)
        n_t_in = int(np.count_nonzero(track_use_inside[ft])) if n_t > 0 else 0
        n_o_in = int(np.count_nonzero(obs_use_inside[fo])) if n_o > 0 else 0
        n_files = int(len(np.unique(track["track_file"][ft]))) if n_t > 0 else 0
        n_src = int(len(np.unique(track["source"][ft]))) if n_t > 0 else 0
        n_tracks = int(len(np.unique(track["track_id"][ft]))) if n_t > 0 else 0
        n_segments = int(len(np.unique(track["segment_id"][ft]))) if n_t > 0 else 0

        if n_t > 0:
            tmin = float(np.nanmin(track["time"][ft]))
            tmax = float(np.nanmax(track["time"][ft]))
        else:
            tmin = np.nan
            tmax = np.nan
        if n_o > 0:
            dmin = float(np.nanmin(obs["depth"][fo]))
            dmax = float(np.nanmax(obs["depth"][fo]))
        else:
            dmin = np.nan
            dmax = np.nan
        type_rows.append(
            {
                "data_type": dt,
                "instrument_name": DATA_TYPE_NAME.get(dt, "Unknown"),
                "n_sources": n_src,
                "n_files": n_files,
                "n_tracks": n_tracks,
                "n_segments": n_segments,
                "n_track_points": n_t,
                "n_track_points_inside": n_t_in,
                "track_inside_ratio": (n_t_in / n_t) if n_t > 0 else np.nan,
                "n_obs_points": n_o,
                "n_obs_points_inside": n_o_in,
                "obs_inside_ratio": (n_o_in / n_o) if n_o > 0 else np.nan,
                "time_min": _fmt_time(tmin),
                "time_max": _fmt_time(tmax),
                "depth_min": dmin,
                "depth_max": dmax,
            }
        )

    # Source summary rows (QC-filtered primary).
    source_rows: List[Dict[str, Any]] = []
    for src in selected_sources:
        fs_t = np.where((track["source"] == src) & track_use)[0]
        fs_o = np.where((obs["source"] == src) & obs_use)[0]
        n_t = len(fs_t)
        n_o = len(fs_o)
        n_t_in = int(np.count_nonzero(track_use_inside[fs_t])) if n_t > 0 else 0
        n_o_in = int(np.count_nonzero(obs_use_inside[fs_o])) if n_o > 0 else 0
        n_files = int(len(np.unique(track["track_file"][fs_t]))) if n_t > 0 else 0
        n_tracks = int(len(np.unique(track["track_id"][fs_t]))) if n_t > 0 else 0
        n_segments = int(len(np.unique(track["segment_id"][fs_t]))) if n_t > 0 else 0
        n_tracks_in = int(
            np.sum(
                [
                    np.any(track_use_inside[np.where((track["track_id"] == tid) & track_use)[0]])
                    for tid in np.unique(track["track_id"][fs_t]).astype(int)
                ]
            )
        ) if n_t > 0 else 0

        if n_t > 0:
            tmin = float(np.nanmin(track["time"][fs_t]))
            tmax = float(np.nanmax(track["time"][fs_t]))
        else:
            tmin = np.nan
            tmax = np.nan
        if n_o > 0:
            dmin = float(np.nanmin(obs["depth"][fs_o]))
            dmax = float(np.nanmax(obs["depth"][fs_o]))
        else:
            dmin = np.nan
            dmax = np.nan

        source_rows.append(
            {
                "source": src,
                "n_files": n_files,
                "n_tracks": n_tracks,
                "n_segments": n_segments,
                "n_tracks_with_inside": n_tracks_in,
                "n_track_points": n_t,
                "n_track_points_inside": n_t_in,
                "track_inside_ratio": (n_t_in / n_t) if n_t > 0 else np.nan,
                "n_obs_points": n_o,
                "n_obs_points_inside": n_o_in,
                "obs_inside_ratio": (n_o_in / n_o) if n_o > 0 else np.nan,
                "time_min": _fmt_time(tmin),
                "time_max": _fmt_time(tmax),
                "depth_min": dmin,
                "depth_max": dmax,
            }
        )

    # Coverage tables (yearly/monthly) using QC-filtered track points.
    if np.any(track_use):
        years = np.array([num2date(float(t)).year for t in track["time"][track_use]], dtype=int)
        months = np.array([num2date(float(t)).month for t in track["time"][track_use]], dtype=int)
        source_cov = track["source"][track_use]
        inside_cov = track_use_inside[track_use]
    else:
        years = np.array([], dtype=int)
        months = np.array([], dtype=int)
        source_cov = np.array([], dtype="U16")
        inside_cov = np.array([], dtype=bool)

    cov_year = Counter()
    cov_month = Counter()
    for s, y, m, inside in zip(source_cov, years, months, inside_cov):
        cov_year[(str(s), int(y), "all")] += 1
        cov_month[(str(s), _to_ym(int(y), int(m)), "all")] += 1
        if inside:
            cov_year[(str(s), int(y), "inside")] += 1
            cov_month[(str(s), _to_ym(int(y), int(m)), "inside")] += 1

    cov_year_rows: List[Dict[str, Any]] = []
    for src in selected_sources:
        ys = sorted({y for (s, y, tag) in cov_year.keys() if s == src and tag == "all"})
        for y in ys:
            cov_year_rows.append(
                {
                    "source": src,
                    "year": y,
                    "n_track_points": cov_year.get((src, y, "all"), 0),
                    "n_track_points_inside": cov_year.get((src, y, "inside"), 0),
                }
            )
    cov_month_rows: List[Dict[str, Any]] = []
    for src in selected_sources:
        yms = sorted({ym for (s, ym, tag) in cov_month.keys() if s == src and tag == "all"})
        for ym in yms:
            cov_month_rows.append(
                {
                    "source": src,
                    "year_month": ym,
                    "n_track_points": cov_month.get((src, ym, "all"), 0),
                    "n_track_points_inside": cov_month.get((src, ym, "inside"), 0),
                }
            )

    # QC filter summary output.
    qc_filter_summary = {
        "keep_codes": [int(x) for x in qc_keep_codes],
        "qc_mode": qc_mode,
        "report_all_qc_codes": bool(cfg.get("report_all_qc_codes", True)),
        "n_obs_total": int(len(obs["time"])),
        "n_obs_qc_pass": int(np.count_nonzero(obs_use)),
        "n_obs_qc_fail": int(len(obs["time"]) - np.count_nonzero(obs_use)),
        "n_obs_inside_total": int(np.count_nonzero(obs_inside)),
        "n_obs_qc_inside": int(np.count_nonzero(obs_use_inside)),
        "n_track_points_total": int(len(track["time"])),
        "n_track_points_qc": int(np.count_nonzero(track_use)),
        "n_track_points_qc_inside": int(np.count_nonzero(track_use_inside)),
    }

    # Write NPZ outputs.
    track_npz = outdir / "trusted_track_points.npz"
    tmask = track_use
    T = zdata()
    T.time = track["time"][tmask]
    T.lon = track["lon"][tmask]
    T.lat = track["lat"][tmask]
    T.source = track["source"][tmask]
    T.track_file = track["track_file"][tmask]
    T.track_id = track["track_id"][tmask]
    T.segment_id = track["segment_id"][tmask]
    T.segment_seq = track["segment_seq"][tmask]
    T.ship = track["ship"][tmask]
    T.data_type = track["data_type"][tmask]
    T.time_index = track["time_index"][tmask]
    T.inside = track_use_inside[tmask].astype(int)
    T.qc_pass = np.ones(np.count_nonzero(tmask), dtype=int)
    savez(str(track_npz), T)
    print(f"[OK] wrote: {track_npz}", flush=True)

    inside_npz = outdir / "trusted_inside_current.npz"
    if bool(cfg.get("save_inside_npz", True)):
        mask = obs_use_inside
        O = zdata()
        O.time = obs["time"][mask]
        O.lon = obs["lon"][mask]
        O.lat = obs["lat"][mask]
        O.depth = obs["depth"][mask]
        O.u = obs["u"][mask]
        O.v = obs["v"][mask]
        O.spd = obs["spd"][mask]
        O.spd_kn = obs["spd_kn"][mask]
        O.dir = obs["dir"][mask]
        O.qc_u = obs["qc_u"][mask]
        O.qc_v = obs["qc_v"][mask]
        O.source = obs["source"][mask]
        O.track_file = obs["track_file"][mask]
        O.track_id = obs["track_id"][mask]
        O.segment_id = obs["segment_id"][mask]
        O.segment_seq = obs["segment_seq"][mask]
        O.ship = obs["ship"][mask]
        O.data_type = obs["data_type"][mask]
        O.time_index = obs["time_index"][mask]
        O.depth_index = obs["depth_index"][mask]
        O.track_ref = obs["track_ref"][mask]
        savez(str(inside_npz), O)
        print(f"[OK] wrote: {inside_npz} (qc+inside obs={int(np.count_nonzero(mask)):,})", flush=True)

    # Write CSV outputs.
    source_csv = outdir / "trusted_source_summary.csv"
    track_csv = outdir / "trusted_track_summary.csv"
    segment_csv = outdir / "trusted_segment_summary.csv"
    type_csv = outdir / "trusted_type_summary.csv"
    yearly_csv = outdir / "trusted_coverage_yearly.csv"
    monthly_csv = outdir / "trusted_coverage_monthly.csv"
    qc_flag_csv = outdir / "trusted_qc_flag_summary.csv"
    qc_filter_json = outdir / "trusted_qc_filter_summary.json"
    colloc_csv = outdir / "trusted_collocation_obs.csv"
    colloc_npz = outdir / "trusted_collocation_obs.npz"

    _write_csv(
        source_csv,
        [
            "source",
            "n_files",
            "n_tracks",
            "n_segments",
            "n_tracks_with_inside",
            "n_track_points",
            "n_track_points_inside",
            "track_inside_ratio",
            "n_obs_points",
            "n_obs_points_inside",
            "obs_inside_ratio",
            "time_min",
            "time_max",
            "depth_min",
            "depth_max",
        ],
        source_rows,
    )
    _write_csv(
        track_csv,
        [
            "source",
            "track_id",
            "track_file",
            "ship",
            "data_type",
            "instrument_name",
            "n_segments",
            "n_track_points",
            "n_track_points_inside",
            "track_inside_ratio",
            "n_obs_points",
            "n_obs_points_inside",
            "obs_inside_ratio",
            "time_min",
            "time_max",
            "span_hours",
            "dt_count",
            "dt_median_min",
            "dt_p90_min",
            "dt_mode_min",
            "dt_mode_ratio",
            "irregular",
        ],
        track_rows,
    )
    _write_csv(
        segment_csv,
        [
            "source",
            "data_type",
            "instrument_name",
            "track_id",
            "segment_id",
            "segment_seq",
            "track_file",
            "ship",
            "n_track_points",
            "n_track_points_inside",
            "track_inside_ratio",
            "n_obs_points",
            "n_obs_points_inside",
            "obs_inside_ratio",
            "time_min",
            "time_max",
            "span_hours",
            "dt_count",
            "dt_median_min",
            "dt_p90_min",
            "dt_mode_min",
            "dt_mode_ratio",
            "irregular",
        ],
        segment_rows,
    )
    _write_csv(
        type_csv,
        [
            "data_type",
            "instrument_name",
            "n_sources",
            "n_files",
            "n_tracks",
            "n_segments",
            "n_track_points",
            "n_track_points_inside",
            "track_inside_ratio",
            "n_obs_points",
            "n_obs_points_inside",
            "obs_inside_ratio",
            "time_min",
            "time_max",
            "depth_min",
            "depth_max",
        ],
        type_rows,
    )
    _write_csv(
        yearly_csv,
        ["source", "year", "n_track_points", "n_track_points_inside"],
        cov_year_rows,
    )
    _write_csv(
        monthly_csv,
        ["source", "year_month", "n_track_points", "n_track_points_inside"],
        cov_month_rows,
    )
    _write_csv(
        qc_flag_csv,
        ["qc_code", "u_qc_count", "v_qc_count", "is_keep_code"],
        qc_flag_rows,
    )
    with qc_filter_json.open("w", encoding="utf-8") as f:
        json.dump(qc_filter_summary, f, indent=2, ensure_ascii=True)

    # Model-ready collocation export: QC-filtered + inside boundary.
    cmask = obs_use_inside
    cidx = np.where(cmask)[0]
    if len(cidx) > 0:
        cidx = cidx[np.argsort(obs["time"][cidx])]
    colloc_fields = [
        "timestamp_utc",
        "time_num",
        "lon",
        "lat",
        "depth",
        "u",
        "v",
        "speed",
        "direction",
        "qc_u",
        "qc_v",
        "source",
        "data_type",
        "track_id",
        "segment_id",
        "track_file",
        "ship",
        "time_index",
        "depth_index",
    ]
    if bool(cfg.get("export_collocation_csv", True)):
        with colloc_csv.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=colloc_fields)
            w.writeheader()
            for i in cidx:
                w.writerow(
                    {
                        "timestamp_utc": _fmt_time(float(obs["time"][i])),
                        "time_num": float(obs["time"][i]),
                        "lon": float(obs["lon"][i]),
                        "lat": float(obs["lat"][i]),
                        "depth": float(obs["depth"][i]),
                        "u": float(obs["u"][i]),
                        "v": float(obs["v"][i]),
                        "speed": float(obs["spd"][i]),
                        "direction": float(obs["dir"][i]),
                        "qc_u": int(obs["qc_u"][i]),
                        "qc_v": int(obs["qc_v"][i]),
                        "source": str(obs["source"][i]),
                        "data_type": str(obs["data_type"][i]),
                        "track_id": int(obs["track_id"][i]),
                        "segment_id": int(obs["segment_id"][i]),
                        "track_file": str(obs["track_file"][i]),
                        "ship": str(obs["ship"][i]),
                        "time_index": int(obs["time_index"][i]),
                        "depth_index": int(obs["depth_index"][i]),
                    }
                )
    if bool(cfg.get("export_collocation_npz", True)):
        C = zdata()
        C.time = obs["time"][cidx]
        C.timestamp_utc = np.asarray([_fmt_time(float(t)) for t in obs["time"][cidx]], dtype="U19")
        C.lon = obs["lon"][cidx]
        C.lat = obs["lat"][cidx]
        C.depth = obs["depth"][cidx]
        C.u = obs["u"][cidx]
        C.v = obs["v"][cidx]
        C.spd = obs["spd"][cidx]
        C.dir = obs["dir"][cidx]
        C.qc_u = obs["qc_u"][cidx]
        C.qc_v = obs["qc_v"][cidx]
        C.source = obs["source"][cidx]
        C.data_type = obs["data_type"][cidx]
        C.track_id = obs["track_id"][cidx]
        C.segment_id = obs["segment_id"][cidx]
        C.track_file = obs["track_file"][cidx]
        C.ship = obs["ship"][cidx]
        C.time_index = obs["time_index"][cidx]
        C.depth_index = obs["depth_index"][cidx]
        C.track_ref = obs["track_ref"][cidx]
        savez(str(colloc_npz), C)

    print(f"[OK] wrote: {source_csv}", flush=True)
    print(f"[OK] wrote: {track_csv}", flush=True)
    print(f"[OK] wrote: {segment_csv}", flush=True)
    print(f"[OK] wrote: {type_csv}", flush=True)
    print(f"[OK] wrote: {yearly_csv}", flush=True)
    print(f"[OK] wrote: {monthly_csv}", flush=True)
    print(f"[OK] wrote: {qc_flag_csv}", flush=True)
    print(f"[OK] wrote: {qc_filter_json}", flush=True)
    if bool(cfg.get("export_collocation_csv", True)):
        print(f"[OK] wrote: {colloc_csv}", flush=True)
    if bool(cfg.get("export_collocation_npz", True)):
        print(f"[OK] wrote: {colloc_npz}", flush=True)

    # Plot outputs.
    plot_outputs: List[str] = []
    if bool(cfg.get("plot", True)):
        try:
            plt = _ensure_plot_backend()
            p1 = outdir / "map_all_trusted_points_with_boundary.png"
            p2 = outdir / "map_inside_points_with_boundary.png"
            p3 = outdir / "map_source_facets_inside.png"
            p4 = outdir / "coverage_yearly_by_source.png"
            p5 = outdir / "track_samples_by_source.png"
            p6 = outdir / "track_time_step_distribution.png"
            p7 = outdir / "segment_speed_time_by_type.png"
            p8 = outdir / "segment_hodograph_by_type.png"
            p9 = outdir / "ca_depth_time_speed.png"
            p10 = outdir / "depthbin_speed_time_by_type.png"
            p11d = outdir / "segment_speed_time_by_source"
            p12d = outdir / "segment_hodograph_by_source"
            p13d = outdir / "segment_individual"

            _plot_map_all(
                plt=plt,
                outpath=p1,
                parts=parts,
                lon=track["lon"][track_use],
                lat=track["lat"][track_use],
                sample_max=int(cfg["plot_sample_max"]),
            )
            _plot_map_inside(
                plt=plt,
                outpath=p2,
                parts=parts,
                lon=track["lon"][track_use],
                lat=track["lat"][track_use],
                source=track["source"][track_use],
                inside=track_use_inside[track_use],
                sources=selected_sources,
                sample_max=int(cfg["plot_sample_max"]),
            )
            _plot_source_facets_inside(
                plt=plt,
                outpath=p3,
                parts=parts,
                lon=track["lon"][track_use],
                lat=track["lat"][track_use],
                source=track["source"][track_use],
                inside=track_use_inside[track_use],
                sources=selected_sources,
                sample_max=max(40000, int(cfg["plot_sample_max"] // 4)),
            )
            _plot_coverage_yearly(
                plt=plt,
                outpath=p4,
                years_arr=years,
                source_arr=source_cov,
                inside_arr=inside_cov,
                sources=selected_sources,
            )
            _plot_track_samples_by_source(
                plt=plt,
                outpath=p5,
                parts=parts,
                source_arr=track["source"][track_use],
                track_id_arr=track["track_id"][track_use],
                time_arr=track["time"][track_use],
                lon_arr=track["lon"][track_use],
                lat_arr=track["lat"][track_use],
                inside_arr=track_use_inside[track_use],
                sources=selected_sources,
                max_track_plots_per_source=int(cfg["max_track_plots_per_source"]),
            )
            _plot_timestep_distribution(
                plt=plt,
                outpath=p6,
                track_rows=track_rows,
                sources=selected_sources,
            )
            _plot_segment_speed_time_by_type(
                plt=plt,
                outpath=p7,
                obs=obs,
                obs_use=obs_use_inside,
                data_types=types_present,
                nmax=int(cfg["max_segment_plots_per_type"]),
            )
            _plot_segment_hodograph_by_type(
                plt=plt,
                outpath=p8,
                obs=obs,
                obs_use=obs_use_inside,
                data_types=types_present,
                nmax=int(cfg["max_segment_plots_per_type"]),
            )
            _plot_ca_depth_time_speed(
                plt=plt,
                outpath=p9,
                obs=obs,
                obs_use=obs_use_inside,
            )
            _plot_depthbin_speed_time_by_type(
                plt=plt,
                outpath=p10,
                obs=obs,
                obs_use=obs_use_inside,
                data_types=types_present,
                depth_bins=depth_bins,
                depth_bin_labels=depth_bin_labels,
            )
            source_seg_speed_paths = _plot_segment_speed_time_by_source(
                plt=plt,
                outdir=p11d,
                obs=obs,
                obs_use=obs_use_inside,
                sources=selected_sources,
                nmax=int(cfg.get("max_segment_plots_per_source", 0)),
            )
            source_seg_hodo_paths = _plot_segment_hodograph_by_source(
                plt=plt,
                outdir=p12d,
                obs=obs,
                obs_use=obs_use_inside,
                sources=selected_sources,
                nmax=int(cfg.get("max_segment_plots_per_source", 0)),
            )
            segment_individual_paths: List[Path] = []
            if bool(cfg.get("plot_each_segment", True)):
                segment_individual_paths = _plot_individual_segments_by_source(
                    plt=plt,
                    outdir=p13d,
                    obs=obs,
                    obs_use=obs_use_inside,
                    sources=selected_sources,
                    nmax=int(cfg.get("max_individual_segment_plots_per_source", 0)),
                )

            plot_outputs = [str(p) for p in [p1, p2, p3, p4, p5, p6, p7, p8, p9, p10]]
            plot_outputs.extend([str(p) for p in source_seg_speed_paths])
            plot_outputs.extend([str(p) for p in source_seg_hodo_paths])
            plot_outputs.extend([str(p) for p in segment_individual_paths])
            print(f"[OK] wrote plots: {len(plot_outputs)} files", flush=True)
        except Exception as exc:
            plot_outputs = []
            print(f"[WARN] plotting failed: {exc}", flush=True)

    depth_bin_stats: Dict[str, Dict[str, int]] = {}
    for dt in types_present:
        fdt = np.where((obs["data_type"] == dt) & obs_use_inside)[0]
        bstat: Dict[str, int] = {}
        for ib in range(len(depth_bins) - 1):
            z0 = float(depth_bins[ib])
            z1 = float(depth_bins[ib + 1])
            nbin = int(np.count_nonzero((obs["depth"][fdt] >= z0) & (obs["depth"][fdt] < z1)))
            bstat[str(depth_bin_labels[ib])] = nbin
        depth_bin_stats[str(dt)] = bstat

    run_summary = {
        "config": {
            "nc_dir": str(nc_dir),
            "trusted_csv": str(trusted_csv),
            "classification_csv": str(classification_csv),
            "boundary_shp": str(boundary_shp),
            "source_selection_mode": source_selection_mode,
            "trusted_tier": trusted_tier,
            "selected_sources": selected_sources,
            "selected_data_types": selected_data_types if len(selected_data_types) > 0 else None,
            "start": _fmt_time(float(date2num([start_dt])[0])) if start_dt is not None else None,
            "end": _fmt_time(float(date2num([end_dt])[0])) if end_dt is not None else None,
            "min_record_length_hours": float(min_record_length_hours),
            "min_depth": cfg.get("min_depth"),
            "max_depth": cfg.get("max_depth"),
            "qc_keep_codes": [int(x) for x in qc_keep_codes],
            "qc_mode": qc_mode,
            "report_all_qc_codes": bool(cfg.get("report_all_qc_codes", True)),
            "inside_mode": cfg.get("inside_mode"),
            "ring_close_tol": cfg.get("ring_close_tol"),
            "depth_bins": [float(x) for x in depth_bins],
            "depth_bin_labels": list(depth_bin_labels),
            "save_inside_npz": bool(cfg.get("save_inside_npz", True)),
            "plot": bool(cfg.get("plot", True)),
            "max_track_plots_per_source": int(cfg.get("max_track_plots_per_source", 3)),
            "max_segment_plots_per_type": int(cfg.get("max_segment_plots_per_type", 4)),
            "max_segment_plots_per_source": int(cfg.get("max_segment_plots_per_source", 0)),
            "plot_each_segment": bool(cfg.get("plot_each_segment", True)),
            "max_individual_segment_plots_per_source": int(cfg.get("max_individual_segment_plots_per_source", 0)),
            "plot_sample_max": int(cfg.get("plot_sample_max", 180000)),
            "export_collocation_csv": bool(cfg.get("export_collocation_csv", True)),
            "export_collocation_npz": bool(cfg.get("export_collocation_npz", True)),
        },
        "counts": {
            **dict(c),
            **dict(stats),
            "n_track_points_raw_total": int(len(track["time"])),
            "n_track_points_raw_inside": int(np.count_nonzero(track_inside)),
            "n_track_points_qc_total": int(np.count_nonzero(track_use)),
            "n_track_points_qc_inside": int(np.count_nonzero(track_use_inside)),
            "n_obs_points_raw_total": int(len(obs["time"])),
            "n_obs_points_raw_inside": int(np.count_nonzero(obs_inside)),
            "n_obs_points_qc_total": int(np.count_nonzero(obs_use)),
            "n_obs_points_qc_inside": int(np.count_nonzero(obs_use_inside)),
            "n_tracks_qc": int(len(np.unique(track["track_id"][track_use]))) if np.any(track_use) else 0,
            "n_segments_qc": int(len(np.unique(track["segment_id"][track_use]))) if np.any(track_use) else 0,
            "data_type_track_counts": {
                str(dt): int(np.count_nonzero((track["data_type"] == dt) & track_use)) for dt in types_present
            },
            "depth_bin_obs_inside_counts": depth_bin_stats,
        },
        "time_range": {
            "track_time_min_raw": _fmt_time(float(np.nanmin(track["time"]))),
            "track_time_max_raw": _fmt_time(float(np.nanmax(track["time"]))),
            "obs_time_min_raw": _fmt_time(float(np.nanmin(obs["time"]))) if len(obs["time"]) > 0 else "",
            "obs_time_max_raw": _fmt_time(float(np.nanmax(obs["time"]))) if len(obs["time"]) > 0 else "",
            "track_time_min_qc": _fmt_time(float(np.nanmin(track["time"][track_use]))) if np.any(track_use) else "",
            "track_time_max_qc": _fmt_time(float(np.nanmax(track["time"][track_use]))) if np.any(track_use) else "",
            "obs_time_min_qc": _fmt_time(float(np.nanmin(obs["time"][obs_use]))) if np.any(obs_use) else "",
            "obs_time_max_qc": _fmt_time(float(np.nanmax(obs["time"][obs_use]))) if np.any(obs_use) else "",
        },
        "boundary": {
            "shp_shape_type": int(shp_info["shape_type"]),
            "bbox": shp_info["bbox"],
            "n_parts": int(len(parts)),
            "n_closed_rings": int(len(rings)),
            "inside_meta": inside_meta,
        },
        "outputs": {
            "track_npz": str(track_npz),
            "inside_npz": str(inside_npz) if bool(cfg.get("save_inside_npz", True)) else None,
            "source_summary_csv": str(source_csv),
            "track_summary_csv": str(track_csv),
            "segment_summary_csv": str(segment_csv),
            "type_summary_csv": str(type_csv),
            "coverage_yearly_csv": str(yearly_csv),
            "coverage_monthly_csv": str(monthly_csv),
            "qc_flag_summary_csv": str(qc_flag_csv),
            "qc_filter_summary_json": str(qc_filter_json),
            "collocation_csv": str(colloc_csv) if bool(cfg.get("export_collocation_csv", True)) else None,
            "collocation_npz": str(colloc_npz) if bool(cfg.get("export_collocation_npz", True)) else None,
            "plots": plot_outputs,
        },
    }

    summary_json = outdir / "trusted_run_summary.json"
    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2, ensure_ascii=True)
    print(f"[OK] wrote: {summary_json}", flush=True)
    print(
        "[INFO] done:"
        f" files={stats['nc_files_selected']:,},"
        f" track_points_raw={len(track['time']):,},"
        f" track_points_qc={int(np.count_nonzero(track_use)):,},"
        f" track_inside_qc={int(np.count_nonzero(track_use_inside)):,},"
        f" obs_points_raw={len(obs['time']):,},"
        f" obs_points_qc={int(np.count_nonzero(obs_use)):,},"
        f" obs_inside_qc={int(np.count_nonzero(obs_use_inside)):,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
