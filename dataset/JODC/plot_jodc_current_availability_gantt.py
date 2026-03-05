#!/usr/bin/env python3
"""
Plot Gantt-style JODC current observation availability.

This script scans selected NC files, extracts valid observation times, optionally
applies QC and inside-boundary filtering, then plots merged availability windows.
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
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pylib import ReadNC, date2num, inside_polygon, num2date


DEFAULT_CFG: Dict[str, Any] = {
    "nc_dir": "/Users/kpark/Documents/Projects/Active/AIMEC_TohokuCoast/01_Data/JODC/Current/vector-140E-145E35N-40N_NCFiles",
    "trusted_csv": "/Users/kpark/Documents/Codes/coastal-ocean-utils/dataset/JODC/nc_trusted_sources.csv",
    "classification_csv": "/Users/kpark/Documents/Codes/coastal-ocean-utils/dataset/JODC/nc_filename_classification.csv",
    "trusted_tier": "core",
    "sources": None,
    "data_types": None,
    "start": "2012-01-01",
    "end": "2024-12-31",
    "row_mode": "source_type",  # source|source_type|file
    "max_rows": 120,
    "min_row_points": 1,
    "sort_by": "coverage",  # coverage|first|last|points|segments|name
    "merge_gap_hours": 72.0,
    "min_segment_hours": 6.0,
    "use_qc": True,
    "qc_keep_codes": [0],
    "qc_mode": "strict_both",  # strict_both|either
    "inside_only": True,
    "boundary_shp": "/Users/kpark/Documents/Projects/Active/AIMEC_TohokuCoast/01_Data/Grid/shp/SOB.shp",
    "inside_mode": "largest_closed_ring",  # largest_closed_ring|all_closed_rings
    "ring_close_tol": 1.0e-6,
    "outdir": "./trusted_current_availability",
    "outfile": "jodc_current_availability_gantt.png",
    "dpi": 220,
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


def _read_nc_quiet(path: Path) -> Any:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=r".*valid_min not used.*", category=UserWarning)
        warnings.filterwarnings("ignore", message=r".*valid_max not used.*", category=UserWarning)
        return ReadNC(str(path))


def _safe_nc_attr(nc: Any, name: str, default: str = "") -> str:
    try:
        return str(nc.getncattr(name)).strip()
    except Exception:
        pass
    try:
        if hasattr(nc, name):
            return str(getattr(nc, name)).strip()
    except Exception:
        pass
    try:
        d = getattr(nc, "__dict__", {})
        if isinstance(d, dict) and name in d:
            return str(d[name]).strip()
    except Exception:
        pass
    return default


def _get_nc_var(nc: Any, name: str) -> Optional[np.ndarray]:
    try:
        if hasattr(nc, "variables") and name in nc.variables:
            v = nc.variables[name]
            if hasattr(v, "val"):
                return np.asarray(v.val)
            return np.asarray(v[:])
    except Exception:
        pass
    try:
        if hasattr(nc, name):
            v = getattr(nc, name)
            if hasattr(v, "val"):
                return np.asarray(v.val)
            return np.asarray(v)
    except Exception:
        pass
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


def _parse_obs_datetime(obs_date: np.ndarray, obs_time: np.ndarray) -> np.ndarray:
    n = len(obs_date)
    out = np.full(n, np.nan, dtype=float)
    idx: List[int] = []
    dts: List[datetime] = []
    for i in range(n):
        try:
            d = int(obs_date[i])
        except Exception:
            continue
        if d < 18000101 or d > 21001231:
            continue
        try:
            t = int(obs_time[i])
        except Exception:
            t = 0
        hh = t // 10000
        mm = (t % 10000) // 100
        ss = t % 100
        if hh > 23 or mm > 59 or ss > 59:
            continue
        try:
            dt = datetime.strptime(str(d), "%Y%m%d").replace(hour=hh, minute=mm, second=ss)
        except Exception:
            continue
        idx.append(i)
        dts.append(dt)
    if len(dts) > 0:
        out[np.asarray(idx, dtype=int)] = np.asarray(date2num(dts), dtype=float)
    return out


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
    seen = set()
    uniq: List[str] = []
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
            if fn != "":
                out[fn] = src
    return out


def _parse_code_list(values: Any, default: Sequence[int]) -> List[int]:
    if values is None:
        return [int(x) for x in default]
    items: List[str]
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
    return out if len(out) > 0 else [int(x) for x in default]


def _qc_pass_mask(qc_u: np.ndarray, qc_v: np.ndarray, keep_codes: Sequence[int], mode: str) -> np.ndarray:
    keep = np.asarray(list(keep_codes), dtype=int)
    mu = np.isin(qc_u, keep)
    mv = np.isin(qc_v, keep)
    if mode == "either":
        return mu | mv
    return mu & mv


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
        raise RuntimeError(f"Expected PolyLine shapefile but got type={shape_type} in {shp_path}")
    xmin, ymin, xmax, ymax = struct.unpack("<4d", header[36:68])
    parts: List[np.ndarray] = []
    pos = 100
    nbytes = len(data)
    while pos + 8 <= nbytes:
        _, content_len_words = struct.unpack(">2i", data[pos : pos + 8])
        pos += 8
        content_len = content_len_words * 2
        if content_len <= 0 or pos + content_len > nbytes:
            break
        content = data[pos : pos + content_len]
        pos += content_len
        if len(content) < 44:
            continue
        rec_shape_type = struct.unpack("<i", content[:4])[0]
        if rec_shape_type == 0:
            continue
        if rec_shape_type not in (3, 13, 23):
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
            if e - s >= 2:
                parts.append(pts[s:e, :])
    if len(parts) == 0:
        raise RuntimeError(f"No polyline parts found in {shp_path}")
    return {"parts": parts, "bbox": (float(xmin), float(ymin), float(xmax), float(ymax)), "shape_type": int(shape_type)}


def _is_closed_ring(xy: np.ndarray, tol: float) -> bool:
    if xy.ndim != 2 or xy.shape[0] < 4:
        return False
    return math.hypot(float(xy[0, 0] - xy[-1, 0]), float(xy[0, 1] - xy[-1, 1])) <= float(tol)


def _polygon_area_abs(xy: np.ndarray) -> float:
    if xy.ndim != 2 or xy.shape[0] < 4:
        return 0.0
    x = xy[:, 0]
    y = xy[:, 1]
    return 0.5 * float(abs(np.sum(x[:-1] * y[1:] - x[1:] * y[:-1])))


def _extract_closed_rings(parts: List[np.ndarray], tol: float) -> List[Dict[str, Any]]:
    rings: List[Dict[str, Any]] = []
    for i, seg in enumerate(parts):
        if _is_closed_ring(seg, tol):
            rings.append({"part_index": i, "xy": seg, "area": _polygon_area_abs(seg)})
    rings.sort(key=lambda r: r["area"], reverse=True)
    return rings


def _inside_from_rings(points_xy: np.ndarray, rings: List[Dict[str, Any]], inside_mode: str) -> np.ndarray:
    if len(rings) == 0:
        raise RuntimeError("No closed polygon ring detected from boundary polyline.")
    mask_inside = np.zeros(points_xy.shape[0], dtype=bool)
    if inside_mode == "largest_closed_ring":
        ring = rings[0]
        raw = np.asarray(inside_polygon(points_xy, ring["xy"][:, 0], ring["xy"][:, 1])).reshape(-1)
        mask_inside = raw == 1
    elif inside_mode == "all_closed_rings":
        for ring in rings:
            raw = np.asarray(inside_polygon(points_xy, ring["xy"][:, 0], ring["xy"][:, 1])).reshape(-1)
            mask_inside |= raw == 1
    else:
        raise ValueError(f"Unsupported inside mode: {inside_mode}")
    return mask_inside


def _inside_from_all_parts_path(points_xy: np.ndarray, parts: List[np.ndarray]) -> np.ndarray:
    blocks: List[np.ndarray] = []
    for seg in parts:
        if seg.ndim != 2 or seg.shape[0] < 2:
            continue
        blocks.append(np.asarray(seg, dtype=float))
        blocks.append(np.array([[np.nan, np.nan]], dtype=float))
    if len(blocks) == 0:
        return np.zeros(points_xy.shape[0], dtype=bool)
    path_xy = np.vstack(blocks[:-1])
    raw = np.asarray(inside_polygon(points_xy, path_xy[:, 0], path_xy[:, 1])).reshape(-1)
    return raw == 1


def _fmt_time(num: float) -> str:
    if not np.isfinite(num):
        return ""
    return num2date(float(num)).strftime("%Y-%m-%d %H:%M:%S")


def _normalize_qc_to_time(q: np.ndarray, tlen: int) -> Optional[np.ndarray]:
    arr = np.asarray(q)
    if arr.ndim == 1:
        if len(arr) != tlen:
            return None
        return arr[:, None]
    if arr.ndim != 2:
        return None
    if arr.shape[0] == tlen:
        return arr
    if arr.shape[1] == tlen:
        return arr.T
    return None


def _merge_segments(times_num: np.ndarray, gap_days: float, min_width_days: float) -> List[Tuple[float, float, int]]:
    arr = np.asarray(times_num, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return []
    arr = np.sort(arr)
    segs: List[Tuple[float, float, int]] = []
    s = float(arr[0])
    e = float(arr[0])
    n = 1
    for t in arr[1:]:
        tv = float(t)
        if (tv - e) <= gap_days:
            e = tv
            n += 1
        else:
            if (e - s) < min_width_days:
                e = s + min_width_days
            segs.append((s, e, n))
            s = tv
            e = tv
            n = 1
    if (e - s) < min_width_days:
        e = s + min_width_days
    segs.append((s, e, n))
    return segs


def _build_row_key(row_mode: str, source: str, data_type: str, filename: str) -> str:
    if row_mode == "source":
        return source
    if row_mode == "source_type":
        return f"{source}:{data_type}"
    if row_mode == "file":
        return filename
    raise ValueError(f"Unsupported row_mode: {row_mode}")


def _ensure_plot_backend():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot JODC current data-availability Gantt chart.")
    p.add_argument("--nc-dir", default=DEFAULT_CFG["nc_dir"])
    p.add_argument("--trusted-csv", default=DEFAULT_CFG["trusted_csv"])
    p.add_argument("--classification-csv", default=DEFAULT_CFG["classification_csv"])
    p.add_argument("--trusted-tier", choices=["core", "extended", "all"], default=DEFAULT_CFG["trusted_tier"])
    p.add_argument("--sources", nargs="+", default=None, help="Explicit sources; overrides --trusted-tier.")
    p.add_argument("--data-types", nargs="+", default=None, help="Optional filter list, e.g., CA CU CD CV.")
    p.add_argument("--start", default=DEFAULT_CFG["start"], help="YYYY-MM-DD[ HH:MM[:SS]]")
    p.add_argument("--end", default=DEFAULT_CFG["end"], help="YYYY-MM-DD[ HH:MM[:SS]]")

    p.add_argument("--row-mode", choices=["source", "source_type", "file"], default=DEFAULT_CFG["row_mode"])
    p.add_argument("--max-rows", type=int, default=DEFAULT_CFG["max_rows"])
    p.add_argument("--min-row-points", type=int, default=DEFAULT_CFG["min_row_points"])
    p.add_argument("--sort-by", choices=["coverage", "first", "last", "points", "segments", "name"], default=DEFAULT_CFG["sort_by"])
    p.add_argument("--merge-gap-hours", type=float, default=DEFAULT_CFG["merge_gap_hours"])
    p.add_argument("--min-segment-hours", type=float, default=DEFAULT_CFG["min_segment_hours"])

    p.add_argument("--use-qc", dest="use_qc", action="store_true", help="Use QC-filtered availability.")
    p.add_argument("--no-use-qc", dest="use_qc", action="store_false", help="Ignore QC flags.")
    p.set_defaults(use_qc=DEFAULT_CFG["use_qc"])
    p.add_argument("--qc-keep-codes", nargs="+", default=DEFAULT_CFG["qc_keep_codes"])
    p.add_argument("--qc-mode", choices=["strict_both", "either"], default=DEFAULT_CFG["qc_mode"])

    p.add_argument("--inside-only", dest="inside_only", action="store_true", help="Keep only points inside boundary.")
    p.add_argument("--no-inside-only", dest="inside_only", action="store_false", help="Do not apply inside filter.")
    p.set_defaults(inside_only=DEFAULT_CFG["inside_only"])
    p.add_argument("--boundary-shp", default=DEFAULT_CFG["boundary_shp"], help="Boundary shapefile used with --inside-only.")
    p.add_argument("--inside-mode", choices=["largest_closed_ring", "all_closed_rings"], default=DEFAULT_CFG["inside_mode"])
    p.add_argument("--ring-close-tol", type=float, default=DEFAULT_CFG["ring_close_tol"])

    p.add_argument("--outdir", default=DEFAULT_CFG["outdir"])
    p.add_argument("--outfile", default=DEFAULT_CFG["outfile"])
    p.add_argument("--dpi", type=int, default=DEFAULT_CFG["dpi"])
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)

    nc_dir = _resolve_path(args.nc_dir)
    trusted_csv = _resolve_path(args.trusted_csv)
    classification_csv = _resolve_path(args.classification_csv)
    outdir = _resolve_path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if not nc_dir.is_dir():
        raise FileNotFoundError(f"nc_dir not found: {nc_dir}")
    if not trusted_csv.is_file():
        raise FileNotFoundError(f"trusted_csv not found: {trusted_csv}")
    if not classification_csv.is_file():
        raise FileNotFoundError(f"classification_csv not found: {classification_csv}")

    start_dt = _parse_bound(args.start, is_end=False)
    end_dt = _parse_bound(args.end, is_end=True)
    if start_dt is not None and end_dt is not None and end_dt < start_dt:
        raise ValueError("end must be >= start")
    start_num = float(date2num([start_dt])[0]) if start_dt is not None else None
    end_num = float(date2num([end_dt])[0]) if end_dt is not None else None

    if args.sources is not None and len(args.sources) > 0:
        selected_sources = []
        seen = set()
        for s in args.sources:
            ss = str(s).strip()
            if ss == "" or ss in seen:
                continue
            seen.add(ss)
            selected_sources.append(ss)
        source_mode = "explicit_sources"
    else:
        selected_sources = _load_trusted_sources(trusted_csv, str(args.trusted_tier).strip().lower())
        source_mode = f"trusted_tier:{str(args.trusted_tier).strip().lower()}"
    if len(selected_sources) == 0:
        raise RuntimeError("No selected source.")

    selected_data_types: List[str] = []
    if args.data_types is not None and len(args.data_types) > 0:
        selected_data_types = sorted({str(x).strip().upper() for x in args.data_types if str(x).strip() != ""})
    qc_keep_codes = _parse_code_list(args.qc_keep_codes, default=[0])
    qc_mode = str(args.qc_mode).strip().lower()

    inside_ctx: Dict[str, Any] = {"enabled": bool(args.inside_only), "fallback_used": False}
    boundary_parts: Optional[List[np.ndarray]] = None
    boundary_rings: Optional[List[Dict[str, Any]]] = None
    if bool(args.inside_only):
        if args.boundary_shp is None:
            raise ValueError("--inside-only requires --boundary-shp.")
        boundary_shp = _resolve_path(args.boundary_shp)
        if not boundary_shp.is_file():
            raise FileNotFoundError(f"boundary_shp not found: {boundary_shp}")
        shp_info = _read_polyline_parts(boundary_shp)
        boundary_parts = shp_info["parts"]
        boundary_rings = _extract_closed_rings(boundary_parts, tol=float(args.ring_close_tol))
        if len(boundary_rings) == 0:
            raise RuntimeError("No closed ring found in boundary shapefile.")
        inside_ctx["boundary_shp"] = str(boundary_shp)
        inside_ctx["shape_type"] = int(shp_info["shape_type"])
        inside_ctx["n_parts"] = int(len(boundary_parts))
        inside_ctx["n_closed_rings"] = int(len(boundary_rings))

    file_source = _load_filename_source_map(classification_csv)
    nc_files = sorted(nc_dir.glob("*.nc"))
    selected_files: List[Tuple[Path, str]] = []
    stats = Counter()
    for fp in nc_files:
        stats["nc_files_total"] += 1
        src = file_source.get(fp.name)
        if src is None:
            stats["nc_files_unmapped"] += 1
            continue
        if src not in selected_sources:
            stats["nc_files_unselected_source"] += 1
            continue
        selected_files.append((fp, src))
    if len(selected_files) == 0:
        raise RuntimeError("No selected NC files after source filtering.")

    print(
        f"[INFO] files selected: {len(selected_files):,}/{len(nc_files):,} "
        f"(source_mode={source_mode}, row_mode={args.row_mode}, use_qc={bool(args.use_qc)}, inside_only={bool(args.inside_only)})",
        flush=True,
    )

    row_times: Dict[str, List[np.ndarray]] = defaultdict(list)
    row_sources: Dict[str, str] = {}
    row_types: Dict[str, str] = {}
    row_files: Dict[str, set] = defaultdict(set)

    for i, (fp, source_token) in enumerate(selected_files, start=1):
        stats["file_seen"] += 1
        try:
            nc = _read_nc_quiet(fp)
        except Exception:
            stats["file_read_fail"] += 1
            continue

        data_type = _safe_nc_attr(nc, "DATA_TYPE", "").upper()
        if data_type == "":
            data_type = "UNK"
        if len(selected_data_types) > 0 and data_type not in selected_data_types:
            stats["file_filtered_data_type"] += 1
            continue

        obs_date = _get_nc_var(nc, "obs_date")
        obs_time = _get_nc_var(nc, "obs_time")
        lon = _get_nc_var(nc, "longitude")
        lat = _get_nc_var(nc, "latitude")
        if obs_date is None:
            stats["file_missing_obs_date"] += 1
            continue
        obs_date = np.asarray(obs_date).reshape(-1)
        if obs_time is None:
            obs_time = np.zeros_like(obs_date)
            stats["file_missing_obs_time"] += 1
        else:
            obs_time = np.asarray(obs_time).reshape(-1)
        if obs_time.size == 1 and obs_date.size > 1:
            obs_time = np.repeat(obs_time, obs_date.size)
        if obs_time.size != obs_date.size:
            stats["file_bad_obs_time_shape"] += 1
            continue

        tnum = _parse_obs_datetime(obs_date, obs_time)
        tlen = len(tnum)
        tmask = np.isfinite(tnum)
        if start_num is not None:
            tmask &= tnum >= float(start_num)
        if end_num is not None:
            tmask &= tnum <= float(end_num)

        if bool(args.use_qc):
            u_qc = _get_nc_var(nc, "u_QC")
            v_qc = _get_nc_var(nc, "v_QC")
            if u_qc is None or v_qc is None:
                stats["file_missing_qc"] += 1
                tmask &= False
            else:
                qu = _normalize_qc_to_time(np.asarray(u_qc), tlen=tlen)
                qv = _normalize_qc_to_time(np.asarray(v_qc), tlen=tlen)
                if qu is None or qv is None or qu.shape != qv.shape:
                    stats["file_bad_qc_shape"] += 1
                    tmask &= False
                else:
                    qpass = _qc_pass_mask(qu.astype(int), qv.astype(int), qc_keep_codes, qc_mode)
                    if qpass.ndim == 2:
                        qtime = np.any(qpass, axis=1)
                    else:
                        qtime = qpass.reshape(-1)
                    if len(qtime) != tlen:
                        stats["file_bad_qc_time_axis"] += 1
                        tmask &= False
                    else:
                        tmask &= qtime

        if bool(args.inside_only):
            if lon is None or lat is None:
                stats["file_missing_lonlat_for_inside"] += 1
                tmask &= False
            else:
                lonv = np.asarray(lon, dtype=float).reshape(-1)
                latv = np.asarray(lat, dtype=float).reshape(-1)
                if lonv.size == 1 and tlen > 1:
                    lonv = np.repeat(lonv, tlen)
                if latv.size == 1 and tlen > 1:
                    latv = np.repeat(latv, tlen)
                if lonv.size != tlen or latv.size != tlen:
                    stats["file_bad_lonlat_shape"] += 1
                    tmask &= False
                else:
                    pt = np.c_[lonv, latv]
                    if boundary_rings is None or boundary_parts is None:
                        raise RuntimeError("Boundary context not initialized.")
                    inmask = _inside_from_rings(pt, boundary_rings, str(args.inside_mode))
                    if int(np.count_nonzero(inmask)) == 0:
                        in_fallback = _inside_from_all_parts_path(pt, boundary_parts)
                        if int(np.count_nonzero(in_fallback)) > 0:
                            inmask = in_fallback
                            inside_ctx["fallback_used"] = True
                        stats["inside_ring_zero_files"] += 1
                    tmask &= inmask

        if not np.any(tmask):
            stats["file_no_valid_time"] += 1
            continue

        tt = tnum[tmask]
        key = _build_row_key(str(args.row_mode), source_token, data_type, fp.name)
        row_times[key].append(np.asarray(tt, dtype=float))
        row_sources[key] = source_token
        row_types[key] = data_type
        row_files[key].add(fp.name)
        stats["valid_points_total"] += int(len(tt))
        stats["file_with_valid_time"] += 1

        if i % 300 == 0:
            print(
                f"[INFO] parsed {i}/{len(selected_files)} files; "
                f"rows={len(row_times):,}, valid_points={stats['valid_points_total']:,}",
                flush=True,
            )

    if len(row_times) == 0:
        raise RuntimeError("No valid availability points after filters.")

    gap_days = float(args.merge_gap_hours) / 24.0
    min_width_days = max(0.0, float(args.min_segment_hours) / 24.0)

    row_info: List[Dict[str, Any]] = []
    row_segments: Dict[str, List[Tuple[float, float, int]]] = {}
    for key, chunks in row_times.items():
        arr = np.concatenate(chunks) if len(chunks) > 1 else chunks[0]
        arr = arr[np.isfinite(arr)]
        if len(arr) == 0:
            continue
        arr = np.sort(arr)
        segs = _merge_segments(arr, gap_days=gap_days, min_width_days=min_width_days)
        if len(arr) < int(args.min_row_points):
            continue
        coverage_days = float(sum((e - s) for s, e, _ in segs))
        row_info.append(
            {
                "row_key": key,
                "source": row_sources.get(key, ""),
                "data_type": row_types.get(key, ""),
                "n_files": int(len(row_files.get(key, set()))),
                "n_points": int(len(arr)),
                "n_segments": int(len(segs)),
                "time_min_num": float(arr[0]),
                "time_max_num": float(arr[-1]),
                "time_min": _fmt_time(float(arr[0])),
                "time_max": _fmt_time(float(arr[-1])),
                "coverage_days": coverage_days,
            }
        )
        row_segments[key] = segs

    if len(row_info) == 0:
        raise RuntimeError("No rows left after min-row-points filtering.")

    sort_by = str(args.sort_by)
    if sort_by == "coverage":
        row_info.sort(key=lambda r: (float(r["coverage_days"]), int(r["n_points"])), reverse=True)
    elif sort_by == "first":
        row_info.sort(key=lambda r: float(r["time_min_num"]))
    elif sort_by == "last":
        row_info.sort(key=lambda r: float(r["time_max_num"]), reverse=True)
    elif sort_by == "points":
        row_info.sort(key=lambda r: int(r["n_points"]), reverse=True)
    elif sort_by == "segments":
        row_info.sort(key=lambda r: int(r["n_segments"]), reverse=True)
    else:
        row_info.sort(key=lambda r: str(r["row_key"]))

    if int(args.max_rows) > 0 and len(row_info) > int(args.max_rows):
        row_info = row_info[: int(args.max_rows)]

    plt = _ensure_plot_backend()
    import matplotlib.dates as mdates
    from matplotlib.patches import Patch

    nrows = len(row_info)
    fig_w = 14.0
    fig_h = max(4.2, 0.42 * nrows + 1.8)
    fig, ax = plt.subplots(1, 1, figsize=(fig_w, fig_h))

    all_sources = sorted({str(r["source"]) for r in row_info})
    cmap = plt.get_cmap("tab20", max(1, len(all_sources)))
    src_color = {s: cmap(i) for i, s in enumerate(all_sources)}

    segment_rows_csv: List[Dict[str, Any]] = []
    req_xmin_num = mdates.date2num(start_dt) if start_dt is not None else None
    req_xmax_num = mdates.date2num(end_dt) if end_dt is not None else None
    for iy, row in enumerate(row_info):
        key = str(row["row_key"])
        segs = row_segments.get(key, [])
        src = str(row["source"])
        fc = src_color.get(src, "#1f77b4")
        for iseg, (s, e, npt) in enumerate(segs, start=1):
            ss = float(s)
            ee = float(e)
            if start_num is not None:
                ss = max(ss, float(start_num))
            if end_num is not None:
                ee = min(ee, float(end_num))
            if ee < ss:
                continue
            x0 = mdates.date2num(num2date(ss))
            x1 = mdates.date2num(num2date(ee))
            width = float(x1 - x0)
            if width <= 0.0:
                width = max(min_width_days, 1.0 / 24.0)
            ax.broken_barh([(x0, width)], (iy - 0.36, 0.72), facecolors=fc, edgecolors="none", alpha=0.88)
            segment_rows_csv.append(
                {
                    "row_key": key,
                    "row_index": int(iy),
                    "source": src,
                    "data_type": str(row.get("data_type", "")),
                    "segment_index": int(iseg),
                    "start_utc": _fmt_time(ss),
                    "end_utc": _fmt_time(ee),
                    "duration_days": float(ee - ss),
                    "n_points": int(npt),
                }
            )

    labels = [str(r["row_key"]) for r in row_info]
    ax.set_yticks(np.arange(nrows))
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.8, nrows - 0.2)
    ax.invert_yaxis()
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(ax.xaxis.get_major_locator()))
    ax.grid(alpha=0.22, axis="x")
    ax.set_xlabel("UTC time")
    ax.set_ylabel(f"Rows ({args.row_mode})")
    subtitle = (
        f"sources={len(selected_sources)}, files={len(selected_files):,}, rows={nrows}, "
        f"use_qc={bool(args.use_qc)}, inside_only={bool(args.inside_only)}"
    )
    ax.set_title(f"JODC Current Availability Gantt\n{subtitle}")
    if req_xmin_num is not None and req_xmax_num is not None:
        ax.set_xlim(req_xmin_num, req_xmax_num)
    elif req_xmin_num is not None:
        ax.set_xlim(left=req_xmin_num)
    elif req_xmax_num is not None:
        ax.set_xlim(right=req_xmax_num)

    if len(all_sources) <= 20:
        handles = [Patch(facecolor=src_color[s], edgecolor="none", label=s) for s in all_sources]
        ax.legend(handles=handles, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0.0, fontsize=8)

    fig.tight_layout()
    out_png = outdir / str(args.outfile)
    fig.savefig(out_png, dpi=int(args.dpi))
    plt.close(fig)

    out_row_csv = outdir / "availability_gantt_rows.csv"
    with out_row_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "row_key",
                "source",
                "data_type",
                "n_files",
                "n_points",
                "n_segments",
                "time_min",
                "time_max",
                "coverage_days",
            ],
        )
        w.writeheader()
        for r in row_info:
            w.writerow(
                {
                    "row_key": r["row_key"],
                    "source": r["source"],
                    "data_type": r["data_type"],
                    "n_files": r["n_files"],
                    "n_points": r["n_points"],
                    "n_segments": r["n_segments"],
                    "time_min": r["time_min"],
                    "time_max": r["time_max"],
                    "coverage_days": f"{float(r['coverage_days']):.6f}",
                }
            )

    out_seg_csv = outdir / "availability_gantt_segments.csv"
    with out_seg_csv.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "row_key",
                "row_index",
                "source",
                "data_type",
                "segment_index",
                "start_utc",
                "end_utc",
                "duration_days",
                "n_points",
            ],
        )
        w.writeheader()
        for r in segment_rows_csv:
            w.writerow(r)

    out_json = outdir / "availability_gantt_summary.json"
    summary = {
        "config": {
            "nc_dir": str(nc_dir),
            "trusted_csv": str(trusted_csv),
            "classification_csv": str(classification_csv),
            "source_mode": source_mode,
            "selected_sources": selected_sources,
            "selected_data_types": selected_data_types if len(selected_data_types) > 0 else None,
            "start": _fmt_time(float(date2num([start_dt])[0])) if start_dt is not None else None,
            "end": _fmt_time(float(date2num([end_dt])[0])) if end_dt is not None else None,
            "row_mode": str(args.row_mode),
            "max_rows": int(args.max_rows),
            "min_row_points": int(args.min_row_points),
            "sort_by": str(args.sort_by),
            "merge_gap_hours": float(args.merge_gap_hours),
            "min_segment_hours": float(args.min_segment_hours),
            "use_qc": bool(args.use_qc),
            "qc_keep_codes": [int(x) for x in qc_keep_codes],
            "qc_mode": str(qc_mode),
            "inside_only": bool(args.inside_only),
            "inside_mode": str(args.inside_mode),
            "inside_context": inside_ctx,
        },
        "counts": {
            **dict(stats),
            "rows_plotted": int(nrows),
            "segments_plotted": int(len(segment_rows_csv)),
        },
        "outputs": {
            "plot_png": str(out_png),
            "rows_csv": str(out_row_csv),
            "segments_csv": str(out_seg_csv),
        },
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)

    print(f"[OK] wrote: {out_png}", flush=True)
    print(f"[OK] wrote: {out_row_csv}", flush=True)
    print(f"[OK] wrote: {out_seg_csv}", flush=True)
    print(f"[OK] wrote: {out_json}", flush=True)
    print(
        "[INFO] done:"
        f" files_seen={stats['file_seen']:,},"
        f" valid_files={stats['file_with_valid_time']:,},"
        f" valid_points={stats['valid_points_total']:,},"
        f" rows={nrows:,},"
        f" segments={len(segment_rows_csv):,}",
        flush=True,
    )


if __name__ == "__main__":
    main()
