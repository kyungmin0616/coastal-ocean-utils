#!/usr/bin/env python3
"""
Parse JODC current observations from TXT (HC/DC/DE), NC files, or both,
and export a unified NPZ bundle for model validation workflows.

Supported sources
- TXT: fixed-format JODC records (HC/DC/DE)
- NC: per-file profile/time datasets under a directory

Notes for NC input
- Many files have placeholder `time` values (-1/0).
- This script derives timestamps from `obs_date` + `obs_time`.
"""
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pylib import ReadNC, date2num, num2date, savez, zdata


KNOTS_TO_MS = 0.514444
FILL_CUTOFF = 9.0e4

HC_RE = re.compile(
    r"^HC"
    r"(?P<lat_deg>\d{2})(?P<lat_min>\d{2})(?P<lat_tenth>\d{2})(?P<lat_h>[NS])"
    r"(?P<lon_deg>\d{3})(?P<lon_min>\d{2})(?P<lon_tenth>\d{2})(?P<lon_h>[EW])"
    r"(?P<dt>\d{12})"
    r"(?P<tail>.*)$"
)


USER_CONFIG: Dict[str, Any] = {
    "enable": True,
    # Input source: txt | nc | both
    "source": "both",
    # TXT input
    "txt_input": "/Users/kpark/Documents/Projects/Active/AIMEC_TohokuCoast/01_Data/JODC/Current/140E-145E35N-40N.txt",
    "max_txt_profiles": None,
    "txt_verbose_every": 100000,
    # NC input
    "nc_dir": "/Users/kpark/Documents/Projects/Active/AIMEC_TohokuCoast/01_Data/JODC/Current/vector-140E-145E35N-40N_NCFiles",
    "nc_glob": "*.nc",
    "nc_recursive": False,
    "max_nc_files": None,
    # If True and QC arrays exist, keep only entries with u_QC>0 and v_QC>0.
    "nc_use_qc": True,
    # Optional additional filter: drop exact zero vectors for NC observations.
    "nc_drop_zero_vector": False,
    # Common filters
    "start": None,  # "YYYY-MM-DD" or "YYYY-MM-DD HH:MM[:SS]"
    "end": None,
    "min_depth": None,
    "max_depth": None,
    # Output
    "outdir": ".",
    "out": "jodc_current_all.npz",
    # Plot
    "plot": True,
    "plot_outdir": "./plots_jodc_current",
    "plot_sample_max": 80000,
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
    try:
        v = nc.getncattr(name)
        return str(v).strip()
    except Exception:
        return default


def _new_sink() -> Dict[str, List[Any]]:
    return {
        "time": [],
        "lon": [],
        "lat": [],
        "depth": [],
        "station": [],
        "profile": [],
        "ship": [],
        "code": [],
        "ndig": [],
        "exp": [],
        "acc": [],
        "pfl": [],
        "qc_u": [],
        "qc_v": [],
        "spd_kn": [],
        "spd": [],
        "dir": [],
        "u": [],
        "v": [],
        "source": [],
        "record_file": [],
    }


def _extend_sink(sink: Dict[str, List[Any]], payload: Dict[str, np.ndarray]) -> None:
    for key, arr in payload.items():
        sink[key].extend(np.asarray(arr).tolist())


def _parse_hc_line(line: str) -> Optional[Dict[str, Any]]:
    m = HC_RE.match(line)
    if m is None:
        return None

    lat = (
        float(m.group("lat_deg"))
        + float(m.group("lat_min")) / 60.0
        + float(m.group("lat_tenth")) / 6000.0
    )
    if m.group("lat_h") == "S":
        lat = -lat

    lon = (
        float(m.group("lon_deg"))
        + float(m.group("lon_min")) / 60.0
        + float(m.group("lon_tenth")) / 6000.0
    )
    if m.group("lon_h") == "W":
        lon = -lon

    dt_token = m.group("dt")
    dt = None
    try:
        dt = datetime.strptime(dt_token, "%Y%m%d%H%M")
    except ValueError:
        dt = None

    ship = ""
    if len(line) >= 40:
        ship = line[36:40].strip()

    return {
        "lat": lat,
        "lon": lon,
        "dt": dt,
        "dt_token": dt_token,
        "ship": ship,
    }


def _parse_data_line(line: str) -> Optional[Dict[str, Any]]:
    if len(line) < 2 or line[:2] not in ("DC", "DE"):
        return None

    code_raw = line[4:11] if len(line) >= 11 else ""
    m_pos = line.find(" m ")
    if m_pos < 0:
        return None

    payload = line[m_pos + 3 :]
    if len(payload) < 2 or not payload[:2].isdigit():
        return None

    ndig = int(payload[0])
    exp = int(payload[1])
    glen = 12 + ndig
    body = payload[2:]

    nfull = len(body) // glen
    rem = len(body) % glen
    segments: List[Tuple[float, float, float, str, str]] = []
    malformed = 0
    for i in range(nfull):
        seg = body[i * glen : (i + 1) * glen]
        s_depth = seg[0:6]
        s_dir = seg[6:10]
        s_spd = seg[10 : 10 + ndig]
        if not (s_depth.isdigit() and s_dir.isdigit() and s_spd.isdigit()):
            malformed += 1
            continue
        depth = int(s_depth) / 10.0
        direction = int(s_dir) / 10.0
        speed_kn = int(s_spd) / (10**exp)
        acc = seg[10 + ndig]
        pfl = seg[11 + ndig]
        segments.append((depth, direction, speed_kn, acc, pfl))

    return {
        "code_raw": code_raw,
        "ndig": ndig,
        "exp": exp,
        "segments": segments,
        "remainder": rem,
        "malformed": malformed,
    }


def _collect_txt(
    txt_path: Path,
    sink: Dict[str, List[Any]],
    c: Counter,
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
    min_depth: Optional[float],
    max_depth: Optional[float],
    profile_start: int,
    max_txt_profiles: Optional[int],
    verbose_every: int,
) -> int:
    profile_id = profile_start - 1
    txt_profiles_seen = 0
    context: Optional[Dict[str, Any]] = None
    active = False

    with txt_path.open("r", encoding="utf-8", errors="ignore") as f:
        for iline, raw in enumerate(f, start=1):
            line = raw.rstrip("\n")
            if not line:
                c["txt_line_blank"] += 1
                continue

            pref = line[:2]
            c[f"txt_line_{pref}"] += 1

            if pref == "HC":
                txt_profiles_seen += 1
                if max_txt_profiles is not None and txt_profiles_seen > int(max_txt_profiles):
                    break

                profile_id += 1
                c["txt_profile_total"] += 1

                hc = _parse_hc_line(line)
                if hc is None:
                    c["txt_profile_bad_hc"] += 1
                    context = None
                    active = False
                    continue

                dt = hc["dt"]
                if dt is None:
                    c["txt_profile_bad_time"] += 1
                    context = None
                    active = False
                    continue

                if start_dt is not None and dt < start_dt:
                    c["txt_profile_before_start"] += 1
                    context = None
                    active = False
                    continue
                if end_dt is not None and dt > end_dt:
                    c["txt_profile_after_end"] += 1
                    context = None
                    active = False
                    continue

                tnum = float(date2num([dt])[0])
                context = {
                    "profile": profile_id,
                    "station": f"TXT_P{profile_id:07d}",
                    "ship": hc["ship"] if hc["ship"] else "UNK",
                    "time": tnum,
                    "lon": float(hc["lon"]),
                    "lat": float(hc["lat"]),
                }
                active = True
                c["txt_profile_active"] += 1
                continue

            if pref not in ("DC", "DE"):
                continue

            if not active or context is None:
                c["txt_data_orphan"] += 1
                continue

            rec = _parse_data_line(line)
            if rec is None:
                c["txt_data_bad_record"] += 1
                continue

            if rec["remainder"] != 0:
                c["txt_data_remainder"] += 1
            if rec["malformed"] > 0:
                c["txt_data_malformed_segment"] += int(rec["malformed"])

            code_raw = rec["code_raw"]
            ndig = int(rec["ndig"])
            exp = int(rec["exp"])
            segments = rec["segments"]
            c["txt_segment_total_raw"] += len(segments)
            c[f"txt_code_{code_raw}"] += len(segments)
            c[f"txt_nexp_{ndig}_{exp}"] += len(segments)

            out = {
                "time": [],
                "lon": [],
                "lat": [],
                "depth": [],
                "station": [],
                "profile": [],
                "ship": [],
                "code": [],
                "ndig": [],
                "exp": [],
                "acc": [],
                "pfl": [],
                "qc_u": [],
                "qc_v": [],
                "spd_kn": [],
                "spd": [],
                "dir": [],
                "u": [],
                "v": [],
                "source": [],
                "record_file": [],
            }

            for depth, dire, spd_kn, acc, pfl in segments:
                if min_depth is not None and depth < float(min_depth):
                    c["txt_segment_skip_min_depth"] += 1
                    continue
                if max_depth is not None and depth > float(max_depth):
                    c["txt_segment_skip_max_depth"] += 1
                    continue

                if not (0.0 <= dire <= 360.0):
                    dire = np.nan
                    c["txt_segment_invalid_dir"] += 1

                spd_ms = spd_kn * KNOTS_TO_MS
                if np.isfinite(dire):
                    rad = np.deg2rad(dire)
                    u = spd_ms * np.sin(rad)
                    v = spd_ms * np.cos(rad)
                else:
                    u = np.nan
                    v = np.nan

                out["time"].append(context["time"])
                out["lon"].append(context["lon"])
                out["lat"].append(context["lat"])
                out["depth"].append(depth)
                out["station"].append(context["station"])
                out["profile"].append(int(context["profile"]))
                out["ship"].append(context["ship"])
                out["code"].append(code_raw)
                out["ndig"].append(ndig)
                out["exp"].append(exp)
                out["acc"].append(acc)
                out["pfl"].append(pfl)
                out["qc_u"].append(-1)
                out["qc_v"].append(-1)
                out["spd_kn"].append(spd_kn)
                out["spd"].append(spd_ms)
                out["dir"].append(dire)
                out["u"].append(u)
                out["v"].append(v)
                out["source"].append("txt")
                out["record_file"].append(txt_path.name)
                c["txt_segment_kept"] += 1

            if len(out["time"]) > 0:
                _extend_sink(sink, {k: np.asarray(v) for k, v in out.items()})

            if pref == "DE":
                c["txt_profile_with_de"] += 1

            if verbose_every > 0 and iline % verbose_every == 0:
                print(
                    f"[INFO] TXT lines={iline:,} profiles={c['txt_profile_total']:,} kept_obs={c['txt_segment_kept']:,}",
                    flush=True,
                )

    return profile_id + 1


def _get_nc_var(nc: Any, name: str) -> Optional[np.ndarray]:
    try:
        return np.asarray(nc.variables[name][:])
    except Exception:
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

        t = 0
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
        nums = np.asarray(date2num(dts), dtype=float)
        out[np.asarray(idx, dtype=int)] = nums
    return out


def _collect_nc(
    nc_dir: Path,
    nc_glob: str,
    nc_recursive: bool,
    sink: Dict[str, List[Any]],
    c: Counter,
    start_dt: Optional[datetime],
    end_dt: Optional[datetime],
    min_depth: Optional[float],
    max_depth: Optional[float],
    profile_start: int,
    max_nc_files: Optional[int],
    nc_use_qc: bool,
    nc_drop_zero_vector: bool,
) -> int:
    files = sorted(nc_dir.rglob(nc_glob) if nc_recursive else nc_dir.glob(nc_glob))
    if max_nc_files is not None:
        files = files[: int(max_nc_files)]

    c["nc_file_selected"] += len(files)
    next_profile = profile_start

    # Convert bounds once for faster mask operations
    start_num = float(date2num([start_dt])[0]) if start_dt is not None else None
    end_num = float(date2num([end_dt])[0]) if end_dt is not None else None

    for i, fp in enumerate(files, start=1):
        c["nc_file_total"] += 1
        try:
            nc = ReadNC(str(fp))
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

        if depth is None or lon is None or lat is None or u is None or v is None:
            c["nc_file_missing_core_var"] += 1
            continue

        depth = np.asarray(depth, dtype=float).reshape(-1)
        lon = np.asarray(lon, dtype=float).reshape(-1)
        lat = np.asarray(lat, dtype=float).reshape(-1)
        u = np.asarray(u, dtype=float)
        v = np.asarray(v, dtype=float)

        if u.ndim == 1:
            u = u[:, None]
        if v.ndim == 1:
            v = v[:, None]

        # Align orientation: expected (time, depth)
        if u.ndim != 2 or v.ndim != 2:
            c["nc_file_bad_uv_shape"] += 1
            continue

        if u.shape != v.shape:
            c["nc_file_uv_shape_mismatch"] += 1
            continue

        tlen, dlen = u.shape
        if depth.size != dlen:
            if depth.size == tlen and dlen != tlen:
                # unlikely but safe transpose attempt
                u = u.T
                v = v.T
                tlen, dlen = u.shape
            if depth.size != dlen:
                c["nc_file_depth_shape_mismatch"] += 1
                continue

        if lon.size == 1:
            lon = np.repeat(lon, tlen)
        if lat.size == 1:
            lat = np.repeat(lat, tlen)
        if lon.size != tlen or lat.size != tlen:
            c["nc_file_lonlat_shape_mismatch"] += 1
            continue

        if obs_date is None:
            c["nc_file_missing_obs_date"] += 1
            continue
        if obs_time is None:
            obs_time = np.zeros_like(obs_date)
            c["nc_file_missing_obs_time"] += 1

        obs_date = np.asarray(obs_date).reshape(-1)
        obs_time = np.asarray(obs_time).reshape(-1)
        if obs_date.size == 1 and tlen > 1:
            obs_date = np.repeat(obs_date, tlen)
        if obs_time.size == 1 and tlen > 1:
            obs_time = np.repeat(obs_time, tlen)
        if obs_date.size != tlen or obs_time.size != tlen:
            c["nc_file_obs_datetime_shape_mismatch"] += 1
            continue

        tnum = _parse_obs_datetime(obs_date, obs_time)

        # QC arrays (optional)
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
                c["nc_file_qc_shape_mismatch"] += 1

        ship = _safe_nc_attr(nc, "SHIP_CODE", "")
        data_type = _safe_nc_attr(nc, "DATA_TYPE", "")
        station = f"NC_{fp.stem}"

        profile_ids = np.arange(next_profile, next_profile + tlen, dtype=int)
        next_profile += tlen

        # Common time/space validity mask by time-index
        tmask = np.isfinite(tnum)
        tmask &= np.isfinite(lon) & np.isfinite(lat)
        tmask &= (np.abs(lon) <= 180.0) & (np.abs(lat) <= 90.0)
        if start_num is not None:
            tmask &= tnum >= float(start_num)
        if end_num is not None:
            tmask &= tnum <= float(end_num)

        if not np.any(tmask):
            c["nc_file_no_valid_time_space"] += 1
            continue

        c["nc_file_kept"] += 1

        for j in range(dlen):
            dval = float(depth[j])
            if min_depth is not None and dval < float(min_depth):
                c["nc_depth_skip_min"] += 1
                continue
            if max_depth is not None and dval > float(max_depth):
                c["nc_depth_skip_max"] += 1
                continue

            ucol = u[:, j].astype(float)
            vcol = v[:, j].astype(float)
            mask = tmask.copy()
            mask &= np.isfinite(ucol) & np.isfinite(vcol)
            mask &= (np.abs(ucol) < FILL_CUTOFF) & (np.abs(vcol) < FILL_CUTOFF)

            qu_col = None
            qv_col = None
            if qu is not None and qv is not None:
                qu_col = qu[:, j]
                qv_col = qv[:, j]
                if nc_use_qc:
                    mask &= (qu_col > 0) & (qv_col > 0)

            if nc_drop_zero_vector:
                mask &= ~((ucol == 0.0) & (vcol == 0.0))

            if not np.any(mask):
                continue

            u_keep = ucol[mask]
            v_keep = vcol[mask]
            spd_kn = np.sqrt(u_keep**2 + v_keep**2)
            spd_ms = spd_kn * KNOTS_TO_MS
            dire = np.rad2deg(np.arctan2(u_keep, v_keep)) % 360.0

            n = int(np.count_nonzero(mask))
            c["nc_segment_kept"] += n
            c[f"nc_code_{data_type if data_type else 'UNK'}"] += n

            out = {
                "time": tnum[mask],
                "lon": lon[mask],
                "lat": lat[mask],
                "depth": np.full(n, dval, dtype=float),
                "station": np.full(n, station, dtype="U64"),
                "profile": profile_ids[mask],
                "ship": np.full(n, ship if ship else "UNK", dtype="U16"),
                "code": np.full(n, data_type if data_type else "NC", dtype="U8"),
                "ndig": np.full(n, -1, dtype=int),
                "exp": np.full(n, -1, dtype=int),
                "acc": np.full(n, "", dtype="U4"),
                "pfl": np.full(n, "", dtype="U4"),
                "qc_u": (qu_col[mask].astype(int) if qu_col is not None else np.full(n, -1, dtype=int)),
                "qc_v": (qv_col[mask].astype(int) if qv_col is not None else np.full(n, -1, dtype=int)),
                "spd_kn": spd_kn,
                "spd": spd_ms,
                "dir": dire,
                "u": u_keep * KNOTS_TO_MS,
                "v": v_keep * KNOTS_TO_MS,
                "source": np.full(n, "nc", dtype="U8"),
                "record_file": np.full(n, fp.name, dtype="U128"),
            }
            _extend_sink(sink, out)

        if i % 200 == 0:
            print(f"[INFO] NC files processed: {i}/{len(files)} kept_obs={c['nc_segment_kept']:,}", flush=True)

    return next_profile


def _sample_indices(n: int, max_n: int) -> np.ndarray:
    if n <= 0:
        return np.array([], dtype=int)
    if max_n <= 0 or n <= max_n:
        return np.arange(n, dtype=int)
    return np.linspace(0, n - 1, max_n, dtype=int)


def _plot_quicklooks(bundle: zdata, outdir: Path, sample_max: int) -> List[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ModuleNotFoundError as exc:
        raise RuntimeError("matplotlib is required for plotting. Install it or run with --no-plot.") from exc

    outdir.mkdir(parents=True, exist_ok=True)
    wrote: List[str] = []

    t = np.asarray(bundle.time, dtype=float)
    lon = np.asarray(bundle.lon, dtype=float)
    lat = np.asarray(bundle.lat, dtype=float)
    depth = np.asarray(bundle.depth, dtype=float)
    spd = np.asarray(bundle.spd, dtype=float)
    dire = np.asarray(bundle.dir, dtype=float)
    profile = np.asarray(bundle.profile, dtype=int)

    idx = _sample_indices(len(t), int(sample_max))

    # 1) map quicklook
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 6.0))
    sc = ax.scatter(lon[idx], lat[idx], c=spd[idx], s=2, cmap="viridis", alpha=0.7)
    cb = plt.colorbar(sc, ax=ax, shrink=0.92)
    cb.set_label("Speed (m/s)")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("JODC Current Observation Locations (sampled)")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fp = outdir / "jodc_current_map.png"
    fig.savefig(fp, dpi=220)
    plt.close(fig)
    wrote.append(str(fp))

    # 2) yearly profile count
    _, first_idx = np.unique(profile, return_index=True)
    t_profile = t[first_idx]
    years = np.array([num2date(float(x)).year for x in t_profile], dtype=int)
    y0 = int(np.nanmin(years))
    y1 = int(np.nanmax(years))
    bins = np.arange(y0, y1 + 2)
    hist, _ = np.histogram(years, bins=bins)
    fig, ax = plt.subplots(1, 1, figsize=(11.0, 4.2))
    ax.bar(bins[:-1], hist, width=0.9, color="#3b82f6")
    ax.set_xlim(y0 - 0.5, y1 + 0.5)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of profiles")
    ax.set_title("JODC Current Profile Coverage by Year")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fp = outdir / "jodc_current_coverage_yearly.png"
    fig.savefig(fp, dpi=220)
    plt.close(fig)
    wrote.append(str(fp))

    # 3) depth vs speed
    fig, ax = plt.subplots(1, 1, figsize=(7.5, 5.8))
    h = ax.hexbin(spd, depth, gridsize=90, bins="log", mincnt=1, cmap="plasma")
    cb = plt.colorbar(h, ax=ax, shrink=0.92)
    cb.set_label("log10(count)")
    ax.set_xlabel("Speed (m/s)")
    ax.set_ylabel("Depth (m)")
    ax.invert_yaxis()
    ax.set_title("Speed-Depth Distribution")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fp = outdir / "jodc_current_speed_depth.png"
    fig.savefig(fp, dpi=220)
    plt.close(fig)
    wrote.append(str(fp))

    # 4) direction histogram
    valid_dir = np.isfinite(dire)
    fig, ax = plt.subplots(1, 1, figsize=(8.0, 4.2))
    ax.hist(dire[valid_dir], bins=np.arange(0, 361, 10), color="#16a34a", alpha=0.9)
    ax.set_xlim(0, 360)
    ax.set_xlabel("Direction (deg)")
    ax.set_ylabel("Count")
    ax.set_title("Direction Histogram")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fp = outdir / "jodc_current_direction_hist.png"
    fig.savefig(fp, dpi=220)
    plt.close(fig)
    wrote.append(str(fp))

    return wrote


def _merge_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(USER_CONFIG) if USER_CONFIG.get("enable", False) else {}
    keys = (
        "source",
        "txt_input",
        "max_txt_profiles",
        "txt_verbose_every",
        "nc_dir",
        "nc_glob",
        "nc_recursive",
        "max_nc_files",
        "nc_use_qc",
        "nc_drop_zero_vector",
        "start",
        "end",
        "min_depth",
        "max_depth",
        "outdir",
        "out",
        "plot",
        "plot_outdir",
        "plot_sample_max",
    )
    for key in keys:
        val = getattr(args, key, None)
        if val is not None:
            cfg[key] = val

    cfg.setdefault("source", "both")
    cfg.setdefault("txt_input", None)
    cfg.setdefault("max_txt_profiles", None)
    cfg.setdefault("txt_verbose_every", 100000)
    cfg.setdefault("nc_dir", None)
    cfg.setdefault("nc_glob", "*.nc")
    cfg.setdefault("nc_recursive", False)
    cfg.setdefault("max_nc_files", None)
    cfg.setdefault("nc_use_qc", True)
    cfg.setdefault("nc_drop_zero_vector", False)
    cfg.setdefault("start", None)
    cfg.setdefault("end", None)
    cfg.setdefault("min_depth", None)
    cfg.setdefault("max_depth", None)
    cfg.setdefault("outdir", ".")
    cfg.setdefault("out", "jodc_current_all.npz")
    cfg.setdefault("plot", False)
    cfg.setdefault("plot_outdir", "./plots_jodc_current")
    cfg.setdefault("plot_sample_max", 80000)

    src = str(cfg["source"]).strip().lower()
    if src not in {"txt", "nc", "both"}:
        raise ValueError("source must be one of: txt, nc, both")
    cfg["source"] = src

    return cfg


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Process JODC current observations from TXT/NC into NPZ.")

    p.add_argument("--source", choices=["txt", "nc", "both"], default=None, help="Input source type")

    p.add_argument("--txt-input", dest="txt_input", default=None, help="Path to JODC TXT file")
    p.add_argument("--max-txt-profiles", dest="max_txt_profiles", type=int, default=None, help="Max TXT HC profiles")
    p.add_argument("--txt-verbose-every", dest="txt_verbose_every", type=int, default=None, help="TXT progress interval")

    p.add_argument("--nc-dir", dest="nc_dir", default=None, help="Directory containing JODC NC files")
    p.add_argument("--nc-glob", dest="nc_glob", default=None, help="Glob pattern for NC files (default: *.nc)")
    p.add_argument("--nc-recursive", dest="nc_recursive", action="store_true", help="Recursively search NC files")
    p.add_argument("--no-nc-recursive", dest="nc_recursive", action="store_false", help="Disable recursive search")
    p.set_defaults(nc_recursive=None)
    p.add_argument("--max-nc-files", dest="max_nc_files", type=int, default=None, help="Limit NC files count")

    p.add_argument("--nc-use-qc", dest="nc_use_qc", action="store_true", help="Use QC>0 mask when QC vars exist")
    p.add_argument("--no-nc-use-qc", dest="nc_use_qc", action="store_false", help="Ignore NC QC arrays")
    p.set_defaults(nc_use_qc=None)

    p.add_argument(
        "--nc-drop-zero-vector",
        dest="nc_drop_zero_vector",
        action="store_true",
        help="Drop NC entries where u==0 and v==0",
    )
    p.add_argument(
        "--no-nc-drop-zero-vector",
        dest="nc_drop_zero_vector",
        action="store_false",
        help="Keep NC zero-vector entries",
    )
    p.set_defaults(nc_drop_zero_vector=None)

    p.add_argument("--start", default=None, help="Start time filter (YYYY-MM-DD[ HH:MM[:SS]])")
    p.add_argument("--end", default=None, help="End time filter (YYYY-MM-DD[ HH:MM[:SS]])")
    p.add_argument("--min-depth", dest="min_depth", type=float, default=None, help="Minimum depth [m]")
    p.add_argument("--max-depth", dest="max_depth", type=float, default=None, help="Maximum depth [m]")

    p.add_argument("--outdir", default=None, help="Output directory")
    p.add_argument("--out", default=None, help="Output NPZ filename")

    p.add_argument("--plot", dest="plot", action="store_true", help="Write quicklook plots")
    p.add_argument("--no-plot", dest="plot", action="store_false", help="Do not write plots")
    p.set_defaults(plot=None)
    p.add_argument("--plot-outdir", dest="plot_outdir", default=None, help="Plot output directory")
    p.add_argument("--plot-sample-max", dest="plot_sample_max", type=int, default=None, help="Max sampled points for map")

    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cfg = _merge_config(args)

    start_dt = _parse_bound(cfg.get("start"), is_end=False)
    end_dt = _parse_bound(cfg.get("end"), is_end=True)
    if start_dt is not None and end_dt is not None and end_dt < start_dt:
        raise ValueError("end must be >= start")

    outdir = _resolve_path(str(cfg["outdir"]))
    outdir.mkdir(parents=True, exist_ok=True)
    out_npz = outdir / str(cfg["out"]) if cfg.get("out") else outdir / "jodc_current_all.npz"

    source = str(cfg["source"]).lower()
    sink = _new_sink()
    c = Counter()

    next_profile = 1

    if source in {"txt", "both"}:
        txt_input = cfg.get("txt_input")
        if txt_input is None:
            raise ValueError("txt_input is required for source=txt/both")
        txt_path = _resolve_path(str(txt_input))
        if not txt_path.is_file():
            raise FileNotFoundError(f"TXT input not found: {txt_path}")

        next_profile = _collect_txt(
            txt_path=txt_path,
            sink=sink,
            c=c,
            start_dt=start_dt,
            end_dt=end_dt,
            min_depth=cfg.get("min_depth"),
            max_depth=cfg.get("max_depth"),
            profile_start=next_profile,
            max_txt_profiles=cfg.get("max_txt_profiles"),
            verbose_every=int(cfg.get("txt_verbose_every") or 0),
        )

    if source in {"nc", "both"}:
        nc_dir_v = cfg.get("nc_dir")
        if nc_dir_v is None:
            raise ValueError("nc_dir is required for source=nc/both")
        nc_dir = _resolve_path(str(nc_dir_v))
        if not nc_dir.is_dir():
            raise FileNotFoundError(f"NC directory not found: {nc_dir}")

        next_profile = _collect_nc(
            nc_dir=nc_dir,
            nc_glob=str(cfg.get("nc_glob", "*.nc")),
            nc_recursive=bool(cfg.get("nc_recursive", False)),
            sink=sink,
            c=c,
            start_dt=start_dt,
            end_dt=end_dt,
            min_depth=cfg.get("min_depth"),
            max_depth=cfg.get("max_depth"),
            profile_start=next_profile,
            max_nc_files=cfg.get("max_nc_files"),
            nc_use_qc=bool(cfg.get("nc_use_qc", True)),
            nc_drop_zero_vector=bool(cfg.get("nc_drop_zero_vector", False)),
        )

    if len(sink["time"]) == 0:
        raise RuntimeError("No observations parsed. Check source, inputs, or filters.")

    S = zdata()
    S.time = np.asarray(sink["time"], dtype=float)
    S.lon = np.asarray(sink["lon"], dtype=float)
    S.lat = np.asarray(sink["lat"], dtype=float)
    S.depth = np.asarray(sink["depth"], dtype=float)
    S.station = np.asarray(sink["station"], dtype="U64")
    S.profile = np.asarray(sink["profile"], dtype=int)
    S.ship = np.asarray(sink["ship"], dtype="U16")
    S.code = np.asarray(sink["code"], dtype="U8")
    S.ndig = np.asarray(sink["ndig"], dtype=int)
    S.exp = np.asarray(sink["exp"], dtype=int)
    S.acc = np.asarray(sink["acc"], dtype="U4")
    S.pfl = np.asarray(sink["pfl"], dtype="U4")
    S.qc_u = np.asarray(sink["qc_u"], dtype=int)
    S.qc_v = np.asarray(sink["qc_v"], dtype=int)
    S.spd_kn = np.asarray(sink["spd_kn"], dtype=float)
    S.spd = np.asarray(sink["spd"], dtype=float)
    S.dir = np.asarray(sink["dir"], dtype=float)
    S.u = np.asarray(sink["u"], dtype=float)
    S.v = np.asarray(sink["v"], dtype=float)
    S.source = np.asarray(sink["source"], dtype="U8")
    S.record_file = np.asarray(sink["record_file"], dtype="U128")

    order = np.argsort(S.time)
    for k, v in list(S.__dict__.items()):
        S.__dict__[k] = np.asarray(v)[order]

    savez(str(out_npz), S)
    print(f"[OK] wrote NPZ: {out_npz}", flush=True)

    summary = {
        "config": {
            "source": source,
            "txt_input": cfg.get("txt_input"),
            "nc_dir": cfg.get("nc_dir"),
            "nc_glob": cfg.get("nc_glob"),
            "nc_recursive": bool(cfg.get("nc_recursive", False)),
            "nc_use_qc": bool(cfg.get("nc_use_qc", True)),
            "nc_drop_zero_vector": bool(cfg.get("nc_drop_zero_vector", False)),
            "start": str(start_dt) if start_dt else None,
            "end": str(end_dt) if end_dt else None,
            "min_depth": cfg.get("min_depth"),
            "max_depth": cfg.get("max_depth"),
        },
        "output_npz": str(out_npz if str(out_npz).endswith(".npz") else f"{out_npz}.npz"),
        "counts": dict(sorted(c.items())),
        "n_obs": int(len(S.time)),
        "n_profiles": int(len(np.unique(S.profile))),
        "n_stations": int(len(np.unique(S.station))),
        "time_min": num2date(float(np.nanmin(S.time))).strftime("%Y-%m-%d %H:%M:%S"),
        "time_max": num2date(float(np.nanmax(S.time))).strftime("%Y-%m-%d %H:%M:%S"),
        "lon_min": float(np.nanmin(S.lon)),
        "lon_max": float(np.nanmax(S.lon)),
        "lat_min": float(np.nanmin(S.lat)),
        "lat_max": float(np.nanmax(S.lat)),
        "depth_min": float(np.nanmin(S.depth)),
        "depth_max": float(np.nanmax(S.depth)),
        "speed_ms_min": float(np.nanmin(S.spd)),
        "speed_ms_max": float(np.nanmax(S.spd)),
        "speed_kn_min": float(np.nanmin(S.spd_kn)),
        "speed_kn_max": float(np.nanmax(S.spd_kn)),
        "source_counts": {
            "txt": int(np.sum(S.source == "txt")),
            "nc": int(np.sum(S.source == "nc")),
        },
    }

    if bool(cfg.get("plot", False)):
        plot_outdir = _resolve_path(str(cfg.get("plot_outdir", "./plots_jodc_current")))
        try:
            plot_files = _plot_quicklooks(S, plot_outdir, int(cfg.get("plot_sample_max", 80000)))
            summary["plot_outdir"] = str(plot_outdir)
            summary["plot_files"] = plot_files
            print(f"[OK] wrote plots: {len(plot_files)} files -> {plot_outdir}", flush=True)
        except RuntimeError as exc:
            summary["plot_warning"] = str(exc)
            print(f"[WARN] plotting skipped: {exc}", flush=True)

    summary_fp = outdir / "jodc_current_summary.json"
    with summary_fp.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=True)
    print(f"[OK] wrote summary: {summary_fp}", flush=True)

    print(
        "[INFO] stats:"
        f" obs={summary['n_obs']:,},"
        f" profiles={summary['n_profiles']:,},"
        f" stations={summary['n_stations']:,},"
        f" time=[{summary['time_min']} .. {summary['time_max']}],"
        f" speed_kn_max={summary['speed_kn_max']:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
