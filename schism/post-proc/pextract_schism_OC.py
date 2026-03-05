#!/usr/bin/env python3
"""
Extract SCHISM current (u, v) collocated at JODC trajectory observations.

Features:
1) Multi-run extraction (RUNS list).
2) Optional model-start UTC offset from run/param.nml.
3) Nearest-time, FEM horizontal, depth interpolation collocation.
4) Extraction-only outputs (paired CSV/NPZ + rejects + manifest).
"""

from __future__ import annotations

# =============================================================================
# Configuration
# =============================================================================
CONFIG = dict(
    # Multi-run mode
    RUNS=[
        {"NAME": "RUN01d", "RUN": "../run/RUN01d"},
    ],
    SNAME_TEMPLATE="./npz/{run_name}_jodc_ca_schism_pairs",

    # Observation input (exactly one is required)
    OBS_CSV=None,
    OBS_NPZ="../../dataset/JODC/npz/trusted_collocation_obs.npz",

    # Observation filters
    OBS_DATA_TYPES=("CA",),
    OBS_SOURCES=None,
    START=None,  # UTC: YYYY-MM-DD[ HH:MM[:SS]]
    END=None,  # UTC: YYYY-MM-DD[ HH:MM[:SS]]
    MIN_DEPTH=None,
    MAX_DEPTH=None,
    REQUIRE_OBS_QC_CODES=(0,),
    REQUIRE_INSIDE=True,

    # Matching controls
    TIME_MATCH="nearest",  # fixed in phase-1
    MAX_TIME_LAG_HOURS=6.0,
    SPACE_MATCH="fem",  # fixed in phase-1
    DEPTH_MATCH="interp",  # fixed in phase-1
    OUTSIDE_DOMAIN="drop",  # drop | nan

    # Stack controls
    STACKS=None,  # None -> all discovered stacks after screening
    STACK_CHECK_MODE="light",  # none | light | size | light+size
    STACK_CHECK_ALL_FILES=False,
    STACK_SIZE_RATIO_MIN=0.70,
    STACK_SIZE_MIN_BYTES=None,

    # Time offset controls
    APPLY_UTC_START=False,

    # Output controls
    WRITE_CSV=True,
    WRITE_NPZ=True,
    WRITE_REJECTS_CSV=True,

    VERBOSE=True,
    DRY_RUN=False,
    MANIFEST=None,  # optional JSON summary path
)

# =============================================================================
# Imports
# =============================================================================
import argparse
import copy
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from pylib import ReadNC, date2num, grd, loadz, num2date, savez, schout_info, zdata


# =============================================================================
# Shared Utilities
# =============================================================================
_COMMON_CANDIDATE_DIRS = []
_env_pylibs_src = os.environ.get("PYLIBS_SRC")
if _env_pylibs_src:
    _COMMON_CANDIDATE_DIRS.append(Path(_env_pylibs_src).expanduser())
try:
    _COMMON_CANDIDATE_DIRS.append(Path(__file__).resolve().parents[3] / "pylibs" / "src")
except Exception:
    pass
_COMMON_CANDIDATE_DIRS.append(Path.home() / "Documents" / "Codes" / "pylibs" / "src")

for _common_dir in _COMMON_CANDIDATE_DIRS:
    if _common_dir.is_dir():
        _common_dir_str = str(_common_dir)
        if _common_dir_str not in sys.path:
            sys.path.insert(0, _common_dir_str)

try:
    from postproc_common import (
        deep_update_dict,
        get_model_start_datenum as common_get_model_start_datenum,
        init_mpi_runtime,
        normalize_stack_list as common_normalize_stack_list,
        normalize_run_specs as common_normalize_run_specs,
        read_stack_times_abs as common_read_stack_times_abs,
        screen_stacks as common_screen_stacks,
    )
except Exception as exc:
    raise ImportError(
        "Shared helpers not found. Set PYLIBS_SRC to pylibs/src or install postproc_common."
    ) from exc

MPI, COMM, RANK, SIZE, USE_MPI = init_mpi_runtime(sys.argv)


# =============================================================================
# Core Helpers
# =============================================================================
def _log(msg: str, verbose: bool = True, rank0_only: bool = True) -> None:
    if not bool(verbose):
        return
    if rank0_only and int(RANK) != 0:
        return
    if int(SIZE) > 1:
        print(f"[rank {RANK}/{SIZE}] {msg}", flush=True)
    else:
        print(str(msg), flush=True)


def _resolve_path(path_like: str) -> Path:
    p = Path(os.path.expanduser(str(path_like)))
    if p.is_absolute():
        return p
    return (Path.cwd() / p).resolve()


def _resolve_output_stem(path_like: str) -> Path:
    p = _resolve_path(path_like)
    if p.suffix.lower() in {".csv", ".npz", ".json"}:
        p = p.with_suffix("")
    return p


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


def _fmt_time(num: float) -> str:
    if not np.isfinite(num):
        return ""
    return num2date(float(num)).strftime("%Y-%m-%d %H:%M:%S")


def _safe_float(v: Any, default: float = np.nan) -> float:
    try:
        if v is None:
            return float(default)
        s = str(v).strip()
        if s == "":
            return float(default)
        return float(s)
    except Exception:
        return float(default)


def _safe_int(v: Any, default: int = -1) -> int:
    try:
        if v is None:
            return int(default)
        s = str(v).strip()
        if s == "":
            return int(default)
        return int(float(s))
    except Exception:
        return int(default)


def _parse_int_set(values: Any, default: Sequence[int]) -> List[int]:
    if values is None:
        return [int(x) for x in default]
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


def _load_json_config(config_path: Optional[str]) -> Dict[str, Any]:
    if config_path is None:
        return {}
    with open(str(config_path), "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"--config must contain a JSON object: {config_path}")
    return obj


def _normalize_run_specs(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    return common_normalize_run_specs(
        cfg,
        run_keys=("RUN", "run", "run_dir"),
        name_keys=("NAME", "name", "RUN_NAME", "run_name"),
        output_keys=("SNAME", "sname", "out_prefix"),
        output_template_key="SNAME_TEMPLATE",
        default_output_template="./npz/{run_name}_jodc_ca_schism_pairs",
        include_keys=("STACKS",),
    )


def _as_stack_list(stacks: Any, dstacks: Any) -> np.ndarray:
    return np.asarray(common_normalize_stack_list(stacks, dstacks), dtype=int)


def _screen_stacks(
    outputs_dir: str,
    stacks: np.ndarray,
    outfmt: int,
    cfg: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    return common_screen_stacks(
        outputs_dir=outputs_dir,
        stacks=np.asarray(stacks, dtype=int),
        outfmt=int(outfmt),
        mode=str(cfg.get("STACK_CHECK_MODE", "light")),
        check_all_files=bool(cfg.get("STACK_CHECK_ALL_FILES", False)),
        ratio_min=float(cfg.get("STACK_SIZE_RATIO_MIN", 0.70)),
        abs_min_bytes=cfg.get("STACK_SIZE_MIN_BYTES"),
        readnc=lambda p: ReadNC(str(p), 1),
        logger=None,
        log_limit=20,
    )


def _time_to_days(tvar: Any) -> np.ndarray:
    arr = np.asarray(tvar[:], dtype=float).ravel()
    units = str(getattr(tvar, "units", "")).strip().lower()
    if "second" in units:
        return arr / 86400.0
    if "day" in units:
        return arr
    return arr / 86400.0


# =============================================================================
# Observation Readers / Filters
# =============================================================================
def _safe_getattr_any(obj: Any, names: Sequence[str]) -> Optional[np.ndarray]:
    for nm in names:
        if hasattr(obj, nm):
            try:
                return np.asarray(getattr(obj, nm))
            except Exception:
                continue
    return None


def _read_obs_npz(path: Path) -> Dict[str, np.ndarray]:
    S = loadz(str(path))

    time_arr = _safe_getattr_any(S, ["time", "obs_time_num"])
    lon_arr = _safe_getattr_any(S, ["lon", "obs_lon"])
    lat_arr = _safe_getattr_any(S, ["lat", "obs_lat"])
    dep_arr = _safe_getattr_any(S, ["depth", "obs_depth"])
    u_arr = _safe_getattr_any(S, ["u", "obs_u"])
    v_arr = _safe_getattr_any(S, ["v", "obs_v"])
    spd_arr = _safe_getattr_any(S, ["spd", "speed", "obs_speed"])
    dir_arr = _safe_getattr_any(S, ["dir", "direction", "obs_dir"])
    qcu_arr = _safe_getattr_any(S, ["qc_u", "obs_qc_u"])
    qcv_arr = _safe_getattr_any(S, ["qc_v", "obs_qc_v"])
    src_arr = _safe_getattr_any(S, ["source", "obs_source"])
    typ_arr = _safe_getattr_any(S, ["data_type", "obs_data_type"])
    tid_arr = _safe_getattr_any(S, ["track_id", "obs_track_id"])
    sid_arr = _safe_getattr_any(S, ["segment_id", "obs_segment_id"])
    tf_arr = _safe_getattr_any(S, ["track_file", "obs_track_file"])
    ins_arr = _safe_getattr_any(S, ["inside", "obs_inside"])

    mandatory = [time_arr, lon_arr, lat_arr, dep_arr, u_arr, v_arr]
    if any(x is None for x in mandatory):
        raise ValueError(
            "Observation NPZ missing required fields. Need at least time/lon/lat/depth/u/v "
            "(or obs_time_num/obs_lon/obs_lat/obs_depth/obs_u/obs_v)."
        )

    n = len(np.asarray(time_arr).reshape(-1))

    def _as_len(arr: Optional[np.ndarray], default: Any, dtype: Any) -> np.ndarray:
        if arr is None:
            return np.full(n, default, dtype=dtype)
        out = np.asarray(arr).reshape(-1)
        if len(out) == n:
            return out.astype(dtype, copy=False)
        if len(out) == 1 and n > 1:
            return np.full(n, out[0], dtype=dtype)
        raise ValueError(f"Inconsistent observation array length: expected {n}, got {len(out)}")

    obs = {
        "time_num": _as_len(time_arr, np.nan, float),
        "lon": _as_len(lon_arr, np.nan, float),
        "lat": _as_len(lat_arr, np.nan, float),
        "depth": _as_len(dep_arr, np.nan, float),
        "u": _as_len(u_arr, np.nan, float),
        "v": _as_len(v_arr, np.nan, float),
        "speed": _as_len(spd_arr, np.nan, float) if spd_arr is not None else np.full(n, np.nan, dtype=float),
        "dir": _as_len(dir_arr, np.nan, float) if dir_arr is not None else np.full(n, np.nan, dtype=float),
        "qc_u": _as_len(qcu_arr, -1, int),
        "qc_v": _as_len(qcv_arr, -1, int),
        "source": _as_len(src_arr, "", "U32"),
        "data_type": _as_len(typ_arr, "", "U16"),
        "track_id": _as_len(tid_arr, -1, int),
        "segment_id": _as_len(sid_arr, -1, int),
        "track_file": _as_len(tf_arr, "", "U128"),
        "inside": None if ins_arr is None else _as_len(ins_arr, 0, int),
    }

    if np.all(~np.isfinite(obs["speed"])):
        obs["speed"] = np.sqrt(obs["u"] ** 2 + obs["v"] ** 2)
    if np.all(~np.isfinite(obs["dir"])):
        obs["dir"] = np.rad2deg(np.arctan2(obs["u"], obs["v"])) % 360.0
    return obs


def _read_obs_csv(path: Path) -> Dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if len(rows) == 0:
        raise ValueError(f"Empty observation CSV: {path}")

    cols = set(rows[0].keys())

    def _pick(*cands: str) -> Optional[str]:
        for c in cands:
            if c in cols:
                return c
        return None

    c_time = _pick("time_num", "obs_time_num", "time")
    c_tutc = _pick("timestamp_utc", "obs_time_utc", "time_utc")
    c_lon = _pick("lon", "obs_lon")
    c_lat = _pick("lat", "obs_lat")
    c_dep = _pick("depth", "obs_depth")
    c_u = _pick("u", "obs_u")
    c_v = _pick("v", "obs_v")
    c_spd = _pick("speed", "spd", "obs_speed")
    c_dir = _pick("direction", "dir", "obs_dir")
    c_qcu = _pick("qc_u", "obs_qc_u")
    c_qcv = _pick("qc_v", "obs_qc_v")
    c_src = _pick("source", "obs_source")
    c_typ = _pick("data_type", "obs_data_type")
    c_tid = _pick("track_id", "obs_track_id")
    c_sid = _pick("segment_id", "obs_segment_id")
    c_tf = _pick("track_file", "obs_track_file")
    c_in = _pick("inside", "obs_inside")

    mandatory = [c_lon, c_lat, c_dep, c_u, c_v]
    if c_time is None and c_tutc is None:
        raise ValueError("Observation CSV requires time_num/obs_time_num/time or timestamp_utc column.")
    if any(c is None for c in mandatory):
        raise ValueError("Observation CSV requires lon/lat/depth/u/v columns.")

    n = len(rows)
    time_num = np.full(n, np.nan, dtype=float)
    if c_time is not None:
        time_num = np.asarray([_safe_float(r.get(c_time), np.nan) for r in rows], dtype=float)
    else:
        parsed = np.full(n, np.nan, dtype=float)
        for i, r in enumerate(rows):
            txt = str(r.get(c_tutc, "")).strip()
            if txt == "":
                continue
            dt = None
            for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
                try:
                    dt = datetime.strptime(txt, fmt)
                    break
                except ValueError:
                    continue
            if dt is not None:
                parsed[i] = float(date2num([dt])[0])
        time_num = parsed

    obs = {
        "time_num": time_num,
        "lon": np.asarray([_safe_float(r.get(c_lon), np.nan) for r in rows], dtype=float),
        "lat": np.asarray([_safe_float(r.get(c_lat), np.nan) for r in rows], dtype=float),
        "depth": np.asarray([_safe_float(r.get(c_dep), np.nan) for r in rows], dtype=float),
        "u": np.asarray([_safe_float(r.get(c_u), np.nan) for r in rows], dtype=float),
        "v": np.asarray([_safe_float(r.get(c_v), np.nan) for r in rows], dtype=float),
        "speed": np.asarray(
            [_safe_float(r.get(c_spd), np.nan) for r in rows], dtype=float
        ) if c_spd is not None else np.full(n, np.nan, dtype=float),
        "dir": np.asarray(
            [_safe_float(r.get(c_dir), np.nan) for r in rows], dtype=float
        ) if c_dir is not None else np.full(n, np.nan, dtype=float),
        "qc_u": np.asarray([_safe_int(r.get(c_qcu), -1) for r in rows], dtype=int),
        "qc_v": np.asarray([_safe_int(r.get(c_qcv), -1) for r in rows], dtype=int),
        "source": np.asarray([str(r.get(c_src, "")).strip() for r in rows], dtype="U32"),
        "data_type": np.asarray([str(r.get(c_typ, "")).strip().upper() for r in rows], dtype="U16"),
        "track_id": np.asarray([_safe_int(r.get(c_tid), -1) for r in rows], dtype=int),
        "segment_id": np.asarray([_safe_int(r.get(c_sid), -1) for r in rows], dtype=int),
        "track_file": np.asarray([str(r.get(c_tf, "")).strip() for r in rows], dtype="U128"),
        "inside": None if c_in is None else np.asarray([_safe_int(r.get(c_in), 0) for r in rows], dtype=int),
    }

    if np.all(~np.isfinite(obs["speed"])):
        obs["speed"] = np.sqrt(obs["u"] ** 2 + obs["v"] ** 2)
    if np.all(~np.isfinite(obs["dir"])):
        obs["dir"] = np.rad2deg(np.arctan2(obs["u"], obs["v"])) % 360.0
    return obs


def _filter_obs(
    obs: Dict[str, np.ndarray],
    obs_data_types: Sequence[str],
    obs_sources: Optional[Sequence[str]],
    start_num: Optional[float],
    end_num: Optional[float],
    min_depth: Optional[float],
    max_depth: Optional[float],
    qc_keep_codes: Sequence[int],
    require_inside: bool,
) -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    n = len(obs["time_num"])
    mask = np.ones(n, dtype=bool)
    mask &= np.isfinite(obs["time_num"])
    mask &= np.isfinite(obs["lon"]) & np.isfinite(obs["lat"])
    mask &= np.isfinite(obs["depth"])
    mask &= np.isfinite(obs["u"]) & np.isfinite(obs["v"])
    mask &= (np.abs(obs["lon"]) <= 180.0) & (np.abs(obs["lat"]) <= 90.0)

    if obs_data_types is not None and len(obs_data_types) > 0:
        types_keep = {str(x).strip().upper() for x in obs_data_types if str(x).strip() != ""}
        if len(types_keep) > 0:
            mask &= np.isin(np.char.upper(obs["data_type"].astype("U16")), list(types_keep))

    if obs_sources is not None and len(obs_sources) > 0:
        src_keep = {str(x).strip() for x in obs_sources if str(x).strip() != ""}
        if len(src_keep) > 0:
            mask &= np.isin(obs["source"], list(src_keep))

    if start_num is not None:
        mask &= obs["time_num"] >= float(start_num)
    if end_num is not None:
        mask &= obs["time_num"] <= float(end_num)
    if min_depth is not None:
        mask &= obs["depth"] >= float(min_depth)
    if max_depth is not None:
        mask &= obs["depth"] <= float(max_depth)

    keep = np.asarray(list(qc_keep_codes), dtype=int)
    mask &= np.isin(obs["qc_u"], keep) & np.isin(obs["qc_v"], keep)

    if require_inside and (obs.get("inside") is not None):
        mask &= np.asarray(obs["inside"], dtype=int) == 1

    idx = np.where(mask)[0]
    out: Dict[str, np.ndarray] = {}
    for k, arr in obs.items():
        if arr is None:
            out[k] = None
            continue
        out[k] = np.asarray(arr)[idx]
    out["obs_index"] = idx.astype(int)
    return out, idx


def _load_and_filter_observations(cfg: Dict[str, Any]) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    obs_csv = cfg.get("OBS_CSV")
    obs_npz = cfg.get("OBS_NPZ")
    if (obs_csv in (None, "")) and (obs_npz in (None, "")):
        raise ValueError("Configure OBS_CSV or OBS_NPZ.")
    if (obs_csv not in (None, "")) and (obs_npz not in (None, "")):
        raise ValueError("Provide only one observation input: OBS_CSV or OBS_NPZ.")

    start_dt = _parse_bound(cfg.get("START"), is_end=False)
    end_dt = _parse_bound(cfg.get("END"), is_end=True)
    if start_dt is not None and end_dt is not None and end_dt < start_dt:
        raise ValueError("END must be >= START.")
    start_num = float(date2num([start_dt])[0]) if start_dt is not None else None
    end_num = float(date2num([end_dt])[0]) if end_dt is not None else None

    qc_keep = _parse_int_set(cfg.get("REQUIRE_OBS_QC_CODES"), default=[0])
    obs_types = [str(x).strip().upper() for x in (cfg.get("OBS_DATA_TYPES") or []) if str(x).strip() != ""]
    obs_sources = [str(x).strip() for x in (cfg.get("OBS_SOURCES") or []) if str(x).strip() != ""]
    if len(obs_sources) == 0:
        obs_sources = None

    if obs_csv not in (None, ""):
        obs_path = _resolve_path(str(obs_csv))
        if not obs_path.is_file():
            raise FileNotFoundError(f"obs_csv not found: {obs_path}")
        obs_raw = _read_obs_csv(obs_path)
    else:
        obs_path = _resolve_path(str(obs_npz))
        if not obs_path.is_file():
            raise FileNotFoundError(f"obs_npz not found: {obs_path}")
        obs_raw = _read_obs_npz(obs_path)

    n_loaded = int(len(obs_raw["time_num"]))
    obs, idx = _filter_obs(
        obs=obs_raw,
        obs_data_types=obs_types,
        obs_sources=obs_sources,
        start_num=start_num,
        end_num=end_num,
        min_depth=cfg.get("MIN_DEPTH"),
        max_depth=cfg.get("MAX_DEPTH"),
        qc_keep_codes=qc_keep,
        require_inside=bool(cfg.get("REQUIRE_INSIDE", True)),
    )
    n_filtered = int(len(obs["time_num"]))
    if n_filtered == 0:
        raise RuntimeError("No observations remain after filters.")

    meta = {
        "obs_source_file": str(obs_path),
        "obs_loaded": n_loaded,
        "obs_filtered": n_filtered,
        "obs_types": obs_types,
        "obs_sources": obs_sources,
        "qc_keep": [int(x) for x in qc_keep],
        "start_num": start_num,
        "end_num": end_num,
        "selected_obs_index": idx.astype(int),
    }
    return obs, meta


# =============================================================================
# SCHISM Matching / Interpolation Helpers
# =============================================================================
def _build_model_time_table(
    outputs_dir: Path,
    stacks: np.ndarray,
    start_dnum: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    times: List[float] = []
    stks: List[int] = []
    tids: List[int] = []

    for stk in stacks.tolist():
        tfile = outputs_dir / f"out2d_{int(stk)}.nc"
        if not tfile.exists():
            continue
        tarr = common_read_stack_times_abs(
            str(tfile),
            start_datenum=start_dnum,
            readnc=lambda p: ReadNC(str(p), 1),
            time_to_days=_time_to_days,
        )
        tarr = np.asarray(tarr, dtype=float).ravel()
        if len(tarr) == 0:
            continue
        for it, tv in enumerate(tarr.tolist()):
            if not np.isfinite(tv):
                continue
            times.append(float(tv))
            stks.append(int(stk))
            tids.append(int(it))

    if len(times) == 0:
        return np.array([], dtype=float), np.array([], dtype=int), np.array([], dtype=int)

    mt = np.asarray(times, dtype=float)
    ms = np.asarray(stks, dtype=int)
    mi = np.asarray(tids, dtype=int)
    order = np.argsort(mt)
    return mt[order], ms[order], mi[order]


def _nearest_model_time(
    obs_time: np.ndarray,
    model_time: np.ndarray,
    model_stack: np.ndarray,
    model_tidx: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(obs_time)
    out_stack = np.full(n, -1, dtype=int)
    out_tidx = np.full(n, -1, dtype=int)
    out_mtime = np.full(n, np.nan, dtype=float)
    out_dt_hours = np.full(n, np.nan, dtype=float)
    if len(model_time) == 0:
        return out_stack, out_tidx, out_mtime, out_dt_hours

    pos = np.searchsorted(model_time, obs_time)
    left = np.clip(pos - 1, 0, len(model_time) - 1)
    right = np.clip(pos, 0, len(model_time) - 1)

    dl = np.abs(model_time[left] - obs_time)
    dr = np.abs(model_time[right] - obs_time)
    take_right = dr < dl
    idx = np.where(take_right, right, left)

    mt = model_time[idx]
    dt_hours = (mt - obs_time) * 24.0
    good = np.isfinite(dt_hours)

    out_stack[good] = model_stack[idx[good]]
    out_tidx[good] = model_tidx[idx[good]]
    out_mtime[good] = mt[good]
    out_dt_hours[good] = dt_hours[good]
    return out_stack, out_tidx, out_mtime, out_dt_hours


def _open_required_stack_files(outputs_dir: Path, stack: int) -> Tuple[Any, Any, Optional[Any], Optional[Any]]:
    fx = outputs_dir / f"horizontalVelX_{int(stack)}.nc"
    fy = outputs_dir / f"horizontalVelY_{int(stack)}.nc"
    if not fx.exists() or not fy.exists():
        raise FileNotFoundError(
            f"Missing model current files for stack {stack}: horizontalVelX/horizontalVelY required."
        )

    cx = ReadNC(str(fx), 1)
    cy = ReadNC(str(fy), 1)

    fz = outputs_dir / f"zCoordinates_{int(stack)}.nc"
    cz = ReadNC(str(fz), 1) if fz.exists() else None

    co = None
    if cz is None:
        fo = outputs_dir / f"out2d_{int(stack)}.nc"
        if not fo.exists():
            raise FileNotFoundError(
                f"Missing zCoordinates and out2d files for stack {stack}; cannot compute vertical coordinates."
            )
        co = ReadNC(str(fo), 1)

    return cx, cy, cz, co


def _read_var_slice_t_node_layer(v: Any, t_idx: int, node_ids: np.ndarray, np_nodes: int) -> np.ndarray:
    try:
        arr = np.asarray(v[t_idx, node_ids, :], dtype=float)
        if arr.ndim == 2 and arr.shape[0] == len(node_ids):
            return arr
    except Exception:
        pass

    try:
        arr = np.asarray(v[t_idx, :, node_ids], dtype=float)
        if arr.ndim == 2:
            if arr.shape[1] == len(node_ids):
                return arr.T
            if arr.shape[0] == len(node_ids):
                return arr
    except Exception:
        pass

    arr2 = np.asarray(v[t_idx], dtype=float)
    if arr2.ndim != 2:
        raise ValueError(f"Unexpected variable slice ndim={arr2.ndim}, need 2D.")
    if arr2.shape[0] == np_nodes:
        return arr2[node_ids, :]
    if arr2.shape[1] == np_nodes:
        return arr2[:, node_ids].T
    raise ValueError(f"Cannot infer node axis for slice shape={arr2.shape}, np={np_nodes}.")


def _read_var_slice_t_node(v: Any, t_idx: int, node_ids: np.ndarray, np_nodes: int) -> np.ndarray:
    try:
        arr = np.asarray(v[t_idx, node_ids], dtype=float).reshape(-1)
        if len(arr) == len(node_ids):
            return arr
    except Exception:
        pass

    arr2 = np.asarray(v[t_idx], dtype=float).reshape(-1)
    if len(arr2) == np_nodes:
        return arr2[node_ids]
    raise ValueError(f"Cannot infer node scalar slice shape={arr2.shape}, np={np_nodes}.")


def _interp_at_obs_depth(z_prof: np.ndarray, val_prof: np.ndarray, obs_depth: float) -> Tuple[float, bool]:
    target = -float(obs_depth)
    z = np.asarray(z_prof, dtype=float).reshape(-1)
    v = np.asarray(val_prof, dtype=float).reshape(-1)

    valid = np.isfinite(z) & np.isfinite(v)
    z = z[valid]
    v = v[valid]
    if len(z) < 2:
        return np.nan, False

    order = np.argsort(z)
    z = z[order]
    v = v[order]

    uz, idx = np.unique(z, return_index=True)
    z = uz
    v = v[idx]
    if len(z) < 2:
        return np.nan, False

    if target < z[0] or target > z[-1]:
        return np.nan, False
    return float(np.interp(target, z, v)), True


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fieldnames), extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def _process_one_run(spec: Dict[str, Any], cfg: Dict[str, Any], obs: Dict[str, np.ndarray], obs_meta: Dict[str, Any]) -> Dict[str, Any]:
    run = str(spec["RUN"])
    run_name = str(spec.get("NAME", os.path.basename(os.path.abspath(run))))
    sname = str(spec.get("SNAME"))
    verbose = bool(cfg.get("VERBOSE", True))

    run_dir = _resolve_path(run)
    outputs_dir = run_dir / "outputs"
    if not run_dir.is_dir() or not outputs_dir.is_dir():
        return {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "output_stem": str(_resolve_output_stem(sname)),
            "status": "skipped_missing_run_or_outputs",
        }

    out_stem = _resolve_output_stem(sname)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    paired_csv = Path(str(out_stem) + ".csv")
    paired_npz = Path(str(out_stem) + ".npz")
    reject_csv = Path(str(out_stem) + "_rejects.csv")

    modules, outfmt, dstacks, dvars, dvars_2d = schout_info(str(outputs_dir), 1)
    _ = modules
    _ = dvars
    _ = dvars_2d
    if int(outfmt) != 0:
        return {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "output_stem": str(out_stem),
            "status": f"skipped_outfmt_{int(outfmt)}",
        }

    stack_candidates = _as_stack_list(spec.get("STACKS"), dstacks)
    valid_stacks, skipped_stacks = _screen_stacks(str(outputs_dir), stack_candidates, int(outfmt), cfg)
    if len(valid_stacks) == 0:
        return {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "output_stem": str(out_stem),
            "status": "skipped_no_valid_stacks",
            "stacks_requested": int(len(stack_candidates)),
            "stacks_valid": 0,
            "stacks_skipped": int(len(skipped_stacks)),
        }

    start_dnum, start_info = common_get_model_start_datenum(
        str(run_dir),
        apply_utc_start=bool(cfg.get("APPLY_UTC_START", False)),
    )
    if start_dnum is None:
        return {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "output_stem": str(out_stem),
            "status": "skipped_missing_param_start",
            "reason": str(start_info),
        }

    model_time, model_stack, model_tidx = _build_model_time_table(
        outputs_dir=outputs_dir,
        stacks=np.asarray(valid_stacks, dtype=int),
        start_dnum=float(start_dnum),
    )
    if len(model_time) == 0:
        return {
            "run_name": run_name,
            "run_dir": str(run_dir),
            "output_stem": str(out_stem),
            "status": "skipped_no_model_times",
        }

    n_obs = int(len(obs["time_num"]))
    mod_stack = np.full(n_obs, -1, dtype=int)
    mod_tidx = np.full(n_obs, -1, dtype=int)
    mod_time = np.full(n_obs, np.nan, dtype=float)
    mod_dt_hours = np.full(n_obs, np.nan, dtype=float)
    reject_reason = np.asarray([""] * n_obs, dtype="U64")

    ms, mtidx, mt, mdt = _nearest_model_time(
        obs_time=obs["time_num"],
        model_time=model_time,
        model_stack=model_stack,
        model_tidx=model_tidx,
    )
    mod_stack[:] = ms
    mod_tidx[:] = mtidx
    mod_time[:] = mt
    mod_dt_hours[:] = mdt

    no_time = mod_stack < 0
    reject_reason[no_time] = "time_no_bracket_or_no_nearest"
    lag_bad = (mod_stack >= 0) & (
        ~np.isfinite(mod_dt_hours) | (np.abs(mod_dt_hours) > float(cfg.get("MAX_TIME_LAG_HOURS", 6.0)))
    )
    reject_reason[lag_bad] = "time_lag_exceeds_threshold"

    gd, vd = grd(str(run_dir), fmt=2)
    pts = np.c_[obs["lon"], obs["lat"]]
    pie, pip, pacor = gd.compute_acor(pts)
    pip = np.asarray(pip).T
    pacor = np.asarray(pacor).T
    pie = np.asarray(pie, dtype=int).reshape(-1)

    if pip.ndim != 2:
        raise RuntimeError(f"Unexpected pip shape: {pip.shape}")
    if pip.shape[0] != 3 and pip.shape[1] == 3:
        pip = pip.T
        pacor = pacor.T
    if pip.shape[0] != 3:
        raise RuntimeError(f"Unexpected pip shape after normalize: {pip.shape}")

    inside_domain = np.zeros(n_obs, dtype=int)
    inside_domain[(pie >= 0) & np.all(pip >= 0, axis=0)] = 1

    mod_u = np.full(n_obs, np.nan, dtype=float)
    mod_v = np.full(n_obs, np.nan, dtype=float)
    mod_speed = np.full(n_obs, np.nan, dtype=float)
    mod_dir = np.full(n_obs, np.nan, dtype=float)
    matched = np.zeros(n_obs, dtype=int)
    depth_interp_ok = np.zeros(n_obs, dtype=int)
    elem_id = pie.copy()

    outside = (inside_domain == 0) & (mod_stack >= 0)
    reject_reason[outside] = "outside_domain"

    need = np.where((mod_stack >= 0) & (inside_domain == 1))[0]
    groups: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for i in need.tolist():
        groups[(int(mod_stack[i]), int(mod_tidx[i]))].append(int(i))

    needed_stacks = sorted({k[0] for k in groups.keys()})
    for stk in needed_stacks:
        fx = outputs_dir / f"horizontalVelX_{int(stk)}.nc"
        fy = outputs_dir / f"horizontalVelY_{int(stk)}.nc"
        if not fx.exists() or not fy.exists():
            return {
                "run_name": run_name,
                "run_dir": str(run_dir),
                "output_stem": str(out_stem),
                "status": "failed_missing_model_var",
                "stack": int(stk),
                "reason": f"Missing {fx.name} and/or {fy.name}",
            }

    for stk in needed_stacks:
        cx, cy, cz, co = _open_required_stack_files(outputs_dir, stack=int(stk))
        try:
            xvar = cx.variables["horizontalVelX"]
            yvar = cy.variables["horizontalVelY"]
            zvar = None if cz is None else cz.variables.get("zCoordinates")
            elev_var = None if co is None else co.variables.get("elevation")
            if zvar is None and elev_var is None:
                raise RuntimeError(
                    f"Stack {stk}: missing zCoordinates and elevation (out2d) for depth interpolation."
                )

            keys = sorted([k for k in groups.keys() if k[0] == stk], key=lambda x: x[1])
            for _, tidx in keys:
                obs_ids = np.asarray(groups[(stk, tidx)], dtype=int)
                if len(obs_ids) == 0:
                    continue

                tri_nodes = pip[:, obs_ids].astype(int)
                unique_nodes = np.unique(tri_nodes.ravel())
                unique_nodes = unique_nodes[unique_nodes >= 0]
                if len(unique_nodes) == 0:
                    reject_reason[obs_ids] = "outside_domain"
                    continue

                u_nodes = _read_var_slice_t_node_layer(xvar, int(tidx), unique_nodes, np_nodes=int(gd.np))
                v_nodes = _read_var_slice_t_node_layer(yvar, int(tidx), unique_nodes, np_nodes=int(gd.np))
                if u_nodes.shape != v_nodes.shape:
                    raise RuntimeError(
                        f"Stack {stk}, t={tidx}: horizontalVelX/Y shape mismatch {u_nodes.shape} vs {v_nodes.shape}"
                    )

                if zvar is not None:
                    z_nodes = _read_var_slice_t_node_layer(zvar, int(tidx), unique_nodes, np_nodes=int(gd.np))
                else:
                    eta_nodes = _read_var_slice_t_node(elev_var, int(tidx), unique_nodes, np_nodes=int(gd.np))
                    dp_nodes = np.asarray(gd.dp, dtype=float)[unique_nodes]
                    if int(vd.ivcor) == 1:
                        sigma_nodes = np.asarray(vd.sigma, dtype=float)[unique_nodes, :]
                        kbp_nodes = np.asarray(vd.kbp, dtype=int)[unique_nodes]
                        z_nodes = vd.compute_zcor(
                            dp_nodes,
                            eta=eta_nodes,
                            fmt=1,
                            method=1,
                            sigma=sigma_nodes,
                            kbp=kbp_nodes,
                        )
                    else:
                        z_nodes = vd.compute_zcor(dp_nodes, eta=eta_nodes, fmt=1)
                    z_nodes = np.asarray(z_nodes, dtype=float)

                node_to_local = {int(nid): ii for ii, nid in enumerate(unique_nodes.tolist())}

                for oi in obs_ids.tolist():
                    if reject_reason[oi] != "":
                        continue
                    tri = pip[:, oi].astype(int)
                    w = np.asarray(pacor[:, oi], dtype=float)
                    if np.any(tri < 0) or not np.all(np.isfinite(w)):
                        reject_reason[oi] = "outside_domain"
                        continue

                    try:
                        li = np.asarray([node_to_local[int(n)] for n in tri], dtype=int)
                    except KeyError:
                        reject_reason[oi] = "outside_domain"
                        continue

                    u_prof = np.sum(u_nodes[li, :] * w[:, None], axis=0)
                    v_prof = np.sum(v_nodes[li, :] * w[:, None], axis=0)
                    z_prof = np.sum(z_nodes[li, :] * w[:, None], axis=0)

                    if not np.any(np.isfinite(z_prof)):
                        reject_reason[oi] = "bad_zcor_or_depth_interp"
                        continue

                    z_surf = z_prof[-1]
                    if not np.isfinite(z_surf):
                        reject_reason[oi] = "bad_zcor_or_depth_interp"
                        continue
                    z_rel = z_prof - z_surf

                    uu, ok_u = _interp_at_obs_depth(z_rel, u_prof, obs_depth=float(obs["depth"][oi]))
                    vv, ok_v = _interp_at_obs_depth(z_rel, v_prof, obs_depth=float(obs["depth"][oi]))
                    if not (ok_u and ok_v and np.isfinite(uu) and np.isfinite(vv)):
                        reject_reason[oi] = "bad_zcor_or_depth_interp"
                        continue

                    mod_u[oi] = float(uu)
                    mod_v[oi] = float(vv)
                    mod_speed[oi] = float(np.sqrt(uu**2 + vv**2))
                    mod_dir[oi] = float(np.rad2deg(np.arctan2(uu, vv)) % 360.0)
                    matched[oi] = 1
                    depth_interp_ok[oi] = 1
        finally:
            for cobj in (cx, cy, cz, co):
                if cobj is not None:
                    try:
                        cobj.close()
                    except Exception:
                        pass

    unmatched = matched == 0
    empty_reason = reject_reason == ""
    reject_reason[unmatched & empty_reason] = "nan_model_value"

    rows_all: List[Dict[str, Any]] = []
    for i in range(n_obs):
        rows_all.append(
            {
                "obs_time_num": float(obs["time_num"][i]),
                "obs_time_utc": _fmt_time(float(obs["time_num"][i])),
                "obs_lon": float(obs["lon"][i]),
                "obs_lat": float(obs["lat"][i]),
                "obs_depth": float(obs["depth"][i]),
                "obs_u": float(obs["u"][i]),
                "obs_v": float(obs["v"][i]),
                "obs_speed": float(obs["speed"][i]),
                "obs_dir": float(obs["dir"][i]),
                "obs_qc_u": int(obs["qc_u"][i]),
                "obs_qc_v": int(obs["qc_v"][i]),
                "obs_source": str(obs["source"][i]),
                "obs_data_type": str(obs["data_type"][i]),
                "obs_track_id": int(obs["track_id"][i]),
                "obs_segment_id": int(obs["segment_id"][i]),
                "obs_track_file": str(obs["track_file"][i]),
                "mod_time_num": float(mod_time[i]) if np.isfinite(mod_time[i]) else np.nan,
                "mod_time_utc": _fmt_time(float(mod_time[i])) if np.isfinite(mod_time[i]) else "",
                "mod_stack": int(mod_stack[i]),
                "mod_time_index": int(mod_tidx[i]),
                "mod_dt_hours": float(mod_dt_hours[i]) if np.isfinite(mod_dt_hours[i]) else np.nan,
                "mod_u": float(mod_u[i]) if np.isfinite(mod_u[i]) else np.nan,
                "mod_v": float(mod_v[i]) if np.isfinite(mod_v[i]) else np.nan,
                "mod_speed": float(mod_speed[i]) if np.isfinite(mod_speed[i]) else np.nan,
                "mod_dir": float(mod_dir[i]) if np.isfinite(mod_dir[i]) else np.nan,
                "matched": int(matched[i]),
                "reject_reason": str(reject_reason[i]),
                "elem_id": int(elem_id[i]),
                "inside_domain": int(inside_domain[i]),
                "depth_interp_ok": int(depth_interp_ok[i]),
            }
        )

    rows = rows_all
    if str(cfg.get("OUTSIDE_DOMAIN", "drop")).lower() == "drop":
        rows = [r for r in rows_all if str(r["reject_reason"]) != "outside_domain"]

    fieldnames = [
        "obs_time_num",
        "obs_time_utc",
        "obs_lon",
        "obs_lat",
        "obs_depth",
        "obs_u",
        "obs_v",
        "obs_speed",
        "obs_dir",
        "obs_qc_u",
        "obs_qc_v",
        "obs_source",
        "obs_data_type",
        "obs_track_id",
        "obs_segment_id",
        "obs_track_file",
        "mod_time_num",
        "mod_time_utc",
        "mod_stack",
        "mod_time_index",
        "mod_dt_hours",
        "mod_u",
        "mod_v",
        "mod_speed",
        "mod_dir",
        "matched",
        "reject_reason",
        "elem_id",
        "inside_domain",
        "depth_interp_ok",
    ]

    if bool(cfg.get("WRITE_CSV", True)):
        _write_csv(paired_csv, fieldnames, rows)
    if bool(cfg.get("WRITE_NPZ", True)):
        P = zdata()
        for fn in fieldnames:
            vals = [r[fn] for r in rows]
            if fn.endswith("_utc") or fn in ("obs_source", "obs_data_type", "obs_track_file", "reject_reason"):
                arr = np.asarray(vals, dtype="U128")
            elif fn in (
                "mod_stack",
                "mod_time_index",
                "matched",
                "obs_qc_u",
                "obs_qc_v",
                "obs_track_id",
                "obs_segment_id",
                "elem_id",
                "inside_domain",
                "depth_interp_ok",
            ):
                arr = np.asarray(vals, dtype=int)
            else:
                arr = np.asarray(vals, dtype=float)
            setattr(P, fn, arr)
        savez(str(paired_npz), P)

    reject_rows = [r for r in rows_all if int(r["matched"]) == 0]
    if bool(cfg.get("WRITE_REJECTS_CSV", True)):
        rej_fields = [
            "obs_time_num",
            "obs_time_utc",
            "obs_lon",
            "obs_lat",
            "obs_depth",
            "obs_source",
            "obs_data_type",
            "obs_track_id",
            "obs_segment_id",
            "obs_track_file",
            "mod_time_num",
            "mod_time_utc",
            "mod_stack",
            "mod_time_index",
            "mod_dt_hours",
            "inside_domain",
            "elem_id",
            "reject_reason",
        ]
        _write_csv(reject_csv, rej_fields, reject_rows)

    reject_counts = Counter([str(r["reject_reason"]) for r in reject_rows])
    matched_rows = [r for r in rows if int(r["matched"]) == 1]

    summary = {
        "run_name": run_name,
        "run_dir": str(run_dir),
        "output_stem": str(out_stem),
        "status": "written",
        "obs_source_file": str(obs_meta["obs_source_file"]),
        "obs_filtered": int(len(obs["time_num"])),
        "matched": int(np.count_nonzero(matched == 1)),
        "rejected": int(np.count_nonzero(matched == 0)),
        "paired_rows_written": int(len(rows)),
        "outside_domain_rows_dropped": int(len(rows_all) - len(rows)),
        "reject_reason_counts": dict(reject_counts),
        "inside_domain_count": int(np.count_nonzero(inside_domain == 1)),
        "outside_domain_count": int(np.count_nonzero(inside_domain == 0)),
        "model_start_info": str(start_info),
        "stacks_requested": [int(x) for x in np.asarray(stack_candidates, dtype=int).tolist()],
        "stacks_valid": [int(x) for x in np.asarray(valid_stacks, dtype=int).tolist()],
        "time_range": {
            "obs_min": _fmt_time(float(np.nanmin(obs["time_num"]))) if len(obs["time_num"]) > 0 else "",
            "obs_max": _fmt_time(float(np.nanmax(obs["time_num"]))) if len(obs["time_num"]) > 0 else "",
            "mod_min_matched": _fmt_time(float(np.nanmin([r["mod_time_num"] for r in matched_rows]))) if len(matched_rows) > 0 else "",
            "mod_max_matched": _fmt_time(float(np.nanmax([r["mod_time_num"] for r in matched_rows]))) if len(matched_rows) > 0 else "",
        },
        "outputs": {
            "paired_csv": str(paired_csv) if bool(cfg.get("WRITE_CSV", True)) else None,
            "paired_npz": str(paired_npz) if bool(cfg.get("WRITE_NPZ", True)) else None,
            "reject_csv": str(reject_csv) if bool(cfg.get("WRITE_REJECTS_CSV", True)) else None,
        },
    }

    _log(
        f"[OK] {run_name}: filtered={summary['obs_filtered']:,}, matched={summary['matched']:,}, "
        f"rejected={summary['rejected']:,}, rows={summary['paired_rows_written']:,}",
        verbose=verbose,
    )
    return summary


# =============================================================================
# CLI
# =============================================================================
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract SCHISM currents collocated at JODC trajectory observations.")
    p.add_argument("--config", help="Optional JSON config overrides.")
    p.add_argument("--manifest", help="Optional output JSON summary path.")

    # Observation input/filter overrides
    p.add_argument("--obs-csv", help="Override CONFIG['OBS_CSV'].")
    p.add_argument("--obs-npz", help="Override CONFIG['OBS_NPZ'].")
    p.add_argument("--obs-data-types", nargs="+", help="Override CONFIG['OBS_DATA_TYPES'].")
    p.add_argument("--obs-sources", nargs="+", help="Override CONFIG['OBS_SOURCES'].")
    p.add_argument("--start", help="UTC start time (YYYY-MM-DD[ HH:MM[:SS]]).")
    p.add_argument("--end", help="UTC end time (YYYY-MM-DD[ HH:MM[:SS]]).")
    p.add_argument("--min-depth", type=float)
    p.add_argument("--max-depth", type=float)
    p.add_argument("--require-obs-qc-codes", nargs="+")
    p.add_argument("--require-inside", dest="require_inside", action="store_true")
    p.add_argument("--no-require-inside", dest="require_inside", action="store_false")

    # Matching/runtime overrides
    p.add_argument("--stacks", nargs="+", type=int, help="Global stack override: [start end] or explicit list.")
    p.add_argument("--stack-check-mode", choices=["none", "light", "size", "light+size"])
    p.add_argument("--max-time-lag-hours", type=float)
    p.add_argument("--outside-domain", choices=["drop", "nan"])
    p.add_argument("--apply-utc-start", dest="apply_utc_start", action="store_true")
    p.add_argument("--no-apply-utc-start", dest="apply_utc_start", action="store_false")

    # Output overrides
    p.add_argument("--write-csv", dest="write_csv", action="store_true")
    p.add_argument("--no-write-csv", dest="write_csv", action="store_false")
    p.add_argument("--write-npz", dest="write_npz", action="store_true")
    p.add_argument("--no-write-npz", dest="write_npz", action="store_false")
    p.add_argument("--write-rejects-csv", dest="write_rejects_csv", action="store_true")
    p.add_argument("--no-write-rejects-csv", dest="write_rejects_csv", action="store_false")

    p.add_argument("--dry-run", action="store_true", help="Resolve runs/stacks and observation counts, then exit.")
    p.add_argument("--verbose", dest="verbose", action="store_true")
    p.add_argument("--quiet", dest="verbose", action="store_false")

    p.set_defaults(
        require_inside=None,
        apply_utc_start=None,
        write_csv=None,
        write_npz=None,
        write_rejects_csv=None,
        verbose=None,
    )
    return p.parse_args(argv)


def _apply_cli(cfg: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    out = deep_update_dict(copy.deepcopy(cfg), _load_json_config(args.config), merge_list_of_dicts=False)

    if args.manifest is not None:
        out["MANIFEST"] = str(args.manifest)

    if args.obs_csv is not None:
        out["OBS_CSV"] = str(args.obs_csv)
        out["OBS_NPZ"] = None
    if args.obs_npz is not None:
        out["OBS_NPZ"] = str(args.obs_npz)
        out["OBS_CSV"] = None
    if args.obs_data_types is not None:
        out["OBS_DATA_TYPES"] = tuple(str(x).strip().upper() for x in args.obs_data_types if str(x).strip() != "")
    if args.obs_sources is not None:
        out["OBS_SOURCES"] = [str(x).strip() for x in args.obs_sources if str(x).strip() != ""]
    if args.start is not None:
        out["START"] = str(args.start)
    if args.end is not None:
        out["END"] = str(args.end)
    if args.min_depth is not None:
        out["MIN_DEPTH"] = float(args.min_depth)
    if args.max_depth is not None:
        out["MAX_DEPTH"] = float(args.max_depth)
    if args.require_obs_qc_codes is not None:
        out["REQUIRE_OBS_QC_CODES"] = _parse_int_set(args.require_obs_qc_codes, default=[0])
    if args.require_inside is not None:
        out["REQUIRE_INSIDE"] = bool(args.require_inside)

    if args.stacks is not None:
        vals = [int(v) for v in args.stacks]
        out["STACKS"] = vals if len(vals) != 2 else [vals[0], vals[1]]
    if args.stack_check_mode is not None:
        out["STACK_CHECK_MODE"] = str(args.stack_check_mode).lower()
    if args.max_time_lag_hours is not None:
        out["MAX_TIME_LAG_HOURS"] = float(args.max_time_lag_hours)
    if args.outside_domain is not None:
        out["OUTSIDE_DOMAIN"] = str(args.outside_domain).lower()
    if args.apply_utc_start is not None:
        out["APPLY_UTC_START"] = bool(args.apply_utc_start)

    if args.write_csv is not None:
        out["WRITE_CSV"] = bool(args.write_csv)
    if args.write_npz is not None:
        out["WRITE_NPZ"] = bool(args.write_npz)
    if args.write_rejects_csv is not None:
        out["WRITE_REJECTS_CSV"] = bool(args.write_rejects_csv)

    if args.dry_run:
        out["DRY_RUN"] = True
    if args.verbose is not None:
        out["VERBOSE"] = bool(args.verbose)
    return out


def _validate_config(cfg: Dict[str, Any]) -> None:
    runs = cfg.get("RUNS")
    if not isinstance(runs, (list, tuple)) or len(runs) == 0:
        raise ValueError("RUNS must be a non-empty list of run specs.")

    if cfg.get("OBS_CSV") in (None, "") and cfg.get("OBS_NPZ") in (None, ""):
        raise ValueError("Configure OBS_CSV or OBS_NPZ.")

    if str(cfg.get("TIME_MATCH", "nearest")).lower() != "nearest":
        raise ValueError("Only TIME_MATCH='nearest' is supported in phase-1.")
    if str(cfg.get("SPACE_MATCH", "fem")).lower() != "fem":
        raise ValueError("Only SPACE_MATCH='fem' is supported in phase-1.")
    if str(cfg.get("DEPTH_MATCH", "interp")).lower() != "interp":
        raise ValueError("Only DEPTH_MATCH='interp' is supported in phase-1.")

    mode = str(cfg.get("STACK_CHECK_MODE", "light")).lower()
    if mode not in {"none", "light", "size", "light+size"}:
        raise ValueError(f"Invalid STACK_CHECK_MODE: {mode}")

    od = str(cfg.get("OUTSIDE_DOMAIN", "drop")).lower()
    if od not in {"drop", "nan"}:
        raise ValueError(f"Invalid OUTSIDE_DOMAIN: {od}")

    qcodes = _parse_int_set(cfg.get("REQUIRE_OBS_QC_CODES"), default=[0])
    if len(qcodes) == 0:
        raise ValueError("REQUIRE_OBS_QC_CODES must not be empty.")


def _dry_run_report(run_specs: List[Dict[str, Any]], cfg: Dict[str, Any], obs_meta: Dict[str, Any]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    verbose = bool(cfg.get("VERBOSE", True))
    for spec in run_specs:
        run = str(spec["RUN"])
        run_name = str(spec.get("NAME", os.path.basename(os.path.abspath(run))))
        out_stem = str(_resolve_output_stem(str(spec.get("SNAME"))))
        outputs_dir = _resolve_path(run) / "outputs"

        if not outputs_dir.is_dir():
            _log(f"[DRY-RUN] {run_name}: missing outputs dir -> {outputs_dir}", verbose=verbose)
            summaries.append(
                {
                    "run_name": run_name,
                    "run_dir": str(_resolve_path(run)),
                    "output_stem": out_stem,
                    "status": "missing_outputs_dir",
                }
            )
            continue

        try:
            modules, outfmt, dstacks, dvars, dvars_2d = schout_info(str(outputs_dir), 1)
            _ = modules
            _ = dvars
            _ = dvars_2d
        except Exception as exc:
            _log(f"[DRY-RUN] {run_name}: schout_info failed: {exc}", verbose=verbose)
            summaries.append(
                {
                    "run_name": run_name,
                    "run_dir": str(_resolve_path(run)),
                    "output_stem": out_stem,
                    "status": "schout_info_failed",
                    "reason": str(exc),
                }
            )
            continue

        cand = _as_stack_list(spec.get("STACKS"), dstacks)
        valid, skipped = _screen_stacks(str(outputs_dir), cand, int(outfmt), cfg)
        _log(
            f"[DRY-RUN] {run_name}: run={_resolve_path(run)}, out={out_stem}, "
            f"obs_filtered={obs_meta['obs_filtered']:,}, candidates={len(cand)}, valid={len(valid)}, skipped={len(skipped)}",
            verbose=verbose,
        )
        summaries.append(
            {
                "run_name": run_name,
                "run_dir": str(_resolve_path(run)),
                "output_stem": out_stem,
                "status": "dry_run",
                "obs_filtered": int(obs_meta["obs_filtered"]),
                "stacks_requested": int(len(cand)),
                "stacks_valid": int(len(valid)),
                "stacks_skipped": int(len(skipped)),
            }
        )
    return summaries


def _write_manifest(cfg: Dict[str, Any], run_specs: List[Dict[str, Any]], run_summaries: List[Dict[str, Any]]) -> None:
    mpath = cfg.get("MANIFEST")
    if not mpath:
        return
    out = _resolve_path(str(mpath))
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "script": str(_resolve_path(__file__)),
        "dry_run": bool(cfg.get("DRY_RUN", False)),
        "run_count": int(len(run_specs)),
        "runs": run_summaries,
    }
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    cfg = _apply_cli(CONFIG, args)
    _validate_config(cfg)

    # Current OC extractor is serial. Under mpirun, execute on rank 0 only.
    if bool(USE_MPI) and int(RANK) != 0:
        return

    verbose = bool(cfg.get("VERBOSE", True))
    run_specs = _normalize_run_specs(cfg)
    if len(run_specs) == 0:
        raise ValueError("No runs configured.")

    obs, obs_meta = _load_and_filter_observations(cfg)
    _log(
        f"[INFO] observations: loaded={obs_meta['obs_loaded']:,}, filtered={obs_meta['obs_filtered']:,}, "
        f"types={obs_meta['obs_types'] if len(obs_meta['obs_types']) > 0 else 'ALL'}, qc={obs_meta['qc_keep']}",
        verbose=verbose,
    )
    _log(f"[INFO] runs to process: {len(run_specs)}", verbose=verbose)

    if bool(cfg.get("DRY_RUN", False)):
        run_summaries = _dry_run_report(run_specs, cfg, obs_meta)
        _write_manifest(cfg, run_specs, run_summaries)
        return

    run_summaries: List[Dict[str, Any]] = []
    for i, spec in enumerate(run_specs, start=1):
        _log(f"---- Run {i}/{len(run_specs)}: {spec['NAME']} ----", verbose=verbose)
        rs = _process_one_run(spec, cfg, obs, obs_meta)
        run_summaries.append(rs)

    _write_manifest(cfg, run_specs, run_summaries)


if __name__ == "__main__":
    main()
