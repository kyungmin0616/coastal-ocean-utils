#!/usr/bin/env python3
"""
Extract SCHISM CTD profiles at TEAMS observation profile locations/times.

Workflow:
1) Read observation NPZ and build profile groups (station_id + time).
2) For each SCHISM run, discover valid stacks and stack times.
3) For each profile, select nearest stack/time and extract SCHISM profile.
4) Interpolate model temp/salt onto observation depths.
5) Save one NPZ with obs + model-ready matched arrays.
"""

import argparse
import os
import re
import time
from glob import glob

import numpy as np
from pylib import *  # noqa: F403

try:
    from mpi4py import MPI  # type: ignore

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except Exception:
    COMM = None
    RANK = 0
    SIZE = 1

CONFIG = dict(
    # Observation NPZ (TEAMS CTD)
    OBS_NPZ="/scratch2/08924/kmpark/post-proc/npz/onagawa_d2_ctd.npz",
    OBS_FIELDS=dict(
        lon="lon",
        lat="lat",
        time="time",  # UTC time recommended; can use "time_local" if needed
        depth="depth",
        temp="temp",
        sal="sal",
        station_id="station_id",
        station_name="station_name",
    ),

    # Run configuration: if RUNS is None, RUN/SNAME are used
    RUN=None,
    RUN_NAME=None,
    SNAME="./npz/ctd_pairs_single",
    SNAME_TEMPLATE="./npz/{run_name}_OB_D2",
    RUNS=[
        dict(name="RUN01b", run_dir="../RUN01b"),
        dict(name="RUN01d", run_dir="../RUN01d"),
    ],
    OUT_NPZ="./npz/ctd_pairs_multirun",
    SAVE_COMBINED_MULTI=False,

    # Time/depth matching
    MAX_TIME_LAG_HOURS=12.0,
    MIN_VALID_LEVELS=3,
    DEPTH_OUTPUT_MODE="both",  # native | interp | both

    # Stack selection
    STACKS=None,  # None -> all discovered stacks
    STACK_CHECK_MODE="light",  # none | light | size | light+size
    STACK_CHECK_ALL_FILES=False,  # False: check primary file only
    STACK_SIZE_RATIO_MIN=0.70,  # for size/light+size
    STACK_SIZE_MIN_BYTES=None,  # optional absolute minimum size

    # Convert model relative time to absolute time using param.nml
    APPLY_PARAM_START_TIME=True,
    APPLY_UTC_START=False,

    VERBOSE=True,
    LOG_SKIP_STACK_DETAILS=10,  # print first N skipped stack reasons
    LOG_EVERY_STACK=1,  # stack progress interval per rank
    )


def _log(msg, all_ranks=False):
    if (not all_ranks) and (RANK != 0):
        return
    prefix = f"[rank {RANK}/{SIZE}] " if SIZE > 1 else ""
    print(prefix + str(msg), flush=True)


def _to_scalar(v, default=None):
    if v is None:
        return default
    if isinstance(v, (list, tuple, np.ndarray)):
        if len(v) == 0:
            return default
        return v[0]
    return v


def _as_stack_list(stacks, dstacks):
    if stacks is None:
        return np.array(sorted(set([int(i) for i in np.array(dstacks).ravel()])), dtype=int)
    if isinstance(stacks, (list, tuple)) and len(stacks) == 2:
        return np.arange(int(stacks[0]), int(stacks[1]) + 1, dtype=int)
    return np.array(sorted(set([int(i) for i in np.array(stacks).ravel()])), dtype=int)


def _parse_args():
    p = argparse.ArgumentParser(description="Extract SCHISM CTD profiles for TEAMS observations.")
    p.add_argument("--obs-npz", help="Override CONFIG['OBS_NPZ']")
    p.add_argument("--out-npz", help="Override output npz path")
    p.add_argument("--sname-template", help="Override per-run output template for multi-run.")
    p.add_argument("--max-lag-hours", type=float, help="Override CONFIG['MAX_TIME_LAG_HOURS']")
    p.add_argument("--time-field", help="Override CONFIG['OBS_FIELDS']['time']")
    return p.parse_args()


def _apply_cli(cfg, args):
    out = dict(cfg)
    out["OBS_FIELDS"] = dict(cfg["OBS_FIELDS"])
    if args.obs_npz:
        out["OBS_NPZ"] = args.obs_npz
    if args.out_npz:
        out["OUT_NPZ"] = args.out_npz
        out["SNAME"] = args.out_npz
    if args.sname_template:
        out["SNAME_TEMPLATE"] = args.sname_template
    if args.max_lag_hours is not None:
        out["MAX_TIME_LAG_HOURS"] = float(args.max_lag_hours)
    if args.time_field:
        out["OBS_FIELDS"]["time"] = str(args.time_field)
    return out


def _ensure_parent(path):
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)


def _parse_run_specs(cfg):
    runs = cfg.get("RUNS")
    specs = []
    combined_npz = cfg.get("OUT_NPZ")
    if runs:
        template = str(cfg.get("SNAME_TEMPLATE", "./npz/ctd_pairs_{run_name}"))
        for i, item in enumerate(runs):
            if isinstance(item, str):
                item = dict(run_dir=item)
            if not isinstance(item, dict):
                raise ValueError(f"RUNS[{i}] must be dict or string")
            run = item.get("run_dir", item.get("RUN"))
            if run is None:
                raise ValueError(f"RUNS[{i}] missing run_dir/RUN")
            name = item.get("name", item.get("NAME", os.path.basename(os.path.abspath(run))))
            out_npz = item.get("SNAME", item.get("sname", item.get("out_npz")))
            if out_npz is None:
                out_npz = template.format(run_name=name, run=run)
            specs.append(dict(name=str(name), run_dir=str(run), out_npz=str(out_npz)))
    else:
        run = cfg.get("RUN")
        if run is None:
            raise ValueError("Either RUNS or RUN must be configured.")
        name = cfg.get("RUN_NAME") or os.path.basename(os.path.abspath(run))
        out_npz = cfg.get("SNAME", cfg["OUT_NPZ"])
        specs.append(dict(name=str(name), run_dir=str(run), out_npz=str(out_npz)))
    return specs, combined_npz


def _get_model_start_datenum(run_dir, apply_utc_start=False):
    pfile = os.path.join(run_dir, "param.nml")
    if not os.path.exists(pfile):
        return None, f"param.nml not found in {run_dir}"
    try:
        p = read_schism_param(pfile, 1)  # noqa: F403
    except Exception as exc:
        return None, f"failed to parse param.nml: {exc}"

    try:
        sy = int(_to_scalar(p.get("start_year")))
        sm = int(_to_scalar(p.get("start_month")))
        sd = int(_to_scalar(p.get("start_day")))
        sh = float(_to_scalar(p.get("start_hour"), 0.0))
        us = float(_to_scalar(p.get("utc_start"), 0.0))
    except Exception as exc:
        return None, f"invalid start fields in param.nml: {exc}"

    d0 = float(datenum(sy, sm, sd))  # noqa: F403
    d0 += sh / 24.0
    if apply_utc_start:
        d0 -= us / 24.0
    return d0, f"{sy:04d}-{sm:02d}-{sd:02d} {sh:05.2f}h utc_start={us}"


def _primary_stack_file(outputs_dir, stack, outfmt):
    if outfmt == 0:
        fn = os.path.join(outputs_dir, f"out2d_{stack}.nc")
        return fn if os.path.exists(fn) else None
    cand = sorted(glob(os.path.join(outputs_dir, f"schout_*_{stack}.nc")))
    if cand:
        return cand[0]
    fn = os.path.join(outputs_dir, f"schout_{stack}.nc")
    return fn if os.path.exists(fn) else None


def _stack_files_for_check(outputs_dir, stack, outfmt, check_all_files=False):
    primary = _primary_stack_file(outputs_dir, stack, outfmt)
    if primary is None:
        return []
    if not check_all_files:
        return [primary]
    files = sorted(glob(os.path.join(outputs_dir, f"*_{stack}.nc")))
    if not files:
        return [primary]
    if primary not in files:
        files.insert(0, primary)
    return files


def _header_time_ok(nc_path):
    c = None
    try:
        c = ReadNC(nc_path, 1)  # noqa: F403
        if "time" not in c.variables:
            return False, "missing time variable"
        tvar = c.variables["time"]
        if hasattr(tvar, "shape") and len(tvar.shape) > 0:
            nt = int(tvar.shape[0])
        else:
            nt = int(len(np.array(tvar)))
        if nt <= 0:
            return False, "empty time variable"
        _ = float(np.array(tvar[0]).ravel()[0])
        return True, "ok"
    except Exception as exc:
        return False, str(exc)
    finally:
        if c is not None:
            try:
                c.close()
            except Exception:
                pass


def _size_ok(path, ref_size, ratio_min=0.70, abs_min_bytes=None):
    try:
        size = int(os.path.getsize(path))
    except Exception as exc:
        return False, f"size check failed: {exc}"
    if abs_min_bytes is not None and size < int(abs_min_bytes):
        return False, f"size={size} < abs_min={int(abs_min_bytes)}"
    thr = float(ratio_min) * float(ref_size)
    if size < thr:
        return False, f"size={size} < ratio_min*median={int(thr)}"
    return True, "ok"


def _screen_stacks(outputs_dir, stacks, outfmt, mode="light", check_all_files=False, ratio_min=0.70, abs_min_bytes=None):
    mode = "none" if mode is None else str(mode).lower()
    stacks = [int(i) for i in np.array(stacks).ravel()]
    if len(stacks) == 0:
        return np.array([], dtype=int), {}

    primary = {}
    for st in stacks:
        p = _primary_stack_file(outputs_dir, st, outfmt)
        if p is not None:
            primary[st] = p

    ref_size = None
    sizes = [os.path.getsize(fp) for fp in primary.values() if os.path.exists(fp)]
    if sizes:
        ref_size = int(np.median(np.array(sizes, dtype=float)))

    valid = []
    skipped = {}
    for st in stacks:
        files = _stack_files_for_check(outputs_dir, st, outfmt, check_all_files=check_all_files)
        if not files:
            skipped[st] = "missing primary stack file"
            continue

        need_light = mode in {"light", "light+size"}
        need_size = mode in {"size", "light+size"}
        ok = True
        reason = ""

        if need_light:
            for fn in files:
                f_ok, f_reason = _header_time_ok(fn)
                if not f_ok:
                    ok = False
                    reason = f"{os.path.basename(fn)}: {f_reason}"
                    break
        if ok and need_size and ref_size is not None:
            for fn in files:
                s_ok, s_reason = _size_ok(fn, ref_size, ratio_min=ratio_min, abs_min_bytes=abs_min_bytes)
                if not s_ok:
                    ok = False
                    reason = f"{os.path.basename(fn)}: {s_reason}"
                    break

        if mode == "none":
            ok = True

        if ok:
            valid.append(st)
        else:
            skipped[st] = reason if reason else "stack check failed"
    return np.array(valid, dtype=int), skipped


def _convert_obs_time_to_datenum(time_arr):
    time_arr = np.asarray(time_arr)
    if np.issubdtype(time_arr.dtype, np.datetime64):
        epoch = np.datetime64("1970-01-01T00:00:00")
        sec = (time_arr.astype("datetime64[s]") - epoch).astype("timedelta64[s]").astype(np.int64)
        return float(datenum(1970, 1, 1)) + sec.astype(float) / 86400.0  # noqa: F403
    out = []
    for t in time_arr.astype("U"):
        out.append(float(datenum(t)))  # noqa: F403
    return np.asarray(out, dtype=float)


def _build_profiles(obs, min_levels=3):
    stid = np.asarray(obs["station_id"]).astype("U")
    stnm = np.asarray(obs["station_name"]).astype("U") if obs.get("station_name") is not None else np.array([""] * len(stid))
    lon = np.asarray(obs["lon"], dtype=float)
    lat = np.asarray(obs["lat"], dtype=float)
    time_raw = np.asarray(obs["time"])
    time_str = time_raw.astype("datetime64[s]").astype("U") if np.issubdtype(time_raw.dtype, np.datetime64) else time_raw.astype("U")
    time_dn = _convert_obs_time_to_datenum(time_raw)
    depth = np.asarray(obs["depth"], dtype=float)
    temp = np.asarray(obs["temp"], dtype=float) if obs.get("temp") is not None else np.full(depth.shape, np.nan)
    sal = np.asarray(obs["sal"], dtype=float) if obs.get("sal") is not None else np.full(depth.shape, np.nan)

    keys = np.char.add(stid, np.char.add("|", time_str))
    uniq, inv = np.unique(keys, return_inverse=True)

    profiles = []
    skipped = dict(
        too_few_levels=0,
        invalid_location=0,
    )
    for i, key in enumerate(uniq):
        mask = inv == i
        if not np.any(mask):
            continue
        d = depth[mask]
        t = temp[mask]
        s = sal[mask]
        valid = np.isfinite(d) & (np.isfinite(t) | np.isfinite(s))
        if valid.sum() < int(min_levels):
            skipped["too_few_levels"] += 1
            continue
        d = d[valid]
        t = t[valid]
        s = s[valid]
        order = np.argsort(d)
        d = d[order]
        t = t[order]
        s = s[order]

        idx0 = np.where(mask)[0][0]
        lon_mask = lon[mask]
        lat_mask = lat[mask]
        fxy = np.isfinite(lon_mask) & np.isfinite(lat_mask)
        if fxy.sum() == 0:
            skipped["invalid_location"] += 1
            continue
        lon_pt = float(np.mean(lon_mask[fxy]))
        lat_pt = float(np.mean(lat_mask[fxy]))

        profiles.append(
            dict(
                profile_id=str(key),
                station_id=str(stid[idx0]),
                station_name=str(stnm[idx0]),
                lon=lon_pt,
                lat=lat_pt,
                time=float(time_dn[idx0]),
                time_str=str(time_str[idx0]),
                depth=d,
                temp=t,
                sal=s,
            )
        )
    return profiles, skipped


def _pack_obs_profiles(profiles):
    nprof = len(profiles)
    max_lev = max([len(p["depth"]) for p in profiles]) if nprof > 0 else 0
    obs_depth = np.full((nprof, max_lev), np.nan, dtype=float)
    obs_temp = np.full((nprof, max_lev), np.nan, dtype=float)
    obs_sal = np.full((nprof, max_lev), np.nan, dtype=float)
    nlev = np.zeros(nprof, dtype=int)

    profile_id = np.empty(nprof, dtype=object)
    station_id = np.empty(nprof, dtype=object)
    station_name = np.empty(nprof, dtype=object)
    lon = np.full(nprof, np.nan, dtype=float)
    lat = np.full(nprof, np.nan, dtype=float)
    time_obs = np.full(nprof, np.nan, dtype=float)

    for i, p in enumerate(profiles):
        nv = len(p["depth"])
        nlev[i] = nv
        obs_depth[i, :nv] = p["depth"]
        obs_temp[i, :nv] = p["temp"]
        obs_sal[i, :nv] = p["sal"]
        profile_id[i] = p["profile_id"]
        station_id[i] = p["station_id"]
        station_name[i] = p["station_name"]
        lon[i] = p["lon"]
        lat[i] = p["lat"]
        time_obs[i] = p["time"]

    return dict(
        nprof=nprof,
        max_lev=max_lev,
        nlev=nlev,
        profile_id=profile_id.astype("U"),
        station_id=station_id.astype("U"),
        station_name=station_name.astype("U"),
        lon=lon,
        lat=lat,
        time_obs=time_obs,
        obs_depth=obs_depth,
        obs_temp=obs_temp,
        obs_sal=obs_sal,
    )


def _time_var_to_days(tvar):
    t = np.asarray(tvar[:], dtype=float).ravel()
    units = str(getattr(tvar, "units", "")).lower()
    if "second" in units:
        return t / 86400.0
    if "hour" in units:
        return t / 24.0
    if "day" in units:
        return t
    if np.nanmax(np.abs(t)) > 1000:
        return t / 86400.0
    return t


def _read_stack_times_abs(nc_path, start_dn):
    c = ReadNC(nc_path, 1)  # noqa: F403
    try:
        if "time" not in c.variables:
            return np.array([], dtype=float)
        t_days = _time_var_to_days(c.variables["time"])
        if start_dn is None:
            return t_days.astype(float)
        return (t_days + float(start_dn)).astype(float)
    finally:
        try:
            c.close()
        except Exception:
            pass


def _assign_profiles_to_stacks(time_obs, stack_ids, stack_tmin, stack_tmax, stack_tmid):
    p2s = np.full(len(time_obs), -1, dtype=int)
    for i, t in enumerate(time_obs):
        inside = np.where((t >= stack_tmin) & (t <= stack_tmax))[0]
        if inside.size == 1:
            p2s[i] = int(stack_ids[inside[0]])
        elif inside.size > 1:
            k = inside[np.argmin(np.abs(stack_tmid[inside] - t))]
            p2s[i] = int(stack_ids[k])
        else:
            k = int(np.argmin(np.abs(stack_tmid - t)))
            p2s[i] = int(stack_ids[k])
    return p2s


def _extract_tp(arr, it, ip, nt=None, npnt=None):
    a = np.asarray(arr)
    if a.ndim == 3:
        sh = a.shape
        time_axis = None
        point_axis = None

        if nt is not None:
            cand_t = [i for i, s in enumerate(sh) if int(s) == int(nt)]
            if len(cand_t) > 0:
                time_axis = cand_t[0]
        if npnt is not None:
            cand_p = [i for i, s in enumerate(sh) if int(s) == int(npnt)]
            if len(cand_p) > 0:
                point_axis = cand_p[0]

        # fallbacks
        if time_axis is None:
            time_axis = 0
        if point_axis is None or point_axis == time_axis:
            point_axis = 1 if time_axis != 1 else 0

        axes_left = [i for i in range(3) if i not in (time_axis, point_axis)]
        if len(axes_left) != 1:
            return np.asarray(a, dtype=float).ravel()
        depth_axis = axes_left[0]

        slc = [slice(None)] * 3
        slc[time_axis] = int(np.clip(it, 0, sh[time_axis] - 1))
        slc[point_axis] = int(np.clip(ip, 0, sh[point_axis] - 1))
        v = np.asarray(a[tuple(slc)], dtype=float)
        if v.ndim == 0:
            return v.reshape(1)
        if v.ndim == 1:
            return v.ravel()
        # Keep only depth direction if any leftover dims remain
        return np.moveaxis(v, depth_axis if depth_axis < v.ndim else 0, 0).ravel()
    if a.ndim == 2:
        sh = a.shape
        i0 = int(np.clip(it, 0, sh[0] - 1))
        i1 = int(np.clip(ip, 0, sh[1] - 1))

        # try explicit matching first
        if nt is not None and sh[0] == int(nt):
            return np.asarray(a[i0, :], dtype=float).ravel()
        if nt is not None and sh[1] == int(nt):
            return np.asarray(a[:, i0], dtype=float).ravel()
        if npnt is not None and sh[0] == int(npnt):
            return np.asarray(a[i1, :], dtype=float).ravel()
        if npnt is not None and sh[1] == int(npnt):
            return np.asarray(a[:, i1], dtype=float).ravel()

        # fallback heuristic
        if sh[0] > sh[1]:
            return np.asarray(a[i0, :], dtype=float).ravel()
        return np.asarray(a[:, i1], dtype=float).ravel()
    return np.asarray(a, dtype=float).ravel()


def _interp_on_obs_depth(model_depth, model_val, obs_depth):
    md = np.asarray(model_depth, dtype=float)
    mv = np.asarray(model_val, dtype=float)
    od = np.asarray(obs_depth, dtype=float)
    out = np.full(od.shape, np.nan, dtype=float)

    valid = np.isfinite(md) & np.isfinite(mv)
    if valid.sum() < 2:
        return out
    md = md[valid]
    mv = mv[valid]
    order = np.argsort(md)
    md = md[order]
    mv = mv[order]
    md_u, idx_u = np.unique(md, return_index=True)
    mv_u = mv[idx_u]
    if md_u.size < 2:
        return out

    inrng = np.isfinite(od) & (od >= md_u[0]) & (od <= md_u[-1])
    if np.any(inrng):
        out[inrng] = np.interp(od[inrng], md_u, mv_u)
    return out


def _extract_run(run_spec, obs_pack, cfg):
    run_name = run_spec["name"]
    run_dir = run_spec["run_dir"]
    outputs_dir = os.path.join(run_dir, "outputs")

    modules, outfmt, dstacks, dvars, _ = schout_info(outputs_dir, 1)  # noqa: F403
    _ = modules
    _ = dvars

    candidates = _as_stack_list(cfg.get("STACKS"), dstacks)
    valid_stacks, skipped = _screen_stacks(
        outputs_dir=outputs_dir,
        stacks=candidates,
        outfmt=outfmt,
        mode=cfg.get("STACK_CHECK_MODE", "light"),
        check_all_files=bool(cfg.get("STACK_CHECK_ALL_FILES", False)),
        ratio_min=float(cfg.get("STACK_SIZE_RATIO_MIN", 0.70)),
        abs_min_bytes=cfg.get("STACK_SIZE_MIN_BYTES"),
    )
    if cfg.get("VERBOSE", True):
        _log(f"[{run_name}] candidates={len(candidates)}, valid={len(valid_stacks)}, skipped={len(skipped)}", all_ranks=True)
        nshow = int(cfg.get("LOG_SKIP_STACK_DETAILS", 0))
        if nshow > 0 and len(skipped) > 0:
            for ist, st in enumerate(sorted(skipped.keys())):
                if ist >= nshow:
                    _log(f"[{run_name}] ... {len(skipped) - nshow} more skipped stacks", all_ranks=True)
                    break
                _log(f"[{run_name}] skip stack {st}: {skipped[st]}", all_ranks=True)
    if len(valid_stacks) == 0:
        raise RuntimeError(f"{run_name}: no valid stacks found")

    start_dn = None
    start_info = "not applied"
    if cfg.get("APPLY_PARAM_START_TIME", True):
        start_dn, start_info = _get_model_start_datenum(run_dir, apply_utc_start=bool(cfg.get("APPLY_UTC_START", False)))
        if cfg.get("VERBOSE", True):
            _log(f"[{run_name}] model start: {start_info}", all_ranks=True)

    stack_ids = []
    stack_tmin = []
    stack_tmax = []
    stack_tmid = []
    stack_times = {}
    for st in valid_stacks:
        fp = _primary_stack_file(outputs_dir, int(st), outfmt)
        if fp is None:
            continue
        tabs = _read_stack_times_abs(fp, start_dn)
        if tabs.size == 0:
            continue
        stack_ids.append(int(st))
        stack_tmin.append(float(np.nanmin(tabs)))
        stack_tmax.append(float(np.nanmax(tabs)))
        stack_tmid.append(float(np.nanmean(tabs)))
        stack_times[int(st)] = tabs

    if len(stack_ids) == 0:
        raise RuntimeError(f"{run_name}: no stack has usable time axis")

    stack_ids = np.asarray(stack_ids, dtype=int)
    stack_tmin = np.asarray(stack_tmin, dtype=float)
    stack_tmax = np.asarray(stack_tmax, dtype=float)
    stack_tmid = np.asarray(stack_tmid, dtype=float)

    p2s = _assign_profiles_to_stacks(obs_pack["time_obs"], stack_ids, stack_tmin, stack_tmax, stack_tmid)

    nprof = obs_pack["nprof"]
    max_lev = obs_pack["max_lev"]
    mtemp = np.full((nprof, max_lev), np.nan, dtype=float)
    msal = np.full((nprof, max_lev), np.nan, dtype=float)
    mtime = np.full(nprof, np.nan, dtype=float)
    mlag_h = np.full(nprof, np.nan, dtype=float)
    mstack = np.full(nprof, -1, dtype=int)
    qc = np.full(nprof, -1, dtype=int)

    max_lag = cfg.get("MAX_TIME_LAG_HOURS")
    max_lag = None if max_lag is None else float(max_lag)
    mode = str(cfg.get("DEPTH_OUTPUT_MODE", "both")).strip().lower()
    want_interp = mode in {"interp", "both"}
    want_native = mode in {"native", "both"}

    native_depth_list = [None] * nprof
    native_temp_list = [None] * nprof
    native_sal_list = [None] * nprof

    cvars = ["zcor", "temp", "salt"]
    uniq_st = np.unique(p2s)
    if cfg.get("VERBOSE", True):
        _log(f"[{run_name}] stacks with assigned profiles: {len(uniq_st[uniq_st >= 0])}", all_ranks=True)
    for ist, st in enumerate(uniq_st):
        if st < 0:
            continue
        pidx = np.where(p2s == st)[0]
        if pidx.size == 0:
            continue
        t_stack0 = time.time()
        xy = np.c_[obs_pack["lon"][pidx], obs_pack["lat"][pidx]]
        try:
            C = read_schism_output(run_dir, cvars, xy, int(st), fmt=1)  # noqa: F403
        except Exception as exc:
            if cfg.get("VERBOSE", True):
                _log(f"[{run_name}] stack={st} read failed: {exc}", all_ranks=True)
            qc[pidx] = 3
            continue

        t_abs = np.asarray(C.time, dtype=float).ravel()
        if start_dn is not None:
            t_abs = t_abs + float(start_dn)
        if t_abs.size == 0:
            qc[pidx] = 4
            continue
        nt = int(t_abs.size)
        npnt = int(len(pidx))

        for j, iobs in enumerate(pidx):
            tobs = obs_pack["time_obs"][iobs]
            it = int(np.argmin(np.abs(t_abs - tobs)))
            lag_h = abs(float(t_abs[it] - tobs)) * 24.0
            mtime[iobs] = float(t_abs[it])
            mlag_h[iobs] = lag_h
            mstack[iobs] = int(st)

            if (max_lag is not None) and (lag_h > max_lag):
                qc[iobs] = 2
                continue

            nz = int(obs_pack["nlev"][iobs])
            od = obs_pack["obs_depth"][iobs, :nz]
            md = -_extract_tp(C.zcor, it, j, nt=nt, npnt=npnt)
            tp = _extract_tp(C.temp, it, j, nt=nt, npnt=npnt)
            sp = _extract_tp(C.salt, it, j, nt=nt, npnt=npnt)

            if want_native:
                native_depth_list[iobs] = np.asarray(md, dtype=float)
                native_temp_list[iobs] = np.asarray(tp, dtype=float)
                native_sal_list[iobs] = np.asarray(sp, dtype=float)

            if want_interp:
                ti = _interp_on_obs_depth(md, tp, od)
                si = _interp_on_obs_depth(md, sp, od)
                mtemp[iobs, :nz] = ti
                msal[iobs, :nz] = si
            qc[iobs] = 0
        if cfg.get("VERBOSE", True):
            log_every = max(1, int(cfg.get("LOG_EVERY_STACK", 1)))
            if ((ist + 1) % log_every == 0) or (ist == len(uniq_st) - 1):
                _log(
                    f"[{run_name}] stack {int(st)} done ({ist + 1}/{len(uniq_st)}), "
                    f"profiles={len(pidx)}, elapsed={time.time() - t_stack0:.2f}s",
                    all_ranks=True,
                )

    nlev_native = np.zeros(nprof, dtype=int)
    mdepth_native = np.full((nprof, 0), np.nan, dtype=float)
    mtemp_native = np.full((nprof, 0), np.nan, dtype=float)
    msal_native = np.full((nprof, 0), np.nan, dtype=float)
    if want_native:
        max_native = 0
        for i in range(nprof):
            arr = native_depth_list[i]
            if arr is None:
                continue
            nlev_native[i] = int(len(arr))
            if nlev_native[i] > max_native:
                max_native = nlev_native[i]
        mdepth_native = np.full((nprof, max_native), np.nan, dtype=float)
        mtemp_native = np.full((nprof, max_native), np.nan, dtype=float)
        msal_native = np.full((nprof, max_native), np.nan, dtype=float)
        for i in range(nprof):
            nn = int(nlev_native[i])
            if nn <= 0:
                continue
            mdepth_native[i, :nn] = native_depth_list[i]
            mtemp_native[i, :nn] = native_temp_list[i]
            msal_native[i, :nn] = native_sal_list[i]

    return dict(
        run_name=run_name,
        run_dir=os.path.abspath(run_dir),
        model_start=np.nan if start_dn is None else float(start_dn),
        stack_count=int(len(stack_ids)),
        model_temp=mtemp,
        model_sal=msal,
        model_time=mtime,
        model_lag_hours=mlag_h,
        model_stack=mstack,
        qc_flag=qc,
        model_depth_native=mdepth_native,
        model_temp_native=mtemp_native,
        model_sal_native=msal_native,
        model_nlev_native=nlev_native,
        has_interp=int(want_interp),
        has_native=int(want_native),
    )


def _read_obs_npz(cfg):
    fields = cfg["OBS_FIELDS"]
    npz = np.load(cfg["OBS_NPZ"], allow_pickle=True)
    out = {}
    for key, name in fields.items():
        out[key] = npz[name] if name in npz.files else None
    required = ("lon", "lat", "time", "depth", "station_id")
    missing = [k for k in required if out.get(k) is None]
    if missing:
        raise KeyError(f"Missing required fields in OBS_NPZ: {missing}")
    return out


def _build_payload(obs_pack, run_results):
    nrun = len(run_results)
    nprof = obs_pack["nprof"]
    max_lev = obs_pack["max_lev"]
    model_temp = np.full((nrun, nprof, max_lev), np.nan, dtype=float)
    model_sal = np.full((nrun, nprof, max_lev), np.nan, dtype=float)
    model_time = np.full((nrun, nprof), np.nan, dtype=float)
    model_lag_h = np.full((nrun, nprof), np.nan, dtype=float)
    model_stack = np.full((nrun, nprof), -1, dtype=int)
    qc_flag = np.full((nrun, nprof), -1, dtype=int)
    model_start = np.full(nrun, np.nan, dtype=float)
    run_names = np.empty(nrun, dtype=object)
    run_dirs = np.empty(nrun, dtype=object)
    has_interp = np.zeros(nrun, dtype=int)
    has_native = np.zeros(nrun, dtype=int)

    for i, rr in enumerate(run_results):
        run_names[i] = rr["run_name"]
        run_dirs[i] = rr["run_dir"]
        model_start[i] = rr["model_start"]
        model_temp[i, :, :] = rr["model_temp"]
        model_sal[i, :, :] = rr["model_sal"]
        model_time[i, :] = rr["model_time"]
        model_lag_h[i, :] = rr["model_lag_hours"]
        model_stack[i, :] = rr["model_stack"]
        qc_flag[i, :] = rr["qc_flag"]
        has_interp[i] = int(rr.get("has_interp", 1))
        has_native[i] = int(rr.get("has_native", 0))

    max_native = 0
    for rr in run_results:
        arr = np.asarray(rr.get("model_nlev_native", np.array([], dtype=int)))
        if arr.size > 0:
            max_native = max(max_native, int(np.nanmax(arr)))
    model_depth_native = np.full((nrun, nprof, max_native), np.nan, dtype=float)
    model_temp_native = np.full((nrun, nprof, max_native), np.nan, dtype=float)
    model_sal_native = np.full((nrun, nprof, max_native), np.nan, dtype=float)
    model_nlev_native = np.zeros((nrun, nprof), dtype=int)

    for i, rr in enumerate(run_results):
        nlev_i = np.asarray(rr.get("model_nlev_native", np.zeros(nprof, dtype=int)), dtype=int)
        d_i = np.asarray(rr.get("model_depth_native", np.full((nprof, 0), np.nan)))
        t_i = np.asarray(rr.get("model_temp_native", np.full((nprof, 0), np.nan)))
        s_i = np.asarray(rr.get("model_sal_native", np.full((nprof, 0), np.nan)))
        model_nlev_native[i, :] = nlev_i
        if d_i.ndim == 2 and d_i.shape[1] > 0:
            nn = min(max_native, d_i.shape[1])
            model_depth_native[i, :, :nn] = d_i[:, :nn]
            model_temp_native[i, :, :nn] = t_i[:, :nn]
            model_sal_native[i, :, :nn] = s_i[:, :nn]

    payload = dict(
        run_names=run_names.astype("U"),
        run_dirs=run_dirs.astype("U"),
        model_start=model_start,
        has_interp=has_interp,
        has_native=has_native,
        time_units=np.array("datenum_utc"),
        profile_id=obs_pack["profile_id"],
        station_id=obs_pack["station_id"],
        station_name=obs_pack["station_name"],
        lon=obs_pack["lon"],
        lat=obs_pack["lat"],
        time_obs=obs_pack["time_obs"],
        nlev=obs_pack["nlev"],
        obs_depth=obs_pack["obs_depth"],
        obs_temp=obs_pack["obs_temp"],
        obs_sal=obs_pack["obs_sal"],
        model_temp=model_temp,
        model_sal=model_sal,
        model_time=model_time,
        model_lag_hours=model_lag_h,
        model_stack=model_stack,
        qc_flag=qc_flag,
        model_depth_native=model_depth_native,
        model_temp_native=model_temp_native,
        model_sal_native=model_sal_native,
        model_nlev_native=model_nlev_native,
    )
    return payload


def _save_payload(npz_path, payload):
    _ensure_parent(npz_path)
    np.savez(npz_path, **payload)
    _log(f"Saved: {npz_path if str(npz_path).endswith('.npz') else npz_path + '.npz'}")


def _npz_path(path):
    return path if str(path).endswith(".npz") else f"{path}.npz"


def _build_combined_from_single_files(run_specs):
    run_axis_keys = {
        "run_names",
        "run_dirs",
        "model_start",
        "model_temp",
        "model_sal",
        "model_time",
        "model_lag_hours",
        "model_stack",
        "qc_flag",
    }

    loaded = []
    for spec in run_specs:
        fp = _npz_path(spec["out_npz"])
        if not os.path.exists(fp):
            raise FileNotFoundError(f"Missing per-run npz for combine: {fp}")
        loaded.append(np.load(fp, allow_pickle=True))
    if len(loaded) == 0:
        raise RuntimeError("No per-run files found for combined output.")

    payload = {}
    for k in loaded[0].files:
        if k not in run_axis_keys:
            payload[k] = loaded[0][k]

    for k in run_axis_keys:
        arrs = [z[k] for z in loaded]
        payload[k] = np.concatenate(arrs, axis=0)

    for z in loaded:
        z.close()
    return payload


def main():
    args = _parse_args()
    cfg = _apply_cli(CONFIG, args)
    run_specs, combined_npz = _parse_run_specs(cfg)

    if cfg.get("VERBOSE", True):
        _log(f"Observation NPZ: {cfg['OBS_NPZ']}")
        _log(f"Runs (all): {[r['name'] for r in run_specs]}")

    obs = _read_obs_npz(cfg)
    profiles, skipped_prof = _build_profiles(obs, min_levels=int(cfg.get("MIN_VALID_LEVELS", 3)))
    if len(profiles) == 0:
        raise RuntimeError("No valid CTD profiles found from observation NPZ.")
    obs_pack = _pack_obs_profiles(profiles)
    if cfg.get("VERBOSE", True):
        _log(f"Valid profiles: {obs_pack['nprof']}, max_levels: {obs_pack['max_lev']}")
        _log(
            "Skipped profiles: "
            f"too_few_levels={skipped_prof['too_few_levels']}, "
            f"invalid_location={skipped_prof['invalid_location']}"
        )

    local_specs = [spec for i, spec in enumerate(run_specs) if (i % SIZE) == RANK]
    if (len(local_specs) > 0) or (RANK == 0):
        _log(f"Assigned runs on this rank: {[r['name'] for r in local_specs]}", all_ranks=True)

    run_results = []
    for spec in local_specs:
        _log(f"Processing run: {spec['name']}", all_ranks=True)
        rr = _extract_run(spec, obs_pack, cfg)
        run_results.append(rr)
        single_payload = _build_payload(obs_pack, [rr])
        _save_payload(spec["out_npz"], single_payload)

    if COMM is not None:
        COMM.Barrier()

    if len(run_specs) > 1 and bool(cfg.get("SAVE_COMBINED_MULTI", False)) and RANK == 0:
        combo_payload = _build_combined_from_single_files(run_specs)
        _save_payload(combined_npz, combo_payload)


if __name__ == "__main__":
    main()
