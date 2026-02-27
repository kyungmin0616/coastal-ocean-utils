#!/usr/bin/env python3
"""
Extract time series at stations or transects from SCHISM outputs.

Features:
1) Single-run or multi-run extraction.
2) Optional model-start time offset from run/param.nml.
3) Auto stack discovery with lightweight incomplete-file screening.
"""

from pylib import *  # noqa: F403
import os
import re
import time
from glob import glob

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
    # Legacy single-run mode (kept for backward compatibility)
    RUN="../RUN01c",
    SNAME="./npz/RUN01c_v1",
    
    # Multi-run mode (if RUNS is set, RUN/SNAME are ignored)
    RUNS=[
         {"NAME": "RUN01b", "RUN": "../RUN01b" },
         {"NAME": "RUN01d", "RUN": "../RUN01d"},
     ],  
    # Example:
    # RUNS=[
    #     {"NAME": "RUN01a", "RUN": "../RUN01a", "SNAME": "./npz/RUN01a"},
    #     {"NAME": "RUN02a", "RUN": "../RUN02a", "SNAME": "./npz/RUN02a"},
    # ]
    SNAME_TEMPLATE="./npz/{run_name}_SB_D2",
    
    SVARS=("temp", "salt"),
    BPFILE="./station_sendai_d2.bp",
    ITYPE=0,  # 0: time series of points @xyz; 1: transects @xy
    IFS=0,  # 0: refer to free surface; 1: fixed depth
    STACKS=None,  # None -> all available stacks after screening
    NSPOOL=1,  # sub-sampling within each stack
    MDT=None,  # time window (day) for averaging output
    RVARS=None,  # rename variables
    PRJ=None,  # e.g. ["epsg:26918", "epsg:4326"]
    
    # Stack screening controls
    STACK_CHECK_MODE="light",  # none | light | size | light+size
    STACK_CHECK_ALL_FILES=False,  # False: check primary stack file only
    STACK_SIZE_RATIO_MIN=0.70,  # for size/light+size: min size ratio against median primary size
    STACK_SIZE_MIN_BYTES=None,  # optional absolute size floor in bytes
    
    # Time offset controls
    APPLY_PARAM_START_TIME=True,
    APPLY_UTC_START=False,  # If True, shift by -utc_start/24 from param.nml
    
    VERBOSE=True,
)


def _log(msg):
    prefix = f"[rank {RANK}/{SIZE}] " if SIZE > 1 else ""
    print(prefix + str(msg), flush=True)


def _as_stack_list(stacks, dstacks):
    if stacks is None:
        return array(sorted(set([int(i) for i in array(dstacks).ravel()])), dtype=int)  # noqa: F403
    if isinstance(stacks, (list, tuple)) and len(stacks) == 2:
        s0 = int(stacks[0])
        s1 = int(stacks[1])
        return arange(s0, s1 + 1).astype(int)  # noqa: F403
    return array(sorted(set([int(i) for i in array(stacks).ravel()])), dtype=int)  # noqa: F403


def _to_scalar(v, default=None):
    if v is None:
        return default
    if isinstance(v, (list, tuple, ndarray)):  # noqa: F405
        if len(v) == 0:
            return default
        return v[0]
    return v


def _get_model_start_datenum(run, apply_utc_start=False):
    pfile = os.path.join(run, "param.nml")
    if not fexist(pfile):  # noqa: F403
        return None, f"param.nml not found in {run}"

    try:
        p = read_schism_param(pfile, 1)  # noqa: F403
    except Exception as exc:
        return None, f"failed to parse param.nml: {exc}"

    keys = ["start_year", "start_month", "start_day", "start_hour"]
    for key in keys:
        if key not in p:
            return None, f"missing {key} in param.nml"

    try:
        sy = int(_to_scalar(p.get("start_year")))
        sm = int(_to_scalar(p.get("start_month")))
        sd = int(_to_scalar(p.get("start_day")))
        sh = float(_to_scalar(p.get("start_hour"), 0.0))
        us = float(_to_scalar(p.get("utc_start"), 0.0))
    except Exception as exc:
        return None, f"invalid start time fields in param.nml: {exc}"

    d0 = float(datenum(sy, sm, sd))  # noqa: F403
    d0 = d0 + sh / 24.0
    if apply_utc_start:
        d0 = d0 - us / 24.0

    return d0, f"{sy:04d}-{sm:02d}-{sd:02d} {sh:05.2f}h (utc_start={us})"


def _stack_num_from_name(path):
    name = os.path.basename(path)
    m = re.search(r"_(\d+)\.nc$", name)
    return int(m.group(1)) if m else None


def _primary_stack_file(outputs_dir, stack, outfmt):
    if outfmt == 0:
        fn = os.path.join(outputs_dir, f"out2d_{stack}.nc")
        return fn if fexist(fn) else None  # noqa: F403

    cand = sorted(glob(os.path.join(outputs_dir, f"schout_*_{stack}.nc")))
    if len(cand) > 0:
        return cand[0]
    fn = os.path.join(outputs_dir, f"schout_{stack}.nc")
    return fn if fexist(fn) else None  # noqa: F403


def _stack_files_for_check(outputs_dir, stack, outfmt, check_all_files):
    primary = _primary_stack_file(outputs_dir, stack, outfmt)
    if primary is None:
        return []
    if not check_all_files:
        return [primary]

    files = sorted(glob(os.path.join(outputs_dir, f"*_{stack}.nc")))
    if len(files) == 0:
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

        # lightweight validity check: dimension + first value
        if hasattr(tvar, "shape") and len(tvar.shape) > 0:
            nt = int(tvar.shape[0])
        else:
            nt = int(len(array(tvar)))  # noqa: F403
        if nt <= 0:
            return False, "empty time variable"

        _ = float(array(tvar[0]).ravel()[0])  # noqa: F403
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


def _screen_stacks(
    outputs_dir,
    stacks,
    outfmt,
    mode="light",
    check_all_files=False,
    ratio_min=0.70,
    abs_min_bytes=None,
    verbose=True,
):
    stacks = [int(i) for i in array(stacks).ravel()]  # noqa: F403
    if len(stacks) == 0:
        return array([], dtype=int), {}  # noqa: F403

    mode = "none" if mode is None else str(mode).lower()
    if mode == "none":
        return array(stacks, dtype=int), {}  # noqa: F403

    primary = {}
    for st in stacks:
        p = _primary_stack_file(outputs_dir, st, outfmt)
        if p is not None:
            primary[st] = p
    ref_size = None
    sizes = [os.path.getsize(fp) for fp in primary.values() if fexist(fp)]  # noqa: F403
    if len(sizes) > 0:
        ref_size = int(median(array(sizes, dtype=float)))  # noqa: F403

    valid = []
    skipped = {}
    for st in stacks:
        files = _stack_files_for_check(outputs_dir, st, outfmt, check_all_files)
        if len(files) == 0:
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
                s_ok, s_reason = _size_ok(
                    fn,
                    ref_size,
                    ratio_min=ratio_min,
                    abs_min_bytes=abs_min_bytes,
                )
                if not s_ok:
                    ok = False
                    reason = f"{os.path.basename(fn)}: {s_reason}"
                    break

        if ok:
            valid.append(st)
        else:
            skipped[st] = reason

    if verbose and RANK == 0:
        _log(
            f"Stack screen ({mode}): requested={len(stacks)}, "
            f"valid={len(valid)}, skipped={len(skipped)}"
        )
        if len(skipped) > 0:
            for st in sorted(skipped)[:20]:
                _log(f"  skip stack {st}: {skipped[st]}")
            if len(skipped) > 20:
                _log(f"  ... {len(skipped) - 20} more skipped stacks")

    return array(valid, dtype=int), skipped  # noqa: F403


def _normalize_run_specs(cfg):
    runs = cfg.get("RUNS")
    specs = []

    if runs is None:
        specs.append(
            dict(
                NAME=os.path.basename(os.path.abspath(cfg["RUN"])),
                RUN=cfg["RUN"],
                SNAME=cfg["SNAME"],
                SVARS=cfg["SVARS"],
                BPFILE=cfg["BPFILE"],
                ITYPE=cfg["ITYPE"],
                IFS=cfg["IFS"],
                STACKS=cfg.get("STACKS"),
                NSPOOL=cfg["NSPOOL"],
                MDT=cfg["MDT"],
                RVARS=cfg["RVARS"],
                PRJ=cfg["PRJ"],
            )
        )
        return specs

    for i, item in enumerate(runs):
        if isinstance(item, str):
            item = {"RUN": item}
        if not isinstance(item, dict):
            raise ValueError(f"RUNS[{i}] must be dict or string")

        run = item.get("RUN", item.get("run", item.get("run_dir")))
        if run is None:
            raise ValueError(f"RUNS[{i}] missing RUN/run_dir")

        name = item.get("NAME", item.get("name", os.path.basename(os.path.abspath(run))))
        sname = item.get("SNAME", item.get("sname"))
        if sname is None:
            template = cfg.get("SNAME_TEMPLATE", "./npz/{run_name}")
            sname = str(template).format(run_name=name, run=run)

        specs.append(
            dict(
                NAME=name,
                RUN=run,
                SNAME=sname,
                SVARS=item.get("SVARS", cfg["SVARS"]),
                BPFILE=item.get("BPFILE", cfg["BPFILE"]),
                ITYPE=item.get("ITYPE", cfg["ITYPE"]),
                IFS=item.get("IFS", cfg["IFS"]),
                STACKS=item.get("STACKS", cfg.get("STACKS")),
                NSPOOL=item.get("NSPOOL", cfg["NSPOOL"]),
                MDT=item.get("MDT", cfg["MDT"]),
                RVARS=item.get("RVARS", cfg["RVARS"]),
                PRJ=item.get("PRJ", cfg["PRJ"]),
            )
        )

    return specs


def _normalize_chunk_2d(arr, tvec, expected_nsta=None):
    a = array(arr)  # noqa: F403
    t = array(tvec, dtype="float64").ravel()  # noqa: F403
    nt = int(len(t))

    if a.ndim == 0:
        a = a.reshape(1, 1)
    elif a.ndim > 2:
        a = squeeze(a)  # noqa: F403
        if a.ndim > 2:
            a = a.reshape(a.shape[0], -1)

    if a.ndim == 1:
        # Most common collapse: single-station chunk saved as (nt,)
        if nt > 0 and a.size == nt:
            a = a.reshape(1, nt)
        # Single-time chunk saved as (nsta,)
        elif expected_nsta is not None and expected_nsta > 0 and a.size == int(expected_nsta):
            a = a.reshape(int(expected_nsta), 1)
        elif nt == 1:
            a = a.reshape(-1, 1)
        else:
            a = a.reshape(1, -1)

    # Keep time on axis=1 so stacks can be concatenated by axis=1.
    if nt > 0 and a.shape[1] != nt and a.shape[0] == nt:
        a = a.T

    if nt > 0 and a.shape[1] != nt:
        raise ValueError(f"cannot align chunk shape={a.shape} with nt={nt}")

    return a.astype("float32"), t


def _process_one_run(spec, cfg):
    run = spec["RUN"]
    svars = tuple(spec["SVARS"])
    bpfile = spec["BPFILE"]
    sname = spec["SNAME"]
    itype = spec["ITYPE"]
    ifs = spec["IFS"]
    stacks = spec["STACKS"]
    nspool = spec["NSPOOL"]
    mdt = spec["MDT"]
    rvars = spec["RVARS"] if spec["RVARS"] is not None else svars
    prj = spec["PRJ"]

    if len(rvars) != len(svars):
        raise ValueError(f"RVARS must match SVARS length for run {run}")

    odir = os.path.dirname(os.path.abspath(sname))
    if RANK == 0 and (not fexist(odir)):  # noqa: F403
        os.makedirs(odir, exist_ok=True)

    if RANK == 0:
        _log(f"RUN={run}")

    outputs_dir = os.path.join(run, "outputs")
    modules, outfmt, dstacks, dvars, dvars_2d = schout_info(outputs_dir, 1)  # noqa: F403
    _ = dvars_2d

    stack_candidates = _as_stack_list(stacks, dstacks)
    if RANK == 0 and cfg["VERBOSE"]:
        mode_desc = "AUTO" if stacks is None else "USER"
        _log(f"Stack mode={mode_desc}; candidates={len(stack_candidates)}")

    valid_stacks, _ = _screen_stacks(
        outputs_dir,
        stack_candidates,
        outfmt,
        mode=cfg.get("STACK_CHECK_MODE", "light"),
        check_all_files=bool(cfg.get("STACK_CHECK_ALL_FILES", False)),
        ratio_min=float(cfg.get("STACK_SIZE_RATIO_MIN", 0.70)),
        abs_min_bytes=cfg.get("STACK_SIZE_MIN_BYTES"),
        verbose=bool(cfg.get("VERBOSE", True)),
    )

    if len(valid_stacks) == 0:
        if RANK == 0:
            _log(f"No valid stacks for run {run}; skip.")
        return

    gd, vd = grd(run, fmt=2)  # noqa: F403
    gd.compute_bnd()

    irec = 0
    oname = os.path.join(odir, ".schout_" + os.path.basename(os.path.abspath(sname)))

    for svar in svars:
        ovars = schvar_info(svar, modules, fmt=outfmt)  # noqa: F403
        if len(ovars) == 0 or ovars[0][1] not in dvars:
            if RANK == 0 and cfg["VERBOSE"]:
                _log(f"Skip {svar}: not found in outputs")
            continue

        for istack in valid_stacks:
            irec += 1
            if (irec % SIZE) != RANK:
                continue

            fname = f"{oname}_{svar}_{int(istack)}"
            t00 = time.time()
            try:
                if mdt is not None:
                    read_schism_output(  # noqa: F403
                        run,
                        svar,
                        bpfile,
                        int(istack),
                        ifs,
                        nspool,
                        fname=fname,
                        hgrid=gd,
                        vgrid=vd,
                        fmt=itype,
                        prj=prj,
                        mdt=mdt,
                    )
                else:
                    read_schism_output(  # noqa: F403
                        run,
                        svar,
                        bpfile,
                        int(istack),
                        ifs,
                        nspool,
                        fname=fname,
                        hgrid=gd,
                        vgrid=vd,
                        fmt=itype,
                        prj=prj,
                    )
                dt = time.time() - t00
                if cfg["VERBOSE"]:
                    _log(f"Finished {svar}_{int(istack)} in {dt:.2f}s")
            except Exception as exc:
                _log(f"Failed {svar}_{int(istack)}: {exc}")

    if COMM is not None:
        COMM.Barrier()

    if RANK == 0:
        t0 = time.time()
        S = zdata()  # noqa: F403
        S.bp = read(bpfile)  # noqa: F403
        S.time = []
        S.run_dir = os.path.abspath(run)
        S.run_name = str(spec.get("NAME", os.path.basename(os.path.abspath(run))))
        S.used_stacks = array(valid_stacks, dtype=int)  # noqa: F403
        fnss = []

        for k, m in zip(svars, rvars):
            fns = [f"{oname}_{k}_{int(n)}.npz" for n in valid_stacks]
            fnss.extend(fns)
            nsta_guess = None
            try:
                if hasattr(S.bp, "x"):
                    nsta_guess = int(len(array(S.bp.x).ravel()))  # noqa: F403
            except Exception:
                nsta_guess = None

            data = []
            mtime = []
            for fn in fns:
                if not fexist(fn):  # noqa: F403
                    continue
                try:
                    ai = read(fn, k)  # noqa: F403
                    ti = read(fn, "time")  # noqa: F403
                    ai, ti = _normalize_chunk_2d(ai, ti, expected_nsta=nsta_guess)
                    data.append(ai)
                    mtime.append(ti)
                except Exception as exc:
                    _log(f"[WARN] Skip malformed chunk {os.path.basename(fn)} ({k}): {exc}")

            if len(data) > 0:
                nrow = int(data[0].shape[0])
                data_ok = []
                mtime_ok = []
                for ai, ti in zip(data, mtime):
                    if ai.shape[0] == nrow:
                        data_ok.append(ai)
                        mtime_ok.append(ti)
                    else:
                        _log(
                            "[WARN] Skip row-mismatched chunk "
                            f"for {k}: rows={ai.shape[0]} expected={nrow}"
                        )

                if len(data_ok) > 0:
                    S.attr(m, concatenate(data_ok, axis=1))  # noqa: F403
                    mtime_cat = concatenate(mtime_ok)  # noqa: F403
                    if len(mtime_cat) > len(S.time):
                        S.time = array(mtime_cat, dtype="float64")  # noqa: F403

        if cfg.get("APPLY_PARAM_START_TIME", True) and len(S.time) > 0:
            start_dn, info = _get_model_start_datenum(
                run,
                apply_utc_start=bool(cfg.get("APPLY_UTC_START", False)),
            )
            if start_dn is None:
                _log(f"[WARN] Cannot apply model start time for {run}: {info}")
            else:
                S.time = S.time + start_dn
                S.model_start_datenum = float(start_dn)
                S.model_start = num2date(start_dn).strftime("%Y-%m-%d %H:%M:%S")  # noqa: F403
                S.time_is_absolute = 1
                _log(f"Applied model start time: {info}")

        for pn in ["param", "icm", "sediment", "cosine", "wwminput"]:
            fn = f"{run}/{pn}.nml"
            if fexist(fn):
                S.attr(pn, read(fn, 3))  # noqa: F403

        S.save(sname)
        for fn in fnss:
            if fexist(fn):
                os.remove(fn)

        dt = time.time() - t0
        _log(f"Wrote {sname}")
        _log(f"Merge/save time: {dt:.2f}s")

    if COMM is not None:
        COMM.Barrier()


def main():
    run_specs = _normalize_run_specs(CONFIG)
    if len(run_specs) == 0:
        raise ValueError("No runs configured.")

    if RANK == 0:
        _log(f"Total runs to process: {len(run_specs)}")

    for i, spec in enumerate(run_specs, start=1):
        if RANK == 0:
            _log(f"---- Run {i}/{len(run_specs)}: {spec['NAME']} ----")
        _process_one_run(spec, CONFIG)

    if COMM is not None:
        COMM.Barrier()


if __name__ == "__main__":
    main()
