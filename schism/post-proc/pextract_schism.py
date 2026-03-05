#!/usr/bin/env python3
"""
Extract time series at stations or transects from SCHISM outputs.

Features:
1) Multi-run extraction.
2) Optional model-start time offset from run/param.nml.
3) Auto stack discovery with lightweight incomplete-file screening.
"""

# =============================================================================
# Configuration
# =============================================================================
CONFIG = dict(
    # Multi-run mode
    RUNS=[
         {"NAME": "RUN01d", "RUN": "../run/RUN01d"},
         {"NAME": "RUN04a", "RUN": "../run/RUN04a"},
         {"NAME": "RUN01g", "RUN": "../run/RUN01g"},
     ],
 
    SNAME_TEMPLATE="./npz/{run_name}_test",

    SVARS=("temp", "salt"),
    BPFILE="./stations/station_sendai_d2.bp",
    ITYPE=0,  # 0: time series of points @xyz; 1: transects @xy
    IFS=0,  # 0: refer to free surface; 1: fixed depth
    STACKS=None,  # None -> all available stacks after screening
    NSPOOL=1,  # sub-sampling within each stack
    MDT=None,  # time window (day) for averaging output
    RVARS=None,  # rename variables
    PRJ=None,  # e.g. ["epsg:26918", "epsg:4326"]

    # Stack screening controls
    STACK_CHECK_MODE="size",  # none | light | size | light+size
    STACK_CHECK_ALL_FILES=False,  # False: check primary stack file only
    STACK_SIZE_RATIO_MIN=0.80,  # for size/light+size: min size ratio against median primary size
    STACK_SIZE_MIN_BYTES=None,  # optional absolute size floor in bytes

    # Time offset controls
    APPLY_PARAM_START_TIME=True,
    APPLY_UTC_START=False,  # If True, shift by -utc_start/24 from param.nml

    VERBOSE=True,
    DRY_RUN=False,
    MANIFEST=None,  # optional JSON summary path
    ASSIGNMENT_SYNC=False,  # True: gather+barrier during assignment report
    LOG_TASK_START=False,  # print "Start <var>_<stack>" before each read
    PHASE_TIMING=False,  # print per-phase elapsed times
    PHASE_TIMING_RANK0_ONLY=True,  # False: print phase timings on all ranks
)

# =============================================================================
# Imports
# =============================================================================
import os
import sys
import time
import argparse
import json
from pathlib import Path

from pylib import *  # noqa: F403


# =============================================================================
# Shared Utilities / MPI Runtime
# =============================================================================
_COMMON_CANDIDATE_DIRS = []
_env_pylibs_src = os.environ.get("PYLIBS_SRC")
if _env_pylibs_src:
    _COMMON_CANDIDATE_DIRS.append(Path(_env_pylibs_src).expanduser())
try:
    # Typical layout: .../Codes/coastal-ocean-utils/schism/post-proc/script.py
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
        init_mpi_runtime,
        rank_log,
        report_work_assignment,
        normalize_stack_list,
        screen_stacks as common_screen_stacks,
        get_model_start_datenum as common_get_model_start_datenum,
        normalize_run_specs as common_normalize_run_specs,
        deep_update_dict,
    )
except Exception as exc:
    missing = [
        name
        for name in (
            "init_mpi_runtime",
            "rank_log",
            "report_work_assignment",
            "normalize_stack_list",
            "common_screen_stacks",
            "common_get_model_start_datenum",
            "common_normalize_run_specs",
            "deep_update_dict",
        )
        if name not in globals()
    ]
    if missing:
        raise ImportError(
            "Shared helpers not found. Set PYLIBS_SRC to pylibs/src or install "
            f"postproc_common. Missing: {', '.join(missing)}"
        ) from exc

MPI, COMM, RANK, SIZE, USE_MPI = init_mpi_runtime(sys.argv)

# =============================================================================
# Core Helpers
# =============================================================================
def _log(msg, rank0_only=False):
    rank_log(str(msg), rank=RANK, size=SIZE, rank0_only=rank0_only)


def _report_extract_assignment(nvars, nstacks, sync=False):
    total_count = int(nvars) * int(nstacks)
    local_indices = [i for i in range(total_count) if ((i + 1) % SIZE) == RANK]
    report_work_assignment(
        tag="extract",
        total_count=total_count,
        local_indices=local_indices,
        rank=RANK,
        size=SIZE,
        comm=COMM,
        mpi_enabled=bool(USE_MPI) and bool(sync),
        logger=_log,
    )


def _as_stack_list(stacks, dstacks):
    return normalize_stack_list(stacks, dstacks)


def _get_model_start_datenum(run, apply_utc_start=False):
    return common_get_model_start_datenum(
        run,
        apply_utc_start=bool(apply_utc_start),
        read_schism_param_func=read_schism_param,  # noqa: F405
        datenum_func=datenum,  # noqa: F405
    )


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
    logger = _log if (verbose and RANK == 0) else None
    return common_screen_stacks(
        outputs_dir=outputs_dir,
        stacks=stacks,
        outfmt=outfmt,
        mode=mode,
        check_all_files=check_all_files,
        ratio_min=ratio_min,
        abs_min_bytes=abs_min_bytes,
        readnc=lambda p: ReadNC(p, 1),  # noqa: F403
        logger=logger,
        log_limit=20,
    )


def _normalize_run_specs(cfg):
    return common_normalize_run_specs(
        cfg,
        run_keys=("RUN", "run", "run_dir"),
        name_keys=("NAME", "name"),
        output_keys=("SNAME", "sname"),
        output_template_key="SNAME_TEMPLATE",
        default_output_template="./npz/{run_name}_SB_D2",
        include_keys=("SVARS", "BPFILE", "ITYPE", "IFS", "STACKS", "NSPOOL", "MDT", "RVARS", "PRJ"),
    )


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


def _phase_log(cfg, label, t0):
    if not bool(cfg.get("PHASE_TIMING", False)):
        return
    rank0_only = bool(cfg.get("PHASE_TIMING_RANK0_ONLY", True))
    _log(f"[TIMER] {label}: {time.time() - float(t0):.2f}s", rank0_only=rank0_only)


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
    run_name = str(spec.get("NAME", os.path.basename(os.path.abspath(run))))

    if len(rvars) != len(svars):
        raise ValueError(f"RVARS must match SVARS length for run {run}")

    odir = os.path.dirname(os.path.abspath(sname))
    if RANK == 0 and (not fexist(odir)):  # noqa: F403
        os.makedirs(odir, exist_ok=True)

    if RANK == 0:
        _log(f"RUN={run}")

    outputs_dir = os.path.join(run, "outputs")
    t_phase = time.time()
    modules, outfmt, dstacks, dvars, dvars_2d = schout_info(outputs_dir, 1)  # noqa: F403
    _ = dvars_2d
    _phase_log(cfg, f"{run_name}: schout_info", t_phase)

    stack_candidates = _as_stack_list(stacks, dstacks)
    if RANK == 0 and cfg["VERBOSE"]:
        mode_desc = "AUTO" if stacks is None else "USER"
        _log(f"Stack mode={mode_desc}; candidates={len(stack_candidates)}")

    t_phase = time.time()
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
    _phase_log(cfg, f"{run_name}: stack_screen", t_phase)

    if len(valid_stacks) == 0:
        if RANK == 0:
            _log(f"No valid stacks for run {run}; skip.")
            return dict(
                run_name=run_name,
                run_dir=os.path.abspath(run),
                output=sname,
                status="skipped_no_valid_stacks",
                valid_stacks=0,
                variables=[],
            )
        return None

    if cfg.get("VERBOSE", True):
        _report_extract_assignment(
            len(svars),
            len(valid_stacks),
            sync=bool(cfg.get("ASSIGNMENT_SYNC", False)),
        )

    t_phase = time.time()
    gd, vd = grd(run, fmt=2)  # noqa: F403
    gd.compute_bnd()
    _phase_log(cfg, f"{run_name}: load_grid", t_phase)

    t_phase = time.time()
    local_tasks = 0
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

            local_tasks += 1
            fname = f"{oname}_{svar}_{int(istack)}"
            t00 = time.time()
            try:
                if cfg.get("LOG_TASK_START", False):
                    _log(f"Start {svar}_{int(istack)}")
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
    _phase_log(cfg, f"{run_name}: extraction(local_tasks={local_tasks})", t_phase)

    if COMM is not None:
        if cfg.get("PHASE_TIMING", False):
            _log("Entering post-extract barrier", rank0_only=bool(cfg.get("PHASE_TIMING_RANK0_ONLY", True)))
        COMM.Barrier()
        if cfg.get("PHASE_TIMING", False):
            _log("Passed post-extract barrier", rank0_only=bool(cfg.get("PHASE_TIMING_RANK0_ONLY", True)))

    if RANK == 0:
        t0 = time.time()
        S = zdata()  # noqa: F403
        S.bp = read(bpfile)  # noqa: F403
        S.time = []
        S.run_dir = os.path.abspath(run)
        S.run_name = run_name
        S.used_stacks = array(valid_stacks, dtype=int)  # noqa: F403
        fnss = []
        extracted_vars = []

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
                    extracted_vars.append(str(m))
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
        return dict(
            run_name=run_name,
            run_dir=os.path.abspath(run),
            output=sname,
            status="written",
            valid_stacks=int(len(valid_stacks)),
            variables=sorted(set(extracted_vars)),
        )

    if COMM is not None:
        if cfg.get("PHASE_TIMING", False):
            _log("Entering final-run barrier", rank0_only=bool(cfg.get("PHASE_TIMING_RANK0_ONLY", True)))
        COMM.Barrier()
        if cfg.get("PHASE_TIMING", False):
            _log("Passed final-run barrier", rank0_only=bool(cfg.get("PHASE_TIMING_RANK0_ONLY", True)))
    return None


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Extract SCHISM station/transect products.")
    p.add_argument("--config", help="Optional JSON config overrides.")
    p.add_argument("--manifest", help="Optional output JSON summary path.")
    p.add_argument("--vars", nargs="+", help="Override CONFIG['SVARS'].")
    p.add_argument(
        "--stacks",
        nargs="+",
        type=int,
        help="Override stack selection: pass 2 ints for [start end], or explicit stack list.",
    )
    p.add_argument(
        "--stack-check-mode",
        choices=["none", "light", "size", "light+size"],
        help="Override CONFIG['STACK_CHECK_MODE'].",
    )
    p.add_argument("--dry-run", action="store_true", help="Resolve runs/stacks/vars and exit.")
    p.add_argument("--verbose", dest="verbose", action="store_true")
    p.add_argument("--quiet", dest="verbose", action="store_false")
    p.add_argument("--apply-param-start-time", dest="apply_param_start_time", action="store_true")
    p.add_argument("--no-apply-param-start-time", dest="apply_param_start_time", action="store_false")
    p.add_argument("--apply-utc-start", dest="apply_utc_start", action="store_true")
    p.add_argument("--no-apply-utc-start", dest="apply_utc_start", action="store_false")
    p.add_argument("--assignment-sync", action="store_true", help="Enable assignment gather/barrier report.")
    p.add_argument("--log-task-start", action="store_true", help="Log start line before each stack read.")
    p.add_argument("--phase-timing", action="store_true", help="Enable coarse per-phase timing logs.")
    p.add_argument(
        "--phase-timing-all-ranks",
        action="store_true",
        help="When --phase-timing is enabled, print timing logs on all ranks.",
    )
    p.set_defaults(verbose=None, apply_param_start_time=None, apply_utc_start=None)
    return p.parse_args(argv)


def _load_json_config(config_path):
    if config_path is None:
        return {}
    with open(str(config_path), "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"--config must contain a JSON object: {config_path}")
    return obj


def _apply_cli(cfg, args):
    out = deep_update_dict(cfg, _load_json_config(args.config), merge_list_of_dicts=False)
    if args.vars:
        out["SVARS"] = tuple(args.vars)
    if args.stacks is not None:
        vals = [int(v) for v in args.stacks]
        out["STACKS"] = vals if len(vals) != 2 else [vals[0], vals[1]]
    if args.stack_check_mode:
        out["STACK_CHECK_MODE"] = str(args.stack_check_mode)
    if args.dry_run:
        out["DRY_RUN"] = True
    if args.manifest:
        out["MANIFEST"] = str(args.manifest)
    if args.verbose is not None:
        out["VERBOSE"] = bool(args.verbose)
    if args.apply_param_start_time is not None:
        out["APPLY_PARAM_START_TIME"] = bool(args.apply_param_start_time)
    if args.apply_utc_start is not None:
        out["APPLY_UTC_START"] = bool(args.apply_utc_start)
    if args.assignment_sync:
        out["ASSIGNMENT_SYNC"] = True
    if args.log_task_start:
        out["LOG_TASK_START"] = True
    if args.phase_timing:
        out["PHASE_TIMING"] = True
    if args.phase_timing_all_ranks:
        out["PHASE_TIMING"] = True
        out["PHASE_TIMING_RANK0_ONLY"] = False
    return out


def _validate_config(cfg):
    runs = cfg.get("RUNS")
    if not isinstance(runs, (list, tuple)) or len(runs) == 0:
        raise ValueError("RUNS must be a non-empty list of run specs.")
    svars = tuple(cfg.get("SVARS", ()))
    if len(svars) == 0:
        raise ValueError("SVARS must not be empty.")
    if cfg.get("RVARS") is not None and len(tuple(cfg.get("RVARS"))) != len(svars):
        raise ValueError("RVARS must have the same length as SVARS.")
    mode = str(cfg.get("STACK_CHECK_MODE", "light")).lower()
    if mode not in {"none", "light", "size", "light+size"}:
        raise ValueError(f"Invalid STACK_CHECK_MODE: {mode}")


def _dry_run_report(run_specs, cfg):
    for spec in run_specs:
        run = spec["RUN"]
        svars = tuple(spec["SVARS"])
        outputs_dir = os.path.join(run, "outputs")
        if not os.path.isdir(outputs_dir):
            _log(f"[DRY-RUN] {spec['NAME']}: missing outputs dir -> {outputs_dir}", rank0_only=True)
            continue

        try:
            modules, outfmt, dstacks, dvars, dvars_2d = schout_info(outputs_dir, 1)  # noqa: F403
            _ = modules
            _ = dvars_2d
        except Exception as exc:
            _log(f"[DRY-RUN] {spec['NAME']}: schout_info failed: {exc}", rank0_only=True)
            continue

        cand = _as_stack_list(spec.get("STACKS"), dstacks)
        valid, skipped = _screen_stacks(
            outputs_dir=outputs_dir,
            stacks=cand,
            outfmt=outfmt,
            mode=cfg.get("STACK_CHECK_MODE", "light"),
            check_all_files=bool(cfg.get("STACK_CHECK_ALL_FILES", False)),
            ratio_min=float(cfg.get("STACK_SIZE_RATIO_MIN", 0.70)),
            abs_min_bytes=cfg.get("STACK_SIZE_MIN_BYTES"),
            verbose=False,
        )
        present = [sv for sv in svars if len(schvar_info(sv, modules, fmt=outfmt)) > 0]  # noqa: F403
        _log(
            f"[DRY-RUN] {spec['NAME']}: run={run}, out={spec['SNAME']}, "
            f"vars={svars}, vars_present={present}, candidates={len(cand)}, "
            f"valid={len(valid)}, skipped={len(skipped)}",
            rank0_only=True,
        )


def _write_manifest(cfg, run_specs, run_summaries):
    mpath = cfg.get("MANIFEST")
    if not mpath:
        return
    mpath = os.path.abspath(str(mpath))
    mdir = os.path.dirname(mpath)
    if mdir and (not os.path.isdir(mdir)):
        os.makedirs(mdir, exist_ok=True)
    payload = dict(
        script=os.path.abspath(__file__),
        dry_run=bool(cfg.get("DRY_RUN", False)),
        run_count=int(len(run_specs)),
        runs=run_summaries,
    )
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _log(f"Wrote manifest: {mpath}", rank0_only=True)


def main():
    args = _parse_args()
    cfg = _apply_cli(CONFIG, args)
    _validate_config(cfg)

    run_specs = _normalize_run_specs(cfg)
    if len(run_specs) == 0:
        raise ValueError("No runs configured.")

    if RANK == 0:
        _log(f"Total runs to process: {len(run_specs)}")

    if cfg.get("DRY_RUN", False):
        _dry_run_report(run_specs, cfg)
        if RANK == 0:
            _write_manifest(
                cfg,
                run_specs,
                [
                    dict(
                        run_name=str(spec.get("NAME", "")),
                        run_dir=os.path.abspath(str(spec.get("RUN", ""))),
                        output=str(spec.get("SNAME", "")),
                        status="dry_run",
                    )
                    for spec in run_specs
                ],
            )
        return

    run_summaries = []
    for i, spec in enumerate(run_specs, start=1):
        if RANK == 0:
            _log(f"---- Run {i}/{len(run_specs)}: {spec['NAME']} ----")
        rs = _process_one_run(spec, cfg)
        if RANK == 0 and isinstance(rs, dict):
            run_summaries.append(rs)

    if COMM is not None:
        COMM.Barrier()

    if RANK == 0:
        _write_manifest(cfg, run_specs, run_summaries)


if __name__ == "__main__":
    main()
