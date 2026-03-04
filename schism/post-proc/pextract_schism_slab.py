#!/usr/bin/env python3
"""
Extract SCHISM slab outputs.

This refactor keeps slab extraction behavior while standardizing runtime flow:
load_config -> normalize_run_specs -> process_run -> merge_and_save.
"""

# =============================================================================
# Configuration
# =============================================================================
CONFIG = dict(
    PATHS=dict(
        RUN="../../run/RUN12p",
        SNAME="./npz/RUN12p-1lv-hvel-temp.npz",
        RUNS=None,  # optional list of run specs
        SNAME_TEMPLATE="./npz/{run_name}-slab.npz",
    ),
    RUN_CONTROL=dict(),
    VARIABLES=dict(
        SVARS=("hvel", "temp"),
        RVARS=None,
        LEVELS=[1],  # 1..nvrt surface->bottom, (>nvrt) means kbp layer
    ),
    STACK_POLICY=dict(
        STACKS=[1, 120],
        STACK_CHECK_MODE="none",  # none | light | size | light+size
        STACK_CHECK_ALL_FILES=False,
        STACK_SIZE_RATIO_MIN=0.70,
        STACK_SIZE_MIN_BYTES=None,
    ),
    TIME_POLICY=dict(
        NSPOOL=1,
        MDT=None,
        APPLY_PARAM_START_TIME=False,
        APPLY_UTC_START=False,
    ),
    MPI_HPC=dict(
        SUBMIT=False,
        WALLTIME="00:30:00",
        QNODE="frontera",
        NNODE=1,
        PPN=56,
        QNAME="development",
        ACCOUNT="OCE22003",
        JOB_PREFIX="Rd_",
        SCR_OUT_TEMPLATE="screen_{run_name}.out",
    ),
    OUTPUT=dict(
        VERBOSE=True,
        DRY_RUN=False,
        CLEAN_CHUNKS=True,
        MANIFEST=None,  # optional JSON summary path
    ),
)

# =============================================================================
# Imports
# =============================================================================
import argparse
import os
import sys
import time
import json
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from pylib import *  # noqa: F403


# =============================================================================
# Shared Utilities / MPI Runtime
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
        init_mpi_runtime,
        rank_log,
        normalize_stack_list,
        screen_stacks as common_screen_stacks,
        normalize_run_specs as common_normalize_run_specs,
        get_model_start_datenum as common_get_model_start_datenum,
        deep_update_dict,
    )
except Exception as exc:
    missing = [
        name
        for name in (
            "init_mpi_runtime",
            "rank_log",
            "normalize_stack_list",
            "common_screen_stacks",
            "common_normalize_run_specs",
            "common_get_model_start_datenum",
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
def _log(msg, all_ranks=False):
    rank_log(str(msg), rank=RANK, size=SIZE, rank0_only=(not bool(all_ranks)))


def _parse_args(argv=None):
    p = argparse.ArgumentParser(description="Extract SCHISM slab outputs.")
    p.add_argument("--config", help="Optional JSON config overrides.")
    p.add_argument("--run", help="Override PATHS.RUN and force single-run mode.")
    p.add_argument("--sname", help="Override PATHS.SNAME for single-run mode.")
    p.add_argument("--manifest", help="Optional output JSON summary path.")
    p.add_argument("--vars", nargs="+", help="Override VARIABLES.SVARS.")
    p.add_argument("--rvars", nargs="+", help="Override VARIABLES.RVARS.")
    p.add_argument("--levels", nargs="+", type=int, help="Override VARIABLES.LEVELS.")
    p.add_argument("--stacks", nargs="+", type=int, help="Override STACK_POLICY.STACKS.")
    p.add_argument("--nspool", type=int, help="Override TIME_POLICY.NSPOOL.")
    p.add_argument("--mdt", type=float, help="Override TIME_POLICY.MDT.")
    p.add_argument(
        "--stack-check-mode",
        choices=["none", "light", "size", "light+size"],
        help="Override STACK_POLICY.STACK_CHECK_MODE.",
    )
    p.add_argument("--dry-run", action="store_true", help="Resolve run/stacks and exit.")
    p.add_argument("--submit", dest="submit", action="store_true", help="Submit job to scheduler.")
    p.add_argument("--no-submit", dest="submit", action="store_false")
    p.add_argument("--verbose", dest="verbose", action="store_true")
    p.add_argument("--quiet", dest="verbose", action="store_false")
    p.set_defaults(submit=None, verbose=None)
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
    out = deep_update_dict(cfg, _load_json_config(args.config), merge_list_of_dicts=True)
    if args.run:
        out["PATHS"]["RUN"] = args.run
        out["PATHS"]["RUNS"] = None
    if args.sname:
        out["PATHS"]["SNAME"] = args.sname
    if args.vars:
        out["VARIABLES"]["SVARS"] = tuple(args.vars)
    if args.rvars:
        out["VARIABLES"]["RVARS"] = tuple(args.rvars)
    if args.levels:
        out["VARIABLES"]["LEVELS"] = [int(v) for v in args.levels]
    if args.stacks is not None:
        vals = [int(v) for v in args.stacks]
        out["STACK_POLICY"]["STACKS"] = vals if len(vals) != 2 else [vals[0], vals[1]]
    if args.nspool is not None:
        out["TIME_POLICY"]["NSPOOL"] = int(args.nspool)
    if args.mdt is not None:
        out["TIME_POLICY"]["MDT"] = float(args.mdt)
    if args.stack_check_mode:
        out["STACK_POLICY"]["STACK_CHECK_MODE"] = str(args.stack_check_mode)
    if args.dry_run:
        out["OUTPUT"]["DRY_RUN"] = True
    if args.manifest:
        out["OUTPUT"]["MANIFEST"] = str(args.manifest)
    if args.submit is not None:
        out["MPI_HPC"]["SUBMIT"] = bool(args.submit)
    if args.verbose is not None:
        out["OUTPUT"]["VERBOSE"] = bool(args.verbose)
    return out


def _flatten_cfg(cfg):
    flat = {}
    for section in ("PATHS", "RUN_CONTROL", "VARIABLES", "STACK_POLICY", "TIME_POLICY", "MPI_HPC", "OUTPUT"):
        flat.update(cfg.get(section, {}))
    return flat


def _validate_config(flat_cfg):
    if flat_cfg.get("RUNS") is None and flat_cfg.get("RUN") is None:
        raise ValueError("Either PATHS.RUNS or PATHS.RUN must be configured.")
    svars = tuple(flat_cfg.get("SVARS", ()))
    if len(svars) == 0:
        raise ValueError("VARIABLES.SVARS must not be empty.")
    levels = list(flat_cfg.get("LEVELS", []))
    if len(levels) == 0:
        raise ValueError("VARIABLES.LEVELS must not be empty.")
    rvars = flat_cfg.get("RVARS")
    if rvars is not None and len(tuple(rvars)) != len(svars):
        raise ValueError("VARIABLES.RVARS must have the same length as VARIABLES.SVARS.")
    mode = str(flat_cfg.get("STACK_CHECK_MODE", "none")).lower()
    if mode not in {"none", "light", "size", "light+size"}:
        raise ValueError(f"Invalid STACK_CHECK_MODE: {mode}")


def _normalize_run_specs(flat_cfg):
    return common_normalize_run_specs(
        flat_cfg,
        run_keys=("RUN", "run", "run_dir"),
        name_keys=("NAME", "name", "RUN_NAME"),
        output_keys=("SNAME", "sname", "out_npz"),
        output_template_key="SNAME_TEMPLATE",
        default_output_template="./npz/{run_name}-slab.npz",
        include_keys=("SVARS", "RVARS", "LEVELS", "STACKS", "NSPOOL", "MDT"),
    )


def _get_schout_info(outputs_dir):
    if "schout_info" in globals():
        return schout_info(outputs_dir, 1)  # noqa: F405
    return get_schism_output_info(outputs_dir, 1)  # noqa: F405


def _get_schvar_info(svar, modules, outfmt):
    if "schvar_info" in globals():
        return schvar_info(svar, modules, fmt=outfmt)  # noqa: F405
    return get_schism_var_info(svar, modules, fmt=outfmt)  # noqa: F405


def _screen_valid_stacks(outputs_dir, stack_candidates, outfmt, flat_cfg):
    return common_screen_stacks(
        outputs_dir=outputs_dir,
        stacks=stack_candidates,
        outfmt=outfmt,
        mode=flat_cfg.get("STACK_CHECK_MODE", "none"),
        check_all_files=bool(flat_cfg.get("STACK_CHECK_ALL_FILES", False)),
        ratio_min=float(flat_cfg.get("STACK_SIZE_RATIO_MIN", 0.70)),
        abs_min_bytes=flat_cfg.get("STACK_SIZE_MIN_BYTES"),
        readnc=lambda p: ReadNC(p, 1),  # noqa: F403
        logger=(_log if (RANK == 0 and bool(flat_cfg.get("VERBOSE", True))) else None),
        log_limit=20,
    )


def _as_list_of_vectors(values):
    arr = np.asarray(values)
    if arr.ndim == 0:
        return [arr.reshape(1)]
    if arr.ndim == 1:
        return [arr]
    return [arr[i, ...] for i in range(arr.shape[0])]


def _maybe_submit_job(flat_cfg, run_specs):
    if not bool(flat_cfg.get("SUBMIT", False)):
        return False
    if os.getenv("job_on_node") is not None:
        return False

    run_name = run_specs[0]["NAME"] if len(run_specs) > 0 else "slab"
    bdir = os.path.abspath(os.curdir)
    bcode = sys.argv[0]
    jname = f"{flat_cfg.get('JOB_PREFIX', 'Rd_')}{run_name}"
    scrout = str(flat_cfg.get("SCR_OUT_TEMPLATE", "screen_{run_name}.out")).format(run_name=run_name)
    scode = get_hpc_command(  # noqa: F405
        bcode,
        bdir,
        jname,
        flat_cfg.get("QNODE", "frontera"),
        int(flat_cfg.get("NNODE", 1)),
        int(flat_cfg.get("PPN", 56)),
        flat_cfg.get("WALLTIME", "00:30:00"),
        scrout,
        fmt=0,
        qname=flat_cfg.get("QNAME", "development"),
    )
    print(scode, flush=True)
    os.system(scode)
    return True


def _dry_run_report(flat_cfg, run_specs):
    for spec in run_specs:
        run = spec["RUN"]
        outputs_dir = os.path.join(run, "outputs")
        if not os.path.isdir(outputs_dir):
            _log(f"[DRY-RUN] {spec['NAME']}: missing outputs dir -> {outputs_dir}", all_ranks=False)
            continue
        try:
            modules, outfmt, dstacks, dvars, dvars_2d = _get_schout_info(outputs_dir)
            _ = modules
            _ = dvars
            _ = dvars_2d
        except Exception as exc:
            _log(f"[DRY-RUN] {spec['NAME']}: schout info failed: {exc}", all_ranks=False)
            continue

        stack_candidates = normalize_stack_list(spec.get("STACKS"), dstacks)
        valid_stacks, skipped = _screen_valid_stacks(outputs_dir, stack_candidates, outfmt, flat_cfg)
        _log(
            f"[DRY-RUN] {spec['NAME']}: run={run}, out={spec['SNAME']}, vars={tuple(spec['SVARS'])}, "
            f"levels={list(spec['LEVELS'])}, candidates={len(stack_candidates)}, "
            f"valid={len(valid_stacks)}, skipped={len(skipped)}",
            all_ranks=False,
        )


def _write_manifest(flat_cfg, run_specs, run_summaries):
    mpath = flat_cfg.get("MANIFEST")
    if not mpath:
        return
    mpath = os.path.abspath(str(mpath))
    mdir = os.path.dirname(mpath)
    if mdir and (not os.path.isdir(mdir)):
        os.makedirs(mdir, exist_ok=True)
    payload = dict(
        script=os.path.abspath(__file__),
        dry_run=bool(flat_cfg.get("DRY_RUN", False)),
        run_count=int(len(run_specs)),
        runs=run_summaries,
    )
    with open(mpath, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    _log(f"Wrote manifest: {mpath}", all_ranks=False)


def _process_one_run(spec, flat_cfg):
    run = spec["RUN"]
    sname = spec["SNAME"]
    svars = tuple(spec["SVARS"])
    rvars = tuple(spec["RVARS"]) if spec.get("RVARS") is not None else svars
    levels = [int(v) for v in spec["LEVELS"]]
    nspool = int(spec.get("NSPOOL", 1))
    mdt = spec.get("MDT")
    run_name = str(spec.get("NAME", os.path.basename(os.path.abspath(run))))

    odir = os.path.dirname(os.path.abspath(sname))
    if RANK == 0 and not os.path.isdir(odir):
        os.makedirs(odir, exist_ok=True)

    outputs_dir = os.path.join(run, "outputs")
    modules, outfmt, dstacks, dvars, _ = _get_schout_info(outputs_dir)
    stack_candidates = normalize_stack_list(spec.get("STACKS"), dstacks)
    valid_stacks, _ = _screen_valid_stacks(outputs_dir, stack_candidates, outfmt, flat_cfg)

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

    if RANK == 0 and bool(flat_cfg.get("VERBOSE", True)):
        _log(f"RUN={run}")
        _log(f"Valid stacks: {len(valid_stacks)}")

    irec = 0
    oname = os.path.join(odir, ".schout_" + os.path.basename(os.path.abspath(sname)))
    for svar in svars:
        ovars = _get_schvar_info(svar, modules, outfmt)
        if len(ovars) == 0 or ovars[0][1] not in dvars:
            continue
        for istack in valid_stacks:
            irec += 1
            if (irec % SIZE) != RANK:
                continue
            fname = f"{oname}_{svar}_{int(istack)}_slab"
            t0 = time.time()
            read_schism_slab(run, svar, levels, int(istack), nspool, mdt, fname=fname)  # noqa: F405
            if bool(flat_cfg.get("VERBOSE", True)):
                _log(
                    f"Finished {svar}_{int(istack)} on rank {RANK}: {time.time() - t0:.2f}s",
                    all_ranks=True,
                )

    if COMM is not None:
        COMM.Barrier()

    if RANK == 0:
        t0 = time.time()
        S = zdata()  # noqa: F405
        S.time = []
        S.run_dir = os.path.abspath(run)
        S.run_name = run_name
        S.used_stacks = np.asarray(valid_stacks, dtype=int)
        tmp_files = []
        extracted_vars = []

        for k, m in zip(svars, rvars):
            data = []
            mtime = []
            for istack in valid_stacks:
                fname = f"{oname}_{k}_{int(istack)}_slab.npz"
                if not os.path.exists(fname):
                    continue
                C = loadz(fname)  # noqa: F405
                if hasattr(C, k):
                    data.extend(_as_list_of_vectors(getattr(C, k)))
                if hasattr(C, "time"):
                    mtime.extend(np.asarray(C.time).ravel().tolist())
                tmp_files.append(fname)
            if len(data) > 0:
                S.attr(m, np.asarray(data))
                extracted_vars.append(str(m))
            if len(mtime) > len(S.time):
                S.time = np.asarray(mtime, dtype="float64")

        if bool(flat_cfg.get("APPLY_PARAM_START_TIME", False)) and len(S.time) > 0:
            start_dn, info = common_get_model_start_datenum(
                run,
                apply_utc_start=bool(flat_cfg.get("APPLY_UTC_START", False)),
                read_schism_param_func=read_schism_param,  # noqa: F405
                datenum_func=datenum,  # noqa: F405
            )
            if start_dn is not None:
                S.time = S.time + float(start_dn)
                S.model_start_datenum = float(start_dn)
                _log(f"Applied model start time: {info}")
            else:
                _log(f"[WARN] Cannot apply model start time: {info}")

        savez(sname, S)  # noqa: F405
        if bool(flat_cfg.get("CLEAN_CHUNKS", True)):
            for fn in tmp_files:
                if os.path.exists(fn):
                    os.remove(fn)
        _log(f"Wrote {sname}")
        _log(f"Merge/save time: {time.time() - t0:.2f}s")
        return dict(
            run_name=run_name,
            run_dir=os.path.abspath(run),
            output=sname,
            status="written",
            valid_stacks=int(len(valid_stacks)),
            variables=sorted(set(extracted_vars)),
        )

    if COMM is not None:
        COMM.Barrier()
    return None


def main():
    args = _parse_args()
    cfg = _apply_cli(CONFIG, args)
    flat_cfg = _flatten_cfg(cfg)
    _validate_config(flat_cfg)
    run_specs = _normalize_run_specs(flat_cfg)

    if len(run_specs) == 0:
        raise ValueError("No runs configured.")

    if _maybe_submit_job(flat_cfg, run_specs):
        return

    if RANK == 0:
        _log(f"Total runs to process: {len(run_specs)}")

    if bool(flat_cfg.get("DRY_RUN", False)):
        _dry_run_report(flat_cfg, run_specs)
        if RANK == 0:
            _write_manifest(
                flat_cfg,
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
        rs = _process_one_run(spec, flat_cfg)
        if RANK == 0 and isinstance(rs, dict):
            run_summaries.append(rs)

    if COMM is not None:
        COMM.Barrier()

    if RANK == 0:
        _write_manifest(flat_cfg, run_specs, run_summaries)


if __name__ == "__main__":
    main()
