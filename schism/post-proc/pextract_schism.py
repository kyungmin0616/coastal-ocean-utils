#!/usr/bin/env python3
"""
Extract time series at stations or transects from SCHISM outputs.

Set CONFIG["SNAME"] to the output npz path.
This script writes a single npz file and does not submit batch jobs.
"""
from pylib import *  # noqa: F403
import os
import time

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
    RUN="../RUN01c",
    SVARS=("elev","temp","salt"),
    BPFILE="./station_sendai_d2.bp",
    SNAME="./npz/RUN01c_v1",
    ITYPE=0, # 0: time series of points @xyz; 1: transects @xy
    IFS=0, # 0: refer to free surface; 1: fixed depth
    STACKS=[1,1681], # e.g., [1, 5]
    NSPOOL=1, # sub-sampling within each stack
    MDT=None, # time window (day) for averaging output
    RVARS=None, # rename variables
    PRJ=None, # e.g., ["epsg:26918", "epsg:4326"]
    VERBOSE=True,
)


def _log(msg):
    prefix = f"[rank {RANK}/{SIZE}] " if SIZE > 1 else ""
    print(prefix + str(msg), flush=True)


def _as_stack_list(stacks, dstacks):
    if stacks is None:
        return dstacks
    if isinstance(stacks, (list, tuple)) and len(stacks) == 2:
        return arange(stacks[0], stacks[1] + 1)  # noqa: F403
    return array(stacks)  # noqa: F403


def main():
    run = CONFIG["RUN"]
    svars = CONFIG["SVARS"]
    bpfile = CONFIG["BPFILE"]
    sname = CONFIG["SNAME"]
    itype = CONFIG["ITYPE"]
    ifs = CONFIG["IFS"]
    stacks = CONFIG["STACKS"]
    nspool = CONFIG["NSPOOL"]
    mdt = CONFIG["MDT"]
    rvars = CONFIG["RVARS"] if CONFIG["RVARS"] is not None else svars
    prj = CONFIG["PRJ"]

    if len(rvars) != len(svars):
        raise ValueError("RVARS must match SVARS length")

    odir = os.path.dirname(os.path.abspath(sname))
    if RANK == 0 and (not fexist(odir)):  # noqa: F403
        os.makedirs(odir, exist_ok=True)

    if RANK == 0:
        t0 = time.time()
        _log(f"RUN={run}")

    modules, outfmt, dstacks, dvars, dvars_2d = schout_info(  # noqa: F403
        os.path.join(run, "outputs"), 1
    )
    stacks = _as_stack_list(stacks, dstacks)

    gd, vd = grd(run, fmt=2)  # noqa: F403
    gd.compute_bnd()

    irec = 0
    oname = os.path.join(odir, ".schout_" + os.path.basename(os.path.abspath(sname)))

    for svar in svars:
        ovars = schvar_info(svar, modules, fmt=outfmt)  # noqa: F403
        if ovars[0][1] not in dvars:
            if RANK == 0 and CONFIG["VERBOSE"]:
                _log(f"Skip {svar}: not found in outputs")
            continue
        for istack in stacks:
            irec += 1
            if (irec % SIZE) != RANK:
                continue
            fname = f"{oname}_{svar}_{istack}"
            t00 = time.time()
            try:
                if mdt is not None:
                    read_schism_output(  # noqa: F403
                        run, svar, bpfile, istack, ifs, nspool,
                        fname=fname, hgrid=gd, vgrid=vd, fmt=itype, prj=prj, mdt=mdt
                    )
                else:
                    read_schism_output(  # noqa: F403
                        run, svar, bpfile, istack, ifs, nspool,
                        fname=fname, hgrid=gd, vgrid=vd, fmt=itype, prj=prj
                    )
                dt = time.time() - t00
                if CONFIG["VERBOSE"]:
                    _log(f"Finished {svar}_{istack} in {dt:.2f}s")
            except Exception as e:
                _log(f"Failed {svar}_{istack}: {e}")

    if COMM is not None:
        COMM.Barrier()

    if RANK == 0:
        S = zdata()  # noqa: F403
        S.bp = read(bpfile)  # noqa: F403
        S.time = []
        fnss = []
        for k, m in zip(svars, rvars):
            fns = [f"{oname}_{k}_{n}.npz" for n in stacks]
            fnss.extend(fns)
            data = [read(fn, k).astype("float32") for fn in fns if fexist(fn)]  # noqa: F403
            mtime = [read(fn, "time") for fn in fns if fexist(fn)]  # noqa: F403
            if len(data) > 0:
                S.attr(m, concatenate(data, axis=1))  # noqa: F403
                mtime = concatenate(mtime)  # noqa: F403
            if len(mtime) > len(S.time):
                S.time = array(mtime)  # noqa: F403

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
        _log(f"Total time: {dt:.2f}s")

    if COMM is not None:
        COMM.Barrier()


if __name__ == "__main__":
    main()
