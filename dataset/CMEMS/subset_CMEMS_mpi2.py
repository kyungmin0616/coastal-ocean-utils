#!/usr/bin/env python
# subset_CMEMS_mpi.py
# MPI-parallel CMEMS extractor with robust filename parsing, progress prints,
# and variable restriction to: zos, so, thetao, uo, vo.

from pylib import *  # provides datenum, num2date, etc.
import os, re, subprocess, builtins, time
import numpy as np
from mpi4py import MPI

# ---------------- user params ----------------
dir_data     = './global'
new_dir_data = './Japan'
StartT       = datenum(1993, 1, 1)
EndT         = datenum(2024, 12, 31)

# subdomains:
# subdm=[-75.97,-67.62,38.49,42.777] # NJ/NY
# subdm=[-62,16,-12,56]              # NAO
# subdm=[-82,-76,23,29]              # FC
# subdm=[-67.2,-55,6.2,47.5]         # ECGOM
# subdm=[-163,-146,16,26]            # HAWAII
#subdm        = [80, 180, -40, 65]    # EastAsia
subdm        = [120, 160, 20, 50]    # Japan
#subdm       = [85, 300, -40, 75]     # Pacific

# Behavior toggles
OVERWRITE    = False     # if True, overwrite existing outputs
QUIET_CDO    = True      # pass -s to cdo (quiet)
# ------------------------------------------------

# MPI init
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Regex helpers
_re_cmems = re.compile(r'^cmems_(\d{4})_(\d{2})_(\d{2})_')
_re_merc  = re.compile(r'mean_(\d{8})_R(\d{8})\.nc$')

def ymd_from_filename(fname):
    """Return (Y, M, D) from supported filenames; raise if not found."""
    m = _re_cmems.match(fname)
    if m:
        return tuple(map(int, m.groups()))
    m = _re_merc.search(fname)
    if m:
        ymd = m.group(1)
        return int(ymd[:4]), int(ymd[4:6]), int(ymd[6:8])
    raise ValueError(f"No recognizable date in: {fname}")

def discover_and_plan():
    """Rank 0 scans filesystem, parses times, dedupes, filters, sorts."""
    fnames, roots = [], []
    for root, dirs, files in os.walk(dir_data):
        for name in files:
            if name.endswith(".nc"):
                roots.append(root)
                fnames.append(name)

    fnames = np.array(fnames, dtype=object)
    roots  = np.array(roots,  dtype=object)

    mti_list, keep_idx, bad_names = [], [], []
    for i, nm in enumerate(fnames):
        try:
            y, m, d = ymd_from_filename(nm)
            mti_list.append(datenum(y, m, d))
            keep_idx.append(i)
        except Exception:
            bad_names.append(nm)

    if bad_names:
        print(f"[rank 0] info: skipped {len(bad_names)} unrecognized filename(s)")

    mti     = np.array(mti_list)
    fnames  = fnames[keep_idx]
    roots   = roots[keep_idx]

    # Unique by time (keep first)
    mti, fpt = np.unique(mti, return_index=True)
    fnames   = fnames[fpt]
    roots    = roots[fpt]

    # Time window (inclusive bounds)
    mask   = (mti >= (StartT - 1)) & (mti < (EndT + 1))
    mti    = mti[mask]
    fnames = fnames[mask]
    roots  = roots[mask]

    # Sort by time
    sind   = np.argsort(mti)
    mti    = mti[sind]
    fnames = fnames[sind]
    roots  = roots[sind]

    return mti, fnames, roots

# Rank 0: plan; others receive
if rank == 0:
    print(f"[rank 0] starting with {size} rank(s)")
    mti, fnames, roots = discover_and_plan()
    total = len(fnames)
    print(f"[rank 0] discovered {total} file(s) after filtering {num2date(StartT).date()}–{num2date(EndT).date()}")
    os.makedirs(new_dir_data, exist_ok=True)
else:
    mti = fnames = roots = None

# Broadcast work arrays to all ranks
mti    = comm.bcast(mti,    root=0)
fnames = comm.bcast(fnames, root=0)
roots  = comm.bcast(roots,  root=0)
N      = 0 if mti is None else len(mti)

# All ranks ensure output dir exists (harmless if exists)
if rank != 0:
    os.makedirs(new_dir_data, exist_ok=True)
comm.Barrier()

# Announce per-rank ownership (striding)
owned = list(range(rank, N, size))
print(f"[rank {rank}] owns {len(owned)} of {N} file(s) via striding (every {size}-th starting at {rank})")

# Per-rank counters and timing
t0 = time.time()
done = skipped = failed = 0

# CDO flags
cdo_quiet = "-s " if QUIET_CDO else ""
cdo_overw = "-O " if OVERWRITE  else ""

for idx_count, nn in enumerate(owned, start=1):
    dt     = num2date(mti[nn])
    fname  = fnames[nn]
    src    = os.path.join(roots[nn], fname)
    nfname = f"cmems_{dt.year}_{dt.month:02d}_{dt.day:02d}_{dt.hour:02d}.nc"
    dst    = os.path.join(new_dir_data, nfname)

    # Progress line
    print(f"[rank {rank}] ({idx_count}/{len(owned)}) {fname} -> {nfname}")

    # Skip logic
    if os.path.isfile(dst) and os.path.getsize(dst) > 0 and not OVERWRITE:
        print(f"[rank {rank}] skip (exists): {dst}")
        skipped += 1
        continue
    # If overwriting, remove any existing file to avoid partial collisions
    if OVERWRITE and os.path.exists(dst):
        try: os.remove(dst)
        except Exception: pass

    # CDO: select variables + subdomain
    # Coordinates (lon,lat,depth,time) are retained automatically.
    #cmd = (
    #    f"cdo {cdo_quiet}{cdo_overw}"
    #    f"selname,zos,so,thetao,uo,vo "
    #    f"-sellonlatbox,{subdm[0]},{subdm[1]},{subdm[2]},{subdm[3]} "
    #    f"\"{src}\" \"{dst}\""
    #)

    cmd = (
        f"cdo -O -f nc4c -z zip_5 -b F32 "
        f"selname,zos,so,thetao,uo,vo "
        f"-sellonlatbox,{subdm[0]},{subdm[1]},{subdm[2]},{subdm[3]} "
        f"\"{src}\" \"{dst}\""
    )

    ret = subprocess.call(cmd, shell=True)
    if ret == 0 and os.path.isfile(dst) and os.path.getsize(dst) > 0:
        print(f"[rank {rank}] done: {dst}")
        done += 1
    else:
        print(f"[rank {rank}] FAILED: {dst} (ret={ret})")
        failed += 1
        # Clean partial file if present
        try:
            if os.path.exists(dst) and os.path.getsize(dst) == 0:
                os.remove(dst)
        except Exception:
            pass

t1 = time.time()
elapsed = t1 - t0

# Gather and summarize
totals = comm.gather((done, skipped, failed, elapsed), root=0)
comm.Barrier()

if rank == 0:
    Tdone = builtins.sum(t[0] for t in totals)
    Tskip = builtins.sum(t[1] for t in totals)
    Tfail = builtins.sum(t[2] for t in totals)
    Ttime = builtins.sum(t[3] for t in totals)
    print(
        f"[summary] ranks={size} files={N} done={Tdone} skipped={Tskip} failed={Tfail} "
        f"wall(sum over ranks)={Ttime:.1f}s"
    )
    print(f"[summary] outputs in: {new_dir_data}")
