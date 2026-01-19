#!/usr/bin/env python3
"""
Generate SCHISM VQS vgrid.in from hgrid.gr3.

Usage examples:
  python gen_vqs_ufs.py --hgrid hgrid.gr3
  python gen_vqs_ufs.py --hgrid hgrid.gr3 --transect
  python gen_vqs_ufs.py --hgrid hgrid.gr3 --transect transect_a.bp --transect transect_b.bp
  python gen_vqs_ufs.py --hgrid hgrid.gr3 --transect --plot-transect
  python gen_vqs_ufs.py --hgrid hgrid.gr3 --compare /path/to/fortran/outputs

Options:
  --hgrid      Path to hgrid.gr3 (default: hgrid.gr3)
  --transect   Enable transect output; repeatable or comma-separated list
  --compare    Directory holding Fortran outputs for numeric diff
  --plot-transect  Plot vgrids on transect using generated outputs
  --plot-prefix    Output prefix for plots (default: vgrid_transect)
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import numpy as np

from pylib import near_pts, read_schism_bpfile, read_schism_hgrid


def build_master_grid():
    m_vqs = 18
    n_sd = 18
    dz_bot_min = 0.1

    hsm = np.array(
        [
            1.0,
            2.0,
            3.0,
            4.0,
            6.0,
            8.0,
            12.0,
            18.0,
            25.0,
            33.0,
            42.0,
            52.0,
            67.0,
            83.0,
            100.0,
            150.0,
            230.0,
            350.0,
        ]
    )
    nv_vqs = np.array(
        [
            2,
            3,
            5,
            7,
            9,
            11,
            13,
            15,
            17,
            17,
            18,
            18,
            19,
            20,
            21,
            22,
            24,
            26,
        ],
        dtype=int,
    )

    if m_vqs < 2:
        raise ValueError(f"Check vgrid.in: m_vqs={m_vqs}")
    if hsm[0] < 0:
        raise ValueError("hsm[0] < 0")
    if np.any(hsm[1:] <= hsm[:-1]):
        raise ValueError("Check hsm: not strictly increasing")

    a_vqs0 = 0.0
    theta_b = 0.0
    etal = 0.0

    nvrt_m = nv_vqs[-1]
    print(f"nvrt in master vgrid={nvrt_m}")
    z_mas = np.full((nvrt_m, m_vqs), -1.0e5, dtype=float)

    for m in range(n_sd):
        m1 = m + 1
        if m1 <= 7:
            theta_f = 0.0001
        elif m1 <= 17:
            theta_f = min(1.0, max(0.0001, (m1 - 4) / 10.0)) * 3.0
        else:
            theta_f = 4.4
        if m1 == 14:
            theta_f -= 0.1
        if m1 == 15:
            theta_f += 0.1
        if m1 == 16:
            theta_f += 0.55
        if m1 == 17:
            theta_f += 0.97

        nvi = nv_vqs[m]
        k = np.arange(nvi, dtype=float)
        sigma = k / (1.0 - nvi)
        cs = (1 - theta_b) * np.sinh(theta_f * sigma) / np.sinh(theta_f) + theta_b * (
            np.tanh(theta_f * (sigma + 0.5)) - np.tanh(theta_f * 0.5)
        ) / (2.0 * np.tanh(theta_f * 0.5))
        z_mas[:nvi, m] = etal * (1.0 + sigma) + hsm[0] * sigma + (hsm[m] - hsm[0]) * cs

    # Force downward steps.
    for m in range(1, 6):
        n_prev = nv_vqs[m - 1]
        n_cur = nv_vqs[m]
        z_mas[:n_prev, m] = np.minimum(z_mas[:n_prev, m], z_mas[:n_prev, m - 1])
        tmp = (z_mas[n_cur - 1, m] - z_mas[n_prev - 1, m]) / (n_cur - n_prev)
        for k in range(n_prev, n_cur):
            z_mas[k, m] = z_mas[k - 1, m] + tmp

    return {
        "m_vqs": m_vqs,
        "n_sd": n_sd,
        "dz_bot_min": dz_bot_min,
        "hsm": hsm,
        "nv_vqs": nv_vqs,
        "a_vqs0": a_vqs0,
        "theta_b": theta_b,
        "etal": etal,
        "nvrt_m": nvrt_m,
        "z_mas": z_mas,
    }


def write_master_outputs(hsm, nv_vqs, z_mas, nvrt_m):
    with open("vgrid_master.out", "w", encoding="utf-8") as f:
        for m in range(len(hsm)):
            line = f" {m + 1:5d} {nv_vqs[m]:5d} {hsm[m]:12.4f}"
            line += "".join(f" {z_mas[k, m]:12.4f}" for k in range(nvrt_m))
            f.write(line + "\n")

    with open("fort.12", "w", encoding="utf-8") as f:
        for k in range(nvrt_m):
            line = f"{k + 1:5d}" + "".join(f" {z_mas[k, m]:12.4f}" for m in range(len(hsm)))
            f.write(line + "\n")


def compute_vgrid(gd, params):
    hsm = params["hsm"]
    nv_vqs = params["nv_vqs"]
    dz_bot_min = params["dz_bot_min"]
    a_vqs0 = params["a_vqs0"]
    etal = params["etal"]
    nvrt_m = params["nvrt_m"]
    z_mas = params["z_mas"]

    dp = gd.dp.copy()
    dpmax = dp.max()
    if dpmax > hsm[-1]:
        raise ValueError(f"Max depth exceeds master depth: {dpmax} > {hsm[-1]}")

    npnt = gd.np
    eta2 = np.full(npnt, etal, dtype=float)
    sigma_vqs = np.full((nvrt_m, npnt), -9.0, dtype=float)
    znd = np.full((nvrt_m, npnt), -1.0e6, dtype=float)
    kbp = np.zeros(npnt, dtype=int)
    m0 = np.zeros(npnt, dtype=int)

    debug_lines = []

    for i in range(npnt):
        if dp[i] <= hsm[0]:
            kbp[i] = nv_vqs[0]
            nvi = nv_vqs[0]
            for k in range(nvi):
                sigma = k / (1.0 - nvi)
                sigma_vqs[k, i] = a_vqs0 * sigma * sigma + (1.0 + a_vqs0) * sigma
                znd[k, i] = sigma_vqs[k, i] * (eta2[i] + dp[i]) + eta2[i]
            continue

        # Find a master vgrid.
        zrat = None
        for m in range(1, len(hsm)):
            if dp[i] > hsm[m - 1] and dp[i] <= hsm[m]:
                m0[i] = m
                zrat = (dp[i] - hsm[m - 1]) / (hsm[m] - hsm[m - 1])
                break
        if zrat is None:
            raise ValueError(f"Failed to find a master vgrid: node={i + 1}, depth={dp[i]}")

        kbp_i = 0
        for k in range(nv_vqs[m0[i]]):
            k_prev = min(k, nv_vqs[m0[i] - 1] - 1)
            z1 = z_mas[k_prev, m0[i] - 1]
            z2 = z_mas[k, m0[i]]
            z3 = z1 + (z2 - z1) * zrat
            if z3 >= -dp[i] + dz_bot_min:
                znd[k, i] = z3
            else:
                kbp_i = k + 1
                break
        if kbp_i == 0:
            msg = f"Failed to find a bottom: node={i + 1}, depth={dp[i]}, z3={z3}"
            raise ValueError(msg)

        kbp[i] = kbp_i
        znd[kbp_i - 1, i] = -dp[i]

        for k in range(1, kbp_i):
            if znd[k - 1, i] <= znd[k, i]:
                msg = (
                    f"Inverted z: node={i + 1}, depth={dp[i]}, m0={m0[i]}, "
                    f"k={k + 1}, z(k-1)={znd[k - 1, i]}, z(k)={znd[k, i]}"
                )
                raise ValueError(msg)

        debug_values = " ".join(f"{znd[k, i]:.6g}" for k in range(kbp_i))
        debug_lines.append(f"Node: {i + 1} {dp[i]:.6g} {debug_values}\n")

    for i in range(npnt):
        if kbp[i] < nvrt_m:
            znd[kbp[i] :, i] = -dp[i]

    return {
        "dp": dp,
        "eta2": eta2,
        "sigma_vqs": sigma_vqs,
        "znd": znd,
        "kbp": kbp,
        "debug_lines": debug_lines,
        "nvrt_m": nvrt_m,
    }


def write_debug_outputs(debug_lines):
    with open("fort.99", "w", encoding="utf-8") as f:
        f.writelines(debug_lines)


def write_transect(transect_path, gd, kbp, dp, znd, out_path):
    bp = read_schism_bpfile(transect_path)

    dist = np.zeros(bp.nsta, dtype=float)
    for i in range(1, bp.nsta):
        dx = bp.x[i] - bp.x[i - 1]
        dy = bp.y[i] - bp.y[i - 1]
        dist[i] = dist[i - 1] + np.hypot(dx, dy)

    sindp = near_pts(np.c_[bp.x, bp.y], np.c_[gd.x, gd.y])

    with open(out_path, "w", encoding="utf-8") as f:
        for i, nd in enumerate(sindp):
            parts = [
                f"{i + 1:6d}",
                f"{int(kbp[nd]):4d}",
                f"{bp.x[i]:16.6e}",
                f"{bp.y[i]:16.6e}",
            ]
            tail = [dist[i], dp[nd], *znd[:, nd]]
            parts.extend(f"{v:12.3f}" for v in tail)
            f.write(" ".join(parts) + "\n")


def write_vgrid_in(gd, params, vgrid_data):
    hsm = params["hsm"]
    nvrt = int(vgrid_data["kbp"].max())
    sigma_vqs = vgrid_data["sigma_vqs"]
    znd = vgrid_data["znd"]
    dp = vgrid_data["dp"]
    eta2 = vgrid_data["eta2"]
    kbp = vgrid_data["kbp"]

    with open("vgrid.in", "w", encoding="utf-8") as f:
        f.write("1\n")
        f.write(f"{nvrt}\n")

        kshift = nvrt + 1 - kbp
        line = "".join(f" {v:10d}" for v in kshift)
        f.write(line + "\n")

        for i in range(gd.np):
            if dp[i] > hsm[0]:
                sigma_vqs[0, i] = 0.0
                sigma_vqs[kbp[i] - 1, i] = -1.0
                for k in range(1, kbp[i] - 1):
                    sigma_vqs[k, i] = (znd[k, i] - eta2[i]) / (eta2[i] + dp[i])

                for k in range(1, kbp[i]):
                    if sigma_vqs[k, i] >= sigma_vqs[k - 1, i]:
                        msg = (
                            f"Inverted sigma: node={i + 1}, k={k + 1}, depth={dp[i]}, "
                            f"sigma(k)={sigma_vqs[k, i]}, sigma(k-1)={sigma_vqs[k - 1, i]}"
                        )
                        raise ValueError(msg)

        for k in range(1, nvrt + 1):
            level = nvrt - k
            line = f"{k:10d}"
            line += "".join(f" {sigma_vqs[level, i]:14.6f}" for i in range(gd.np))
            f.write(line + "\n")

    return nvrt


def write_nlev_gr3(gd, kbp):
    with open("nlev.gr3", "w", encoding="utf-8") as f:
        f.write("# of levels at each node\n")
        f.write(f"{gd.ne} {gd.np}\n")
        for i in range(gd.np):
            f.write(f"{i + 1} {gd.x[i]} {gd.y[i]} {kbp[i]}\n")
        for i in range(gd.ne):
            nodes = gd.elnode[i, : gd.i34[i]] + 1
            node_str = " ".join(str(n) for n in nodes)
            f.write(f"{i + 1} {gd.i34[i]} {node_str}\n")


_NUM_RE = re.compile(r"[-+]?(?:\d+\.\d*|\.\d+|\d+)(?:[eEdD][-+]?\d+)?")


def _extract_numbers(path):
    nums = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for tok in _NUM_RE.findall(line):
                nums.append(float(tok.replace("D", "E").replace("d", "E")))
    return np.array(nums, dtype=float)


def _numeric_diff(ref_path, gen_path):
    ref = _extract_numbers(ref_path)
    gen = _extract_numbers(gen_path)
    if ref.size == 0 or gen.size == 0:
        return None
    n = min(ref.size, gen.size)
    diff = ref[:n] - gen[:n]
    max_abs = float(np.max(np.abs(diff)))
    rms = float(np.sqrt(np.mean(diff**2)))
    denom = np.maximum(np.maximum(np.abs(ref[:n]), np.abs(gen[:n])), 1.0e-12)
    max_rel = float(np.max(np.abs(diff) / denom))
    return {
        "ref_count": int(ref.size),
        "gen_count": int(gen.size),
        "n": int(n),
        "max_abs": max_abs,
        "rms": rms,
        "max_rel": max_rel,
    }


def compare_outputs(ref_dir, gen_dir="."):
    targets = [
        "vgrid.in",
        "vgrid_master.out",
        "nlev.gr3",
        "transect1.out",
        "fort.12",
        "fort.99",
    ]
    print("Numeric diff vs Fortran outputs:")
    for name in targets:
        ref_path = os.path.join(ref_dir, name)
        gen_path = os.path.join(gen_dir, name)
        if not os.path.exists(ref_path) or not os.path.exists(gen_path):
            print(f"- {name}: skip (missing reference or generated file)")
            continue
        stats = _numeric_diff(ref_path, gen_path)
        if stats is None:
            print(f"- {name}: skip (no numeric tokens)")
            continue
        count_note = ""
        if stats["ref_count"] != stats["gen_count"]:
            count_note = f" count(ref/gen)={stats['ref_count']}/{stats['gen_count']}"
        print(
            f"- {name}: max_abs={stats['max_abs']:.6g} rms={stats['rms']:.6g} "
            f"max_rel={stats['max_rel']:.6g}{count_note}"
        )


def plot_transect_vgrid(
    hgrid_path="hgrid.gr3",
    bp_path="transect.bp",
    master_path="vgrid_master.out",
    transect_path="transect1.out",
    out_prefix="vgrid_transect",
):
    import matplotlib.pyplot as plt
    try:
        import cartopy.crs as ccrs
    except Exception:
        ccrs = None

    if not os.path.exists(transect_path):
        raise FileNotFoundError("Missing transect output")

    nvrt_m = None
    if os.path.exists(master_path):
        z_m = np.loadtxt(master_path)
        if z_m.ndim == 1:
            z_m = z_m[None, :]
        nvrt_m = z_m.shape[1] - 3

    z1 = np.loadtxt(transect_path)
    if z1.ndim == 1:
        z1 = z1[None, :]
    header_count = None
    for hc in (6, 7):
        if nvrt_m is not None and z1.shape[1] - hc == nvrt_m:
            header_count = hc
            break
    if header_count is None:
        header_count = 6
    zcor1 = z1[:, header_count:]
    dist = z1[:, 4]
    depth = z1[:, 5]

    gd = None
    bp = None
    is_lonlat = False
    xvals = None
    yvals = None
    if os.path.exists(hgrid_path) and os.path.exists(bp_path):
        gd = read_schism_hgrid(hgrid_path)
        bp = read_schism_bpfile(bp_path)
        xvals = np.asarray(bp.x if bp.nsta > 0 else gd.x)
        yvals = np.asarray(bp.y if bp.nsta > 0 else gd.y)
        is_lonlat = (
            np.nanmin(xvals) >= -360.0
            and np.nanmax(xvals) <= 360.0
            and np.nanmin(yvals) >= -90.0
            and np.nanmax(yvals) <= 90.0
        )

    fig = plt.figure(figsize=(10, 9))
    gs = fig.add_gridspec(2, 1)
    use_cartopy = ccrs is not None and is_lonlat
    if ccrs is None and is_lonlat:
        print("cartopy is not available; using a simple lon/lat aspect correction.", file=sys.stderr)
    if use_cartopy:
        ax_map = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
        map_crs = ccrs.PlateCarree()
    else:
        ax_map = fig.add_subplot(gs[0, 0])
        map_crs = None
    ax_tran = fig.add_subplot(gs[1, 0])

    if gd is not None and bp is not None:
        if use_cartopy:
            xy1, xy2 = gd.lines(1, xy=0)
            ax_map.plot(xy1[:, 0], xy1[:, 1], "k", lw=0.5, transform=map_crs)
            ax_map.plot(xy2[:, 0], xy2[:, 1], "k", lw=0.5, transform=map_crs)
            ax_map.plot(bp.x, bp.y, "r.", transform=map_crs)
            xmin, xmax = float(np.nanmin(gd.x)), float(np.nanmax(gd.x))
            ymin, ymax = float(np.nanmin(gd.y)), float(np.nanmax(gd.y))
            padx = (xmax - xmin) * 0.05 if xmax > xmin else 1.0
            pady = (ymax - ymin) * 0.05 if ymax > ymin else 1.0
            ax_map.set_extent([xmin - padx, xmax + padx, ymin - pady, ymax + pady], crs=map_crs)
        else:
            gd.plot_bnd(ax=ax_map)
            ax_map.plot(bp.x, bp.y, "r.")
            if is_lonlat:
                lat0 = float(np.nanmean(yvals))
                lat0 = np.clip(lat0, -89.0, 89.0)
                aspect = 1.0 / np.cos(np.deg2rad(lat0))
                ax_map.set_aspect(aspect, adjustable="datalim")
            else:
                ax_map.set_aspect("equal", adjustable="datalim")
            xmin, xmax = float(np.nanmin(gd.x)), float(np.nanmax(gd.x))
            ymin, ymax = float(np.nanmin(gd.y)), float(np.nanmax(gd.y))
            padx = (xmax - xmin) * 0.05 if xmax > xmin else 1.0
            pady = (ymax - ymin) * 0.05 if ymax > ymin else 1.0
            ax_map.set_xlim(xmin - padx, xmax + padx)
            ax_map.set_ylim(ymin - pady, ymax + pady)
        ax_map.set_title("Transect map")
    else:
        ax_map.text(0.5, 0.5, "Missing hgrid or transect bp file", ha="center", va="center")
        ax_map.set_axis_off()

    for i in range(z1.shape[0]):
        ax_tran.plot(dist[i] * np.ones(zcor1.shape[1]), zcor1[i, :], "k", linewidth=0.5)
    ax_tran.plot(dist, zcor1, "k-", linewidth=0.5)
    ax_tran.plot(dist, -depth, "r.", markersize=6)
    ax_tran.set_title("Transect")
    ax_tran.set_xlabel("Along transect distance (m)")
    ax_tran.set_ylabel("Depth (m)")

    fig.tight_layout()
    fig.savefig(f"{out_prefix}.png", dpi=200)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Generate SCHISM VQS vgrid.in using gen_vqs_ufs logic."
    )
    parser.add_argument("--hgrid", default="hgrid.gr3", help="Path to hgrid.gr3")
    parser.add_argument(
        "--transect",
        action="append",
        nargs="?",
        const="transect.bp",
        default=None,
        help="Transect bp file; repeat for multiple transects or use comma-separated list.",
    )
    parser.add_argument(
        "--compare",
        default=None,
        help="Directory holding Fortran outputs to diff against.",
    )
    parser.add_argument(
        "--plot-transect",
        action="store_true",
        help="Plot vgrids on transect using generated outputs.",
    )
    parser.add_argument(
        "--plot-prefix",
        default="vgrid_transect",
        help="Output prefix for plots when --plot-transect is set.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.hgrid):
        print(f"Missing hgrid file: {args.hgrid}", file=sys.stderr)
        return 1

    params = build_master_grid()
    write_master_outputs(params["hsm"], params["nv_vqs"], params["z_mas"], params["nvrt_m"])

    gd = read_schism_hgrid(args.hgrid)
    vgrid_data = compute_vgrid(gd, params)
    write_debug_outputs(vgrid_data["debug_lines"])

    transects = []
    if args.transect:
        for entry in args.transect:
            for part in str(entry).split(","):
                part = part.strip()
                if part:
                    transects.append(part)

    transect_outputs = []
    if transects:
        for idx, bp_path in enumerate(transects, start=1):
            if not os.path.exists(bp_path):
                print(f"Missing transect file: {bp_path}", file=sys.stderr)
                return 1
            out_path = f"transect{idx}.out"
            write_transect(bp_path, gd, vgrid_data["kbp"], vgrid_data["dp"], vgrid_data["znd"], out_path)
            transect_outputs.append((bp_path, out_path))
            print(f"Transect output: {out_path} <- {bp_path}")

    nprism = 0
    for i in range(gd.ne):
        nodes = gd.elnode[i, : gd.i34[i]]
        nprism += int(vgrid_data["kbp"][nodes].max())

    avg_layers = nprism / gd.ne
    print(f"Final nvrt={int(vgrid_data['kbp'].max())}")
    print(f"# of prisms={nprism}")
    print(f"Average # of layers={avg_layers}")

    nvrt = write_vgrid_in(gd, params, vgrid_data)
    write_nlev_gr3(gd, vgrid_data["kbp"])

    print(f"vgrid.in written with nvrt={nvrt}")
    if args.compare:
        compare_outputs(args.compare, os.getcwd())
    if args.plot_transect:
        if not transect_outputs:
            print("No transect outputs available for plotting.", file=sys.stderr)
            return 1
        for idx, (bp_path, out_path) in enumerate(transect_outputs, start=1):
            plot_transect_vgrid(
                hgrid_path=args.hgrid,
                bp_path=bp_path,
                transect_path=out_path,
                out_prefix=f"{args.plot_prefix}_{idx}",
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
