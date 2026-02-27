#!/usr/bin/env python3
from __future__ import annotations

"""
Blend M7000 and ETOPO bathymetry on a SCHISM grid using density-based zones.

Expected workflow:
1) Run compare_m7000_etopo_on_schism.py and tune density thresholds
2) Use this script to create merged TP-depth bathymetry on SCHISM nodes
3) Write a new hgrid.gr3 (does not overwrite original by default)

Important:
- ETOPO is treated as provisional unless it has been converted to the same vertical datum (TP).
- This script assumes input depths are positive-down.
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np

try:
    from pylib import read  # type: ignore
except Exception:  # pragma: no cover
    read = None
try:
    from pylib import grd2sms  # type: ignore
except Exception:  # pragma: no cover
    grd2sms = None


# ----------------------------
# Config (CLI overrides)
# ----------------------------
COMPARE_NPZ = "/Users/kpark/Downloads/m7000_etopo_compare/m7000_etopo_on_schism.npz"
SOURCE_HGRID = "/Users/kpark/Documents/Projects/Active/AIMEC_TohokuCoast/01_Data/Grid/03.gr3"
OUTPUT_HGRID = "/Users/kpark/Downloads/hgrid_m7000_etopo_tp_merged.gr3"
OUT_DIR = "/Users/kpark/Downloads/m7000_etopo_merge"
OUT_PREFIX = "m7000_etopo_merge"
SAVE_SOURCE_INTERP_GRIDS = True
SOURCE_EXPORT_FILL_FROM_ORIGINAL_HGRID = True
M7000_INTERP_HGRID_OUT = "/Users/kpark/Downloads/hgrid_m7000_on_schism.gr3"
ETOPO_INTERP_HGRID_OUT = "/Users/kpark/Downloads/hgrid_etopo_on_schism.gr3"
M7000_INTERP_2DM_OUT = "/Users/kpark/Downloads/hgrid_m7000_on_schism.2dm"
ETOPO_INTERP_2DM_OUT = "/Users/kpark/Downloads/hgrid_etopo_on_schism.2dm"

# Density-based zone / blend settings (d_k in km)
DENSITY_K = 4
M7000_ZONE_DK_MAX_KM = 0.4
ETOPO_ZONE_DK_MIN_KM = 1.0
BLEND_SHAPE = "smoothstep"  # smoothstep | linear

# Optional provisional ETOPO depth adjustment (applied before blend)
ETOPO_BIAS_M = 0.0

# Fallback behavior where neither source is valid
FILL_FROM_ORIGINAL_HGRID = True

# Plot / QA
PLOT = True
SHOW = True
MAX_PLOT_POINTS = 500000
DEPTH_CBAR_LIMS = None   # e.g. (0, 3000)
WEIGHT_CBAR_LIMS = (0.0, 1.0)
DIFF_CBAR_LIMS = None    # e.g. (-20, 20)
DEPTH_CMAP = "viridis_r"
WEIGHT_CMAP = "viridis"
DIFF_CMAP = "RdBu_r"
PLOT_SOURCE_INTERP = True


def _to_float_array(x):
    return np.asarray(x, dtype=np.float64).ravel()


def _to_bool_mask(a):
    aa = np.asarray(a)
    if aa.dtype == np.bool_:
        return aa
    return aa.astype(np.int8) != 0


def _load_grid(gr3_path):
    if read is None:
        raise SystemExit("pylib is required to read/write SCHISM grid (.gr3)")
    gd = read(gr3_path)
    x = _to_float_array(gd.x)
    y = _to_float_array(gd.y)
    dp = _to_float_array(gd.dp)
    return gd, x, y, dp


def _stats(arr):
    a = np.asarray(arr, dtype=np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    return {
        "count": int(a.size),
        "min": float(np.min(a)),
        "p05": float(np.percentile(a, 5)),
        "median": float(np.median(a)),
        "mean": float(np.mean(a)),
        "p95": float(np.percentile(a, 95)),
        "max": float(np.max(a)),
        "std": float(np.std(a)),
    }


def _print_stats(label, arr):
    s = _stats(arr)
    if s is None:
        print(f"{label}: no valid values")
        return
    print(
        f"{label}: n={s['count']} min={s['min']:.3f} p05={s['p05']:.3f} "
        f"median={s['median']:.3f} mean={s['mean']:.3f} p95={s['p95']:.3f} "
        f"max={s['max']:.3f} std={s['std']:.3f}"
    )


def _classify_density_zones(dk_km, m7000_dk_max_km, etopo_dk_min_km):
    dk = np.asarray(dk_km, dtype=np.float64)
    z = np.zeros(dk.shape, dtype=np.int8)
    finite = np.isfinite(dk)
    if not np.any(finite):
        return z
    a = float(m7000_dk_max_km)
    b = float(etopo_dk_min_km)
    if a > b:
        a, b = b, a
    z[finite & (dk <= a)] = 1
    z[finite & (dk >= b)] = 3
    z[finite & (dk > a) & (dk < b)] = 2
    return z


def _smoothstep(t):
    t = np.clip(np.asarray(t, dtype=np.float64), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _blend_weight_from_dk(dk_km, zone_code, a_km, b_km, shape="smoothstep"):
    """
    Returns M7000 weight in [0,1], NaN where undefined.
    zone_code: 1=M7000, 2=blend, 3=ETOPO
    """
    dk = np.asarray(dk_km, dtype=np.float64)
    zone = np.asarray(zone_code, dtype=np.int8)
    w = np.full(dk.shape, np.nan, dtype=np.float32)

    w[zone == 1] = 1.0
    w[zone == 3] = 0.0

    blend = (zone == 2) & np.isfinite(dk)
    if np.any(blend):
        a = float(a_km)
        b = float(b_km)
        if a > b:
            a, b = b, a
        if b <= a:
            w[blend] = 0.5
        else:
            t = (dk[blend] - a) / (b - a)  # 0 near M7000, 1 near ETOPO
            if shape == "smoothstep":
                t = _smoothstep(t)
            else:
                t = np.clip(t, 0.0, 1.0)
            w[blend] = (1.0 - t).astype(np.float32)
    return w


def _load_compare_npz(npz_path):
    z = np.load(npz_path, allow_pickle=True)
    required = ["x", "y", "m7000_depth", "etopo_depth", "valid_m7000", "valid_etopo"]
    missing = [k for k in required if k not in z]
    if missing:
        raise SystemExit(f"Compare NPZ missing keys: {missing}")
    return z


def _validate_node_alignment(compare_x, compare_y, grid_x, grid_y, tol=1e-8):
    if compare_x.size != grid_x.size or compare_y.size != grid_y.size:
        raise SystemExit(
            f"Node count mismatch between compare NPZ ({compare_x.size}) and hgrid ({grid_x.size})"
        )
    dx = np.nanmax(np.abs(compare_x - grid_x))
    dy = np.nanmax(np.abs(compare_y - grid_y))
    if not (np.isfinite(dx) and np.isfinite(dy)) or dx > tol or dy > tol:
        raise SystemExit(
            f"Node coordinate mismatch between compare NPZ and hgrid (max |dx|={dx}, |dy|={dy})"
        )


def _plot_single_map(
    gd, x, y, z, valid_mask, out_png, title, cbar_label, cmap, cbar_lims, show, max_plot_points
):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print(f"matplotlib not available; skip {Path(out_png).name}")
        return

    idx = np.where(valid_mask & np.isfinite(z))[0]
    if idx.size == 0:
        print(f"No valid values to plot for {Path(out_png).name}")
        return
    if idx.size > max_plot_points:
        idx = idx[:: max(1, idx.size // max_plot_points)]

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    try:
        gd.plot_bnd(ax=ax)
    except Exception:
        gd.plot_bnd()
        ax = plt.gca()
    sc = ax.scatter(
        x[idx], y[idx], c=z[idx], s=3, alpha=0.7, cmap=cmap,
        vmin=(cbar_lims[0] if cbar_lims else None),
        vmax=(cbar_lims[1] if cbar_lims else None),
    )
    cb = fig.colorbar(sc, ax=ax)
    cb.set_label(cbar_label)
    ax.set_title(title)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    try:
        ax.set_xlim(float(np.nanmin(gd.x)), float(np.nanmax(gd.x)))
        ax.set_ylim(float(np.nanmin(gd.y)), float(np.nanmax(gd.y)))
    except Exception:
        pass
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(out_png, dpi=150)
    print("Saved", out_png)
    if show:
        plt.show()
    plt.close(fig)


def _plot_zone_map(gd, x, y, zone_code, out_png, show, max_plot_points):
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap, BoundaryNorm
    except Exception:
        print(f"matplotlib not available; skip {Path(out_png).name}")
        return
    idx = np.where(zone_code > 0)[0]
    if idx.size == 0:
        print(f"No classified zones to plot for {Path(out_png).name}")
        return
    if idx.size > max_plot_points:
        idx = idx[:: max(1, idx.size // max_plot_points)]

    fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
    try:
        gd.plot_bnd(ax=ax)
    except Exception:
        gd.plot_bnd()
        ax = plt.gca()
    cmap_zone = ListedColormap(["#1f77b4", "#ffbf00", "#d62728"])
    norm = BoundaryNorm([0.5, 1.5, 2.5, 3.5], cmap_zone.N)
    sc = ax.scatter(x[idx], y[idx], c=zone_code[idx], s=3, alpha=0.8, cmap=cmap_zone, norm=norm)
    cb = fig.colorbar(sc, ax=ax, ticks=[1, 2, 3])
    cb.ax.set_yticklabels(["M7000", "Blend", "ETOPO"])
    cb.set_label("Density zone")
    ax.set_title("Density-derived blend zones on SCHISM nodes")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_aspect("equal", adjustable="box")
    fig.savefig(out_png, dpi=150)
    print("Saved", out_png)
    if show:
        plt.show()
    plt.close(fig)


def _write_grid_exports(gd_template, dp, out_gr3=None, out_2dm=None):
    """
    Write SCHISM grid with provided node depths to .gr3 and optionally .2dm.
    Attempts multiple grd2sms call styles to match pylib variants.
    """
    g = copy.deepcopy(gd_template)
    g.dp = np.asarray(dp, dtype=np.float64)

    if out_gr3:
        p = Path(out_gr3)
        p.parent.mkdir(parents=True, exist_ok=True)
        g.write_hgrid(str(p))
        print("Wrote gr3:", p)

    if out_2dm:
        p2 = Path(out_2dm)
        p2.parent.mkdir(parents=True, exist_ok=True)
        ok = False
        err_last = None

        # pylib as method on grid object
        if hasattr(g, "grd2sms"):
            try:
                g.grd2sms(str(p2))  # type: ignore[attr-defined]
                ok = True
            except Exception as exc:
                err_last = exc

        # pylib as module function
        if not ok and grd2sms is not None:
            for args in ((g, str(p2)), (str(p2), g)):
                try:
                    grd2sms(*args)  # type: ignore[misc]
                    ok = True
                    break
                except Exception as exc:
                    err_last = exc

        if ok:
            print("Wrote 2dm:", p2)
        else:
            print(f"[WARN] Could not write 2dm via pylib.grd2sms for {p2}: {err_last}")


def main():
    ap = argparse.ArgumentParser(description="Blend M7000 and ETOPO bathymetry on SCHISM grid using density zones.")
    ap.add_argument("--compare-npz", default=COMPARE_NPZ, help="Output NPZ from compare_m7000_etopo_on_schism.py")
    ap.add_argument("--source-hgrid", default=SOURCE_HGRID, help="Source SCHISM hgrid.gr3 (geometry/topology)")
    ap.add_argument("--output-hgrid", default=OUTPUT_HGRID, help="Output merged hgrid.gr3 path")
    ap.add_argument("--out-dir", default=OUT_DIR, help="QA output directory")
    ap.add_argument("--out-prefix", default=OUT_PREFIX, help="QA output prefix")
    ap.add_argument("--save-source-interp-grids", default=SAVE_SOURCE_INTERP_GRIDS, action=argparse.BooleanOptionalAction, help="Write M7000-only and ETOPO-only SCHISM-grid exports (.gr3/.2dm)")
    ap.add_argument("--source-export-fill-from-original-hgrid", default=SOURCE_EXPORT_FILL_FROM_ORIGINAL_HGRID, action=argparse.BooleanOptionalAction, help="Fill invalid nodes in M7000/ETOPO source exports with original hgrid depths")
    ap.add_argument("--m7000-interp-hgrid-out", default=M7000_INTERP_HGRID_OUT, help="Output .gr3 for M7000 interpolated to SCHISM nodes")
    ap.add_argument("--etopo-interp-hgrid-out", default=ETOPO_INTERP_HGRID_OUT, help="Output .gr3 for ETOPO interpolated to SCHISM nodes")
    ap.add_argument("--m7000-interp-2dm-out", default=M7000_INTERP_2DM_OUT, help="Output .2dm for M7000 interpolated to SCHISM nodes")
    ap.add_argument("--etopo-interp-2dm-out", default=ETOPO_INTERP_2DM_OUT, help="Output .2dm for ETOPO interpolated to SCHISM nodes")
    ap.add_argument("--density-k", default=DENSITY_K, type=int, help="Density k for d_k thresholding")
    ap.add_argument("--m7000-zone-dk-max-km", default=M7000_ZONE_DK_MAX_KM, type=float, help="d_k threshold for M7000 zone")
    ap.add_argument("--etopo-zone-dk-min-km", default=ETOPO_ZONE_DK_MIN_KM, type=float, help="d_k threshold for ETOPO zone")
    ap.add_argument("--blend-shape", default=BLEND_SHAPE, choices=["smoothstep", "linear"], help="Blend shape in blend zone")
    ap.add_argument("--etopo-bias-m", default=ETOPO_BIAS_M, type=float, help="Optional constant bias added to ETOPO depth before blending")
    ap.add_argument("--fill-from-original-hgrid", default=FILL_FROM_ORIGINAL_HGRID, action=argparse.BooleanOptionalAction, help="Use source hgrid depth where neither source is valid")
    ap.add_argument("--plot", default=PLOT, action=argparse.BooleanOptionalAction, help="Write QA plots")
    ap.add_argument("--plot-source-interp", default=PLOT_SOURCE_INTERP, action=argparse.BooleanOptionalAction, help="Plot M7000-only and ETOPO-only interpolated SCHISM-node depths")
    ap.add_argument("--show", default=SHOW, action=argparse.BooleanOptionalAction, help="Show plots interactively")
    ap.add_argument("--max-plot-points", default=MAX_PLOT_POINTS, type=int, help="Max nodes plotted")
    ap.add_argument("--depth-cbar-lims", nargs=2, type=float, default=DEPTH_CBAR_LIMS, metavar=("VMIN", "VMAX"), help="Colorbar limits for merged depth map")
    ap.add_argument("--weight-cbar-lims", nargs=2, type=float, default=WEIGHT_CBAR_LIMS, metavar=("VMIN", "VMAX"), help="Colorbar limits for M7000 blend weight map")
    ap.add_argument("--diff-cbar-lims", nargs=2, type=float, default=DIFF_CBAR_LIMS, metavar=("VMIN", "VMAX"), help="Colorbar limits for diff QA maps")
    ap.add_argument("--depth-cmap", default=DEPTH_CMAP, help="Colormap for depth maps")
    ap.add_argument("--weight-cmap", default=WEIGHT_CMAP, help="Colormap for weight map")
    ap.add_argument("--diff-cmap", default=DIFF_CMAP, help="Colormap for diff QA maps")
    args = ap.parse_args()

    gd, gx, gy, dp0 = _load_grid(args.source_hgrid)
    print(f"Loaded source hgrid: {args.source_hgrid} (nodes={gx.size})")

    zc = _load_compare_npz(args.compare_npz)
    cx = _to_float_array(zc["x"])
    cy = _to_float_array(zc["y"])
    _validate_node_alignment(cx, cy, gx, gy)
    print(f"Loaded compare NPZ: {args.compare_npz}")

    m7000_depth = np.asarray(zc["m7000_depth"], dtype=np.float32).ravel()
    etopo_depth = np.asarray(zc["etopo_depth"], dtype=np.float32).ravel()
    if args.etopo_bias_m != 0.0:
        etopo_depth = (etopo_depth + np.float32(args.etopo_bias_m)).astype(np.float32)
        print(f"Applied ETOPO provisional bias: {args.etopo_bias_m:+.3f} m")

    valid_m = _to_bool_mask(zc["valid_m7000"]).ravel()
    valid_e = _to_bool_mask(zc["valid_etopo"]).ravel()
    n = gx.size

    if "m7000_density_dk_km" not in zc:
        raise SystemExit(
            "Compare NPZ missing 'm7000_density_dk_km'. Re-run compare_m7000_etopo_on_schism.py "
            "with point-cloud M7000 and density diagnostics enabled."
        )
    dk_km = np.asarray(zc["m7000_density_dk_km"], dtype=np.float32).ravel()
    d1_km = np.asarray(zc["m7000_density_d1_km"], dtype=np.float32).ravel() if "m7000_density_d1_km" in zc else np.full(n, np.nan, dtype=np.float32)

    # Recompute zone code from current thresholds (preferred), but preserve compare output for QA if desired.
    zone_code = _classify_density_zones(dk_km, args.m7000_zone_dk_max_km, args.etopo_zone_dk_min_km)
    zone_code[~(valid_m | valid_e)] = 0

    w_m7000 = _blend_weight_from_dk(
        dk_km=dk_km,
        zone_code=zone_code,
        a_km=args.m7000_zone_dk_max_km,
        b_km=args.etopo_zone_dk_min_km,
        shape=args.blend_shape,
    )

    merged = np.full(n, np.nan, dtype=np.float32)

    m_only = valid_m & ~valid_e
    e_only = valid_e & ~valid_m
    both = valid_m & valid_e
    none = ~(valid_m | valid_e)

    merged[m_only] = m7000_depth[m_only]
    merged[e_only] = etopo_depth[e_only]

    if np.any(both):
        wb = np.asarray(w_m7000[both], dtype=np.float32)
        zb = np.asarray(m7000_depth[both], dtype=np.float32)
        ze = np.asarray(etopo_depth[both], dtype=np.float32)
        # Fallback if weight is undefined in overlap
        badw = ~np.isfinite(wb)
        if np.any(badw):
            wb[badw] = 0.5
        merged[both] = (wb * zb + (1.0 - wb) * ze).astype(np.float32)

    if np.any(none):
        if args.fill_from_original_hgrid:
            merged[none] = dp0[none].astype(np.float32)
        else:
            print(f"[WARN] {int(np.sum(none))} nodes have neither M7000 nor ETOPO values; merged hgrid will contain NaNs.")

    # QA arrays
    diff_vs_m7000 = np.full(n, np.nan, dtype=np.float32)
    diff_vs_etopo = np.full(n, np.nan, dtype=np.float32)
    diff_vs_m7000[valid_m & np.isfinite(merged)] = (merged[valid_m & np.isfinite(merged)] - m7000_depth[valid_m & np.isfinite(merged)]).astype(np.float32)
    diff_vs_etopo[valid_e & np.isfinite(merged)] = (merged[valid_e & np.isfinite(merged)] - etopo_depth[valid_e & np.isfinite(merged)]).astype(np.float32)
    diff_vs_original = np.full(n, np.nan, dtype=np.float32)
    diff_vs_original[np.isfinite(merged)] = (merged[np.isfinite(merged)] - dp0[np.isfinite(merged)]).astype(np.float32)

    # Source-only SCHISM-grid exports for inspection / SMS
    m7000_on_schism = np.full(n, np.nan, dtype=np.float32)
    etopo_on_schism = np.full(n, np.nan, dtype=np.float32)
    m7000_on_schism[valid_m] = m7000_depth[valid_m]
    etopo_on_schism[valid_e] = etopo_depth[valid_e]
    if args.source_export_fill_from_original_hgrid:
        miss_m = ~np.isfinite(m7000_on_schism)
        miss_e = ~np.isfinite(etopo_on_schism)
        m7000_on_schism[miss_m] = dp0[miss_m].astype(np.float32)
        etopo_on_schism[miss_e] = dp0[miss_e].astype(np.float32)

    print(f"Valid M7000 nodes: {int(np.sum(valid_m))}")
    print(f"Valid ETOPO nodes: {int(np.sum(valid_e))}")
    print(f"Overlap nodes: {int(np.sum(both))}")
    print(f"M7000-only nodes: {int(np.sum(m_only))}")
    print(f"ETOPO-only nodes: {int(np.sum(e_only))}")
    print(f"Neither valid nodes: {int(np.sum(none))}")
    print(
        "Density zones:",
        f"M7000={int(np.sum(zone_code == 1))}",
        f"blend={int(np.sum(zone_code == 2))}",
        f"ETOPO={int(np.sum(zone_code == 3))}",
    )
    _print_stats("Merged depth", merged)
    _print_stats("M7000 weight (overlap)", w_m7000[both])
    _print_stats("Merged - M7000 (valid M7000)", diff_vs_m7000)
    _print_stats("Merged - ETOPO (valid ETOPO)", diff_vs_etopo)
    _print_stats("Merged - original hgrid", diff_vs_original)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    qa_npz = out_dir / f"{args.out_prefix}.npz"
    np.savez_compressed(
        qa_npz,
        x=gx.astype(np.float64),
        y=gy.astype(np.float64),
        source_hgrid_dp=dp0.astype(np.float32),
        m7000_depth=m7000_depth.astype(np.float32),
        etopo_depth=etopo_depth.astype(np.float32),
        merged_depth=merged.astype(np.float32),
        valid_m7000=valid_m.astype(np.uint8),
        valid_etopo=valid_e.astype(np.uint8),
        m_only=m_only.astype(np.uint8),
        e_only=e_only.astype(np.uint8),
        overlap=both.astype(np.uint8),
        none_valid=none.astype(np.uint8),
        m7000_weight=w_m7000.astype(np.float32),
        density_d1_km=d1_km.astype(np.float32),
        density_dk_km=dk_km.astype(np.float32),
        density_zone_code=zone_code.astype(np.int8),
        diff_vs_m7000=diff_vs_m7000.astype(np.float32),
        diff_vs_etopo=diff_vs_etopo.astype(np.float32),
        diff_vs_original_hgrid=diff_vs_original.astype(np.float32),
        density_k=np.asarray(int(args.density_k)),
        m7000_zone_dk_max_km=np.asarray(float(args.m7000_zone_dk_max_km)),
        etopo_zone_dk_min_km=np.asarray(float(args.etopo_zone_dk_min_km)),
        blend_shape=np.asarray(str(args.blend_shape)),
        etopo_bias_m=np.asarray(float(args.etopo_bias_m)),
        note=np.asarray("ETOPO is provisional unless converted to TP."),
    )
    print("Saved QA NPZ:", qa_npz)

    qa_json = out_dir / f"{args.out_prefix}_settings.json"
    qa_json.write_text(
        json.dumps(
            {
                "compare_npz": str(args.compare_npz),
                "source_hgrid": str(args.source_hgrid),
                "output_hgrid": str(args.output_hgrid),
                "save_source_interp_grids": bool(args.save_source_interp_grids),
                "source_export_fill_from_original_hgrid": bool(args.source_export_fill_from_original_hgrid),
                "m7000_interp_hgrid_out": str(args.m7000_interp_hgrid_out),
                "etopo_interp_hgrid_out": str(args.etopo_interp_hgrid_out),
                "m7000_interp_2dm_out": str(args.m7000_interp_2dm_out),
                "etopo_interp_2dm_out": str(args.etopo_interp_2dm_out),
                "density_k": int(args.density_k),
                "m7000_zone_dk_max_km": float(args.m7000_zone_dk_max_km),
                "etopo_zone_dk_min_km": float(args.etopo_zone_dk_min_km),
                "blend_shape": str(args.blend_shape),
                "etopo_bias_m": float(args.etopo_bias_m),
                "fill_from_original_hgrid": bool(args.fill_from_original_hgrid),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print("Saved settings JSON:", qa_json)

    if np.any(~np.isfinite(merged)):
        raise SystemExit("Merged depth contains NaN values. Adjust inputs or enable --fill-from-original-hgrid.")

    if args.save_source_interp_grids:
        if np.any(~np.isfinite(m7000_on_schism)):
            print("[WARN] M7000 source export contains NaN values; consider --source-export-fill-from-original-hgrid.")
        if np.any(~np.isfinite(etopo_on_schism)):
            print("[WARN] ETOPO source export contains NaN values; consider --source-export-fill-from-original-hgrid.")
        _write_grid_exports(
            gd_template=gd,
            dp=m7000_on_schism,
            out_gr3=args.m7000_interp_hgrid_out,
            out_2dm=args.m7000_interp_2dm_out,
        )
        _write_grid_exports(
            gd_template=gd,
            dp=etopo_on_schism,
            out_gr3=args.etopo_interp_hgrid_out,
            out_2dm=args.etopo_interp_2dm_out,
        )

    gd.dp = merged.astype(np.float64)
    out_hgrid = Path(args.output_hgrid)
    out_hgrid.parent.mkdir(parents=True, exist_ok=True)
    gd.write_hgrid(str(out_hgrid))
    print("Wrote merged hgrid:", out_hgrid)

    if args.plot:
        if args.plot_source_interp:
            _plot_single_map(
                gd, gx, gy, m7000_on_schism, np.isfinite(m7000_on_schism),
                out_dir / f"{args.out_prefix}_m7000_on_schism.png",
                "M7000 interpolated to SCHISM nodes",
                "Depth (m)", args.depth_cmap, args.depth_cbar_lims, args.show, args.max_plot_points,
            )
            _plot_single_map(
                gd, gx, gy, etopo_on_schism, np.isfinite(etopo_on_schism),
                out_dir / f"{args.out_prefix}_etopo_on_schism.png",
                "ETOPO interpolated to SCHISM nodes (provisional)",
                "Depth (m)", args.depth_cmap, args.depth_cbar_lims, args.show, args.max_plot_points,
            )
        _plot_single_map(
            gd, gx, gy, merged, np.isfinite(merged),
            out_dir / f"{args.out_prefix}_merged_depth.png",
            "Merged bathymetry on SCHISM nodes (TP depth, positive-down)",
            "Depth (m)", args.depth_cmap, args.depth_cbar_lims, args.show, args.max_plot_points,
        )
        _plot_single_map(
            gd, gx, gy, w_m7000, both & np.isfinite(w_m7000),
            out_dir / f"{args.out_prefix}_m7000_weight.png",
            "M7000 blend weight on overlap nodes",
            "M7000 weight (0..1)", args.weight_cmap, args.weight_cbar_lims, args.show, args.max_plot_points,
        )
        _plot_zone_map(
            gd, gx, gy, zone_code,
            out_dir / f"{args.out_prefix}_zone_map.png",
            args.show, args.max_plot_points,
        )
        _plot_single_map(
            gd, gx, gy, diff_vs_m7000, valid_m & np.isfinite(diff_vs_m7000),
            out_dir / f"{args.out_prefix}_diff_vs_m7000.png",
            "Merged - M7000 on SCHISM nodes",
            "Depth difference (m)", args.diff_cmap, args.diff_cbar_lims, args.show, args.max_plot_points,
        )
        _plot_single_map(
            gd, gx, gy, diff_vs_etopo, valid_e & np.isfinite(diff_vs_etopo),
            out_dir / f"{args.out_prefix}_diff_vs_etopo.png",
            "Merged - ETOPO on SCHISM nodes",
            "Depth difference (m)", args.diff_cmap, args.diff_cbar_lims, args.show, args.max_plot_points,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
