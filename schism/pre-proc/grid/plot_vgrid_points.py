#!/usr/bin/env python3
"""
Plot SCHISM vertical grid structure at user-provided lon/lat points.

Inputs:
- hgrid.gr3
- vgrid.in
- point locations from:
  1) config `points` list, and/or
  2) bp file

Outputs:
- per-point figure showing:
  - map panel (input point and nearest SCHISM node)
  - vertical interfaces through the water column
  - layer-thickness distribution
- summary CSV
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pylib import read_schism_bpfile, read_schism_hgrid


USER_CONFIG: Dict[str, Any] = {
    "enable": True,
    "hgrid": "/Users/kpark/Documents/Projects/Active/AIMEC_TohokuCoast/01_Data/Grid/03/hgrid.gr3",  # required
    "vgrid": "/Users/kpark/Documents/Projects/Active/AIMEC_TohokuCoast/01_Data/Grid/03/vgrid.in",  # required
    "outdir": "/Users/kpark/Documents/Projects/Active/AIMEC_TohokuCoast/01_Data/Grid/vgrid_point_plots",
    # Optional point sources:
    "bpfile": "/Users/kpark/Documents/Projects/Active/AIMEC_TohokuCoast/01_Data/TEAMS/station_sendai_d2.in",  # e.g., "/path/to/points.bp"
    # points: [{"name": "P1", "lon": 141.0, "lat": 38.2}, ...]
    "points": None,
    # Optional observation depths for reference lines (m, positive downward):
    # {"P1": 4.2, "P2": 8.0}
    "obs_depths": None,
    "dpi": 220,
    "save_overview_map": True,
    "overview_map_name": "vgrid_points_overview.png",
    "summary_csv": "vgrid_points_summary.csv",
}


def _safe_name(name: str) -> str:
    keep = []
    for ch in str(name):
        if ch.isalnum() or ch in ("-", "_", "."):
            keep.append(ch)
        else:
            keep.append("_")
    out = "".join(keep).strip("._")
    return out or "point"


def _iter_tokens(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for tok in line.strip().split():
                yield tok


def read_vgrid_in(path: str, npnt: int) -> Dict[str, Any]:
    toks = _iter_tokens(path)
    try:
        ivcor = int(next(toks))
        nvrt = int(next(toks))
    except StopIteration as exc:
        raise ValueError(f"{path}: invalid vgrid.in header") from exc

    kshift = np.zeros(npnt, dtype=int)
    for i in range(npnt):
        try:
            kshift[i] = int(next(toks))
        except StopIteration as exc:
            raise ValueError(f"{path}: failed reading kshift (node {i + 1}/{npnt})") from exc

    sigma = np.full((nvrt, npnt), np.nan, dtype=float)
    for k in range(nvrt):
        try:
            _ = int(next(toks))
        except StopIteration as exc:
            raise ValueError(f"{path}: missing sigma level index at k={k + 1}") from exc
        for i in range(npnt):
            try:
                sigma[k, i] = float(next(toks))
            except StopIteration as exc:
                raise ValueError(f"{path}: missing sigma value at k={k + 1}, node={i + 1}") from exc

    return {"ivcor": ivcor, "nvrt": nvrt, "kshift": kshift, "sigma": sigma}


def _looks_like_lonlat(x: np.ndarray, y: np.ndarray) -> bool:
    return bool(
        np.all(np.isfinite(x))
        and np.all(np.isfinite(y))
        and np.nanmin(x) >= -360.0
        and np.nanmax(x) <= 360.0
        and np.nanmin(y) >= -90.0
        and np.nanmax(y) <= 90.0
    )


def _haversine_m_vec(lon: float, lat: float, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    r_earth = 6371000.0
    lon1 = np.deg2rad(float(lon))
    lat1 = np.deg2rad(float(lat))
    lon2r = np.deg2rad(np.asarray(lon2, dtype=float))
    lat2r = np.deg2rad(np.asarray(lat2, dtype=float))
    dlon = lon2r - lon1
    dlat = lat2r - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2r) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return r_earth * c


def _nearest_node(gx: np.ndarray, gy: np.ndarray, px: float, py: float, lonlat: bool) -> Tuple[int, float]:
    if lonlat:
        d = _haversine_m_vec(px, py, gx, gy)
    else:
        d = np.hypot(gx - float(px), gy - float(py))
    i = int(np.nanargmin(d))
    return i, float(d[i])


def _is_generic_point_name(name: str, idx1: Optional[int] = None) -> bool:
    txt = str(name).strip()
    if txt == "":
        return True
    low = txt.lower()
    if low in {"point", "station", "sta"}:
        return True
    if re.fullmatch(r"(?:p(?:oint)?|sta(?:tion)?)?\d+", low):
        return True
    if idx1 is not None:
        if low in {f"p{idx1}", f"point{idx1}", f"station{idx1}", str(idx1)}:
            return True
    return False


def _read_bp_names_fallback(bpfile: str, nsta: int) -> List[str]:
    """
    Best-effort station-name parser from raw bp text.
    Priority:
    1) comment text after '!' (common in SCHISM bp files)
    2) trailing tokens after first 4 numeric columns
    3) empty string
    """
    out: List[str] = []
    try:
        with open(bpfile, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
    except Exception:
        return [""] * int(nsta)

    if len(lines) <= 2:
        return [""] * int(nsta)

    recs = lines[2:]
    for i in range(int(nsta)):
        if i >= len(recs):
            out.append("")
            continue
        line = recs[i].strip()
        if line == "":
            out.append("")
            continue

        cmt = ""
        body = line
        if "!" in line:
            body, cmt = line.split("!", 1)
            cmt = cmt.strip()

        if cmt:
            out.append(cmt)
            continue

        toks = body.split()
        if len(toks) > 4:
            out.append(" ".join(toks[4:]).strip())
        else:
            out.append("")

    if len(out) < int(nsta):
        out.extend([""] * (int(nsta) - len(out)))
    return out


def _point_list_from_bp(bpfile: str) -> List[Dict[str, Any]]:
    bp = read_schism_bpfile(bpfile)
    out: List[Dict[str, Any]] = []
    labels = np.asarray(bp.station).astype("U").tolist() if hasattr(bp, "station") else []
    nsta = int(bp.nsta)
    raw_names = _read_bp_names_fallback(bpfile, nsta)
    for i in range(nsta):
        name = f"P{i + 1}"
        # Prefer explicit name parsed from raw bp text.
        nm_raw = str(raw_names[i]).strip() if i < len(raw_names) else ""
        if nm_raw and not _is_generic_point_name(nm_raw, i + 1):
            name = nm_raw

        # Fallback to read_schism_bpfile station label.
        if i < len(labels):
            entry = str(labels[i]).strip()
            if entry:
                parts = entry.split()
                candidates = []
                if len(parts) > 0:
                    candidates.append(parts[0])
                if len(parts) > 1:
                    candidates.append(parts[1])
                candidates.append(entry)
                for cand in candidates:
                    cc = str(cand).strip()
                    if cc and (not _is_generic_point_name(cc, i + 1)):
                        name = cc
                        break
        out.append({"name": name, "lon": float(bp.x[i]), "lat": float(bp.y[i])})
    return out


def _parse_inline_points(raw_points: Any) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if raw_points is None:
        return out
    if not isinstance(raw_points, (list, tuple)):
        raise ValueError("points must be a list of dicts with keys name/lon/lat")
    for i, p in enumerate(raw_points):
        if not isinstance(p, dict):
            raise ValueError(f"points[{i}] must be dict")
        if "lon" not in p or "lat" not in p:
            raise ValueError(f"points[{i}] missing lon/lat")
        name = str(p.get("name", f"P{i + 1}")).strip() or f"P{i + 1}"
        out.append({"name": name, "lon": float(p["lon"]), "lat": float(p["lat"])})
    return out


def _extract_z_interfaces(depth: float, sigma_col: np.ndarray, kshift: int) -> np.ndarray:
    nvrt = int(sigma_col.size)
    levels = np.arange(1, nvrt + 1)
    valid = levels >= int(kshift)
    valid &= np.isfinite(sigma_col) & (sigma_col >= -1.5) & (sigma_col <= 0.5)
    z = np.asarray(sigma_col[valid], dtype=float) * float(depth)  # eta=0
    if z.size == 0:
        return np.array([0.0, -abs(float(depth))], dtype=float)
    # Sort from surface to bottom and remove tiny duplicates.
    z = np.sort(z)[::-1]
    keep = [0]
    for i in range(1, z.size):
        if abs(z[i] - z[keep[-1]]) > 1.0e-8:
            keep.append(i)
    return z[keep]


def _plot_point_figure(
    gd: Any,
    info: Dict[str, Any],
    gx: np.ndarray,
    gy: np.ndarray,
    lonlat: bool,
    outpath: Path,
    dpi: int,
    obs_depth: Optional[float] = None,
) -> None:
    name = str(info["name"])
    x0 = float(info["x_in"])
    y0 = float(info["y_in"])
    xn = float(info["x_node"])
    yn = float(info["y_node"])
    depth = float(info["depth_m"])
    nlevels = int(info["n_levels"])
    nlayers = int(info["n_layers"])
    dist = float(info["distance_to_node"])
    z = np.asarray(info["z_interfaces"], dtype=float)
    thickness = np.asarray(info["layer_thickness"], dtype=float)

    fig = plt.figure(figsize=(11.5, 6.0), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], height_ratios=[1.0, 0.85])
    ax_map = fig.add_subplot(gs[:, 0])
    ax_prof = fig.add_subplot(gs[0, 1])
    ax_thk = fig.add_subplot(gs[1, 1])

    # map panel
    try:
        gd.plot_bnd(ax=ax_map, color="k", lw=0.8)
    except Exception:
        pass
    if gx.size > 50000:
        step = max(1, gx.size // 50000)
        ax_map.plot(gx[::step], gy[::step], ".", color="0.80", ms=1.5, zorder=1)
    else:
        ax_map.plot(gx, gy, ".", color="0.80", ms=1.5, zorder=1)
    ax_map.plot(xn, yn, "bo", ms=6, label="Nearest node", zorder=3)
    ax_map.plot(x0, y0, "r*", ms=10, label="Input point", zorder=4)
    ax_map.plot([x0, xn], [y0, yn], "k--", lw=1.0, alpha=0.8, zorder=2)

    span = max(abs(xn - x0), abs(yn - y0))
    if not np.isfinite(span) or span <= 0:
        span = 0.01 if lonlat else 500.0
    pad = max(span * 5.0, 0.02 if lonlat else 1000.0)
    ax_map.set_xlim(min(x0, xn) - pad, max(x0, xn) + pad)
    ax_map.set_ylim(min(y0, yn) - pad, max(y0, yn) + pad)
    ax_map.set_xlabel("Longitude" if lonlat else "X")
    ax_map.set_ylabel("Latitude" if lonlat else "Y")
    ax_map.set_title(f"{name}: point-to-node match")
    ax_map.grid(alpha=0.3)
    ax_map.legend(loc="best", fontsize=8)

    # vertical interfaces
    for zz in z:
        ax_prof.plot([0.0, 1.0], [zz, zz], color="k", lw=0.9)
    if z.size > 1:
        for k in range(z.size - 1):
            c = "0.90" if (k % 2 == 0) else "0.82"
            ax_prof.fill_between([0.0, 1.0], [z[k], z[k]], [z[k + 1], z[k + 1]], color=c, alpha=0.65)
    if obs_depth is not None and np.isfinite(obs_depth):
        ax_prof.axhline(-abs(float(obs_depth)), color="crimson", ls="--", lw=1.3, label="Obs depth")
        ax_prof.legend(loc="lower right", fontsize=8)
    ax_prof.set_xlim(0.0, 1.0)
    ax_prof.set_xticks([])
    ax_prof.set_ylabel("z (m, 0=MSL, downward negative)")
    ax_prof.set_title("Vertical interfaces")
    ax_prof.grid(alpha=0.25)

    txt = (
        f"node={int(info['node_index']) + 1}\n"
        f"depth={depth:.3f} m\n"
        f"levels={nlevels}, layers={nlayers}\n"
        f"distance={dist:.2f} {'m' if lonlat else 'grid units'}"
    )
    ax_prof.text(
        0.02,
        0.98,
        txt,
        transform=ax_prof.transAxes,
        va="top",
        ha="left",
        fontsize=8,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "0.6"},
    )

    # thickness panel
    if thickness.size > 0:
        idx = np.arange(1, thickness.size + 1)
        ax_thk.bar(idx, thickness, color="tab:blue", alpha=0.85, width=0.85)
        ax_thk.set_xlim(0.5, thickness.size + 0.5)
    ax_thk.set_xlabel("Layer Index (surface to bottom)")
    ax_thk.set_ylabel("Layer Thickness (m)")
    ax_thk.set_title("Layer-thickness distribution")
    ax_thk.grid(alpha=0.3, axis="y")

    fig.suptitle(f"Point {name}: vertical-grid diagnostics", fontsize=12, weight="bold")
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def _plot_overview_map(
    gd: Any,
    gx: np.ndarray,
    gy: np.ndarray,
    lonlat: bool,
    infos: List[Dict[str, Any]],
    outpath: Path,
    dpi: int,
) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8.5, 7.0), constrained_layout=True)
    try:
        gd.plot_bnd(ax=ax, color="k", lw=0.8)
    except Exception:
        pass
    if gx.size > 100000:
        step = max(1, gx.size // 100000)
        ax.plot(gx[::step], gy[::step], ".", color="0.82", ms=1.2, zorder=1)
    else:
        ax.plot(gx, gy, ".", color="0.82", ms=1.2, zorder=1)
    for info in infos:
        ax.plot(float(info["x_in"]), float(info["y_in"]), "r*", ms=8, zorder=3)
        ax.plot(float(info["x_node"]), float(info["y_node"]), "bo", ms=4, zorder=2)
        ax.text(float(info["x_in"]), float(info["y_in"]), str(info["name"]), fontsize=8, va="bottom", ha="left")
    ax.set_xlabel("Longitude" if lonlat else "X")
    ax.set_ylabel("Latitude" if lonlat else "Y")
    ax.set_title("Requested points and matched SCHISM nodes")
    ax.grid(alpha=0.3)
    fig.savefig(outpath, dpi=dpi)
    plt.close(fig)


def _merge_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(USER_CONFIG) if USER_CONFIG.get("enable", False) else {}
    if args.hgrid is not None:
        cfg["hgrid"] = args.hgrid
    if args.vgrid is not None:
        cfg["vgrid"] = args.vgrid
    if args.outdir is not None:
        cfg["outdir"] = args.outdir
    if args.bpfile is not None:
        cfg["bpfile"] = args.bpfile
    if args.point is not None:
        pts = []
        for i, item in enumerate(args.point):
            parts = [s.strip() for s in str(item).split(",")]
            if len(parts) == 2:
                name = f"P{i + 1}"
                lon = float(parts[0])
                lat = float(parts[1])
            elif len(parts) >= 3:
                name = parts[0] or f"P{i + 1}"
                lon = float(parts[1])
                lat = float(parts[2])
            else:
                raise ValueError(f"Invalid --point format: {item}. Use lon,lat or name,lon,lat")
            pts.append({"name": name, "lon": lon, "lat": lat})
        cfg["points"] = pts
    if args.dpi is not None:
        cfg["dpi"] = int(args.dpi)
    if args.save_overview_map is not None:
        cfg["save_overview_map"] = bool(args.save_overview_map)

    if cfg.get("hgrid") is None or cfg.get("vgrid") is None:
        raise ValueError("Both hgrid and vgrid are required.")
    cfg.setdefault("outdir", "./vgrid_point_plots")
    cfg.setdefault("dpi", 220)
    cfg.setdefault("save_overview_map", True)
    cfg.setdefault("overview_map_name", "vgrid_points_overview.png")
    cfg.setdefault("summary_csv", "vgrid_points_summary.csv")
    cfg.setdefault("obs_depths", None)
    return cfg


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot SCHISM vertical-grid diagnostics at selected points.")
    p.add_argument("--hgrid", help="Path to hgrid.gr3")
    p.add_argument("--vgrid", help="Path to vgrid.in")
    p.add_argument("--outdir", help="Output directory")
    p.add_argument("--bpfile", help="BP file with point locations")
    p.add_argument(
        "--point",
        action="append",
        help="Point as 'lon,lat' or 'name,lon,lat'. Repeat this option for multiple points.",
    )
    p.add_argument("--dpi", type=int, help="Figure dpi")
    p.add_argument("--save-overview-map", dest="save_overview_map", action="store_true")
    p.add_argument("--no-overview-map", dest="save_overview_map", action="store_false")
    p.set_defaults(save_overview_map=None)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = parse_args(argv)
    cfg = _merge_config(args)

    hgrid_path = str(Path(cfg["hgrid"]).expanduser().resolve())
    vgrid_path = str(Path(cfg["vgrid"]).expanduser().resolve())
    outdir = Path(cfg["outdir"]).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    gd = read_schism_hgrid(hgrid_path)
    gx = np.asarray(gd.x, dtype=float).ravel()
    gy = np.asarray(gd.y, dtype=float).ravel()
    lonlat = _looks_like_lonlat(gx, gy)

    vgrid = read_vgrid_in(vgrid_path, int(gd.np))
    sigma = np.asarray(vgrid["sigma"], dtype=float)
    kshift = np.asarray(vgrid["kshift"], dtype=int)

    points: List[Dict[str, Any]] = []
    points.extend(_parse_inline_points(cfg.get("points")))
    if cfg.get("bpfile"):
        bp_path = str(Path(cfg["bpfile"]).expanduser().resolve())
        points.extend(_point_list_from_bp(bp_path))
    if len(points) == 0:
        raise ValueError("No points provided. Use config points and/or bpfile (or --point/--bpfile).")

    # Ensure unique display names while preserving all points.
    seen_counts: Dict[str, int] = {}
    uniq_points: List[Dict[str, Any]] = []
    for i, p in enumerate(points, start=1):
        base = str(p.get("name", "")).strip()
        if base == "":
            base = f"P{i}"
        cnt = int(seen_counts.get(base, 0)) + 1
        seen_counts[base] = cnt
        if cnt == 1:
            name = base
        else:
            name = f"{base}_{cnt}"
        uniq_points.append({"name": name, "lon": float(p["lon"]), "lat": float(p["lat"])})
    points = uniq_points

    obs_depths_cfg = cfg.get("obs_depths") or {}
    obs_depths = {str(k): float(v) for k, v in obs_depths_cfg.items()} if isinstance(obs_depths_cfg, dict) else {}

    infos: List[Dict[str, Any]] = []
    for p in points:
        name = str(p["name"])
        x_in = float(p["lon"])
        y_in = float(p["lat"])
        nd, dist = _nearest_node(gx, gy, x_in, y_in, lonlat=lonlat)
        depth = float(np.asarray(gd.dp, dtype=float).ravel()[nd])
        z_interfaces = _extract_z_interfaces(depth, sigma[:, nd], int(kshift[nd]))
        layer_thickness = np.abs(np.diff(z_interfaces))
        info = {
            "name": name,
            "x_in": x_in,
            "y_in": y_in,
            "node_index": int(nd),
            "x_node": float(gx[nd]),
            "y_node": float(gy[nd]),
            "distance_to_node": float(dist),
            "depth_m": float(depth),
            "n_levels": int(z_interfaces.size),
            "n_layers": int(max(0, z_interfaces.size - 1)),
            "z_interfaces": z_interfaces,
            "layer_thickness": layer_thickness,
        }
        infos.append(info)

    for info in infos:
        name = str(info["name"])
        obs_depth = obs_depths.get(name, None)
        out_png = outdir / f"vgrid_point_{_safe_name(name)}.png"
        _plot_point_figure(
            gd=gd,
            info=info,
            gx=gx,
            gy=gy,
            lonlat=lonlat,
            outpath=out_png,
            dpi=int(cfg["dpi"]),
            obs_depth=obs_depth,
        )
        print(
            f"[OK] {name}: node={int(info['node_index']) + 1}, depth={info['depth_m']:.3f} m, "
            f"levels={info['n_levels']}, layers={info['n_layers']}, "
            f"distance={info['distance_to_node']:.2f} {'m' if lonlat else 'grid'} -> {out_png}",
            flush=True,
        )

    if bool(cfg.get("save_overview_map", True)):
        map_png = outdir / str(cfg.get("overview_map_name", "vgrid_points_overview.png"))
        _plot_overview_map(gd=gd, gx=gx, gy=gy, lonlat=lonlat, infos=infos, outpath=map_png, dpi=int(cfg["dpi"]))
        print(f"[OK] overview map: {map_png}", flush=True)

    summary_csv = outdir / str(cfg.get("summary_csv", "vgrid_points_summary.csv"))
    fields = [
        "name",
        "x_in",
        "y_in",
        "node_index_1based",
        "x_node",
        "y_node",
        "distance_to_node",
        "depth_m",
        "n_levels",
        "n_layers",
    ]
    with open(summary_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for info in infos:
            w.writerow(
                {
                    "name": info["name"],
                    "x_in": info["x_in"],
                    "y_in": info["y_in"],
                    "node_index_1based": int(info["node_index"]) + 1,
                    "x_node": info["x_node"],
                    "y_node": info["y_node"],
                    "distance_to_node": info["distance_to_node"],
                    "depth_m": info["depth_m"],
                    "n_levels": info["n_levels"],
                    "n_layers": info["n_layers"],
                }
            )
    print(f"[OK] summary csv: {summary_csv}", flush=True)


if __name__ == "__main__":
    main()
