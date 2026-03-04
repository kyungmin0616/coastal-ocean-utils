#!/usr/bin/env python3
"""
Generate SCHISM VQS `vgrid.in` from `hgrid.gr3`.

Design notes:
- Runtime output is intentionally minimal: only `vgrid.in`.
- Debug/intermediate text outputs (`fort.*`, `vgrid_master.out`, `transect*.out`, `nlev.gr3`) are not written.
- QC is done via plots:
  1) 2D map of number of vertical layers
  2) transect cross-section plots from configured bp files
- `--compare` compares generated `vgrid.in` with reference `vgrid.in` in a directory.
- Multi-vgrid compare is also available via `DEFAULT_COMPARE_CFG`.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Iterable

import numpy as np

# Use non-interactive backend only in headless environments.
if not os.environ.get("DISPLAY"):
    os.environ.setdefault("MPLBACKEND", "Agg")

from pylib import near_pts, read_schism_bpfile, read_schism_hgrid


DEFAULT_VQS_CFG = {
    # Paired config: (master depth [m], target #levels at that depth)
    "master_levels": [
        (1.0, 2),
        (2.0, 3),
        (3.0, 5),
        (4.0, 7),
        (6.0, 9),
        (8.0, 11),
        (12.0, 13),
        (18.0, 15),
        (25.0, 17),
        (33.0, 17),
        (42.0, 18),
        (52.0, 18),
        (67.0, 19),
        (83.0, 20),
        (100.0, 21),
        (150.0, 22),
        (230.0, 24),
        (360.0, 26),
    ],
    "dz_bot_min": 0.1,
    "n_generated_masters": None,  # None: auto-use all master_levels; int N: apply auto-stretch only to first N masters (must be 1..len(master_levels)).
    "require_non_decreasing_layers": False,  # True: enforce nlev(i+1) >= nlev(i) with increasing depth; False: allow local decreases if intentionally specified.
}

DEFAULT_PLOT_CFG = {
    "enable": True,
    "plot_layer_map": False,
    "plot_transect_sections": True,
    "save_plots": True,
    "show_plots": False,
    # Configure desired transects here.
    "transect_paths": ["KL_transect.bp","OB_transect.bp","SB_transect.bp","OO_LON_transect.bp","OO_LAT_transect.bp"],
    "layer_map_path": "vgrid_layers_map.png",
    "transect_prefix": "vgrid",
    "dpi": 200,
    "node_size": 8,
    "transect_marker_size": 6,
}

DEFAULT_COMPARE_CFG = {
    "enable": True,
    # Files to compare. Use "{generated}" to refer to the vgrid.in generated in this run.
    "vgrid_paths": ["{generated}","vgrid_1.in"],
    # Optional labels for vgrid_paths. If None, basenames are used.
    "labels": None,
    # Which item in vgrid_paths is reference.
    "baseline_index": 0,
    # Optional override; if None, uses --hgrid.
    "hgrid_path": None,
    "save_csv": False,
    "csv_path": "vgrid_compare.csv",
    # Optional transect comparison plots for all vgrids in vgrid_paths.
    "plot_transect_sections": True,
    # If None, use DEFAULT_PLOT_CFG["transect_paths"].
    "transect_paths": ["KL_transect.bp","OB_transect.bp","SB_transect.bp","OO_LON_transect.bp","OO_LAT_transect.bp"],
    "transect_prefix": "vgrid_compare_transect",
}


def _parse_master_levels(master_levels):
    if not isinstance(master_levels, (list, tuple)) or len(master_levels) < 2:
        raise ValueError("master_levels must be a list/tuple with at least 2 (depth, nlev) pairs")
    hsm = []
    nv_vqs = []
    for i, pair in enumerate(master_levels):
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"master_levels[{i}] must be (depth, nlev)")
        depth = float(pair[0])
        nlev_raw = pair[1]
        nlev = int(nlev_raw)
        if abs(float(nlev_raw) - nlev) > 1e-12:
            raise ValueError(f"master_levels[{i}] nlev must be an integer-like value")
        hsm.append(depth)
        nv_vqs.append(nlev)
    return np.asarray(hsm, dtype=float), np.asarray(nv_vqs, dtype=int)


def build_master_grid(vqs_cfg=None):
    vqs_cfg = vqs_cfg or DEFAULT_VQS_CFG
    hsm, nv_vqs = _parse_master_levels(vqs_cfg.get("master_levels", []))
    m_vqs = int(len(hsm))
    dz_bot_min = float(vqs_cfg.get("dz_bot_min", 0.1))
    n_gen_cfg = vqs_cfg.get("n_generated_masters", None)
    n_generated_masters = m_vqs if n_gen_cfg is None else int(n_gen_cfg)
    require_non_decreasing = bool(vqs_cfg.get("require_non_decreasing_layers", False))

    if m_vqs < 2:
        raise ValueError(f"Check vgrid.in: m_vqs={m_vqs}")
    if hsm[0] < 0:
        raise ValueError("hsm[0] < 0")
    if np.any(hsm[1:] <= hsm[:-1]):
        raise ValueError("Check hsm: not strictly increasing")
    if np.any(nv_vqs < 2):
        raise ValueError("Check master_levels: all nlev must be >= 2")
    if require_non_decreasing and np.any(nv_vqs[1:] < nv_vqs[:-1]):
        raise ValueError("Check master_levels: nlev must be non-decreasing with depth")
    if n_generated_masters < 1 or n_generated_masters > m_vqs:
        raise ValueError(
            f"n_generated_masters must be within [1, m_vqs={m_vqs}], got {n_generated_masters}"
        )

    a_vqs0 = 0.0
    theta_b = 0.0
    etal = 0.0

    nvrt_m = int(nv_vqs[-1])
    print(f"nvrt in master vgrid={nvrt_m}")
    z_mas = np.full((nvrt_m, m_vqs), -1.0e5, dtype=float)

    # Build first n_generated_masters master grids from formula.
    for m in range(n_generated_masters):
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

        nvi = int(nv_vqs[m])
        k = np.arange(nvi, dtype=float)
        sigma = k / (1.0 - nvi)
        cs = (1 - theta_b) * np.sinh(theta_f * sigma) / np.sinh(theta_f) + theta_b * (
            np.tanh(theta_f * (sigma + 0.5)) - np.tanh(theta_f * 0.5)
        ) / (2.0 * np.tanh(theta_f * 0.5))
        z_mas[:nvi, m] = etal * (1.0 + sigma) + hsm[0] * sigma + (hsm[m] - hsm[0]) * cs

    # Force downward steps for first few master grids (legacy behavior).
    for m in range(1, min(6, m_vqs)):
        n_prev = int(nv_vqs[m - 1])
        n_cur = int(nv_vqs[m])
        z_mas[:n_prev, m] = np.minimum(z_mas[:n_prev, m], z_mas[:n_prev, m - 1])
        if n_cur > n_prev:
            tmp = (z_mas[n_cur - 1, m] - z_mas[n_prev - 1, m]) / (n_cur - n_prev)
            for k in range(n_prev, n_cur):
                z_mas[k, m] = z_mas[k - 1, m] + tmp

    return {
        "m_vqs": m_vqs,
        "n_generated_masters": n_generated_masters,
        "dz_bot_min": dz_bot_min,
        "hsm": hsm,
        "nv_vqs": nv_vqs,
        "a_vqs0": a_vqs0,
        "theta_b": theta_b,
        "etal": etal,
        "nvrt_m": nvrt_m,
        "z_mas": z_mas,
    }


def compute_vgrid(gd, params):
    hsm = params["hsm"]
    nv_vqs = params["nv_vqs"]
    dz_bot_min = params["dz_bot_min"]
    a_vqs0 = params["a_vqs0"]
    etal = params["etal"]
    nvrt_m = params["nvrt_m"]
    z_mas = params["z_mas"]

    dp = np.asarray(gd.dp, dtype=float).copy()
    dpmax = float(dp.max())
    if dpmax > float(hsm[-1]):
        raise ValueError(f"Max depth exceeds master depth: {dpmax} > {hsm[-1]}")

    npnt = int(gd.np)
    eta2 = np.full(npnt, etal, dtype=float)
    sigma_vqs = np.full((nvrt_m, npnt), -9.0, dtype=float)
    znd = np.full((nvrt_m, npnt), -1.0e6, dtype=float)
    kbp = np.zeros(npnt, dtype=int)
    m0 = np.zeros(npnt, dtype=int)

    for i in range(npnt):
        if dp[i] <= hsm[0]:
            kbp[i] = int(nv_vqs[0])
            nvi = int(nv_vqs[0])
            for k in range(nvi):
                sigma = k / (1.0 - nvi)
                sigma_vqs[k, i] = a_vqs0 * sigma * sigma + (1.0 + a_vqs0) * sigma
                znd[k, i] = sigma_vqs[k, i] * (eta2[i] + dp[i]) + eta2[i]
            continue

        zrat = None
        for m in range(1, len(hsm)):
            if dp[i] > hsm[m - 1] and dp[i] <= hsm[m]:
                m0[i] = m
                zrat = (dp[i] - hsm[m - 1]) / (hsm[m] - hsm[m - 1])
                break
        if zrat is None:
            raise ValueError(f"Failed to find a master vgrid: node={i + 1}, depth={dp[i]}")

        kbp_i = 0
        for k in range(int(nv_vqs[m0[i]])):
            k_prev = min(k, int(nv_vqs[m0[i] - 1]) - 1)
            z1 = z_mas[k_prev, m0[i] - 1]
            z2 = z_mas[k, m0[i]]
            z3 = z1 + (z2 - z1) * zrat
            if z3 >= -dp[i] + dz_bot_min:
                znd[k, i] = z3
            else:
                kbp_i = k + 1
                break
        if kbp_i == 0:
            raise ValueError(f"Failed to find a bottom: node={i + 1}, depth={dp[i]}")

        kbp[i] = kbp_i
        znd[kbp_i - 1, i] = -dp[i]

        for k in range(1, kbp_i):
            if znd[k - 1, i] <= znd[k, i]:
                raise ValueError(
                    f"Inverted z: node={i + 1}, depth={dp[i]}, k={k + 1}, "
                    f"z(k-1)={znd[k - 1, i]}, z(k)={znd[k, i]}"
                )

    for i in range(npnt):
        if kbp[i] < nvrt_m:
            znd[kbp[i] :, i] = -dp[i]

    return {
        "dp": dp,
        "eta2": eta2,
        "sigma_vqs": sigma_vqs,
        "znd": znd,
        "kbp": kbp,
        "nvrt_m": nvrt_m,
    }


def write_vgrid_in(gd, params, vgrid_data, path="vgrid.in"):
    hsm = params["hsm"]
    nvrt = int(vgrid_data["kbp"].max())
    sigma_vqs = np.asarray(vgrid_data["sigma_vqs"], dtype=float)
    znd = np.asarray(vgrid_data["znd"], dtype=float)
    dp = np.asarray(vgrid_data["dp"], dtype=float)
    eta2 = np.asarray(vgrid_data["eta2"], dtype=float)
    kbp = np.asarray(vgrid_data["kbp"], dtype=int)

    with open(path, "w", encoding="utf-8") as f:
        f.write("1\n")
        f.write(f"{nvrt}\n")

        kshift = nvrt + 1 - kbp
        f.write("".join(f" {v:10d}" for v in kshift) + "\n")

        for i in range(int(gd.np)):
            if dp[i] > hsm[0]:
                sigma_vqs[0, i] = 0.0
                sigma_vqs[kbp[i] - 1, i] = -1.0
                for k in range(1, kbp[i] - 1):
                    sigma_vqs[k, i] = (znd[k, i] - eta2[i]) / (eta2[i] + dp[i])

                for k in range(1, kbp[i]):
                    if sigma_vqs[k, i] >= sigma_vqs[k - 1, i]:
                        raise ValueError(
                            f"Inverted sigma: node={i + 1}, k={k + 1}, depth={dp[i]}, "
                            f"sigma(k)={sigma_vqs[k, i]}, sigma(k-1)={sigma_vqs[k - 1, i]}"
                        )

        for k in range(1, nvrt + 1):
            level = nvrt - k
            line = f"{k:10d}" + "".join(f" {sigma_vqs[level, i]:14.6f}" for i in range(int(gd.np)))
            f.write(line + "\n")

    return nvrt


def _iter_tokens(path: str) -> Iterable[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            for tok in line.strip().split():
                yield tok


def read_vgrid_in(path: str, npnt: int):
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
            _ = int(next(toks))  # level index
        except StopIteration as exc:
            raise ValueError(f"{path}: missing sigma level index at k={k + 1}") from exc
        for i in range(npnt):
            try:
                sigma[k, i] = float(next(toks))
            except StopIteration as exc:
                raise ValueError(f"{path}: missing sigma value at k={k + 1}, node={i + 1}") from exc

    return {"ivcor": ivcor, "nvrt": nvrt, "kshift": kshift, "sigma": sigma}


def compare_vgrid_in(ref_vgrid_path: str, gen_vgrid_path: str, hgrid_path: str):
    gd = read_schism_hgrid(hgrid_path)
    npnt = int(gd.np)

    ref = read_vgrid_in(ref_vgrid_path, npnt)
    gen = read_vgrid_in(gen_vgrid_path, npnt)

    print("Comparison (vgrid.in):")
    print(f"- ref: {ref_vgrid_path}")
    print(f"- gen: {gen_vgrid_path}")
    print(f"- np (from hgrid): {npnt}")

    if ref["ivcor"] != gen["ivcor"]:
        print(f"- ivcor mismatch: ref={ref['ivcor']} gen={gen['ivcor']}")

    if ref["nvrt"] != gen["nvrt"]:
        print(f"- nvrt mismatch: ref={ref['nvrt']} gen={gen['nvrt']}")

    # kshift comparison
    nk = min(ref["kshift"].size, gen["kshift"].size)
    kdiff = gen["kshift"][:nk] - ref["kshift"][:nk]
    print(f"- kshift: max_abs={int(np.max(np.abs(kdiff)))} mismatch_count={int(np.count_nonzero(kdiff))}/{nk}")

    # sigma comparison over overlap
    ks = min(ref["sigma"].shape[0], gen["sigma"].shape[0])
    ns = min(ref["sigma"].shape[1], gen["sigma"].shape[1])
    d = gen["sigma"][:ks, :ns] - ref["sigma"][:ks, :ns]
    max_abs = float(np.nanmax(np.abs(d)))
    rms = float(np.sqrt(np.nanmean(d * d)))
    denom = np.maximum(np.maximum(np.abs(ref["sigma"][:ks, :ns]), np.abs(gen["sigma"][:ks, :ns])), 1.0e-12)
    max_rel = float(np.nanmax(np.abs(d) / denom))
    print(f"- sigma: overlap=({ks},{ns}) max_abs={max_abs:.6g} rms={rms:.6g} max_rel={max_rel:.6g}")


def _compare_two_vgrids(ref: dict, cmp: dict):
    out = {
        "ivcor_ref": int(ref["ivcor"]),
        "ivcor_cmp": int(cmp["ivcor"]),
        "ivcor_equal": int(ref["ivcor"] == cmp["ivcor"]),
        "nvrt_ref": int(ref["nvrt"]),
        "nvrt_cmp": int(cmp["nvrt"]),
        "nvrt_diff": int(cmp["nvrt"] - ref["nvrt"]),
    }

    nk = min(ref["kshift"].size, cmp["kshift"].size)
    kdiff = cmp["kshift"][:nk] - ref["kshift"][:nk]
    out["kshift_n"] = int(nk)
    out["kshift_mismatch_count"] = int(np.count_nonzero(kdiff))
    out["kshift_max_abs"] = int(np.max(np.abs(kdiff))) if nk > 0 else 0

    ks = min(ref["sigma"].shape[0], cmp["sigma"].shape[0])
    ns = min(ref["sigma"].shape[1], cmp["sigma"].shape[1])
    out["sigma_overlap_k"] = int(ks)
    out["sigma_overlap_n"] = int(ns)
    if ks > 0 and ns > 0:
        d = cmp["sigma"][:ks, :ns] - ref["sigma"][:ks, :ns]
        out["sigma_max_abs"] = float(np.nanmax(np.abs(d)))
        out["sigma_rms"] = float(np.sqrt(np.nanmean(d * d)))
        denom = np.maximum(
            np.maximum(np.abs(ref["sigma"][:ks, :ns]), np.abs(cmp["sigma"][:ks, :ns])),
            1.0e-12,
        )
        out["sigma_max_rel"] = float(np.nanmax(np.abs(d) / denom))
    else:
        out["sigma_max_abs"] = np.nan
        out["sigma_rms"] = np.nan
        out["sigma_max_rel"] = np.nan
    return out


def _sigma_note(max_abs, rms, max_rel):
    if not (np.isfinite(max_abs) and np.isfinite(rms) and np.isfinite(max_rel)):
        return "not comparable (no sigma overlap found)"
    if max_abs <= 1.0e-12 and rms <= 1.0e-12:
        return "exact match (within file read precision)"
    if max_abs <= 1.0e-6:
        return "very small numerical differences"
    return "meaningful vertical-coordinate differences"


def _is_identical_compare(stat):
    sigma_ok = (
        np.isfinite(stat["sigma_max_abs"])
        and np.isfinite(stat["sigma_rms"])
        and float(stat["sigma_max_abs"]) <= 1.0e-12
        and float(stat["sigma_rms"]) <= 1.0e-12
    )
    return bool(
        int(stat["ivcor_equal"]) == 1
        and int(stat["nvrt_diff"]) == 0
        and int(stat["kshift_mismatch_count"]) == 0
        and sigma_ok
    )


def _status(flag):
    return "PASS" if bool(flag) else "DIFF"


def compare_vgrid_collection(compare_cfg, generated_vgrid_path, hgrid_default, plot_cfg=None):
    cfg = dict(DEFAULT_COMPARE_CFG)
    if compare_cfg:
        cfg.update(compare_cfg)
    if not bool(cfg.get("enable", False)):
        return

    raw_paths = list(cfg.get("vgrid_paths", []))
    if len(raw_paths) < 2:
        print("[WARN] compare config skipped: need at least 2 vgrid_paths", file=sys.stderr)
        return

    vgrid_paths = []
    for p in raw_paths:
        sp = str(p).strip()
        if sp in {"{generated}", "generated", "@generated"}:
            sp = str(generated_vgrid_path)
        vgrid_paths.append(sp)

    labels = cfg.get("labels")
    if labels is None:
        labels = [os.path.basename(p) for p in vgrid_paths]
    else:
        labels = [str(x) for x in labels]
        if len(labels) != len(vgrid_paths):
            raise ValueError("COMPARE_CFG.labels length must match COMPARE_CFG.vgrid_paths length")

    baseline_index = int(cfg.get("baseline_index", 0))
    if baseline_index < 0 or baseline_index >= len(vgrid_paths):
        raise ValueError(f"COMPARE_CFG.baseline_index out of range: {baseline_index}")

    hgrid_path = str(cfg.get("hgrid_path") or hgrid_default)
    if not os.path.exists(hgrid_path):
        raise FileNotFoundError(f"Compare hgrid not found: {hgrid_path}")

    for p in vgrid_paths:
        if not os.path.exists(p):
            raise FileNotFoundError(f"Compare vgrid not found: {p}")

    gd = read_schism_hgrid(hgrid_path)
    npnt = int(gd.np)
    parsed = [read_vgrid_in(p, npnt) for p in vgrid_paths]

    ref = parsed[baseline_index]
    ref_label = labels[baseline_index]

    print("Comparison (multi-vgrid):")
    print(f"- hgrid: {hgrid_path}")
    print(f"- baseline: [{baseline_index}] {ref_label} -> {vgrid_paths[baseline_index]}")

    rows = []
    for i, (lbl, path, cur) in enumerate(zip(labels, vgrid_paths, parsed)):
        if i == baseline_index:
            continue
        stat = _compare_two_vgrids(ref, cur)
        identical = _is_identical_compare(stat)
        sigma_note = _sigma_note(stat["sigma_max_abs"], stat["sigma_rms"], stat["sigma_max_rel"])
        row = {
            "baseline_label": ref_label,
            "baseline_path": vgrid_paths[baseline_index],
            "compare_label": lbl,
            "compare_path": path,
            "overall_identical": int(identical),
            "overall_result": "IDENTICAL" if identical else "DIFFERENT",
            "sigma_note": sigma_note,
            **stat,
        }
        rows.append(row)
        print(f"- vs [{i}] {lbl}")
        print(f"  file: {path}")
        print(f"  overall: {row['overall_result']} (1 means same structure and sigma values)")
        print(
            f"  ivcor (vertical-coordinate type): "
            f"{_status(row['ivcor_equal'] == 1)} "
            f"(baseline={row['ivcor_ref']}, compare={row['ivcor_cmp']})"
        )
        print(
            f"  nvrt (total vertical levels): "
            f"{_status(row['nvrt_diff'] == 0)} "
            f"(baseline={row['nvrt_ref']}, compare={row['nvrt_cmp']}, diff={row['nvrt_diff']})"
        )
        print(
            f"  kshift (bottom level index per node): "
            f"{_status(row['kshift_mismatch_count'] == 0)} "
            f"({row['kshift_mismatch_count']} mismatches out of {row['kshift_n']} nodes, "
            f"max_abs_diff={row['kshift_max_abs']})"
        )
        print(
            f"  sigma overlap checked: {row['sigma_overlap_k']} levels x {row['sigma_overlap_n']} nodes"
        )
        print(
            f"  sigma differences: max_abs={row['sigma_max_abs']:.6g}, "
            f"rms={row['sigma_rms']:.6g}, max_rel={row['sigma_max_rel']:.6g}"
        )
        print(f"  sigma interpretation: {row['sigma_note']}")

    if bool(cfg.get("save_csv", True)) and len(rows) > 0:
        csv_path = str(cfg.get("csv_path", "vgrid_compare.csv"))
        fields = [
            "baseline_label",
            "baseline_path",
            "compare_label",
            "compare_path",
            "overall_identical",
            "overall_result",
            "ivcor_ref",
            "ivcor_cmp",
            "ivcor_equal",
            "nvrt_ref",
            "nvrt_cmp",
            "nvrt_diff",
            "kshift_n",
            "kshift_mismatch_count",
            "kshift_max_abs",
            "sigma_overlap_k",
            "sigma_overlap_n",
            "sigma_max_abs",
            "sigma_rms",
            "sigma_max_rel",
            "sigma_note",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for row in rows:
                w.writerow(row)
        print(f"- compare csv written: {csv_path}")

    if bool(cfg.get("plot_transect_sections", False)):
        if plot_cfg is None:
            plot_cfg = {}
        tr_paths_cfg = cfg.get("transect_paths")
        if tr_paths_cfg is None:
            tr_paths = _parse_transect_list(DEFAULT_PLOT_CFG.get("transect_paths"), None)
        else:
            tr_paths = _parse_transect_list(tr_paths_cfg, None)
        tr_paths = [p for p in tr_paths if os.path.exists(p)]
        if len(tr_paths) == 0:
            print("[WARN] compare transect plot skipped: no valid transect bp files", file=sys.stderr)
            return
        vgrid_items = [{"label": lbl, "path": path, "vgrid": vg} for lbl, path, vg in zip(labels, vgrid_paths, parsed)]
        saved = plot_transect_sections_compare(gd, vgrid_items, tr_paths, plot_cfg, cfg)
        if len(saved) > 0:
            print(
                f"- compare transect plots saved ({len(saved)} files) with prefix: "
                f"{cfg.get('transect_prefix', 'vgrid_compare_transect')}"
            )
        elif bool(plot_cfg.get("show_plots", False)):
            print("- compare transect plots shown (not saved)")


def _parse_transect_list(cfg_paths, arg_paths):
    out = []
    if cfg_paths:
        out.extend([str(p).strip() for p in cfg_paths if str(p).strip()])
    if arg_paths:
        for entry in arg_paths:
            for part in str(entry).split(","):
                part = part.strip()
                if part:
                    out.append(part)
    # Keep order, drop duplicates.
    return list(dict.fromkeys(out))


def _looks_like_lonlat(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return bool(
        np.all(np.isfinite(x))
        and np.all(np.isfinite(y))
        and np.nanmin(x) >= -360.0
        and np.nanmax(x) <= 360.0
        and np.nanmin(y) >= -90.0
        and np.nanmax(y) <= 90.0
    )


def _haversine_m(lon1, lat1, lon2, lat2):
    # Great-circle distance on a spherical Earth.
    r_earth = 6371000.0
    lon1 = np.deg2rad(float(lon1))
    lat1 = np.deg2rad(float(lat1))
    lon2 = np.deg2rad(float(lon2))
    lat2 = np.deg2rad(float(lat2))
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(a))
    return r_earth * c


def _build_transect_profile(bp_path, gd, kbp, dp, znd):
    bp = read_schism_bpfile(bp_path)

    nsta = int(bp.nsta)
    if nsta < 2:
        raise ValueError(f"{bp_path}: transect needs at least 2 points")

    is_lonlat = _looks_like_lonlat(bp.x, bp.y)
    dist = np.zeros(nsta, dtype=float)
    for i in range(1, nsta):
        dx = float(bp.x[i] - bp.x[i - 1])
        dy = float(bp.y[i] - bp.y[i - 1])
        if is_lonlat:
            dist[i] = dist[i - 1] + _haversine_m(bp.x[i - 1], bp.y[i - 1], bp.x[i], bp.y[i])
        else:
            dist[i] = dist[i - 1] + np.hypot(dx, dy)

    imap = near_pts(np.c_[bp.x, bp.y], np.c_[gd.x, gd.y])
    depth = np.asarray(dp[imap], dtype=float)
    nlev = np.asarray(kbp[imap], dtype=int)
    nvrt = int(znd.shape[0])

    zplot = np.full((nsta, nvrt), np.nan, dtype=float)
    for i, nd in enumerate(imap):
        nl = int(nlev[i])
        zplot[i, :nl] = znd[:nl, nd]

    seg_flag = None
    if hasattr(bp, "dp"):
        try:
            seg_flag = np.asarray(bp.dp, dtype=float)
        except Exception:
            seg_flag = None

    segments = []
    start = 0
    if seg_flag is not None and seg_flag.size == nsta:
        for i in range(1, nsta):
            if abs(float(seg_flag[i]) - float(seg_flag[i - 1])) > 1.0e-6:
                if i - start >= 2:
                    segments.append((start, i - 1))
                start = i
    if nsta - start >= 2:
        segments.append((start, nsta - 1))
    if not segments:
        segments = [(0, nsta - 1)]

    return {
        "bp_path": bp_path,
        "x": np.asarray(bp.x, dtype=float),
        "y": np.asarray(bp.y, dtype=float),
        "dist": dist,
        "dist_units": "m" if is_lonlat else "grid",
        "depth": depth,
        "zplot": zplot,
        "segments": segments,
    }


def _build_transect_profile_from_vgrid(bp_path, gd, vgrid):
    bp = read_schism_bpfile(bp_path)

    nsta = int(bp.nsta)
    if nsta < 2:
        raise ValueError(f"{bp_path}: transect needs at least 2 points")

    is_lonlat = _looks_like_lonlat(bp.x, bp.y)
    dist = np.zeros(nsta, dtype=float)
    for i in range(1, nsta):
        dx = float(bp.x[i] - bp.x[i - 1])
        dy = float(bp.y[i] - bp.y[i - 1])
        if is_lonlat:
            dist[i] = dist[i - 1] + _haversine_m(bp.x[i - 1], bp.y[i - 1], bp.x[i], bp.y[i])
        else:
            dist[i] = dist[i - 1] + np.hypot(dx, dy)

    imap = near_pts(np.c_[bp.x, bp.y], np.c_[gd.x, gd.y])
    depth = np.asarray(gd.dp[imap], dtype=float)

    sigma = np.asarray(vgrid["sigma"], dtype=float)  # [nvrt, np]
    kshift = np.asarray(vgrid["kshift"], dtype=int)  # [np]
    nvrt = int(vgrid["nvrt"])

    zplot = np.full((nsta, nvrt), np.nan, dtype=float)
    for i, nd in enumerate(imap):
        sig = sigma[:, nd]
        ks = int(kshift[nd])  # 1-based first valid level index in file order
        valid = np.arange(1, nvrt + 1) >= ks
        valid &= np.isfinite(sig) & (sig >= -1.5) & (sig <= 0.5)
        zvals = sig * depth[i]  # eta=0 reference
        zplot[i, valid] = zvals[valid]

    seg_flag = None
    if hasattr(bp, "dp"):
        try:
            seg_flag = np.asarray(bp.dp, dtype=float)
        except Exception:
            seg_flag = None

    segments = []
    start = 0
    if seg_flag is not None and seg_flag.size == nsta:
        for i in range(1, nsta):
            if abs(float(seg_flag[i]) - float(seg_flag[i - 1])) > 1.0e-6:
                if i - start >= 2:
                    segments.append((start, i - 1))
                start = i
    if nsta - start >= 2:
        segments.append((start, nsta - 1))
    if not segments:
        segments = [(0, nsta - 1)]

    return {
        "bp_path": bp_path,
        "x": np.asarray(bp.x, dtype=float),
        "y": np.asarray(bp.y, dtype=float),
        "dist": dist,
        "dist_units": "m" if is_lonlat else "grid",
        "depth": depth,
        "zplot": zplot,
        "segments": segments,
    }


def _plot_transect_axis(ax, tr, marker_size):
    dist = tr["dist"]
    zplot = tr["zplot"]
    depth = tr["depth"]

    for i in range(zplot.shape[0]):
        zi = zplot[i, :]
        valid = np.isfinite(zi)
        if np.any(valid):
            ax.plot(np.full(valid.sum(), dist[i]), zi[valid], "k", lw=0.4)

    for s0, s1 in tr["segments"]:
        ax.plot(dist[s0 : s1 + 1], zplot[s0 : s1 + 1, :], "k-", lw=0.5)

    ax.plot(dist, -depth, "r.", ms=marker_size)
    dunits = str(tr.get("dist_units", "m"))
    if dunits == "m":
        ax.set_xlabel("Along transect distance (m)")
    else:
        ax.set_xlabel("Along transect distance (grid units)")
    ax.set_ylabel("Depth (m)")
    ax.grid(alpha=0.25)


def plot_layer_map(gd, kbp, transect_profiles, plot_cfg):
    import matplotlib.pyplot as plt

    save_plots = bool(plot_cfg.get("save_plots", True))
    show_plots = bool(plot_cfg.get("show_plots", False))

    fig, ax = plt.subplots(1, 1, figsize=(8.5, 7.5))
    try:
        gd.plot_bnd(ax=ax)
    except Exception:
        pass

    sc = ax.scatter(
        np.asarray(gd.x, dtype=float),
        np.asarray(gd.y, dtype=float),
        c=np.asarray(kbp, dtype=float),
        s=float(plot_cfg.get("node_size", 8)),
        cmap="viridis",
        edgecolors="none",
        alpha=0.95,
    )

    for tr in transect_profiles:
        ax.plot(tr["x"], tr["y"], "r-", lw=1.2)
        ax.plot(tr["x"], tr["y"], "r.", ms=3)

    cbar = fig.colorbar(sc, ax=ax)
    cbar.set_label("# vertical layers")
    ax.set_title("Node-wise vertical layers")
    ax.set_xlabel("X / Lon")
    ax.set_ylabel("Y / Lat")
    ax.set_aspect("equal", adjustable="datalim")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    out_png = None
    if save_plots:
        out_png = str(plot_cfg.get("layer_map_path", "vgrid_layers_map.png"))
        fig.savefig(out_png, dpi=int(plot_cfg.get("dpi", 200)))
    if show_plots:
        plt.show()
    plt.close(fig)
    return out_png


def plot_transect_sections(gd, transect_profiles, plot_cfg):
    import matplotlib.pyplot as plt

    save_plots = bool(plot_cfg.get("save_plots", True))
    show_plots = bool(plot_cfg.get("show_plots", False))

    dpi = int(plot_cfg.get("dpi", 200))
    prefix = str(plot_cfg.get("transect_prefix", "vgrid_transect"))
    ms = float(plot_cfg.get("transect_marker_size", 6))
    saved = []

    for idx, tr in enumerate(transect_profiles, start=1):
        fig = plt.figure(figsize=(10, 9))
        gs = fig.add_gridspec(2, 1)
        ax_map = fig.add_subplot(gs[0, 0])
        ax_tran = fig.add_subplot(gs[1, 0])

        try:
            gd.plot_bnd(ax=ax_map)
        except Exception:
            pass
        ax_map.plot(tr["x"], tr["y"], "r-", lw=1.2)
        ax_map.plot(tr["x"], tr["y"], "r.", ms=4)
        ax_map.set_title(f"Transect map: {os.path.basename(tr['bp_path'])}")
        ax_map.set_xlabel("X / Lon")
        ax_map.set_ylabel("Y / Lat")
        ax_map.set_aspect("equal", adjustable="datalim")

        _plot_transect_axis(ax_tran, tr, ms)
        ax_tran.set_title("Transect vertical structure")

        fig.tight_layout()
        out_png = f"{prefix}_{idx}.png"
        if save_plots:
            fig.savefig(out_png, dpi=dpi)
            saved.append(out_png)
        if show_plots:
            plt.show()
        plt.close(fig)
    return saved


def plot_transect_sections_compare(gd, vgrid_items, transect_paths, plot_cfg, compare_cfg):
    import matplotlib.pyplot as plt

    save_plots = bool(plot_cfg.get("save_plots", True))
    show_plots = bool(plot_cfg.get("show_plots", False))
    dpi = int(plot_cfg.get("dpi", 200))
    ms = float(plot_cfg.get("transect_marker_size", 6))
    prefix = str(compare_cfg.get("transect_prefix", "vgrid_compare_transect"))
    saved = []

    for idx, bp_path in enumerate(transect_paths, start=1):
        per_grid = []
        for item in vgrid_items:
            try:
                per_grid.append(
                    (item["label"], _build_transect_profile_from_vgrid(bp_path, gd, item["vgrid"]))
                )
            except Exception as exc:
                print(
                    f"[WARN] Failed compare transect profile for {bp_path} ({item['label']}): {exc}",
                    file=sys.stderr,
                )
        if len(per_grid) == 0:
            continue

        nrows = 1 + len(per_grid)
        fig = plt.figure(figsize=(10.5, 3.0 * nrows))
        gs = fig.add_gridspec(nrows, 1)
        ax_map = fig.add_subplot(gs[0, 0])
        try:
            gd.plot_bnd(ax=ax_map)
        except Exception:
            pass
        tr0 = per_grid[0][1]
        ax_map.plot(tr0["x"], tr0["y"], "r-", lw=1.2)
        ax_map.plot(tr0["x"], tr0["y"], "r.", ms=4)
        ax_map.set_title(f"Transect map: {os.path.basename(bp_path)}")
        ax_map.set_xlabel("X / Lon")
        ax_map.set_ylabel("Y / Lat")
        ax_map.set_aspect("equal", adjustable="datalim")

        for r, (label, tr) in enumerate(per_grid, start=1):
            ax = fig.add_subplot(gs[r, 0])
            _plot_transect_axis(ax, tr, ms)
            ax.set_title(f"Transect vertical structure: {label}")

        fig.tight_layout()
        out_png = f"{prefix}_{idx}.png"
        if save_plots:
            fig.savefig(out_png, dpi=dpi)
            saved.append(out_png)
        if show_plots:
            plt.show()
        plt.close(fig)
    return saved


def main():
    parser = argparse.ArgumentParser(description="Generate SCHISM VQS vgrid.in and QC plots.")
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
        help="Directory containing reference vgrid.in for comparison.",
    )
    parser.add_argument("--no-plot", action="store_true", help="Disable all plotting.")
    parser.add_argument("--no-layer-map", action="store_true", help="Disable 2D layer map plot.")
    parser.add_argument("--no-transect-plot", action="store_true", help="Disable transect section plots.")
    parser.add_argument("--show-plots", action="store_true", help="Show plots interactively.")
    parser.add_argument("--save-plots", dest="save_plots", action="store_true", help="Save plots to image files.")
    parser.add_argument("--no-save-plots", dest="save_plots", action="store_false", help="Do not save plot image files.")
    parser.set_defaults(save_plots=None)
    parser.add_argument("--plot-prefix", default=None, help="Override transect plot prefix.")
    args = parser.parse_args()

    if not os.path.exists(args.hgrid):
        print(f"Missing hgrid file: {args.hgrid}", file=sys.stderr)
        return 1

    params = build_master_grid(DEFAULT_VQS_CFG)
    gd = read_schism_hgrid(args.hgrid)
    vgrid_data = compute_vgrid(gd, params)

    nprism = 0
    for i in range(int(gd.ne)):
        nodes = gd.elnode[i, : gd.i34[i]]
        nprism += int(vgrid_data["kbp"][nodes].max())

    avg_layers = nprism / float(gd.ne)
    print(f"Final nvrt={int(vgrid_data['kbp'].max())}")
    print(f"# of prisms={nprism}")
    print(f"Average # of layers={avg_layers}")

    nvrt = write_vgrid_in(gd, params, vgrid_data, path="vgrid.in")
    print(f"vgrid.in written with nvrt={nvrt}")

    plot_cfg = dict(DEFAULT_PLOT_CFG)
    if args.no_plot:
        plot_cfg["enable"] = False
    if args.no_layer_map:
        plot_cfg["plot_layer_map"] = False
    if args.no_transect_plot:
        plot_cfg["plot_transect_sections"] = False
    if args.show_plots:
        plot_cfg["show_plots"] = True
    if args.save_plots is not None:
        plot_cfg["save_plots"] = bool(args.save_plots)
    if args.plot_prefix:
        plot_cfg["transect_prefix"] = str(args.plot_prefix)

    transect_paths = _parse_transect_list(plot_cfg.get("transect_paths"), args.transect)
    transect_profiles = []
    for bp_path in transect_paths:
        if not os.path.exists(bp_path):
            print(f"[WARN] Missing transect file: {bp_path}; skip", file=sys.stderr)
            continue
        try:
            transect_profiles.append(_build_transect_profile(bp_path, gd, vgrid_data["kbp"], vgrid_data["dp"], vgrid_data["znd"]))
        except Exception as exc:
            print(f"[WARN] Failed transect profile build for {bp_path}: {exc}", file=sys.stderr)

    if bool(plot_cfg.get("enable", True)):
        if bool(plot_cfg.get("plot_layer_map", True)):
            layer_map = plot_layer_map(gd, vgrid_data["kbp"], transect_profiles, plot_cfg)
            if layer_map is not None:
                print(f"Layer map saved: {layer_map}")
            elif bool(plot_cfg.get("show_plots", False)):
                print("Layer map shown (not saved).")
        if bool(plot_cfg.get("plot_transect_sections", True)) and transect_profiles:
            saved = plot_transect_sections(gd, transect_profiles, plot_cfg)
            if len(saved) > 0:
                print(f"Transect section plots saved ({len(saved)} files) with prefix: {plot_cfg.get('transect_prefix', 'vgrid_transect')}")
            elif bool(plot_cfg.get("show_plots", False)):
                print("Transect section plots shown (not saved).")

    if args.compare:
        ref_vgrid = os.path.join(args.compare, "vgrid.in")
        if not os.path.exists(ref_vgrid):
            print(f"[WARN] compare skipped: missing reference {ref_vgrid}", file=sys.stderr)
        else:
            compare_vgrid_in(ref_vgrid, "vgrid.in", args.hgrid)
    compare_vgrid_collection(DEFAULT_COMPARE_CFG, "vgrid.in", args.hgrid, plot_cfg=plot_cfg)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
