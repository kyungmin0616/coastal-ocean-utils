#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compare SCHISM ocean-current collocation outputs against observation currents.

Inputs are paired files produced by pextract_schism_OC.py (CSV or NPZ), where
observation and model values are already collocated in space/time/depth.
"""

from __future__ import annotations

# =============================================================================
# Configuration
# =============================================================================
CONFIG = dict(
    model={
        "pair_files": [
            "./npz/RUN01d_jodc_ca_schism_pairs.npz",
        ],
        "labels": ["RUN01d"],
    },
    filter={
        "data_types": ["CA"],
        "sources": None,
        "start": None,  # UTC: YYYY-MM-DD[ HH:MM[:SS]]
        "end": None,  # UTC: YYYY-MM-DD[ HH:MM[:SS]]
        "min_depth": None,
        "max_depth": None,
        "max_abs_dt_hours": 6.0,
        "require_matched": True,
        "require_inside_domain": True,
        "exclude_reject_reasons": ["outside_domain"],
    },
    time={
        "resample": None,  # None|H|D|M...
    },
    output={
        "dir": "./CompObs/CompTEAMS_OC",
        "task_name": "oc",
        "experiment_id": None,
        "write_task_metrics": True,
        "save_plots": True,
        "write_scatter_plots": True,
        "metrics_raw_name": "OC_metrics_raw.csv",
        "metrics_segment_name": "OC_stats_by_segment.csv",
        "metrics_model_name": "OC_stats_by_model.csv",
        "manifest_name": "OC_manifest.json",
    },
    plot={
        "top_segments_per_source": 8,
        "scatter_alpha": 0.55,
        "scatter_size": 8,
        "scatter_cmap": "viridis",
    },
    debug={
        "times": False,
    },
)

# =============================================================================
# Imports
# =============================================================================
import argparse
import copy
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from pylib import (
    loadz,
    num2date,
    deep_update_dict,
    init_mpi_runtime,
    rank_log,
    compute_skill_metrics,
    write_csv_rows,
)


OC_RAW_FIELDS = [
    "task",
    "experiment_id",
    "model",
    "source",
    "data_type",
    "segment_key",
    "segment_id",
    "track_id",
    "track_file",
    "time_utc",
    "depth",
    "obs_u",
    "obs_v",
    "obs_speed",
    "mod_u",
    "mod_v",
    "mod_speed",
    "error_u",
    "error_v",
    "error_speed",
    "mod_dt_hours",
]

OC_SEGMENT_FIELDS = [
    "model",
    "source",
    "data_type",
    "segment_key",
    "segment_id",
    "track_id",
    "track_file",
    "n",
    "time_start",
    "time_end",
    "depth_mean",
    "depth_min",
    "depth_max",
    "bias_u",
    "rmse_u",
    "corr_u",
    "bias_v",
    "rmse_v",
    "corr_v",
    "bias_speed",
    "rmse_speed",
    "corr_speed",
    "vector_rmse",
    "dir_mae_deg",
]

OC_MODEL_FIELDS = [
    "model",
    "source",
    "data_type",
    "n",
    "bias_u",
    "rmse_u",
    "corr_u",
    "bias_v",
    "rmse_v",
    "corr_v",
    "bias_speed",
    "rmse_speed",
    "corr_speed",
    "vector_rmse",
    "dir_mae_deg",
]


# =============================================================================
# MPI setup
# =============================================================================
MPI, COMM, RANK, SIZE, USE_MPI = init_mpi_runtime(sys.argv)
PATH_BASE_DIR = Path.cwd()


# =============================================================================
# Core helpers
# =============================================================================
def rank_print(*args: Any, **kwargs: Any) -> None:
    rank0_only = bool(kwargs.pop("rank0_only", True))
    msg = " ".join(str(a) for a in args)
    rank_log(msg, rank=RANK, size=SIZE, rank0_only=rank0_only)


def _set_path_base(config_path_like: Optional[str]) -> Optional[Path]:
    global PATH_BASE_DIR
    if config_path_like:
        cfg_path = Path(str(config_path_like)).expanduser()
        if not cfg_path.is_absolute():
            cfg_path = (Path.cwd() / cfg_path).resolve()
        else:
            cfg_path = cfg_path.resolve()
        PATH_BASE_DIR = cfg_path.parent
        return cfg_path
    PATH_BASE_DIR = Path.cwd()
    return None


def _resolve_path(path_like: str) -> Path:
    path = Path(str(path_like)).expanduser()
    if path.is_absolute():
        return path
    return (PATH_BASE_DIR / path).resolve()


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    return deep_update_dict(base, override, merge_list_of_dicts=False)


def _sanitize_name(text: str) -> str:
    out = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in str(text))
    out = out.strip("_")
    return out if out else "x"


def _parse_time_bound(value: Optional[str], is_end: bool) -> Optional[pd.Timestamp]:
    if value is None:
        return None
    txt = str(value).strip()
    if txt == "":
        return None
    ts = pd.to_datetime(txt, utc=True, errors="coerce")
    if pd.isna(ts):
        raise ValueError(f"Invalid datetime string: {value}")
    if is_end and len(txt) == 10:
        ts = ts + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    return ts


def _coerce_datetime(df: pd.DataFrame) -> pd.Series:
    if "obs_time_utc" in df.columns:
        t = pd.to_datetime(df["obs_time_utc"], utc=True, errors="coerce")
    else:
        t = pd.Series(pd.NaT, index=df.index, dtype="datetime64[ns, UTC]")

    if "obs_time_num" in df.columns:
        miss = t.isna()
        if miss.any():
            nums = pd.to_numeric(df.loc[miss, "obs_time_num"], errors="coerce").to_numpy()
            vals: List[pd.Timestamp] = []
            for x in nums:
                if np.isfinite(x):
                    vals.append(pd.Timestamp(num2date(float(x)), tz="UTC"))
                else:
                    vals.append(pd.NaT)
            t.loc[miss] = vals
    return t


def _from_npz(path: Path) -> pd.DataFrame:
    S = loadz(str(path))
    cols = [
        "obs_time_num",
        "obs_time_utc",
        "obs_lon",
        "obs_lat",
        "obs_depth",
        "obs_u",
        "obs_v",
        "obs_speed",
        "obs_dir",
        "obs_qc_u",
        "obs_qc_v",
        "obs_source",
        "obs_data_type",
        "obs_track_id",
        "obs_segment_id",
        "obs_track_file",
        "mod_time_num",
        "mod_time_utc",
        "mod_stack",
        "mod_time_index",
        "mod_dt_hours",
        "mod_u",
        "mod_v",
        "mod_speed",
        "mod_dir",
        "matched",
        "reject_reason",
        "elem_id",
        "inside_domain",
        "depth_interp_ok",
    ]
    data = {}
    n = None
    for c in cols:
        if hasattr(S, c):
            arr = np.asarray(getattr(S, c)).reshape(-1)
            data[c] = arr
            if n is None:
                n = len(arr)
    if not data:
        raise ValueError(f"No known paired fields found in NPZ: {path}")
    if n is None:
        raise ValueError(f"Empty NPZ content: {path}")
    for c, arr in list(data.items()):
        if len(arr) == n:
            continue
        if len(arr) == 1 and n > 1:
            data[c] = np.repeat(arr, n)
        else:
            raise ValueError(f"Inconsistent NPZ field length ({c}): {len(arr)} != {n}")
    return pd.DataFrame(data)


def _load_pair_table(path: Path, label: str) -> pd.DataFrame:
    if path.suffix.lower() == ".npz":
        df = _from_npz(path)
    else:
        df = pd.read_csv(path)

    df["model"] = str(label)
    if "obs_source" not in df.columns:
        df["obs_source"] = ""
    if "obs_data_type" not in df.columns:
        df["obs_data_type"] = ""
    if "obs_track_file" not in df.columns:
        df["obs_track_file"] = ""

    for c in [
        "obs_u",
        "obs_v",
        "mod_u",
        "mod_v",
        "obs_depth",
        "mod_dt_hours",
        "obs_segment_id",
        "obs_track_id",
        "matched",
        "inside_domain",
    ]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    if "obs_speed" not in df.columns:
        df["obs_speed"] = np.nan
    if "mod_speed" not in df.columns:
        df["mod_speed"] = np.nan

    miss_obs_speed = ~np.isfinite(pd.to_numeric(df["obs_speed"], errors="coerce"))
    df.loc[miss_obs_speed, "obs_speed"] = np.sqrt(df.loc[miss_obs_speed, "obs_u"] ** 2 + df.loc[miss_obs_speed, "obs_v"] ** 2)

    miss_mod_speed = ~np.isfinite(pd.to_numeric(df["mod_speed"], errors="coerce"))
    df.loc[miss_mod_speed, "mod_speed"] = np.sqrt(df.loc[miss_mod_speed, "mod_u"] ** 2 + df.loc[miss_mod_speed, "mod_v"] ** 2)

    if "reject_reason" not in df.columns:
        df["reject_reason"] = ""
    df["reject_reason"] = df["reject_reason"].fillna("").astype(str)

    t = _coerce_datetime(df)
    df["time_utc"] = t

    seg = pd.to_numeric(df.get("obs_segment_id", np.nan), errors="coerce")
    trk = pd.to_numeric(df.get("obs_track_id", np.nan), errors="coerce")
    seg_key = []
    for i in range(len(df)):
        s = seg.iloc[i] if hasattr(seg, "iloc") else np.nan
        k = trk.iloc[i] if hasattr(trk, "iloc") else np.nan
        if np.isfinite(s) and int(s) > 0:
            seg_key.append(f"seg_{int(s)}")
        elif np.isfinite(k) and int(k) > 0:
            seg_key.append(f"track_{int(k)}")
        else:
            tf = str(df.iloc[i].get("obs_track_file", "")).strip()
            if tf != "":
                seg_key.append(f"file_{_sanitize_name(os.path.basename(tf))}")
            else:
                seg_key.append(f"row_{i}")
    df["segment_key"] = np.asarray(seg_key, dtype=object)

    return df


def _load_models(cfg: Dict[str, Any]) -> pd.DataFrame:
    pair_files = list(cfg.get("pair_files") or [])
    labels = cfg.get("labels")
    if labels is None or len(labels) == 0:
        labels = [Path(str(p)).stem for p in pair_files]
    if len(labels) != len(pair_files):
        raise ValueError("model.labels must have same length as model.pair_files")

    all_df: List[pd.DataFrame] = []
    for p, lb in zip(pair_files, labels):
        path = _resolve_path(str(p))
        if not path.is_file():
            raise FileNotFoundError(f"Pair file not found: {path}")
        rank_print(f"Loading {path}", rank0_only=True)
        dfi = _load_pair_table(path, str(lb))
        all_df.append(dfi)

    if len(all_df) == 0:
        raise RuntimeError("No paired files loaded.")

    df = pd.concat(all_df, axis=0, ignore_index=True)
    return df


def _apply_filters(df: pd.DataFrame, cfg: Dict[str, Any]) -> pd.DataFrame:
    out = df.copy()

    for c in ["obs_u", "obs_v", "mod_u", "mod_v", "obs_speed", "mod_speed", "obs_depth"]:
        out[c] = pd.to_numeric(out.get(c), errors="coerce")

    m = pd.Series(True, index=out.index)
    m &= out["time_utc"].notna()
    m &= np.isfinite(out["obs_u"]) & np.isfinite(out["obs_v"])
    m &= np.isfinite(out["mod_u"]) & np.isfinite(out["mod_v"])

    dtypes = cfg.get("data_types")
    if dtypes:
        keep = {str(x).strip().upper() for x in dtypes if str(x).strip() != ""}
        if keep:
            m &= out["obs_data_type"].astype(str).str.upper().isin(keep)

    sources = cfg.get("sources")
    if sources:
        sk = {str(x).strip() for x in sources if str(x).strip() != ""}
        if sk:
            m &= out["obs_source"].astype(str).isin(sk)

    ts0 = _parse_time_bound(cfg.get("start"), is_end=False)
    ts1 = _parse_time_bound(cfg.get("end"), is_end=True)
    if ts0 is not None:
        m &= out["time_utc"] >= ts0
    if ts1 is not None:
        m &= out["time_utc"] <= ts1

    z0 = cfg.get("min_depth")
    z1 = cfg.get("max_depth")
    if z0 is not None:
        m &= out["obs_depth"] >= float(z0)
    if z1 is not None:
        m &= out["obs_depth"] <= float(z1)

    if bool(cfg.get("require_matched", True)) and "matched" in out.columns:
        m &= pd.to_numeric(out["matched"], errors="coerce").fillna(0).astype(int) == 1

    if bool(cfg.get("require_inside_domain", True)) and "inside_domain" in out.columns:
        m &= pd.to_numeric(out["inside_domain"], errors="coerce").fillna(0).astype(int) == 1

    max_abs_dt = cfg.get("max_abs_dt_hours")
    if max_abs_dt is not None and "mod_dt_hours" in out.columns:
        dt = pd.to_numeric(out["mod_dt_hours"], errors="coerce")
        m &= np.isfinite(dt) & (np.abs(dt) <= float(max_abs_dt))

    rr = cfg.get("exclude_reject_reasons")
    if rr:
        bad = {str(x).strip() for x in rr if str(x).strip() != ""}
        if bad:
            m &= ~out["reject_reason"].astype(str).isin(bad)

    out = out.loc[m].copy()
    out.sort_values(["model", "obs_source", "segment_key", "time_utc"], inplace=True)
    out.reset_index(drop=True, inplace=True)
    return out


def _compute_oc_metrics(obs_u: np.ndarray, obs_v: np.ndarray, mod_u: np.ndarray, mod_v: np.ndarray) -> Dict[str, float]:
    obs_u = np.asarray(obs_u, dtype=float)
    obs_v = np.asarray(obs_v, dtype=float)
    mod_u = np.asarray(mod_u, dtype=float)
    mod_v = np.asarray(mod_v, dtype=float)

    m_u = compute_skill_metrics(obs_u, mod_u, min_n=2)
    m_v = compute_skill_metrics(obs_v, mod_v, min_n=2)

    obs_s = np.sqrt(obs_u ** 2 + obs_v ** 2)
    mod_s = np.sqrt(mod_u ** 2 + mod_v ** 2)
    m_s = compute_skill_metrics(obs_s, mod_s, min_n=2)

    valid = np.isfinite(obs_u) & np.isfinite(obs_v) & np.isfinite(mod_u) & np.isfinite(mod_v)
    if np.any(valid):
        du = mod_u[valid] - obs_u[valid]
        dv = mod_v[valid] - obs_v[valid]
        vector_rmse = float(np.sqrt(np.mean(du ** 2 + dv ** 2)))

        od = (np.degrees(np.arctan2(obs_u[valid], obs_v[valid])) + 360.0) % 360.0
        md = (np.degrees(np.arctan2(mod_u[valid], mod_v[valid])) + 360.0) % 360.0
        d = ((md - od + 180.0) % 360.0) - 180.0
        dir_mae = float(np.mean(np.abs(d)))
    else:
        vector_rmse = np.nan
        dir_mae = np.nan

    return {
        "n": int(np.count_nonzero(valid)),
        "bias_u": float(m_u.get("bias", np.nan)),
        "rmse_u": float(m_u.get("rmse", np.nan)),
        "corr_u": float(m_u.get("corr", np.nan)),
        "bias_v": float(m_v.get("bias", np.nan)),
        "rmse_v": float(m_v.get("rmse", np.nan)),
        "corr_v": float(m_v.get("corr", np.nan)),
        "bias_speed": float(m_s.get("bias", np.nan)),
        "rmse_speed": float(m_s.get("rmse", np.nan)),
        "corr_speed": float(m_s.get("corr", np.nan)),
        "vector_rmse": vector_rmse,
        "dir_mae_deg": dir_mae,
    }


def _resample_df(df: pd.DataFrame, freq: Optional[str]) -> pd.DataFrame:
    if not freq:
        return df
    d = df[["time_utc", "obs_u", "obs_v", "mod_u", "mod_v", "obs_depth", "mod_dt_hours"]].copy()
    d = d.dropna(subset=["time_utc"])
    if len(d) == 0:
        return df.iloc[0:0].copy()
    d = d.set_index("time_utc").sort_index()
    r = d.resample(str(freq)).mean(numeric_only=True)
    r["obs_speed"] = np.sqrt(r["obs_u"] ** 2 + r["obs_v"] ** 2)
    r["mod_speed"] = np.sqrt(r["mod_u"] ** 2 + r["mod_v"] ** 2)
    r = r.reset_index()
    return r


def _plot_segment_time_history(seg: pd.DataFrame, out_png: Path, title: str, resample_freq: Optional[str]) -> bool:
    d0 = seg[["time_utc", "obs_u", "obs_v", "mod_u", "mod_v", "obs_speed", "mod_speed"]].copy()
    d0 = d0.dropna(subset=["time_utc"]).sort_values("time_utc")
    if len(d0) < 2:
        return False

    if resample_freq:
        rr = _resample_df(seg, resample_freq)
        if len(rr) >= 2:
            d0 = rr

    fig, axs = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    axs[0].plot(d0["time_utc"], d0["obs_speed"], "o", ms=2.5, alpha=0.75, label="Obs speed")
    axs[0].plot(d0["time_utc"], d0["mod_speed"], "-", lw=1.2, alpha=0.9, label="SCHISM speed")
    axs[0].set_ylabel("Speed (m/s)")
    axs[0].grid(True, alpha=0.3)
    axs[0].legend(loc="best")

    axs[1].plot(d0["time_utc"], d0["obs_u"], "o", ms=2.2, alpha=0.75, label="Obs u")
    axs[1].plot(d0["time_utc"], d0["mod_u"], "-", lw=1.2, alpha=0.9, label="SCHISM u")
    axs[1].set_ylabel("u (m/s)")
    axs[1].grid(True, alpha=0.3)
    axs[1].legend(loc="best")

    axs[2].plot(d0["time_utc"], d0["obs_v"], "o", ms=2.2, alpha=0.75, label="Obs v")
    axs[2].plot(d0["time_utc"], d0["mod_v"], "-", lw=1.2, alpha=0.9, label="SCHISM v")
    axs[2].set_ylabel("v (m/s)")
    axs[2].set_xlabel("UTC datetime")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend(loc="best")

    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def _plot_segment_hodograph(seg: pd.DataFrame, out_png: Path, title: str) -> bool:
    d = seg[["obs_u", "obs_v", "mod_u", "mod_v", "time_utc"]].copy()
    d = d.dropna()
    if len(d) < 3:
        return False

    tnum = np.linspace(0.0, 1.0, len(d))

    fig, ax = plt.subplots(figsize=(6.5, 6))
    so = ax.scatter(d["obs_u"], d["obs_v"], c=tnum, cmap="viridis", s=12, alpha=0.8, label="Obs")
    sm = ax.scatter(d["mod_u"], d["mod_v"], c=tnum, cmap="plasma", s=12, alpha=0.8, label="SCHISM")

    ax.plot(d["obs_u"], d["obs_v"], color="0.5", lw=0.8, alpha=0.6)
    ax.plot(d["mod_u"], d["mod_v"], color="0.2", lw=0.8, alpha=0.6)

    lim = np.nanmax(np.abs(np.r_[d["obs_u"].to_numpy(), d["obs_v"].to_numpy(), d["mod_u"].to_numpy(), d["mod_v"].to_numpy()]))
    if np.isfinite(lim) and lim > 0:
        lim = float(lim) * 1.1
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    ax.axhline(0.0, color="k", lw=0.6, alpha=0.5)
    ax.axvline(0.0, color="k", lw=0.6, alpha=0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, alpha=0.25)
    ax.set_xlabel("u (m/s)")
    ax.set_ylabel("v (m/s)")
    ax.set_title(title)
    ax.legend(loc="best")

    cbar1 = fig.colorbar(so, ax=ax, fraction=0.046, pad=0.04)
    cbar1.set_label("Obs time progression")
    cbar2 = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.11)
    cbar2.set_label("Model time progression")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return True


def _write_integrated_scatter(df: pd.DataFrame, cfg: Dict[str, Any], outdir: Path) -> List[str]:
    files: List[str] = []
    if len(df) == 0:
        return files

    alpha = float(cfg.get("scatter_alpha", 0.55))
    size = float(cfg.get("scatter_size", 8))

    for model_name, dm in df.groupby("model"):
        fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.8))
        specs = [
            ("obs_u", "mod_u", "u (m/s)"),
            ("obs_v", "mod_v", "v (m/s)"),
            ("obs_speed", "mod_speed", "speed (m/s)"),
        ]

        for ax, (ox, mx, ttl) in zip(axs, specs):
            x = pd.to_numeric(dm[ox], errors="coerce").to_numpy()
            y = pd.to_numeric(dm[mx], errors="coerce").to_numpy()
            valid = np.isfinite(x) & np.isfinite(y)
            if not np.any(valid):
                ax.set_title(f"{ttl}: no data")
                continue
            x = x[valid]
            y = y[valid]
            ax.scatter(x, y, s=size, alpha=alpha)
            vmin = float(np.nanmin(np.r_[x, y]))
            vmax = float(np.nanmax(np.r_[x, y]))
            if np.isfinite(vmin) and np.isfinite(vmax):
                ax.plot([vmin, vmax], [vmin, vmax], "k--", lw=1.0, alpha=0.8)
            mm = compute_skill_metrics(x, y, min_n=2)
            ax.text(
                0.03,
                0.97,
                f"R={mm.get('corr', np.nan):.2f}\nRMSE={mm.get('rmse', np.nan):.3f}\nBias={mm.get('bias', np.nan):.3f}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=9,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
            )
            ax.grid(True, alpha=0.3)
            ax.set_xlabel(f"Obs {ttl}")
            ax.set_ylabel(f"Model {ttl}")
            ax.set_title(ttl)

        fig.suptitle(f"{model_name}: integrated OC scatter", fontsize=12)
        fig.tight_layout(rect=[0, 0, 1, 0.94])
        fp = outdir / f"{_sanitize_name(str(model_name))}_OC_scatter.png"
        fig.savefig(fp, dpi=260, bbox_inches="tight")
        plt.close(fig)
        files.append(str(fp))

    return files


def _build_segment_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    gcols = ["model", "obs_source", "obs_data_type", "segment_key"]
    for keys, d in df.groupby(gcols):
        model, src, dt, seg_key = keys
        if len(d) < 2:
            continue
        mm = _compute_oc_metrics(d["obs_u"].to_numpy(), d["obs_v"].to_numpy(), d["mod_u"].to_numpy(), d["mod_v"].to_numpy())
        rows.append(
            {
                "model": str(model),
                "source": str(src),
                "data_type": str(dt),
                "segment_key": str(seg_key),
                "segment_id": int(d["obs_segment_id"].iloc[0]) if "obs_segment_id" in d.columns and np.isfinite(d["obs_segment_id"].iloc[0]) else -1,
                "track_id": int(d["obs_track_id"].iloc[0]) if "obs_track_id" in d.columns and np.isfinite(d["obs_track_id"].iloc[0]) else -1,
                "track_file": str(d["obs_track_file"].iloc[0]) if "obs_track_file" in d.columns else "",
                "n": int(mm["n"]),
                "time_start": str(d["time_utc"].min()),
                "time_end": str(d["time_utc"].max()),
                "depth_mean": float(pd.to_numeric(d["obs_depth"], errors="coerce").mean()),
                "depth_min": float(pd.to_numeric(d["obs_depth"], errors="coerce").min()),
                "depth_max": float(pd.to_numeric(d["obs_depth"], errors="coerce").max()),
                **mm,
            }
        )
    return rows


def _build_model_rows(df: pd.DataFrame) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for keys, d in df.groupby(["model", "obs_source", "obs_data_type"]):
        model, src, dt = keys
        if len(d) < 2:
            continue
        mm = _compute_oc_metrics(d["obs_u"].to_numpy(), d["obs_v"].to_numpy(), d["mod_u"].to_numpy(), d["mod_v"].to_numpy())
        rows.append(
            {
                "model": str(model),
                "source": str(src),
                "data_type": str(dt),
                **mm,
            }
        )
    return rows


def _build_raw_rows(df: pd.DataFrame, cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    task = str(cfg.get("task_name", "oc"))
    rows: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        model = str(r.get("model", ""))
        exp_id = cfg.get("experiment_id") or model
        ou = float(r.get("obs_u", np.nan))
        ov = float(r.get("obs_v", np.nan))
        osd = float(r.get("obs_speed", np.nan))
        mu = float(r.get("mod_u", np.nan))
        mv = float(r.get("mod_v", np.nan))
        msd = float(r.get("mod_speed", np.nan))
        rows.append(
            {
                "task": task,
                "experiment_id": str(exp_id),
                "model": model,
                "source": str(r.get("obs_source", "")),
                "data_type": str(r.get("obs_data_type", "")),
                "segment_key": str(r.get("segment_key", "")),
                "segment_id": int(r.get("obs_segment_id", -1)) if np.isfinite(pd.to_numeric(r.get("obs_segment_id", np.nan), errors="coerce")) else -1,
                "track_id": int(r.get("obs_track_id", -1)) if np.isfinite(pd.to_numeric(r.get("obs_track_id", np.nan), errors="coerce")) else -1,
                "track_file": str(r.get("obs_track_file", "")),
                "time_utc": str(r.get("time_utc", "")),
                "depth": float(r.get("obs_depth", np.nan)),
                "obs_u": ou,
                "obs_v": ov,
                "obs_speed": osd,
                "mod_u": mu,
                "mod_v": mv,
                "mod_speed": msd,
                "error_u": float(mu - ou) if np.isfinite(mu) and np.isfinite(ou) else np.nan,
                "error_v": float(mv - ov) if np.isfinite(mv) and np.isfinite(ov) else np.nan,
                "error_speed": float(msd - osd) if np.isfinite(msd) and np.isfinite(osd) else np.nan,
                "mod_dt_hours": float(r.get("mod_dt_hours", np.nan)),
            }
        )
    return rows


def _plot_segments(df: pd.DataFrame, cfg: Dict[str, Any], outdir: Path) -> List[str]:
    files: List[str] = []
    if len(df) == 0:
        return files

    nmax = int(cfg.get("top_segments_per_source", 8))
    resamp = cfg.get("resample")

    for (model, src), ds in df.groupby(["model", "obs_source"]):
        stats = (
            ds.groupby("segment_key")
            .size()
            .sort_values(ascending=False)
            .head(nmax)
        )
        if len(stats) == 0:
            continue

        for seg_key in stats.index.tolist():
            dd = ds.loc[ds["segment_key"] == seg_key].copy()
            if len(dd) < 3:
                continue
            dsort = dd.sort_values("time_utc")
            sid = dsort["obs_segment_id"].iloc[0] if "obs_segment_id" in dsort.columns else np.nan
            tid = dsort["obs_track_id"].iloc[0] if "obs_track_id" in dsort.columns else np.nan
            title = f"{model} | {src} | {seg_key} | n={len(dsort)}"

            subdir = outdir / "segments" / _sanitize_name(str(model)) / _sanitize_name(str(src))
            f1 = subdir / f"{_sanitize_name(str(seg_key))}_timeseries.png"
            f2 = subdir / f"{_sanitize_name(str(seg_key))}_hodograph.png"

            if _plot_segment_time_history(dsort, f1, title, resample_freq=resamp):
                files.append(str(f1))
            htitle = f"{model} | {src} | seg={sid} track={tid}"
            if _plot_segment_hodograph(dsort, f2, htitle):
                files.append(str(f2))

    return files


# =============================================================================
# Config builders
# =============================================================================
def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare SCHISM OC collocation output against observations.")
    p.add_argument("--config", help="Optional JSON config overrides.")
    p.add_argument("--pairs", nargs="+", help="Pair file paths (CSV/NPZ) overriding model.pair_files.")
    p.add_argument("--labels", nargs="+", help="Labels overriding model.labels.")
    p.add_argument("--outdir", help="Output directory override.")
    p.add_argument("--data-types", nargs="+", help="Data type filter (e.g., CA).")
    p.add_argument("--sources", nargs="+", help="Source filter list.")
    p.add_argument("--start", help="UTC start time.")
    p.add_argument("--end", help="UTC end time.")
    p.add_argument("--min-depth", type=float)
    p.add_argument("--max-depth", type=float)
    p.add_argument("--max-lag-hours", type=float)
    p.add_argument("--resample", help="Resample frequency for plots (e.g., H, D).")
    p.add_argument("--top-segments", type=int, help="Top segments per source for plots.")
    p.add_argument("--experiment-id", help="Experiment ID for metrics rows.")
    p.add_argument("--disable-plots", action="store_true", help="Skip segment plots.")
    p.add_argument("--disable-scatter", action="store_true", help="Skip integrated scatter plots.")
    p.add_argument("--disable-metrics", action="store_true", help="Skip metrics CSV writing.")
    p.add_argument("--debug-times", action="store_true", help="Enable extra time-window logs.")
    return p.parse_args(argv)


def _build_canonical_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = copy.deepcopy(CONFIG)
    config_path = _set_path_base(args.config)
    if config_path:
        with config_path.open("r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        cfg = _deep_update(cfg, user_cfg)

    if args.pairs:
        cfg.setdefault("model", {})
        cfg["model"]["pair_files"] = list(args.pairs)
    if args.labels:
        cfg.setdefault("model", {})
        cfg["model"]["labels"] = list(args.labels)
    if args.outdir:
        cfg.setdefault("output", {})
        cfg["output"]["dir"] = args.outdir

    if args.data_types is not None:
        cfg.setdefault("filter", {})
        cfg["filter"]["data_types"] = list(args.data_types)
    if args.sources is not None:
        cfg.setdefault("filter", {})
        cfg["filter"]["sources"] = list(args.sources)
    if args.start is not None:
        cfg.setdefault("filter", {})
        cfg["filter"]["start"] = args.start
    if args.end is not None:
        cfg.setdefault("filter", {})
        cfg["filter"]["end"] = args.end
    if args.min_depth is not None:
        cfg.setdefault("filter", {})
        cfg["filter"]["min_depth"] = float(args.min_depth)
    if args.max_depth is not None:
        cfg.setdefault("filter", {})
        cfg["filter"]["max_depth"] = float(args.max_depth)
    if args.max_lag_hours is not None:
        cfg.setdefault("filter", {})
        cfg["filter"]["max_abs_dt_hours"] = float(args.max_lag_hours)

    if args.resample is not None:
        cfg.setdefault("time", {})
        cfg["time"]["resample"] = str(args.resample)
    if args.top_segments is not None:
        cfg.setdefault("plot", {})
        cfg["plot"]["top_segments_per_source"] = int(args.top_segments)
    if args.experiment_id is not None:
        cfg.setdefault("output", {})
        cfg["output"]["experiment_id"] = str(args.experiment_id)

    if args.disable_plots:
        cfg.setdefault("output", {})
        cfg["output"]["save_plots"] = False
    if args.disable_scatter:
        cfg.setdefault("output", {})
        cfg["output"]["write_scatter_plots"] = False
    if args.disable_metrics:
        cfg.setdefault("output", {})
        cfg["output"]["write_task_metrics"] = False
    if args.debug_times:
        cfg.setdefault("debug", {})
        cfg["debug"]["times"] = True

    return cfg


def _canonical_to_runtime_config(canonical_cfg: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = dict(canonical_cfg.get("model", {}))
    flt_cfg = dict(canonical_cfg.get("filter", {}))
    t_cfg = dict(canonical_cfg.get("time", {}))
    out_cfg = dict(canonical_cfg.get("output", {}))
    p_cfg = dict(canonical_cfg.get("plot", {}))
    dbg_cfg = dict(canonical_cfg.get("debug", {}))

    cfg = {
        "pair_files": list(model_cfg.get("pair_files") or []),
        "labels": model_cfg.get("labels"),
        "data_types": flt_cfg.get("data_types"),
        "sources": flt_cfg.get("sources"),
        "start": flt_cfg.get("start"),
        "end": flt_cfg.get("end"),
        "min_depth": flt_cfg.get("min_depth"),
        "max_depth": flt_cfg.get("max_depth"),
        "max_abs_dt_hours": flt_cfg.get("max_abs_dt_hours"),
        "require_matched": bool(flt_cfg.get("require_matched", True)),
        "require_inside_domain": bool(flt_cfg.get("require_inside_domain", True)),
        "exclude_reject_reasons": list(flt_cfg.get("exclude_reject_reasons") or []),
        "resample": t_cfg.get("resample"),
        "outdir": out_cfg.get("dir"),
        "task_name": out_cfg.get("task_name", "oc"),
        "experiment_id": out_cfg.get("experiment_id"),
        "write_task_metrics": bool(out_cfg.get("write_task_metrics", True)),
        "save_plots": bool(out_cfg.get("save_plots", True)),
        "write_scatter_plots": bool(out_cfg.get("write_scatter_plots", True)),
        "metrics_raw_name": out_cfg.get("metrics_raw_name", "OC_metrics_raw.csv"),
        "metrics_segment_name": out_cfg.get("metrics_segment_name", "OC_stats_by_segment.csv"),
        "metrics_model_name": out_cfg.get("metrics_model_name", "OC_stats_by_model.csv"),
        "manifest_name": out_cfg.get("manifest_name", "OC_manifest.json"),
        "top_segments_per_source": int(p_cfg.get("top_segments_per_source", 8)),
        "scatter_alpha": float(p_cfg.get("scatter_alpha", 0.55)),
        "scatter_size": float(p_cfg.get("scatter_size", 8)),
        "scatter_cmap": str(p_cfg.get("scatter_cmap", "viridis")),
        "debug_times": bool(dbg_cfg.get("times", False)),
    }

    if len(cfg["pair_files"]) == 0:
        raise ValueError("model.pair_files is empty")
    if cfg.get("labels") is not None and len(list(cfg["labels"])) != len(cfg["pair_files"]):
        raise ValueError("model.labels length must match model.pair_files")

    return cfg


# =============================================================================
# Main
# =============================================================================
def main(argv: Optional[Sequence[str]] = None) -> None:
    args = _parse_args(argv)
    canonical_cfg = _build_canonical_config(args)
    cfg = _canonical_to_runtime_config(canonical_cfg)

    # Keep behavior deterministic when launched with mpirun.
    if bool(USE_MPI) and int(RANK) != 0:
        return

    outdir = _resolve_path(str(cfg["outdir"]))
    outdir.mkdir(parents=True, exist_ok=True)

    with (outdir / "oc_config_used.json").open("w", encoding="utf-8") as f:
        json.dump(canonical_cfg, f, indent=2)

    df_all = _load_models(cfg)
    n_loaded = int(len(df_all))

    if cfg.get("debug_times", False):
        rank_print(
            f"Loaded rows={n_loaded:,}, time_min={df_all['time_utc'].min()}, time_max={df_all['time_utc'].max()}",
            rank0_only=True,
        )

    df = _apply_filters(df_all, cfg)
    n_used = int(len(df))
    if n_used == 0:
        raise RuntimeError("No rows remain after OC filters.")

    rank_print(f"OC rows: loaded={n_loaded:,}, used={n_used:,}", rank0_only=True)

    raw_rows = _build_raw_rows(df, cfg)
    segment_rows = _build_segment_rows(df)
    model_rows = _build_model_rows(df)

    raw_path = outdir / str(cfg.get("metrics_raw_name", "OC_metrics_raw.csv"))
    seg_path = outdir / str(cfg.get("metrics_segment_name", "OC_stats_by_segment.csv"))
    mod_path = outdir / str(cfg.get("metrics_model_name", "OC_stats_by_model.csv"))

    if cfg.get("write_task_metrics", True):
        write_csv_rows(raw_path, raw_rows, OC_RAW_FIELDS)
        write_csv_rows(seg_path, segment_rows, OC_SEGMENT_FIELDS)
        write_csv_rows(mod_path, model_rows, OC_MODEL_FIELDS)
        rank_print(f"Wrote raw pairs: {raw_path}", rank0_only=True)
        rank_print(f"Wrote segment stats: {seg_path}", rank0_only=True)
        rank_print(f"Wrote model stats: {mod_path}", rank0_only=True)

    plot_files: List[str] = []
    scatter_files: List[str] = []
    if cfg.get("save_plots", True):
        plot_files = _plot_segments(df, cfg, outdir)
        rank_print(f"Wrote segment plots: {len(plot_files)}", rank0_only=True)
    if cfg.get("write_scatter_plots", True):
        scatter_files = _write_integrated_scatter(df, cfg, outdir)
        rank_print(f"Wrote scatter plots: {len(scatter_files)}", rank0_only=True)

    manifest = {
        "summary": {
            "rows_loaded": n_loaded,
            "rows_used": n_used,
            "models": sorted(df["model"].astype(str).unique().tolist()),
            "sources": sorted(df["obs_source"].astype(str).unique().tolist()),
            "data_types": sorted(df["obs_data_type"].astype(str).unique().tolist()),
            "segment_count": int(df["segment_key"].nunique()),
            "raw_rows": len(raw_rows),
            "segment_rows": len(segment_rows),
            "model_rows": len(model_rows),
            "plot_files": len(plot_files),
            "scatter_files": len(scatter_files),
            "mpi_size": int(SIZE),
        },
        "files": {
            "raw_pairs": str(raw_path) if cfg.get("write_task_metrics", True) else None,
            "segment_stats": str(seg_path) if cfg.get("write_task_metrics", True) else None,
            "model_stats": str(mod_path) if cfg.get("write_task_metrics", True) else None,
            "segment_plots": plot_files,
            "scatter_plots": scatter_files,
        },
    }
    mpath = outdir / str(cfg.get("manifest_name", "OC_manifest.json"))
    with mpath.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    rank_print(f"Wrote manifest: {mpath}", rank0_only=True)


if __name__ == "__main__":
    main()
