#!/usr/bin/env python3
# =============================================================================
# Quick Start (Read This First)
# =============================================================================
# This script orchestrates model-vs-observation validation for multiple runs
# defined in validation/experiments.csv.
#
# Main idea:
#   1) Define defaults in CONFIG below (recommended workflow).
#   2) Run this script once per campaign.
#   3) Review outputs under: validation/runs/<campaign_id>/
#
# Catalog requirement (CSV columns):
#   experiment_id,is_baseline,grid,bathy,forcing,river
#
# Baseline:
#   Exactly one row must have is_baseline=1.
#   Scorecards compute skill gain relative to this baseline.
#
# Produced outputs:
#   - run_plan.csv
#   - run_status.csv
#   - campaign_summary.json
#   - metrics/merged_metrics_raw.csv
#   - metrics/aggregated_metrics_station_equal.csv
#   - metrics/aggregated_metrics_sample_count.csv
#   - metrics/scorecard_station_equal.csv
#   - metrics/scorecard_sample_count.csv
#
# Metrics/scorecard behavior:
#   - NRMSE = RMSE / std(obs)
#   - Total cost J = (NRMSE_WL + NRMSE_Salt + NRMSE_Temp) / 3
#   - Two weighting modes:
#       station_equal  : each station contributes equally
#       sample_count   : station contribution weighted by sample count n
#
# -----------------------------------------------------------------------------
# Examples
# -----------------------------------------------------------------------------
# 1) Use CONFIG only (recommended):
#    python model_evaluation.py
#
# 2) Dry-run planning only:
#    python model_evaluation.py --dry-run --campaign-id plan_only
#
# 3) Execute WL only:
#    python model_evaluation.py \
#      --execute-wl \
#      --wl-run-template '/scratch/npz/{experiment_id}.npz'
#
# 4) Execute CTD + TH + WL:
#    python model_evaluation.py \
#      --execute-ctd --execute-th --execute-wl \
#      --ctd-run-dir-template '/scratch/runs/{experiment_id}' \
#      --th-schism-template '/scratch/npz/{experiment_id}.npz' \
#      --wl-run-template '/scratch/npz/{experiment_id}.npz' \
#      --th-teams-path '/scratch/npz/sendai_d2_timeseries.npz' \
#      --ctd-teams-path '/scratch/npz/onagawa_d1_ctd.npz'
#
# 5) Rebuild scorecards only (no task execution):
#    python model_evaluation.py \
#      --no-execute-ctd --no-execute-th --no-execute-wl --execute-scorecard \
#      --campaign-id rerank_only
# =============================================================================
"""
Controller for TEAMS/SCHISM validation campaigns.

What this script does:
1) Reads experiment catalog (with baseline/factor columns).
2) Builds and runs per-experiment CTD/TH/WL tasks.
3) Merges task metrics into campaign-level tables.
4) Creates scorecards with:
   - NRMSE = RMSE / std(obs)
   - Equal-weight total cost J = (NRMSE_WL + NRMSE_Salt + NRMSE_Temp) / 3
   - Two weighting modes:
     - station_equal
     - sample_count

How configuration works:
1) Edit the top-level CONFIG block for your standard workflow.
2) Use CLI flags only when you need one-off overrides.
3) CLI flags take precedence over CONFIG values.

Expected run outputs per campaign:
- run_plan.csv
- run_status.csv
- campaign_summary.json
- metrics/merged_metrics_raw.csv
- metrics/aggregated_metrics_*.csv
- metrics/scorecard_*.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent


# ---------------------------------------------------------------------------
# User-configurable defaults
# ---------------------------------------------------------------------------
CONFIG: Dict[str, Any] = {
    # Campaign basics
    "CATALOG": "validation/experiments.csv",
    "OUT_ROOT": "validation/runs",
    "CAMPAIGN_ID": None,  # None -> timestamp-based campaign id
    "ONLY": None,  # Example: ["RUN00", "RUN01"]
    "PYTHON": sys.executable,
    "CONTINUE_ON_ERROR": False,
    "DRY_RUN": False,

    # Task execution switches
    "EXECUTE_CTD": False,
    "EXECUTE_TH": False,
    "EXECUTE_WL": False,
    "EXECUTE_SCORECARD": True,

    # Script paths
    "CTD_SCRIPT": "SCHISMvsTEAMS_CTD.py",
    "TH_SCRIPT": "SCHISMvsTEAMS_TH.py",
    "WL_SCRIPT": "SCHISMvsJODC_WL.py",

    # CTD task config
    "CTD_RUN_DIR_TEMPLATE": None,  # Example: "/scratch/runs/{experiment_id}"
    "CTD_TEAMS_PATH": None,  # Example: "/scratch/npz/onagawa_d1_ctd.npz"
    "CTD_START": None,  # "YYYY-MM-DD"
    "CTD_END": None,  # "YYYY-MM-DD"
    "CTD_ENABLE_GLOBAL_MODEL": False,
    "CTD_COLOR": "b",

    # Teams time histories task config
    "TH_SCHISM_TEMPLATE": None,  # Example: "/scratch/npz/{experiment_id}.npz"
    "TH_TEAMS_PATH": None,  # Example: "/scratch/npz/sendai_d2_timeseries.npz"
    "TH_BPFILE": "station_sendai_d2.bp",
    "TH_GRID": None,
    "TH_START": None,
    "TH_END": None,
    "TH_RESAMPLE": "h",
    "TH_VARS": ["temp", "sal"],

    # JODC water level task config
    "WL_RUN_TEMPLATE": None,  # Example: "/scratch/npz/{experiment_id}.npz"
    "WL_BPFILE": "station_jodc.bp",
    "WL_OBS_PATH": "npz/jodc_tide_all.npz",
    "WL_START": "2022-01-14 00:00:00",
    "WL_END": "2022-03-30 00:00:00",
    "WL_MODEL_START": "2022-01-02 00:00:00",

    # Scorecard config
    "WEIGHTING_MODES": ["station_equal", "sample_count"],
    "J_WEIGHTS": {"wl": 1.0 / 3.0, "salt": 1.0 / 3.0, "temp": 1.0 / 3.0},
}


REQUIRED_COLUMNS = ["experiment_id", "is_baseline", "grid", "bathy", "forcing", "river"]


def _resolve(path_like: str) -> Path:
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p
    return (SCRIPT_DIR / p).resolve()


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _write_csv_rows(
    path: Path,
    rows: List[Dict[str, Any]],
    fieldnames: Optional[List[str]] = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if rows:
            fieldnames = list(rows[0].keys())
        else:
            fieldnames = []
    with open(path, "w", encoding="utf-8", newline="") as f:
        if not fieldnames:
            return
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _safe_float(value: Any) -> float:
    try:
        if value is None:
            return float("nan")
        if isinstance(value, str) and value.strip() == "":
            return float("nan")
        return float(value)
    except Exception:
        return float("nan")


def _to_ymd_list(text: str) -> List[int]:
    token = str(text).strip().split()[0]
    parts = token.split("-")
    if len(parts) != 3:
        raise ValueError(f"Expected date like YYYY-MM-DD, got: {text}")
    return [int(parts[0]), int(parts[1]), int(parts[2])]


def _is_finite(v: Any) -> bool:
    x = _safe_float(v)
    return math.isfinite(x)


def _mean(values: Iterable[float]) -> float:
    vals = [v for v in values if math.isfinite(v)]
    if not vals:
        return float("nan")
    return sum(vals) / float(len(vals))


def _weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    pairs = [(v, w) for v, w in zip(values, weights) if math.isfinite(v) and math.isfinite(w) and w > 0]
    if not pairs:
        return float("nan")
    sw = sum(w for _, w in pairs)
    if sw <= 0:
        return float("nan")
    return sum(v * w for v, w in pairs) / sw


def _normalize_var(var_raw: Any) -> str:
    v = str(var_raw).strip().lower()
    if v in {"sal", "salt", "s"}:
        return "salt"
    if v in {"temp", "temperature", "t"}:
        return "temp"
    if v in {"wl", "elev", "water_level", "waterlevel"}:
        return "wl"
    return v


def _validate_catalog(rows: List[Dict[str, str]]) -> None:
    if not rows:
        raise ValueError("Catalog is empty.")
    header = set(rows[0].keys())
    missing = [c for c in REQUIRED_COLUMNS if c not in header]
    if missing:
        raise ValueError(f"Catalog missing required columns: {missing}")
    ids = [str(r["experiment_id"]).strip() for r in rows]
    if len(ids) != len(set(ids)):
        seen = set()
        dupes = []
        for sid in ids:
            if sid in seen:
                dupes.append(sid)
            seen.add(sid)
        raise ValueError(f"Duplicate experiment_id in catalog: {dupes}")
    n_base = sum(1 for r in rows if int(str(r.get("is_baseline", "0")).strip() or "0") == 1)
    if n_base != 1:
        raise ValueError("Catalog must have exactly one baseline row with is_baseline=1.")


def _status_record(exp_id: str, task: str, status: str, message: str, cmd: List[str]) -> Dict[str, Any]:
    return {
        "experiment_id": exp_id,
        "task": task,
        "status": status,
        "message": message,
        "command": " ".join(cmd),
        "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def _run_cmd(cmd: List[str], cwd: Path, stdout_log: Path, stderr_log: Path) -> Tuple[bool, str]:
    stdout_log.parent.mkdir(parents=True, exist_ok=True)
    stderr_log.parent.mkdir(parents=True, exist_ok=True)
    with open(stdout_log, "w", encoding="utf-8") as fo, open(stderr_log, "w", encoding="utf-8") as fe:
        proc = subprocess.run(cmd, cwd=str(cwd), stdout=fo, stderr=fe, check=False)
    if proc.returncode == 0:
        return True, "completed"
    return False, f"failed with return code {proc.returncode}"


def _prepare_ctd_config(cfg: Dict[str, Any], exp_id: str, exp_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "schism": [
            {
                "enabled": True,
                "label": exp_id,
                "color": cfg["CTD_COLOR"],
                "run_dir": "",
            }
        ],
        "global_model": {"enabled": bool(cfg["CTD_ENABLE_GLOBAL_MODEL"])},
        "output": {
            "dir": str((exp_dir / "ctd").resolve()),
            "experiment_id": exp_id,
            "task_name": "ctd",
            "write_task_metrics": True,
            "write_scatter_plots": True,
            "save_profile_plots": True,
        },
    }
    if cfg.get("CTD_RUN_DIR_TEMPLATE"):
        out["schism"][0]["run_dir"] = str(cfg["CTD_RUN_DIR_TEMPLATE"]).format(experiment_id=exp_id)
    if cfg.get("CTD_TEAMS_PATH"):
        out["teams"] = {"npz_path": str(_resolve(cfg["CTD_TEAMS_PATH"]))}
    if cfg.get("CTD_START") or cfg.get("CTD_END"):
        dr: Dict[str, Any] = {}
        if cfg.get("CTD_START"):
            dr["start"] = _to_ymd_list(str(cfg["CTD_START"]))
        if cfg.get("CTD_END"):
            dr["end"] = _to_ymd_list(str(cfg["CTD_END"]))
        out["date_range"] = dr
    return out


def _prepare_th_config(cfg: Dict[str, Any], exp_id: str, exp_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "schism_npzs": [],
        "schism_labels": [exp_id],
        "bpfile": str(_resolve(cfg["TH_BPFILE"])),
        "outdir": str((exp_dir / "th").resolve()),
        "vars": list(cfg["TH_VARS"]),
        "resample": str(cfg["TH_RESAMPLE"]),
        "debug_times": True,
        "task_name": "th",
        "experiment_id": exp_id,
        "write_task_metrics": True,
        "write_integrated_scatter": True,
        "save_plots": True,
    }
    if cfg.get("TH_SCHISM_TEMPLATE"):
        out["schism_npzs"] = [str(cfg["TH_SCHISM_TEMPLATE"]).format(experiment_id=exp_id)]
    if cfg.get("TH_TEAMS_PATH"):
        out["teams_npz"] = str(_resolve(cfg["TH_TEAMS_PATH"]))
    if cfg.get("TH_GRID"):
        out["grid"] = str(_resolve(cfg["TH_GRID"]))
    if cfg.get("TH_START"):
        out["start"] = str(cfg["TH_START"])
    if cfg.get("TH_END"):
        out["end"] = str(cfg["TH_END"])
    return out


def _prepare_wl_config(cfg: Dict[str, Any], exp_id: str, exp_dir: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "runs": [],
        "tags": [exp_id],
        "bpfile": str(_resolve(cfg["WL_BPFILE"])),
        "outdir": str((exp_dir / "wl").resolve()),
        "obs_path": str(_resolve(cfg["WL_OBS_PATH"])),
        "start": str(cfg["WL_START"]),
        "end": str(cfg["WL_END"]),
        "model_start": str(cfg["WL_MODEL_START"]),
        "resample_obs": "h",
        "resample_model": "h",
        "demean": True,
        "save_plots": True,
        "progress_every": 20,
    }
    if cfg.get("WL_RUN_TEMPLATE"):
        out["runs"] = [str(cfg["WL_RUN_TEMPLATE"]).format(experiment_id=exp_id)]
    return out


def _load_metrics_from_wl_csv(path: Path, exp_id: str) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = _read_csv_rows(path)
    if not rows:
        return []
    has_self_model = any(str(r.get("model", "")).strip() == exp_id for r in rows)
    out: List[Dict[str, Any]] = []
    for r in rows:
        model = str(r.get("model", "")).strip() or exp_id
        if has_self_model and model != exp_id:
            continue
        rmse = _safe_float(r.get("rmse"))
        obs_std = _safe_float(r.get("obs_std"))
        nrmse = _safe_float(r.get("nrmse_std"))
        if (not math.isfinite(nrmse)) and math.isfinite(rmse) and math.isfinite(obs_std) and obs_std > 0:
            nrmse = rmse / obs_std
        out.append(
            {
                "experiment_id": exp_id,
                "task": "wl",
                "model": model,
                "var": "wl",
                "station": str(r.get("station", "")).strip(),
                "n": _safe_float(r.get("n")),
                "bias": _safe_float(r.get("bias")),
                "rmse": rmse,
                "corr": _safe_float(r.get("corr")),
                "obs_std": obs_std,
                "mod_std": _safe_float(r.get("mod_std")),
                "nrmse_std": nrmse,
                "wss": _safe_float(r.get("wss")),
                "crmsd": _safe_float(r.get("crmsd")),
            }
        )
    return out


def _load_metrics_from_generic_csv(path: Path, exp_id: str, task: str) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = _read_csv_rows(path)
    if not rows:
        return []
    has_self_model = any(str(r.get("model", "")).strip() == exp_id for r in rows)
    out: List[Dict[str, Any]] = []
    for r in rows:
        model = str(r.get("model", "")).strip() or exp_id
        if has_self_model and model != exp_id:
            continue
        var = _normalize_var(r.get("var", "unknown"))
        if var not in {"temp", "salt", "wl"}:
            continue
        station = (
            str(r.get("station_id_full", "")).strip()
            or str(r.get("station_id", "")).strip()
            or str(r.get("station_name", "")).strip()
        )
        rmse = _safe_float(r.get("rmse"))
        obs_std = _safe_float(r.get("obs_std"))
        nrmse = _safe_float(r.get("nrmse_std"))
        if (not math.isfinite(nrmse)) and math.isfinite(rmse) and math.isfinite(obs_std) and obs_std > 0:
            nrmse = rmse / obs_std
        out.append(
            {
                "experiment_id": exp_id,
                "task": task,
                "model": model,
                "var": var,
                "station": station,
                "n": _safe_float(r.get("n")),
                "bias": _safe_float(r.get("bias")),
                "rmse": rmse,
                "corr": _safe_float(r.get("corr")),
                "obs_std": obs_std,
                "mod_std": _safe_float(r.get("mod_std")),
                "nrmse_std": nrmse,
                "wss": _safe_float(r.get("wss")),
                "crmsd": _safe_float(r.get("crmsd")),
            }
        )
    return out


def _collect_campaign_metrics(campaign_dir: Path, experiments: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    raw: List[Dict[str, Any]] = []
    for row in experiments:
        exp_id = str(row["experiment_id"]).strip()
        exp_dir = campaign_dir / exp_id
        raw.extend(_load_metrics_from_wl_csv(exp_dir / "wl" / "WL_stats.csv", exp_id))
        raw.extend(_load_metrics_from_generic_csv(exp_dir / "th" / "SCHISMvsTEAMS_stats.csv", exp_id, "th"))
        ctd_candidates = [
            exp_dir / "ctd" / "CTD_stats.csv",
            exp_dir / "ctd" / "ctd_stats.csv",
            exp_dir / "ctd" / "SCHISMvsTEAMS_CTD_stats.csv",
        ]
        for cfp in ctd_candidates:
            if cfp.exists():
                raw.extend(_load_metrics_from_generic_csv(cfp, exp_id, "ctd"))
                break
    return raw


def _aggregate_metrics(raw_rows: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    if mode not in {"station_equal", "sample_count"}:
        raise ValueError(f"Unknown weighting mode: {mode}")
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for r in raw_rows:
        exp_id = str(r["experiment_id"]).strip()
        var = _normalize_var(r["var"])
        key = (exp_id, var)
        grouped.setdefault(key, []).append(r)

    out: List[Dict[str, Any]] = []
    metric_names = ["bias", "rmse", "corr", "obs_std", "mod_std", "nrmse_std", "wss", "crmsd"]

    for (exp_id, var), rows in grouped.items():
        stations = sorted(set(str(r.get("station", "")).strip() for r in rows))
        n_total = sum(_safe_float(r.get("n")) for r in rows if math.isfinite(_safe_float(r.get("n"))))
        agg: Dict[str, Any] = {
            "experiment_id": exp_id,
            "var": var,
            "weighting_mode": mode,
            "stations": len([s for s in stations if s]),
            "samples": int(round(n_total)),
        }
        if mode == "sample_count":
            weights = [_safe_float(r.get("n")) for r in rows]
            for m in metric_names:
                vals = [_safe_float(r.get(m)) for r in rows]
                agg[m] = _weighted_mean(vals, weights)
        else:
            # station_equal
            by_station: Dict[str, List[Dict[str, Any]]] = {}
            for r in rows:
                sid = str(r.get("station", "")).strip()
                by_station.setdefault(sid, []).append(r)
            for m in metric_names:
                station_values: List[float] = []
                for sid_rows in by_station.values():
                    v = _mean([_safe_float(rr.get(m)) for rr in sid_rows])
                    if math.isfinite(v):
                        station_values.append(v)
                agg[m] = _mean(station_values)
        out.append(agg)

    out.sort(key=lambda r: (r["experiment_id"], r["var"], r["weighting_mode"]))
    return out


def _build_scorecard(
    agg_rows: List[Dict[str, Any]],
    experiments: List[Dict[str, str]],
    baseline_id: str,
    mode: str,
    weights: Dict[str, float],
) -> List[Dict[str, Any]]:
    # Build lookup (exp,var)->nrmse
    nrmse_lookup: Dict[Tuple[str, str], float] = {}
    for r in agg_rows:
        if str(r.get("weighting_mode")) != mode:
            continue
        exp_id = str(r["experiment_id"])
        var = _normalize_var(r["var"])
        nrmse_lookup[(exp_id, var)] = _safe_float(r.get("nrmse_std"))

    rows: List[Dict[str, Any]] = []
    for exp in experiments:
        exp_id = str(exp["experiment_id"]).strip()
        n_wl = nrmse_lookup.get((exp_id, "wl"), float("nan"))
        n_t = nrmse_lookup.get((exp_id, "temp"), float("nan"))
        n_s = nrmse_lookup.get((exp_id, "salt"), float("nan"))
        terms = [n_wl, n_s, n_t]
        finite_terms = [v for v in terms if math.isfinite(v)]
        j = (
            weights["wl"] * n_wl + weights["salt"] * n_s + weights["temp"] * n_t
            if all(math.isfinite(v) for v in terms)
            else float("nan")
        )
        j_partial = _mean(finite_terms)
        row = {
            "experiment_id": exp_id,
            "is_baseline": int(str(exp.get("is_baseline", "0")).strip() or "0"),
            "grid": str(exp.get("grid", "")),
            "bathy": str(exp.get("bathy", "")),
            "forcing": str(exp.get("forcing", "")),
            "river": str(exp.get("river", "")),
            "weighting_mode": mode,
            "nrmse_wl": n_wl,
            "nrmse_salt": n_s,
            "nrmse_temp": n_t,
            "j_equal_weight": j,
            "j_partial_mean": j_partial,
            "j_terms_available": len(finite_terms),
        }
        rows.append(row)

    baseline_row = next((r for r in rows if r["experiment_id"] == baseline_id), None)
    b_wl = _safe_float(baseline_row["nrmse_wl"]) if baseline_row else float("nan")
    b_s = _safe_float(baseline_row["nrmse_salt"]) if baseline_row else float("nan")
    b_t = _safe_float(baseline_row["nrmse_temp"]) if baseline_row else float("nan")
    b_j = _safe_float(baseline_row["j_equal_weight"]) if baseline_row else float("nan")

    for r in rows:
        n_wl = _safe_float(r["nrmse_wl"])
        n_s = _safe_float(r["nrmse_salt"])
        n_t = _safe_float(r["nrmse_temp"])
        j = _safe_float(r["j_equal_weight"])
        r["gain_wl_vs_baseline"] = (b_wl - n_wl) if math.isfinite(b_wl) and math.isfinite(n_wl) else float("nan")
        r["gain_salt_vs_baseline"] = (b_s - n_s) if math.isfinite(b_s) and math.isfinite(n_s) else float("nan")
        r["gain_temp_vs_baseline"] = (b_t - n_t) if math.isfinite(b_t) and math.isfinite(n_t) else float("nan")
        r["gain_j_vs_baseline"] = (b_j - j) if math.isfinite(b_j) and math.isfinite(j) else float("nan")

    # Rank by J (lower is better)
    finite_j = sorted(
        [(idx, _safe_float(r["j_equal_weight"])) for idx, r in enumerate(rows) if math.isfinite(_safe_float(r["j_equal_weight"]))],
        key=lambda x: x[1],
    )
    rank_map = {idx: rank for rank, (idx, _) in enumerate(finite_j, start=1)}
    for i, r in enumerate(rows):
        r["rank_j"] = rank_map.get(i, None)

    rows.sort(key=lambda rr: (_safe_float(rr["j_equal_weight"]) if math.isfinite(_safe_float(rr["j_equal_weight"])) else 1e300, rr["experiment_id"]))
    return rows


def _save_metrics_and_scorecards(
    campaign_dir: Path,
    experiments: List[Dict[str, str]],
    baseline_id: str,
    weights: Dict[str, float],
    weighting_modes: List[str],
) -> Dict[str, Any]:
    metrics_dir = campaign_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    raw_rows = _collect_campaign_metrics(campaign_dir, experiments)
    _write_csv_rows(
        metrics_dir / "merged_metrics_raw.csv",
        raw_rows,
        fieldnames=[
            "experiment_id",
            "task",
            "model",
            "var",
            "station",
            "n",
            "bias",
            "rmse",
            "corr",
            "obs_std",
            "mod_std",
            "nrmse_std",
            "wss",
            "crmsd",
        ],
    )

    saved: Dict[str, Any] = {
        "raw_rows": len(raw_rows),
        "weighting_modes": [],
    }

    for mode in weighting_modes:
        agg = _aggregate_metrics(raw_rows, mode=mode)
        _write_csv_rows(
            metrics_dir / f"aggregated_metrics_{mode}.csv",
            agg,
            fieldnames=[
                "experiment_id",
                "var",
                "weighting_mode",
                "stations",
                "samples",
                "bias",
                "rmse",
                "corr",
                "obs_std",
                "mod_std",
                "nrmse_std",
                "wss",
                "crmsd",
            ],
        )
        score = _build_scorecard(
            agg_rows=agg,
            experiments=experiments,
            baseline_id=baseline_id,
            mode=mode,
            weights=weights,
        )
        _write_csv_rows(
            metrics_dir / f"scorecard_{mode}.csv",
            score,
            fieldnames=[
                "experiment_id",
                "is_baseline",
                "grid",
                "bathy",
                "forcing",
                "river",
                "weighting_mode",
                "nrmse_wl",
                "nrmse_salt",
                "nrmse_temp",
                "j_equal_weight",
                "j_partial_mean",
                "j_terms_available",
                "gain_wl_vs_baseline",
                "gain_salt_vs_baseline",
                "gain_temp_vs_baseline",
                "gain_j_vs_baseline",
                "rank_j",
            ],
        )
        saved["weighting_modes"].append(
            {
                "mode": mode,
                "aggregated_rows": len(agg),
                "scorecard_rows": len(score),
                "scorecard_file": str(metrics_dir / f"scorecard_{mode}.csv"),
            }
        )

    return saved


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    class _Formatter(argparse.ArgumentDefaultsHelpFormatter, argparse.RawTextHelpFormatter):
        pass

    epilog = (
        "Examples:\n"
        "  1) Use in-file CONFIG only (recommended daily workflow):\n"
        "     python model_evaluation.py\n\n"
        "  2) Dry-run planning with defaults from CONFIG:\n"
        "     python model_evaluation.py --dry-run --campaign-id plan_only\n\n"
        "  3) Execute WL only for all experiments:\n"
        "     python model_evaluation.py \\\n"
        "       --execute-wl \\\n"
        "       --wl-run-template '/scratch/npz/{experiment_id}.npz'\n\n"
        "  4) Execute CTD+TH+WL with one-off path overrides:\n"
        "     python model_evaluation.py \\\n"
        "       --execute-ctd --execute-th --execute-wl \\\n"
        "       --ctd-run-dir-template '/scratch/runs/{experiment_id}' \\\n"
        "       --th-schism-template '/scratch/npz/{experiment_id}.npz' \\\n"
        "       --wl-run-template '/scratch/npz/{experiment_id}.npz' \\\n"
        "       --th-teams-path '/scratch/npz/sendai_d2_timeseries.npz' \\\n"
        "       --ctd-teams-path '/scratch/npz/onagawa_d1_ctd.npz'\n\n"
        "  5) Scorecard-only rerun from existing campaign outputs:\n"
        "     python model_evaluation.py \\\n"
        "       --no-execute-ctd --no-execute-th --no-execute-wl --execute-scorecard \\\n"
        "       --campaign-id rerank_only\n"
    )

    p = argparse.ArgumentParser(
        description="Validation controller for CTD/TH/WL with merged metrics and scorecards.",
        formatter_class=_Formatter,
        epilog=epilog,
    )

    p.add_argument(
        "--catalog",
        help=(
            "Experiment catalog CSV. Must include columns:\n"
            "experiment_id,is_baseline,grid,bathy,forcing,river"
        ),
    )
    p.add_argument(
        "--out-root",
        help="Root directory where campaign folders are created.",
    )
    p.add_argument(
        "--campaign-id",
        help=(
            "Campaign folder name. If omitted, a timestamp-based ID is used.\n"
            "Useful for reproducible reruns and scorecard-only postprocessing."
        ),
    )
    p.add_argument(
        "--only",
        nargs="+",
        help=(
            "Subset of experiment IDs to run, e.g. --only RUN00 RUN02.\n"
            "Note: filtered selection must still include exactly one baseline (is_baseline=1)."
        ),
    )
    p.add_argument(
        "--python",
        help="Python executable used to launch child task scripts.",
    )

    p.add_argument(
        "--continue-on-error",
        dest="continue_on_error",
        action="store_true",
        help="Continue running other experiments/tasks even if one task fails.",
    )
    p.add_argument(
        "--stop-on-error",
        dest="continue_on_error",
        action="store_false",
        help="Fail-fast behavior (default). Stop on first failure.",
    )
    p.set_defaults(continue_on_error=None)

    p.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        help="Generate configs/plans/scorecards but do not execute CTD/TH/WL child scripts.",
    )
    p.add_argument(
        "--no-dry-run",
        dest="dry_run",
        action="store_false",
        help="Execute enabled tasks (subject to required path templates).",
    )
    p.set_defaults(dry_run=None)

    p.add_argument(
        "--execute-ctd",
        dest="execute_ctd",
        action="store_true",
        help="Enable CTD task execution via SCHISMvsTEAMS_CTD.py.",
    )
    p.add_argument(
        "--no-execute-ctd",
        dest="execute_ctd",
        action="store_false",
        help="Disable CTD task execution.",
    )
    p.set_defaults(execute_ctd=None)

    p.add_argument(
        "--execute-th",
        dest="execute_th",
        action="store_true",
        help="Enable TH task execution via SCHISMvsTEAMS_TH.py.",
    )
    p.add_argument(
        "--no-execute-th",
        dest="execute_th",
        action="store_false",
        help="Disable TH task execution.",
    )
    p.set_defaults(execute_th=None)

    p.add_argument(
        "--execute-wl",
        dest="execute_wl",
        action="store_true",
        help="Enable WL task execution via SCHISMvsJODC_WL.py.",
    )
    p.add_argument(
        "--no-execute-wl",
        dest="execute_wl",
        action="store_false",
        help="Disable WL task execution.",
    )
    p.set_defaults(execute_wl=None)

    p.add_argument(
        "--execute-scorecard",
        dest="execute_scorecard",
        action="store_true",
        help=(
            "Enable merged metrics and scorecard generation stage.\n"
            "This can be run independently after task outputs already exist."
        ),
    )
    p.add_argument(
        "--no-execute-scorecard",
        dest="execute_scorecard",
        action="store_false",
        help="Disable merged metrics/scorecard generation.",
    )
    p.set_defaults(execute_scorecard=None)

    p.add_argument(
        "--ctd-script",
        help="Path to SCHISMvsTEAMS_CTD.py script.",
    )
    p.add_argument(
        "--ctd-run-dir-template",
        help=(
            "Template for CTD SCHISM run directory, e.g. '/scratch/runs/{experiment_id}'.\n"
            "Required when --execute-ctd is enabled."
        ),
    )
    p.add_argument(
        "--ctd-teams-path",
        help="Override TEAMS CTD NPZ path.",
    )
    p.add_argument(
        "--ctd-start",
        help="Override CTD start date in YYYY-MM-DD.",
    )
    p.add_argument(
        "--ctd-end",
        help="Override CTD end date in YYYY-MM-DD.",
    )
    p.add_argument(
        "--ctd-enable-global-model",
        dest="ctd_enable_global_model",
        action="store_true",
        help="Enable CTD global model comparison.",
    )
    p.add_argument(
        "--ctd-disable-global-model",
        dest="ctd_enable_global_model",
        action="store_false",
        help="Disable CTD global model comparison.",
    )
    p.set_defaults(ctd_enable_global_model=None)
    p.add_argument(
        "--ctd-color",
        help="Plot color for experiment line in CTD figures (single-run mode).",
    )

    p.add_argument(
        "--th-script",
        help="Path to SCHISMvsTEAMS_TH.py script.",
    )
    p.add_argument(
        "--th-schism-template",
        help=(
            "Template for TH SCHISM NPZ, e.g. '/scratch/npz/{experiment_id}.npz'.\n"
            "Required when --execute-th is enabled."
        ),
    )
    p.add_argument(
        "--th-teams-path",
        help="Override TEAMS TH NPZ path.",
    )
    p.add_argument(
        "--th-bpfile",
        help="BP file for TH station mapping.",
    )
    p.add_argument(
        "--th-grid",
        help="Optional grid file for TH map panel.",
    )
    p.add_argument(
        "--th-start",
        help="Override TH start datetime.",
    )
    p.add_argument(
        "--th-end",
        help="Override TH end datetime.",
    )
    p.add_argument(
        "--th-resample",
        help="Resample frequency for TH (e.g., h, d).",
    )
    p.add_argument(
        "--th-vars",
        nargs="+",
        help="TH variables to compare, e.g. --th-vars temp sal",
    )

    p.add_argument(
        "--wl-script",
        help="Path to SCHISMvsJODC_WL.py.",
    )
    p.add_argument(
        "--wl-run-template",
        help=(
            "Template for WL SCHISM NPZ/staout, e.g. '/scratch/npz/{experiment_id}.npz'.\n"
            "Required when --execute-wl is enabled."
        ),
    )
    p.add_argument(
        "--wl-bpfile",
        help="BP file for WL station mapping.",
    )
    p.add_argument(
        "--wl-obs-path",
        help="WL observation NPZ path (jodc_tide_all.npz).",
    )
    p.add_argument(
        "--wl-start",
        help="WL start datetime.",
    )
    p.add_argument(
        "--wl-end",
        help="WL end datetime.",
    )
    p.add_argument(
        "--wl-model-start",
        help="Model reference datetime used to offset model time in WL script.",
    )

    return p.parse_args(argv)


def _build_runtime_config(args: argparse.Namespace) -> Dict[str, Any]:
    cfg = dict(CONFIG)
    mapping = {
        "catalog": "CATALOG",
        "out_root": "OUT_ROOT",
        "campaign_id": "CAMPAIGN_ID",
        "only": "ONLY",
        "python": "PYTHON",
        "continue_on_error": "CONTINUE_ON_ERROR",
        "dry_run": "DRY_RUN",
        "execute_ctd": "EXECUTE_CTD",
        "execute_th": "EXECUTE_TH",
        "execute_wl": "EXECUTE_WL",
        "execute_scorecard": "EXECUTE_SCORECARD",
        "ctd_script": "CTD_SCRIPT",
        "ctd_run_dir_template": "CTD_RUN_DIR_TEMPLATE",
        "ctd_teams_path": "CTD_TEAMS_PATH",
        "ctd_start": "CTD_START",
        "ctd_end": "CTD_END",
        "ctd_enable_global_model": "CTD_ENABLE_GLOBAL_MODEL",
        "ctd_color": "CTD_COLOR",
        "th_script": "TH_SCRIPT",
        "th_schism_template": "TH_SCHISM_TEMPLATE",
        "th_teams_path": "TH_TEAMS_PATH",
        "th_bpfile": "TH_BPFILE",
        "th_grid": "TH_GRID",
        "th_start": "TH_START",
        "th_end": "TH_END",
        "th_resample": "TH_RESAMPLE",
        "th_vars": "TH_VARS",
        "wl_script": "WL_SCRIPT",
        "wl_run_template": "WL_RUN_TEMPLATE",
        "wl_bpfile": "WL_BPFILE",
        "wl_obs_path": "WL_OBS_PATH",
        "wl_start": "WL_START",
        "wl_end": "WL_END",
        "wl_model_start": "WL_MODEL_START",
    }
    for arg_name, key in mapping.items():
        val = getattr(args, arg_name, None)
        if val is not None:
            cfg[key] = val
    return cfg


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)
    cfg = _build_runtime_config(args)

    catalog_path = _resolve(str(cfg["CATALOG"]))
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")
    experiments = _read_csv_rows(catalog_path)
    _validate_catalog(experiments)

    if cfg.get("ONLY"):
        keep = {str(x).strip() for x in cfg["ONLY"]}
        experiments = [r for r in experiments if str(r.get("experiment_id", "")).strip() in keep]
        if not experiments:
            raise ValueError("No experiments left after ONLY filter.")

    baseline_candidates = [
        str(r["experiment_id"]).strip()
        for r in experiments
        if int(str(r.get("is_baseline", "0")).strip() or "0") == 1
    ]
    if len(baseline_candidates) != 1:
        raise ValueError("Filtered selection must include exactly one baseline row.")
    baseline_id = baseline_candidates[0]

    campaign_id = cfg.get("CAMPAIGN_ID") or datetime.now().strftime("campaign_%Y%m%d_%H%M%S")
    campaign_dir = _resolve(str(cfg["OUT_ROOT"])) / str(campaign_id)
    campaign_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_rows(campaign_dir / "experiments_used.csv", experiments)

    ctd_script = _resolve(str(cfg["CTD_SCRIPT"]))
    th_script = _resolve(str(cfg["TH_SCRIPT"]))
    wl_script = _resolve(str(cfg["WL_SCRIPT"]))

    plan_rows: List[Dict[str, Any]] = []
    statuses: List[Dict[str, Any]] = []

    for exp in experiments:
        exp_id = str(exp["experiment_id"]).strip()
        exp_dir = campaign_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "experiment_id": exp_id,
            "is_baseline": int(str(exp.get("is_baseline", "0")).strip() or "0"),
            "baseline_id": baseline_id,
            "grid": str(exp.get("grid", "")),
            "bathy": str(exp.get("bathy", "")),
            "forcing": str(exp.get("forcing", "")),
            "river": str(exp.get("river", "")),
        }
        with open(exp_dir / "experiment_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        ctd_cfg = _prepare_ctd_config(cfg, exp_id, exp_dir)
        th_cfg = _prepare_th_config(cfg, exp_id, exp_dir)
        wl_cfg = _prepare_wl_config(cfg, exp_id, exp_dir)

        ctd_cfg_path = exp_dir / "ctd_config.json"
        th_cfg_path = exp_dir / "th_config.json"
        wl_cfg_path = exp_dir / "wl_config.json"
        with open(ctd_cfg_path, "w", encoding="utf-8") as f:
            json.dump(ctd_cfg, f, indent=2)
        with open(th_cfg_path, "w", encoding="utf-8") as f:
            json.dump(th_cfg, f, indent=2)
        with open(wl_cfg_path, "w", encoding="utf-8") as f:
            json.dump(wl_cfg, f, indent=2)

        plan_rows.append(
            {
                "experiment_id": exp_id,
                "is_baseline": meta["is_baseline"],
                "grid": meta["grid"],
                "bathy": meta["bathy"],
                "forcing": meta["forcing"],
                "river": meta["river"],
                "ctd_config_path": str(ctd_cfg_path),
                "th_config_path": str(th_cfg_path),
                "wl_config_path": str(wl_cfg_path),
            }
        )

        task_specs = [
            {
                "task": "ctd",
                "enabled": bool(cfg["EXECUTE_CTD"]),
                "required_ok": bool(cfg.get("CTD_RUN_DIR_TEMPLATE")),
                "required_msg": "Missing CTD_RUN_DIR_TEMPLATE.",
                "run_path": ((ctd_cfg.get("schism") or [{}])[0]).get("run_dir", ""),
                "cmd": [str(cfg["PYTHON"]), str(ctd_script), "--config", str(ctd_cfg_path)],
                "stdout": exp_dir / "ctd" / "stdout.log",
                "stderr": exp_dir / "ctd" / "stderr.log",
            },
            {
                "task": "th",
                "enabled": bool(cfg["EXECUTE_TH"]),
                "required_ok": bool(cfg.get("TH_SCHISM_TEMPLATE")),
                "required_msg": "Missing TH_SCHISM_TEMPLATE.",
                "run_path": (th_cfg.get("schism_npzs") or [""])[0],
                "cmd": [str(cfg["PYTHON"]), str(th_script), "--config", str(th_cfg_path)],
                "stdout": exp_dir / "th" / "stdout.log",
                "stderr": exp_dir / "th" / "stderr.log",
            },
            {
                "task": "wl",
                "enabled": bool(cfg["EXECUTE_WL"]),
                "required_ok": bool(cfg.get("WL_RUN_TEMPLATE")),
                "required_msg": "Missing WL_RUN_TEMPLATE.",
                "run_path": (wl_cfg.get("runs") or [""])[0],
                "cmd": [str(cfg["PYTHON"]), str(wl_script), "--config", str(wl_cfg_path)],
                "stdout": exp_dir / "wl" / "stdout.log",
                "stderr": exp_dir / "wl" / "stderr.log",
            },
        ]

        for spec in task_specs:
            task = spec["task"]
            cmd = spec["cmd"]
            if not spec["enabled"] or bool(cfg["DRY_RUN"]):
                statuses.append(
                    _status_record(
                        exp_id,
                        task,
                        "planned",
                        "Execution skipped (dry-run or task not enabled).",
                        cmd,
                    )
                )
                continue

            if not spec["required_ok"]:
                msg = str(spec["required_msg"])
                statuses.append(_status_record(exp_id, task, "failed", msg, cmd))
                if not bool(cfg["CONTINUE_ON_ERROR"]):
                    raise RuntimeError(f"[{exp_id}:{task}] {msg}")
                continue

            run_path = Path(str(spec["run_path"])).expanduser()
            if not run_path.exists() and not str(run_path).endswith("staout"):
                msg = f"Model run input not found: {run_path}"
                statuses.append(_status_record(exp_id, task, "failed", msg, cmd))
                if not bool(cfg["CONTINUE_ON_ERROR"]):
                    raise FileNotFoundError(f"[{exp_id}:{task}] {msg}")
                continue

            ok, msg = _run_cmd(cmd, SCRIPT_DIR, spec["stdout"], spec["stderr"])
            statuses.append(_status_record(exp_id, task, "ok" if ok else "failed", msg, cmd))
            if (not ok) and (not bool(cfg["CONTINUE_ON_ERROR"])):
                raise RuntimeError(f"[{exp_id}:{task}] {msg}")

    _write_csv_rows(campaign_dir / "run_plan.csv", plan_rows)
    _write_csv_rows(
        campaign_dir / "run_status.csv",
        statuses,
        fieldnames=["experiment_id", "task", "status", "message", "command", "timestamp"],
    )

    metrics_info: Dict[str, Any] = {"enabled": bool(cfg["EXECUTE_SCORECARD"]), "saved": False}
    if bool(cfg["EXECUTE_SCORECARD"]):
        metrics_info = _save_metrics_and_scorecards(
            campaign_dir=campaign_dir,
            experiments=experiments,
            baseline_id=baseline_id,
            weights=dict(cfg["J_WEIGHTS"]),
            weighting_modes=list(cfg["WEIGHTING_MODES"]),
        )
        metrics_info["enabled"] = True
        metrics_info["saved"] = True

    summary = {
        "campaign_id": str(campaign_id),
        "catalog": str(catalog_path),
        "campaign_dir": str(campaign_dir),
        "baseline_id": baseline_id,
        "experiments": len(experiments),
        "execute_ctd": bool(cfg["EXECUTE_CTD"] and not cfg["DRY_RUN"]),
        "execute_th": bool(cfg["EXECUTE_TH"] and not cfg["DRY_RUN"]),
        "execute_wl": bool(cfg["EXECUTE_WL"] and not cfg["DRY_RUN"]),
        "execute_scorecard": bool(cfg["EXECUTE_SCORECARD"]),
        "metrics_info": metrics_info,
        "weights": dict(cfg["J_WEIGHTS"]),
    }
    with open(campaign_dir / "campaign_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Campaign prepared: {campaign_dir}")
    print(f"Baseline experiment: {baseline_id}")
    print(f"Plan CSV: {campaign_dir / 'run_plan.csv'}")
    print(f"Status CSV: {campaign_dir / 'run_status.csv'}")
    if bool(cfg["EXECUTE_SCORECARD"]):
        print(f"Merged metrics: {campaign_dir / 'metrics' / 'merged_metrics_raw.csv'}")
        print(f"Scorecards: {campaign_dir / 'metrics' / 'scorecard_station_equal.csv'}")
        print(f"            {campaign_dir / 'metrics' / 'scorecard_sample_count.csv'}")


if __name__ == "__main__":
    main()
