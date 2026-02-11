#!/usr/bin/env python3
"""
Controller for TEAMS/SCHISM validation campaigns.

Current implementation:
  - Reads experiment catalog with baseline/factor columns.
  - Generates per-experiment configs for CTD/TH/WL tasks.
  - Executes selected tasks via adapters (CTD/TH) and direct runner (WL).
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CATALOG = SCRIPT_DIR / "validation" / "experiments.csv"
DEFAULT_RUN_ROOT = SCRIPT_DIR / "validation" / "runs"

DEFAULT_CTD_SCRIPT = SCRIPT_DIR / "SCHISMvsTEAMS-CTD.py"
DEFAULT_CTD_AUTO = SCRIPT_DIR / "SCHISMvsTEAMS-CTD_auto.py"
DEFAULT_TH_SCRIPT = SCRIPT_DIR / "SCHISMvsTEAMS-TH.py"
DEFAULT_TH_AUTO = SCRIPT_DIR / "SCHISMvsTEAMS-TH_auto.py"
DEFAULT_WL_SCRIPT = SCRIPT_DIR / "comp_schism_th_wl.py"

DEFAULT_BPFILE_WL = SCRIPT_DIR / "station_jodc.bp"
DEFAULT_BPFILE_TH = SCRIPT_DIR / "station_sendai_d2.bp"
DEFAULT_OBS_WL = SCRIPT_DIR / "npz" / "jodc_tide_all.npz"

REQUIRED_COLUMNS = ["experiment_id", "is_baseline", "grid", "bathy", "forcing", "river"]


def _resolve(path_like: str) -> Path:
    p = Path(path_like).expanduser()
    if p.is_absolute():
        return p
    return (SCRIPT_DIR / p).resolve()


def _read_csv_rows(path: Path) -> List[Dict[str, str]]:
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader)


def _write_csv_rows(path: Path, rows: List[Dict[str, Any]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        if rows:
            fieldnames = list(rows[0].keys())
        else:
            fieldnames = []
    with open(path, "w", encoding="utf-8", newline="") as f:
        if not fieldnames:
            return
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


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
    nb = sum(1 for r in rows if int(str(r.get("is_baseline", "0")).strip() or "0") == 1)
    if nb != 1:
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


def _require_or_none(path_like: Optional[str]) -> Optional[str]:
    if path_like is None:
        return None
    return str(_resolve(path_like))


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validation controller for CTD/TH/WL.")
    p.add_argument("--catalog", default=str(DEFAULT_CATALOG), help="Experiment catalog CSV.")
    p.add_argument("--out-root", default=str(DEFAULT_RUN_ROOT), help="Campaign output root.")
    p.add_argument("--campaign-id", help="Optional campaign folder name.")
    p.add_argument("--only", nargs="+", help="Optional experiment IDs to run.")
    p.add_argument("--python", default=sys.executable, help="Python executable for subprocess tasks.")
    p.add_argument("--continue-on-error", action="store_true", help="Continue after task failure.")
    p.add_argument("--dry-run", action="store_true", help="Plan only; no task execution.")

    # CTD task
    p.add_argument("--execute-ctd", action="store_true", help="Execute CTD task.")
    p.add_argument("--ctd-auto-script", default=str(DEFAULT_CTD_AUTO), help="Path to CTD adapter script.")
    p.add_argument("--ctd-script", default=str(DEFAULT_CTD_SCRIPT), help="Path to SCHISMvsTEAMS-CTD.py")
    p.add_argument("--ctd-run-dir-template", help="Template, e.g. /path/to/{experiment_id}")
    p.add_argument("--ctd-teams-path", help="Override TEAMS CTD npz path.")
    p.add_argument("--ctd-start", help="Override CTD start date YYYY-MM-DD.")
    p.add_argument("--ctd-end", help="Override CTD end date YYYY-MM-DD.")
    p.add_argument(
        "--ctd-enable-global-model",
        action="store_true",
        help="If set, keep global model comparison enabled in CTD.",
    )
    p.add_argument("--ctd-color", default="b", help="Model color for CTD plotting.")

    # TH task
    p.add_argument("--execute-th", action="store_true", help="Execute TH task.")
    p.add_argument("--th-auto-script", default=str(DEFAULT_TH_AUTO), help="Path to TH adapter script.")
    p.add_argument("--th-script", default=str(DEFAULT_TH_SCRIPT), help="Path to SCHISMvsTEAMS-TH.py")
    p.add_argument("--th-schism-template", help="Template, e.g. /path/to/{experiment_id}.npz")
    p.add_argument("--th-teams-path", help="Override TEAMS TH npz path.")
    p.add_argument("--th-bpfile", default=str(DEFAULT_BPFILE_TH), help="BP file for TH task.")
    p.add_argument("--th-grid", help="Optional grid path for TH map panel.")
    p.add_argument("--th-start", help="Override TH start datetime.")
    p.add_argument("--th-end", help="Override TH end datetime.")
    p.add_argument("--th-resample", default="h", help="TH resample frequency.")
    p.add_argument("--th-vars", nargs="+", default=["temp", "sal"], help="TH variables.")

    # WL task
    p.add_argument("--execute-wl", action="store_true", help="Execute WL task.")
    p.add_argument("--wl-script", default=str(DEFAULT_WL_SCRIPT), help="Path to comp_schism_th_wl.py.")
    p.add_argument("--wl-run-template", help="Template, e.g. /path/to/{experiment_id}.npz")
    p.add_argument("--wl-bpfile", default=str(DEFAULT_BPFILE_WL), help="BP file for WL task.")
    p.add_argument("--wl-obs-path", default=str(DEFAULT_OBS_WL), help="Observation file path for WL.")
    p.add_argument("--wl-start", default="2022-01-14 00:00:00", help="WL start datetime.")
    p.add_argument("--wl-end", default="2022-03-30 00:00:00", help="WL end datetime.")
    p.add_argument("--wl-model-start", default="2022-01-02 00:00:00", help="WL model start datetime.")
    return p.parse_args(argv)


def _prepare_ctd_config(args: argparse.Namespace, exp_id: str, exp_dir: Path) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "_auto_single_run": {
            "label": exp_id,
            "run_dir": "",
            "color": args.ctd_color,
        },
        "_auto_disable_global_model": not bool(args.ctd_enable_global_model),
        "output": {"dir": str((exp_dir / "ctd").resolve())},
    }
    if args.ctd_run_dir_template:
        cfg["_auto_single_run"]["run_dir"] = args.ctd_run_dir_template.format(experiment_id=exp_id)
    if args.ctd_teams_path:
        cfg["teams"] = {"npz_path": str(_resolve(args.ctd_teams_path))}
    if args.ctd_start or args.ctd_end:
        dr: Dict[str, Any] = {}
        if args.ctd_start:
            dr["start"] = args.ctd_start
        if args.ctd_end:
            dr["end"] = args.ctd_end
        cfg["date_range"] = dr
    return cfg


def _prepare_th_config(args: argparse.Namespace, exp_id: str, exp_dir: Path) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "schism_npzs": [],
        "schism_labels": [exp_id],
        "bpfile": str(_resolve(args.th_bpfile)),
        "outdir": str((exp_dir / "th").resolve()),
        "vars": list(args.th_vars),
        "resample": args.th_resample,
        "debug_times": True,
    }
    if args.th_schism_template:
        cfg["schism_npzs"] = [args.th_schism_template.format(experiment_id=exp_id)]
    if args.th_teams_path:
        cfg["teams_npz"] = str(_resolve(args.th_teams_path))
    if args.th_grid:
        cfg["grid"] = str(_resolve(args.th_grid))
    if args.th_start:
        cfg["start"] = args.th_start
    if args.th_end:
        cfg["end"] = args.th_end
    return cfg


def _prepare_wl_config(args: argparse.Namespace, exp_id: str, exp_dir: Path) -> Dict[str, Any]:
    cfg: Dict[str, Any] = {
        "runs": [],
        "tags": [exp_id],
        "bpfile": str(_resolve(args.wl_bpfile)),
        "outdir": str((exp_dir / "wl").resolve()),
        "obs_path": str(_resolve(args.wl_obs_path)),
        "start": args.wl_start,
        "end": args.wl_end,
        "model_start": args.wl_model_start,
        "resample_obs": "h",
        "resample_model": "h",
        "demean": True,
        "save_plots": True,
        "progress_every": 20,
    }
    if args.wl_run_template:
        cfg["runs"] = [args.wl_run_template.format(experiment_id=exp_id)]
    return cfg


def main(argv: Optional[List[str]] = None) -> None:
    args = _parse_args(argv)

    catalog_path = _resolve(args.catalog)
    if not catalog_path.exists():
        raise FileNotFoundError(f"Catalog not found: {catalog_path}")
    rows = _read_csv_rows(catalog_path)
    _validate_catalog(rows)

    if args.only:
        keep = {x.strip() for x in args.only}
        rows = [r for r in rows if str(r.get("experiment_id", "")).strip() in keep]
        if not rows:
            raise ValueError("No experiments left after --only filter.")

    baseline_candidates = [r["experiment_id"] for r in rows if int(str(r.get("is_baseline", "0")).strip() or "0") == 1]
    if len(baseline_candidates) != 1:
        raise ValueError("Filtered selection must include exactly one baseline row.")
    baseline_id = str(baseline_candidates[0]).strip()

    campaign_id = args.campaign_id or datetime.now().strftime("campaign_%Y%m%d_%H%M%S")
    campaign_dir = _resolve(args.out_root) / campaign_id
    campaign_dir.mkdir(parents=True, exist_ok=True)
    _write_csv_rows(campaign_dir / "experiments_used.csv", rows)

    ctd_auto = _resolve(args.ctd_auto_script)
    ctd_script = _resolve(args.ctd_script)
    th_auto = _resolve(args.th_auto_script)
    th_script = _resolve(args.th_script)
    wl_script = _resolve(args.wl_script)

    plan_rows: List[Dict[str, Any]] = []
    statuses: List[Dict[str, Any]] = []

    for row in rows:
        exp_id = str(row["experiment_id"]).strip()
        exp_dir = campaign_dir / exp_id
        exp_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "experiment_id": exp_id,
            "is_baseline": int(str(row["is_baseline"]).strip() or "0"),
            "baseline_id": baseline_id,
            "grid": str(row["grid"]),
            "bathy": str(row["bathy"]),
            "forcing": str(row["forcing"]),
            "river": str(row["river"]),
        }
        with open(exp_dir / "experiment_meta.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        ctd_cfg = _prepare_ctd_config(args, exp_id, exp_dir)
        th_cfg = _prepare_th_config(args, exp_id, exp_dir)
        wl_cfg = _prepare_wl_config(args, exp_id, exp_dir)

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
                "enabled": bool(args.execute_ctd),
                "required_ok": bool(args.ctd_run_dir_template),
                "required_msg": "Missing --ctd-run-dir-template.",
                "run_path": ctd_cfg.get("_auto_single_run", {}).get("run_dir", ""),
                "cmd": [args.python, str(ctd_auto), "--script", str(ctd_script), "--config", str(ctd_cfg_path)],
                "stdout": exp_dir / "ctd" / "stdout.log",
                "stderr": exp_dir / "ctd" / "stderr.log",
            },
            {
                "task": "th",
                "enabled": bool(args.execute_th),
                "required_ok": bool(args.th_schism_template),
                "required_msg": "Missing --th-schism-template.",
                "run_path": (th_cfg.get("schism_npzs") or [""])[0],
                "cmd": [args.python, str(th_auto), "--script", str(th_script), "--config", str(th_cfg_path)],
                "stdout": exp_dir / "th" / "stdout.log",
                "stderr": exp_dir / "th" / "stderr.log",
            },
            {
                "task": "wl",
                "enabled": bool(args.execute_wl),
                "required_ok": bool(args.wl_run_template),
                "required_msg": "Missing --wl-run-template.",
                "run_path": (wl_cfg.get("runs") or [""])[0],
                "cmd": [args.python, str(wl_script), "--config", str(wl_cfg_path)],
                "stdout": exp_dir / "wl" / "stdout.log",
                "stderr": exp_dir / "wl" / "stderr.log",
            },
        ]

        for spec in task_specs:
            task = spec["task"]
            cmd = spec["cmd"]
            if not spec["enabled"] or args.dry_run:
                statuses.append(
                    _status_record(
                        exp_id,
                        task,
                        "planned",
                        "Execution skipped (dry-run or task flag not set).",
                        cmd,
                    )
                )
                continue

            if not spec["required_ok"]:
                msg = str(spec["required_msg"])
                statuses.append(_status_record(exp_id, task, "failed", msg, cmd))
                if not args.continue_on_error:
                    raise RuntimeError(f"[{exp_id}:{task}] {msg}")
                continue

            run_path = Path(str(spec["run_path"])).expanduser()
            if not run_path.exists() and not str(run_path).endswith("staout"):
                msg = f"Model run input not found: {run_path}"
                statuses.append(_status_record(exp_id, task, "failed", msg, cmd))
                if not args.continue_on_error:
                    raise FileNotFoundError(f"[{exp_id}:{task}] {msg}")
                continue

            ok, msg = _run_cmd(cmd, SCRIPT_DIR, spec["stdout"], spec["stderr"])
            statuses.append(_status_record(exp_id, task, "ok" if ok else "failed", msg, cmd))
            if (not ok) and (not args.continue_on_error):
                raise RuntimeError(f"[{exp_id}:{task}] {msg}")

    _write_csv_rows(campaign_dir / "run_plan.csv", plan_rows)
    _write_csv_rows(
        campaign_dir / "run_status.csv",
        statuses,
        fieldnames=["experiment_id", "task", "status", "message", "command", "timestamp"],
    )

    summary = {
        "campaign_id": campaign_id,
        "catalog": str(catalog_path),
        "campaign_dir": str(campaign_dir),
        "baseline_id": baseline_id,
        "experiments": len(rows),
        "execute_ctd": bool(args.execute_ctd and not args.dry_run),
        "execute_th": bool(args.execute_th and not args.dry_run),
        "execute_wl": bool(args.execute_wl and not args.dry_run),
        "ctd_script": str(ctd_script),
        "th_script": str(th_script),
        "wl_script": str(wl_script),
    }
    with open(campaign_dir / "campaign_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Campaign prepared: {campaign_dir}")
    print(f"Baseline experiment: {baseline_id}")
    print(f"Plan CSV: {campaign_dir / 'run_plan.csv'}")
    print(f"Status CSV: {campaign_dir / 'run_status.csv'}")


if __name__ == "__main__":
    main()
