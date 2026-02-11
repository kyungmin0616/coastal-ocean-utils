#!/usr/bin/env python3
"""
Adapter runner for SCHISMvsTEAMS-CTD.py.

Purpose:
  - Keep SCHISMvsTEAMS-CTD.py unchanged.
  - Apply per-experiment config overrides from JSON.
  - Run main() with updated CONFIG.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SCRIPT = SCRIPT_DIR / "SCHISMvsTEAMS-CTD.py"


def _load_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("schism_vs_teams_ctd_mod", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {script_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _parse_ymd(value: str):
    parts = value.strip().split("-")
    if len(parts) != 3:
        raise ValueError(f"Expected YYYY-MM-DD, got: {value}")
    return [int(parts[0]), int(parts[1]), int(parts[2])]


def _parse_args():
    p = argparse.ArgumentParser(description="Run SCHISMvsTEAMS-CTD.py with JSON overrides.")
    p.add_argument("--script", default=str(DEFAULT_SCRIPT), help="Path to SCHISMvsTEAMS-CTD.py")
    p.add_argument("--config", required=True, help="JSON override file.")
    p.add_argument("--dump-used-config", help="Optional path to save merged config JSON.")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    script_path = Path(args.script).expanduser().resolve()
    cfg_path = Path(args.config).expanduser().resolve()

    if not script_path.exists():
        raise FileNotFoundError(f"Script not found: {script_path}")
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        user_cfg = json.load(f)

    mod = _load_module(script_path)
    if not hasattr(mod, "CONFIG") or not hasattr(mod, "main"):
        raise RuntimeError(f"{script_path} does not expose CONFIG/main as expected.")

    base_cfg = copy.deepcopy(mod.CONFIG)
    auto_run = user_cfg.pop("_auto_single_run", None)
    auto_disable_global = user_cfg.pop("_auto_disable_global_model", None)

    merged = _deep_update(base_cfg, user_cfg)

    if auto_run is not None:
        sch_list = merged.get("schism", [])
        if not isinstance(sch_list, list) or len(sch_list) == 0:
            sch_list = [{}]
        first = copy.deepcopy(sch_list[0])
        first["enabled"] = True
        if "label" in auto_run:
            first["label"] = auto_run["label"]
        if "run_dir" in auto_run:
            first["run_dir"] = auto_run["run_dir"]
        if "color" in auto_run:
            first["color"] = auto_run["color"]
        merged["schism"] = [first]

    if auto_disable_global is not None:
        gm = copy.deepcopy(merged.get("global_model", {}))
        gm["enabled"] = not bool(auto_disable_global)
        merged["global_model"] = gm

    # Convenience parse for date_range strings.
    dr = merged.get("date_range", {})
    if isinstance(dr, dict):
        if isinstance(dr.get("start"), str):
            dr["start"] = _parse_ymd(dr["start"])
        if isinstance(dr.get("end"), str):
            dr["end"] = _parse_ymd(dr["end"])
        merged["date_range"] = dr

    mod.CONFIG = merged

    dump_path = Path(args.dump_used_config).expanduser().resolve() if args.dump_used_config else None
    if dump_path is None:
        out_dir = Path(str(merged.get("output", {}).get("dir", "."))).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        dump_path = out_dir / "ctd_config_used_auto.json"
    else:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dump_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    mod.main()


if __name__ == "__main__":
    main()
