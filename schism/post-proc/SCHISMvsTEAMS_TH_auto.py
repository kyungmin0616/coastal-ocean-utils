#!/usr/bin/env python3
"""
Adapter runner for SCHISMvsTEAMS_TH.py.

Purpose:
  - Keep SCHISMvsTEAMS_TH.py unchanged.
  - Apply per-experiment config overrides from JSON.
  - Run main() with updated CONFIG defaults.
"""

from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SCRIPT = SCRIPT_DIR / "SCHISMvsTEAMS_TH.py"


def _load_module(script_path: Path):
    spec = importlib.util.spec_from_file_location("schism_vs_teams_th_mod", script_path)
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


def _parse_args():
    p = argparse.ArgumentParser(description="Run SCHISMvsTEAMS_TH.py with JSON overrides.")
    p.add_argument("--script", default=str(DEFAULT_SCRIPT), help="Path to SCHISMvsTEAMS_TH.py")
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

    merged = _deep_update(copy.deepcopy(mod.CONFIG), user_cfg)
    mod.CONFIG = merged

    dump_path = Path(args.dump_used_config).expanduser().resolve() if args.dump_used_config else None
    if dump_path is None:
        out_dir = Path(str(merged.get("outdir", "."))).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        dump_path = out_dir / "th_config_used_auto.json"
    else:
        dump_path.parent.mkdir(parents=True, exist_ok=True)
    with open(dump_path, "w", encoding="utf-8") as f:
        json.dump(merged, f, indent=2)

    mod.main([])


if __name__ == "__main__":
    main()
