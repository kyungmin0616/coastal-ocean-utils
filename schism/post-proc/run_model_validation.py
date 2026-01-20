#!/usr/bin/env python3
"""Controller to run comp_schism.py and comp_ww3.py for multiple experiments."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union


SCRIPT_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable


def _expand_path(value: Union[str, Path]) -> Path:
    return Path(os.path.expanduser(str(value)))


# ---------------------------------------------------------------------------
# User configuration
# ---------------------------------------------------------------------------

# Experiment names to process. Edit this list to control which runs execute.
exps: List[Union[str, Dict[str, Any]]] = [
    "RUN04d",
]

# Toggle individual model comparisons.
RUN_SCHISM = True
RUN_WW3 = True

# MPI launcher for comp_schism.py. Example: ["mpiexec", "-n", "8"]. Leave empty
# to run in serial mode (comp_schism.py auto-detects MPI when launched this way).
MPI_LAUNCH: Sequence[str] = ["mpirun","-np", "8"]

# --- SCHISM comparison settings ------------------------------------------------

SCHISM_TAG_TEMPLATE = "{exp}"
SCHISM_LINE_WIDTH = 2
SCHISM_BPFILE_DEFAULT = _expand_path("/scratch3/projects/CATUFS/KyungminPark/post-proc/Round4/stations/stationExp")
SCHISM_BPFILE_OVERRIDES: Dict[str, Union[str, Path]] = {
}
SCHISM_PLOT_ROOT_DEFAULT = _expand_path("/scratch3/projects/CATUFS/KyungminPark/post-proc/Round4/images/RUN03a-8-10-11-12")
SCHISM_PLOT_ROOT_OVERRIDES: Dict[str, Union[str, Path]] = {
}
SCHISM_OBS_PATHS_DEFAULT: Dict[str, Union[str, Path]] = {
    "WL": "/scratch3/projects/CATUFS/KyungminPark/post-proc/Round4/npz/twl-ufs-2021.npz",
    "VEL":"/scratch3/projects/CATUFS/KyungminPark/post-proc/Round4/npz/current-ufs-2021.npz",
    "TEMP": "/scratch3/projects/CATUFS/KyungminPark/post-proc/Round4/npz/temp-ufs.npz",
    "SALT": "/scratch3/projects/CATUFS/KyungminPark/post-proc/Round4/npz/salt-ufs.npz",
}
SCHISM_OBS_PATHS_OVERRIDES: Dict[str, Dict[str, Union[str, Path]]] = {
}

# Data source selection for SCHISM. Default source applies when an experiment
# does not set its own preference via SCHISM_OVERRIDES. Each entry in
# SCHISM_SOURCE_TEMPLATES is a list of candidate path templates formatted with
# {exp}. Update the staout template to match your directory layout if needed.
SCHISM_DEFAULT_SOURCE = "staout"  # options: "npz", "staout"
SCHISM_SOURCE_OVERRIDES: Dict[str, str] = {
    # "RUN02a": "staout",
}
SCHISM_SOURCE_TEMPLATES: Dict[str, List[Union[str, Path]]] = {
    "npz": ["npz/{exp}_schism.npz"],
    "staout": ["/scratch3/projects/CATUFS/KyungminPark/run/{exp}/outputs/staout"],
}

# Provide start/end windows for SCHISM plots. A single entry automatically
# replicates across the four panels used by comp_schism.py. Override per
# experiment through SCHISM_OVERRIDES if needed.
SCHISM_TIME_WINDOWS = {
    "stts": ["2021-07-1"],
    "edts": ["2021-9-25"],
}

# Optional per-experiment overrides. Values support {exp} formatting.
SCHISM_OVERRIDES: Dict[str, Dict[str, Any]] = {
    # "RUN02a": {
    #     "source": "staout",  # switch to staout templates for this run
    #     "runs": ["/path/to/{exp}/outputs/staout"],  # optional explicit paths
    #     "tags": ["{exp}_alt"],
    #     "stts": ["2021-07-01"],
    #     "edts": ["2021-08-01"],
    #     "sname": "images/custom/{exp}_schism",
    #     "lw": 1.5,
    # }
}

# --- WW3 comparison settings ---------------------------------------------------

WW3_RUN_TEMPLATE = "/scratch3/projects/CATUFS/KyungminPark/post-proc/Round4/npz/{exp}-wave.npz"
WW3_TAG_TEMPLATE = "{exp}-ww3"
WW3_OBS_NPZ = "/scratch3/projects/CATUFS/KyungminPark/post-proc/Round4/npz/NDBC_NYNJ.npz"
WW3_PLOT_ROOT = _expand_path("/scratch3/projects/CATUFS/KyungminPark/post-proc/Round4/images/RUN04")
WW3_TIME_WINDOW = {"start": "2021-07-1", "end": "2021-09-29"}

# Optional per-experiment WW3 overrides. Each entry is a list of dictionaries
# that mirrors comp_ww3.py's "runs" configuration.
WW3_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "RUN03a-10": {
        "station_mapping": {
            "44065": "44065",
            "44025": "44025",
            "44091": "44091",
        },
    },
}

# Optional global station mapping override for WW3.
WW3_STATION_MAPPING: Optional[Dict[str, str]] = None

# When False, the controller continues after a comparison fails.
FAIL_FAST = True


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _ensure_sequence(value: Union[Sequence[Any], Any]) -> List[Any]:
    if isinstance(value, Path):
        return [value]
    if isinstance(value, (str, bytes)):
        return [value]
    if isinstance(value, Sequence):
        return list(value)
    return [value]


def _format_value(obj: Any, exp: str) -> Any:
    if isinstance(obj, str):
        return obj.format(exp=exp)
    if isinstance(obj, Path):
        return obj
    if isinstance(obj, Mapping):
        return {key: _format_value(val, exp) for key, val in obj.items()}
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        return [_format_value(item, exp) for item in obj]
    return obj


def _slugify(value: str) -> str:
    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    return "".join(ch if ch in allowed else "_" for ch in value)


def _normalize_group_entry(entry: Union[str, Dict[str, Any], Sequence[Any]]) -> Dict[str, Any]:
    if isinstance(entry, (list, tuple)):
        experiments = [str(item) for item in entry]
        base: Dict[str, Any] = {"name": "__".join(experiments)}
    else:
        base = _coerce_experiment(entry)
        experiments = base.get("experiments")
        if experiments is None:
            experiments = [base.get("name")]
        elif isinstance(experiments, (str, bytes)):
            experiments = [str(experiments)]
        else:
            experiments = [str(item) for item in experiments]
    experiments = [exp for exp in experiments if exp]
    base["experiments"] = experiments
    if not base.get("name"):
        base["name"] = "__".join(experiments)
    joined = "__".join(experiments) if experiments else base["name"]
    base.setdefault("label", base["name"])
    if len(experiments) > 1 and base["label"] == base["name"]:
        base["label"] = joined
    base.setdefault("output_id", _slugify(joined))
    if not base["output_id"]:
        base["output_id"] = _slugify(base.get("label") or base.get("name") or "group")
    return base


def _is_group_definition(entry: Any) -> bool:
    if isinstance(entry, str):
        return False
    return isinstance(entry, (dict, list, tuple))


def _prepare_groups(entries: Sequence[Union[str, Dict[str, Any], Sequence[Any]]]) -> List[Dict[str, Any]]:
    filtered = [entry for entry in entries if entry is not None]
    if not filtered:
        return []
    explicit = any(_is_group_definition(entry) for entry in filtered)
    if not explicit:
        experiments = [str(entry) for entry in filtered if str(entry)]
        combined = {"name": "__".join(experiments), "experiments": experiments}
        return [_normalize_group_entry(combined)] if experiments else []
    return [_normalize_group_entry(entry) for entry in filtered]


def _normalize_times(values: Union[Sequence[Any], Any], length: int) -> List[Any]:
    seq = _ensure_sequence(values)
    if len(seq) == 1 and length > 1:
        seq = seq * length
    if len(seq) != length:
        raise ValueError(f"Expected {length} time values but received {len(seq)}")
    normalized: List[Any] = []
    for item in seq:
        if isinstance(item, (int, float)):
            normalized.append(float(item))
        else:
            normalized.append(str(item))
    return normalized


def _resolve_path(path_like: Union[str, Path]) -> Path:
    path = _expand_path(path_like)
    return path if path.is_absolute() else (SCRIPT_DIR / path).resolve()


def _json_dumps(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, default=str)


def _maybe_queue_exception(exp: str, phase: str, exc: Exception) -> None:
    message = f"[{exp}] {phase} failed: {exc}"
    if FAIL_FAST:
        raise RuntimeError(message) from exc
    print(message)


def _is_valid_schism_run(path_like: Union[str, Path]) -> bool:
    path = _resolve_path(path_like)
    if path.exists():
        return True
    if path.name.endswith("staout"):
        parent = path.parent
        if not parent.exists():
            return False
        pattern = f"{path.name}_*"
        return any(parent.glob(pattern))
    return False


# ---------------------------------------------------------------------------
# SCHISM helpers
# ---------------------------------------------------------------------------

def _coerce_experiment(entry: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    if isinstance(entry, str):
        return {"name": entry}
    return dict(entry)


def build_schism_config(exp: str, overrides: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    override_runs = overrides.get("runs")
    if override_runs is not None:
        runs = _ensure_sequence(_format_value(override_runs, exp))
    else:
        source = overrides.get("source") or SCHISM_SOURCE_OVERRIDES.get(exp, SCHISM_DEFAULT_SOURCE)
        templates = overrides.get("templates")
        if templates is not None:
            template_list = _ensure_sequence(_format_value(templates, exp))
        else:
            template_list = SCHISM_SOURCE_TEMPLATES.get(source, [])
        runs = []
        for tmpl in template_list:
            runs.extend(_ensure_sequence(_format_value(tmpl, exp)))

    existing_runs = [str(_resolve_path(run)) for run in runs if _is_valid_schism_run(run)]
    if not existing_runs:
        print(f"[{exp}] Skipping SCHISM comparison (no NPZ files found): {runs}")
        return None

    override_tags = overrides.get("tags")
    if override_tags is None:
        tags = [_format_value(SCHISM_TAG_TEMPLATE, exp) for _ in existing_runs]
    else:
        tags = _ensure_sequence(_format_value(override_tags, exp))
        if len(tags) != len(existing_runs):
            raise ValueError(f"[{exp}] SCHISM tags length must match runs length")

    stts_source = overrides.get("stts", SCHISM_TIME_WINDOWS.get("stts"))
    edts_source = overrides.get("edts", SCHISM_TIME_WINDOWS.get("edts"))
    if stts_source is None or edts_source is None:
        raise ValueError("SCHISM time windows must provide both stts and edts")
    stts = _normalize_times(_format_value(stts_source, exp), length=4)
    edts = _normalize_times(_format_value(edts_source, exp), length=4)

    sname_default = SCHISM_PLOT_ROOT_OVERRIDES.get(exp, SCHISM_PLOT_ROOT_DEFAULT / exp)
    sname_value = overrides.get("sname", sname_default)
    sname = Path(_format_value(sname_value, exp))
    if not sname.is_absolute():
        sname = (SCRIPT_DIR / sname).resolve()
    sname.mkdir(parents=True, exist_ok=True)

    bpfile_default = SCHISM_BPFILE_OVERRIDES.get(exp, SCHISM_BPFILE_DEFAULT)
    bpfile_value = overrides.get("bpfile", bpfile_default)
    bpfile = _format_value(bpfile_value, exp)
    if not _resolve_path(bpfile).exists():
        print(f"[{exp}] Warning: SCHISM bpfile not found: {bpfile}")

    obs_paths = {key: str(_format_value(val, exp)) for key, val in SCHISM_OBS_PATHS_DEFAULT.items()}
    obs_paths.update({k: str(_format_value(v, exp)) for k, v in SCHISM_OBS_PATHS_OVERRIDES.get(exp, {}).items()})
    if "obs_paths" in overrides:
        override_obs = _format_value(overrides["obs_paths"], exp)
        obs_paths.update({k: str(v) for k, v in override_obs.items()})

    config: Dict[str, Any] = {
        "runs": existing_runs,
        "tags": tags,
        "bpfile": str(bpfile),
        "stts": stts,
        "edts": edts,
        "sname": str(sname),
        "lw": overrides.get("lw", SCHISM_LINE_WIDTH),
        "obs_paths": obs_paths,
    }

    return config


def run_schism(exp: str, config: Dict[str, Any]) -> None:
    env = os.environ.copy()
    env["COMP_SCHISM_CONFIG"] = _json_dumps(config)
    cmd: List[str] = list(MPI_LAUNCH) + [PYTHON, str(SCRIPT_DIR / "comp_schism.py")]
    print(f"[{exp}] Running comp_schism.py")
    subprocess.run(cmd, cwd=SCRIPT_DIR, check=True, env=env)


def build_schism_group_config(group: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    experiments = group.get("experiments", [])
    collected: List[Dict[str, Any]] = []
    for exp in experiments:
        overrides = SCHISM_OVERRIDES.get(exp, {})
        cfg = build_schism_config(exp, overrides)
        if cfg:
            collected.append({"exp": exp, "config": cfg})

    if not collected:
        print(f"[{group['label']}] Skipping SCHISM comparison (no runs found)")
        return None

    base = collected[0]["config"]
    combined_runs: List[str] = []
    combined_tags: List[str] = []
    combined_obs = dict(base.get("obs_paths", {}))

    for item in collected:
        cfg = item["config"]
        combined_runs.extend(cfg["runs"])
        combined_tags.extend(cfg["tags"])

        if cfg["bpfile"] != base["bpfile"]:
            raise ValueError(f"[{group['label']}] Inconsistent SCHISM bpfile between experiments")
        if cfg["stts"] != base["stts"] or cfg["edts"] != base["edts"]:
            raise ValueError(f"[{group['label']}] Inconsistent time windows between experiments")

        for key, value in cfg.get("obs_paths", {}).items():
            existing = combined_obs.get(key)
            if existing is not None and existing != value:
                raise ValueError(f"[{group['label']}] Conflicting observation path for {key}")
            combined_obs[key] = value

    plot_dir_override = group.get("schism_plot_dir")
    if plot_dir_override is None:
        plot_dir_override = _resolve_path(SCHISM_PLOT_ROOT_DEFAULT) / group["output_id"]
    plot_dir = _resolve_path(plot_dir_override)
    plot_dir.mkdir(parents=True, exist_ok=True)

    combined_config = {
        **base,
        "runs": combined_runs,
        "tags": combined_tags,
        "sname": str(plot_dir),
        "obs_paths": combined_obs,
    }

    if "schism_overrides" in group:
        overrides = group["schism_overrides"]
        for forbidden in ("runs", "tags"):
            if forbidden in overrides:
                raise ValueError(f"[{group['label']}] schism_overrides cannot set '{forbidden}'")
        combined_config.update(overrides)

    return combined_config


# ---------------------------------------------------------------------------
# WW3 helpers
# ---------------------------------------------------------------------------

def build_ww3_config(exp: str, overrides: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    run_entries = overrides.get("runs")
    if run_entries is None:
        run_entries = [{
            "path": WW3_RUN_TEMPLATE,
            "tag": WW3_TAG_TEMPLATE,
            "stt": WW3_TIME_WINDOW.get("start"),
            "edt": WW3_TIME_WINDOW.get("end"),
        }]
    formatted_runs = _format_value(run_entries, exp)
    runs: List[Dict[str, Any]] = []
    for entry in formatted_runs:
        path = entry.get("path")
        if path is None:
            continue
        if not _resolve_path(path).exists():
            print(f"[{exp}] Skipping WW3 run (missing NPZ): {path}")
            continue
        if "tag" not in entry:
            entry["tag"] = f"{exp}-ww3"
        if entry.get("stt") is None:
            entry["stt"] = WW3_TIME_WINDOW.get("start")
        if entry.get("edt") is None:
            entry["edt"] = WW3_TIME_WINDOW.get("end")
        entry["path"] = str(path)
        runs.append(entry)

    if not runs:
        print(f"[{exp}] Skipping comp_ww3.py (no runs configured)")
        return None

    plot_dir_value = overrides.get("plot_dir", WW3_PLOT_ROOT / exp)
    plot_dir = Path(_format_value(plot_dir_value, exp))
    if not plot_dir.is_absolute():
        plot_dir = (SCRIPT_DIR / plot_dir).resolve()
    plot_dir.mkdir(parents=True, exist_ok=True)

    config: Dict[str, Any] = {
        "runs": runs,
        "obs_npz": _format_value(overrides.get("obs_npz", WW3_OBS_NPZ), exp),
        "default_start": overrides.get("default_start", WW3_TIME_WINDOW.get("start")),
        "default_end": overrides.get("default_end", WW3_TIME_WINDOW.get("end")),
        "plot_dir": str(plot_dir),
    }

    if not _resolve_path(config["obs_npz"]).exists():
        print(f"[{exp}] Skipping WW3 comparison (obs NPZ not found): {config['obs_npz']}")
        return None

    station_map = overrides.get("station_mapping", WW3_STATION_MAPPING)
    if station_map is not None:
        config["station_mapping"] = station_map

    return config


def run_ww3(exp: str, config: Dict[str, Any]) -> None:
    env = os.environ.copy()
    env["COMP_WW3_CONFIG"] = _json_dumps(config)
    cmd = [PYTHON, str(SCRIPT_DIR / "comp_ww3.py")]
    print(f"[{exp}] Running comp_ww3.py")
    subprocess.run(cmd, cwd=SCRIPT_DIR, check=True, env=env)


def build_ww3_group_config(group: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    experiments = group.get("experiments", [])
    collected: List[Dict[str, Any]] = []
    for exp in experiments:
        overrides = WW3_OVERRIDES.get(exp, {})
        cfg = build_ww3_config(exp, overrides)
        if cfg:
            collected.append({"exp": exp, "config": cfg})

    if not collected:
        print(f"[{group['label']}] Skipping comp_ww3.py (no runs found)")
        return None

    combined_runs: List[Dict[str, Any]] = []
    obs_npz: Optional[str] = None
    default_start: Optional[Any] = None
    default_end: Optional[Any] = None
    station_mapping: Dict[str, str] = {}

    for item in collected:
        cfg = item["config"]
        combined_runs.extend(dict(run) for run in cfg["runs"])

        if obs_npz is None:
            obs_npz = cfg["obs_npz"]
        elif obs_npz != cfg["obs_npz"]:
            raise ValueError(f"[{group['label']}] Inconsistent WW3 observation NPZ between experiments")

        if cfg.get("default_start") is not None:
            if default_start is None or str(cfg["default_start"]) < str(default_start):
                default_start = cfg["default_start"]

        if cfg.get("default_end") is not None:
            if default_end is None or str(cfg["default_end"]) > str(default_end):
                default_end = cfg["default_end"]

        station_mapping.update(cfg.get("station_mapping", {}))

    plot_dir_override = group.get("ww3_plot_dir")
    if plot_dir_override is None:
        plot_dir_override = _resolve_path(WW3_PLOT_ROOT) / group["output_id"]
    plot_dir = _resolve_path(plot_dir_override)
    plot_dir.mkdir(parents=True, exist_ok=True)

    base = collected[0]["config"]
    combined_config = {
        **base,
        "runs": combined_runs,
        "obs_npz": obs_npz,
        "default_start": default_start,
        "default_end": default_end,
        "plot_dir": str(plot_dir),
    }

    if station_mapping:
        combined_config["station_mapping"] = station_mapping

    if "ww3_overrides" in group:
        overrides = group["ww3_overrides"]
        if "runs" in overrides:
            raise ValueError(f"[{group['label']}] ww3_overrides cannot set 'runs'")
        combined_config.update(overrides)

    return combined_config


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    groups = _prepare_groups(exps)

    for group in groups:
        label = group.get("label") or group.get("name") or "group"

        if RUN_SCHISM:
            try:
                schism_config = build_schism_group_config(group)
                if schism_config:
                    run_schism(label, schism_config)
            except Exception as exc:  # noqa: BLE001 - propagate or log per FAIL_FAST
                _maybe_queue_exception(label, "SCHISM", exc)
                if FAIL_FAST:
                    break

        if RUN_WW3:
            try:
                ww3_config = build_ww3_group_config(group)
                if ww3_config:
                    run_ww3(label, ww3_config)
            except Exception as exc:  # noqa: BLE001
                _maybe_queue_exception(label, "WW3", exc)
                if FAIL_FAST:
                    break


if __name__ == "__main__":
    main()
