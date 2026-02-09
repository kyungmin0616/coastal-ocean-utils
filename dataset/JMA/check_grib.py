#!/usr/bin/env python3
import os
import shlex
import subprocess
import re
from collections import defaultdict
from datetime import datetime, timedelta

# ==========================================
#      POINT THIS TO ONE GRIB FILE
# ==========================================
FILE_PATH = "msm_data/2011/Z__C_RJTD_20111231000000_MSM_GPV_Rjp_Lsurf_FH00-15_grib2.bin"
# ==========================================

# HPC module required by user environment
WGRIB2_MODULE = "wgrib2_netcdf4/"


def run_wgrib2_inventory(file_path: str, module_name: str) -> list[str]:
    quoted = shlex.quote(file_path)
    cmd = (
        f"module load {shlex.quote(module_name)} >/dev/null 2>&1; "
        f"wgrib2 {quoted} -v"
    )
    proc = subprocess.run(
        ["bash", "-lc", cmd],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "wgrib2 inventory failed.\n"
            f"Command: {cmd}\n"
            f"stdout:\n{proc.stdout}\n"
            f"stderr:\n{proc.stderr}"
        )
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def parse_ref_time(token: str):
    # token like d=2011123100
    if not token.startswith("d="):
        return None
    try:
        return datetime.strptime(token[2:], "%Y%m%d%H")
    except ValueError:
        return None


def step_to_hours(step: str):
    s = step.lower()
    if "anl" in s:
        return 0.0, 0.0

    units = {
        "hour": 1.0,
        "hours": 1.0,
        "hr": 1.0,
        "day": 24.0,
        "days": 24.0,
        "minute": 1.0 / 60.0,
        "minutes": 1.0 / 60.0,
        "min": 1.0 / 60.0,
    }

    m = re.search(r"(\d+)\s*-\s*(\d+)\s*([a-z]+)", s)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        fac = units.get(m.group(3))
        if fac is not None:
            return a * fac, b * fac

    m = re.search(r"(\d+)\s*([a-z]+)", s)
    if m:
        a = float(m.group(1))
        fac = units.get(m.group(2))
        if fac is not None:
            return a * fac, a * fac

    return None


def parse_inventory(lines: list[str]):
    by_var = defaultdict(lambda: {
        "long_name": "",
        "levels": set(),
        "steps": set(),
        "ref_times": [],
        "valid_start": None,
        "valid_end": None,
    })

    for line in lines:
        # Typical -v format:
        # rec:byte:d=yyyymmddhh:VAR LONG_NAME:level:ftime:...
        parts = line.split(":")
        if len(parts) < 6:
            continue

        var_and_name = parts[3].strip()
        if not var_and_name:
            continue

        pieces = var_and_name.split(None, 1)
        short = pieces[0]
        long_name = pieces[1].strip() if len(pieces) > 1 else short

        ref_time = parse_ref_time(parts[2].strip())
        level = parts[4].strip()
        step = parts[5].strip()

        info = by_var[short]
        if not info["long_name"]:
            info["long_name"] = long_name
        info["levels"].add(level)
        info["steps"].add(step)
        if ref_time is not None:
            info["ref_times"].append(ref_time)
            step_window = step_to_hours(step)
            if step_window is not None:
                valid0 = ref_time + timedelta(hours=step_window[0])
                valid1 = ref_time + timedelta(hours=step_window[1])
                if info["valid_start"] is None or valid0 < info["valid_start"]:
                    info["valid_start"] = valid0
                if info["valid_end"] is None or valid1 > info["valid_end"]:
                    info["valid_end"] = valid1

    return by_var


def compact_list(values, max_items=4):
    values = sorted(values)
    if len(values) <= max_items:
        return ", ".join(values)
    head = ", ".join(values[:max_items])
    return f"{head}, ... (+{len(values) - max_items})"


def main():
    print(f"--- INSPECTING: {os.path.basename(FILE_PATH)} ---")
    if not os.path.exists(FILE_PATH):
        raise FileNotFoundError(f"GRIB file not found: {FILE_PATH}")

    print(f"Using module: {WGRIB2_MODULE}")
    lines = run_wgrib2_inventory(FILE_PATH, WGRIB2_MODULE)
    if not lines:
        print("No inventory lines returned by wgrib2.")
        return

    by_var = parse_inventory(lines)
    print(f"Total GRIB messages: {len(lines)}")
    print(f"Unique variables: {len(by_var)}")
    print("-" * 120)
    print("VAR    | LONG NAME                       | LEVELS                                | TIME COVERAGE")
    print("-" * 120)

    for short in sorted(by_var):
        info = by_var[short]
        levels_txt = compact_list(info["levels"])

        if info["valid_start"] is not None and info["valid_end"] is not None:
            coverage = (
                f"{info['valid_start'].strftime('%Y-%m-%d %H:%M')} -> "
                f"{info['valid_end'].strftime('%Y-%m-%d %H:%M')}"
            )
        elif info["ref_times"]:
            r0 = min(info["ref_times"]).strftime("%Y-%m-%d %H:%M")
            r1 = max(info["ref_times"]).strftime("%Y-%m-%d %H:%M")
            coverage = f"ref {r0} -> {r1}"
        else:
            coverage = compact_list(info["steps"])

        print(f"{short:<6} | {info['long_name'][:30]:<30} | {levels_txt[:37]:<37} | {coverage}")


if __name__ == "__main__":
    main()
