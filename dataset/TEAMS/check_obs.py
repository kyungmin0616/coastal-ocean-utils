#!/usr/bin/env python3
"""
Plot observation station locations on top of SCHISM grid boundary.
"""
import os

from pylib import read

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


CONFIG = {
    "GRID_PATH": "/Users/kpark/Documents/Projects/Active/AIMEC_TohokuCoast/01_Data/01.gr3",
    "STATION_BP": "/Users/kpark/Documents/Codes/coastal-ocean-utils/schism/post-proc/TEAMS/station_onagawa_d3.in",
    "OUT_PNG": "/Users/kpark/Documents/Projects/Active/AIMEC_TohokuCoast/01_Data/station_map.png",
    "FIGSIZE": (7, 7),
    "MARKER": "r*",
    "MARKER_SIZE": 7,
    "LABEL_FONTSIZE": 8,
    "LABEL_OFFSET": (3, 3),  # offset in points (x, y)
    "X_LIM": (141.4127, 141.6027),  # (xmin, xmax)
    "Y_LIM": (38.3298, 38.4992),  # (ymin, ymax)
    "SAVE": False,
    "SHOW": True,
}


def read_station_in(path):
    with open(path, "r", encoding="utf-8") as f:
        lines = [line.strip() for line in f if line.strip()]
    if len(lines) < 3:
        raise ValueError(f"Invalid station file: {path}")
    try:
        nsta = int(lines[1].split()[0])
    except Exception as exc:
        raise ValueError(f"Invalid station count line: {lines[1]}") from exc
    xs = []
    ys = []
    names = []
    for line in lines[2:2 + nsta]:
        if "#" in line:
            left, name = line.split("#", 1)
            name = name.strip()
        else:
            left = line
            name = ""
        parts = left.split()
        if len(parts) < 4:
            continue
        try:
            lon = float(parts[1])
            lat = float(parts[2])
        except Exception:
            continue
        xs.append(lon)
        ys.append(lat)
        names.append(name or str(parts[0]))
    return xs, ys, names


def main():
    if plt is None:
        raise SystemExit("matplotlib is required for plotting")

    if not os.path.exists(CONFIG["GRID_PATH"]):
        raise SystemExit(f"Missing grid: {CONFIG['GRID_PATH']}")
    if not os.path.exists(CONFIG["STATION_BP"]):
        raise SystemExit(f"Missing station file: {CONFIG['STATION_BP']}")

    gd = read(CONFIG["GRID_PATH"])
    xs, ys, names = read_station_in(CONFIG["STATION_BP"])

    fig, ax = plt.subplots(figsize=CONFIG["FIGSIZE"])
    gd.plot_bnd()
    ax.plot(xs, ys, CONFIG["MARKER"], ms=CONFIG["MARKER_SIZE"], label="Stations")
    dx, dy = CONFIG["LABEL_OFFSET"]
    for i, label in enumerate(names):
        ax.annotate(
            label,
            xy=(xs[i], ys[i]),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=CONFIG["LABEL_FONTSIZE"],
            ha="left",
            va="bottom",
            clip_on=True,
        )
    ax.legend(loc="best")
    if CONFIG["X_LIM"]:
        ax.set_xlim(*CONFIG["X_LIM"])
    if CONFIG["Y_LIM"]:
        ax.set_ylim(*CONFIG["Y_LIM"])
    ax.set_aspect("equal", adjustable="box")

    fig.tight_layout()
    if CONFIG["SAVE"]:
        out_png = CONFIG["OUT_PNG"]
        os.makedirs(os.path.dirname(out_png), exist_ok=True)
        fig.savefig(out_png, dpi=150)
        print(f"Saved station map to {out_png}")
    if CONFIG["SHOW"]:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
