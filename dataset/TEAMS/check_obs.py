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
    # Optional multi-layer station config. If non-empty, this overrides STATION_BP.
    # Each layer can set its own marker and legend label.
    # Example:
    # "STATION_LAYERS": [
    #     {
    #         "PATH": "/path/to/station_a.in",
    #         "LABEL": "Case A",
    #         "MARKER": "r*",
    #         "MARKER_SIZE": 7,
    #         "SHOW_TEXT_LABELS": True,
    #     },
    #     {
    #         "PATH": "/path/to/station_b.in",
    #         "LABEL": "Case B",
    #         "MARKER": "bo",
    #         "MARKER_SIZE": 5,
    #         "SHOW_TEXT_LABELS": False,
    #     },
    # ],
    "STATION_LAYERS": [         {
             "PATH": "/Users/kpark/Documents/Projects/Active/AIMEC_TohokuCoast/01_Data/TEAMS/station_sendai_d1.in",
             "LABEL": "SD1",
             "MARKER": "r*",
             "MARKER_SIZE": 7,
             "SHOW_TEXT_LABELS": False,
         },
         {
             "PATH": "/Users/kpark/Documents/Projects/Active/AIMEC_TohokuCoast/01_Data/TEAMS/station_sendai_d2.in",
             "LABEL": "SD2",
             "MARKER": "b*",
             "MARKER_SIZE": 7,
             "SHOW_TEXT_LABELS": False,
         },
         {
             "PATH": "/Users/kpark/Documents/DEM/M7000/M7005/station_offset_TP.in",
             "LABEL": "SD2",
             "MARKER": "y*",
             "MARKER_SIZE": 7,
             "SHOW_TEXT_LABELS": False,
         },
     ],
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


def get_station_layers():
    """Return normalized station layer configs, preserving legacy single-file config."""
    layers = CONFIG.get("STATION_LAYERS") or []
    normalized = []

    if layers:
        for i, layer in enumerate(layers):
            if not isinstance(layer, dict):
                raise ValueError(f"STATION_LAYERS[{i}] must be a dict")
            path = layer.get("PATH")
            if not path:
                raise ValueError(f"STATION_LAYERS[{i}] missing PATH")
            normalized.append(
                {
                    "PATH": path,
                    "LABEL": layer.get("LABEL", os.path.basename(path)),
                    "MARKER": layer.get("MARKER", CONFIG["MARKER"]),
                    "MARKER_SIZE": layer.get("MARKER_SIZE", CONFIG["MARKER_SIZE"]),
                    "SHOW_TEXT_LABELS": layer.get("SHOW_TEXT_LABELS", True),
                }
            )
        return normalized

    return [
        {
            "PATH": CONFIG["STATION_BP"],
            "LABEL": "Stations",
            "MARKER": CONFIG["MARKER"],
            "MARKER_SIZE": CONFIG["MARKER_SIZE"],
            "SHOW_TEXT_LABELS": True,
        }
    ]


def main():
    if plt is None:
        raise SystemExit("matplotlib is required for plotting")

    if not os.path.exists(CONFIG["GRID_PATH"]):
        raise SystemExit(f"Missing grid: {CONFIG['GRID_PATH']}")
    station_layers = get_station_layers()
    for layer in station_layers:
        if not os.path.exists(layer["PATH"]):
            raise SystemExit(f"Missing station file: {layer['PATH']}")

    gd = read(CONFIG["GRID_PATH"])

    fig, ax = plt.subplots(figsize=CONFIG["FIGSIZE"])
    gd.plot_bnd()
    dx, dy = CONFIG["LABEL_OFFSET"]
    for layer in station_layers:
        xs, ys, names = read_station_in(layer["PATH"])
        ax.plot(xs, ys, layer["MARKER"], ms=layer["MARKER_SIZE"], label=layer["LABEL"])
        if layer["SHOW_TEXT_LABELS"]:
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
