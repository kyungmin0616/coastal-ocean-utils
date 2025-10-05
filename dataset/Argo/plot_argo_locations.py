#!/usr/bin/env python3
"""Plot locations of downloaded Argo profiles on a world map.

The script scans a directory for Argo NetCDF profile files (the flattened output
produced by ``download_argo_data.py``) and extracts the profile latitude and
longitude. The positions are then rendered on a global map using Cartopy's
coastlines.

All runtime parameters live in ``DEFAULT_CONFIG`` below (directory, file pattern,
figure styling, etc.) and can be overridden either via ``--config`` (JSON) or
command-line arguments.

Example usage::

    python plot_argo_locations.py --argo-dir argo_downloads --output argo_map.png
    python plot_argo_locations.py --config my_plot_config.json --show

Install dependencies (cartopy, matplotlib, netCDF4) beforehand to use this
script.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, MutableMapping, Optional, Sequence, Tuple

import numpy as np

try:  # Optional heavy imports guarded to provide a clear error message
    import cartopy.crs as ccrs
    import matplotlib.pyplot as plt
except Exception as exc:  # pragma: no cover - import guard for missing deps
    raise SystemExit(
        "Cartopy and Matplotlib are required. Install them (e.g. `pip install cartopy matplotlib`)."
    ) from exc

try:
    from netCDF4 import Dataset
except ImportError as exc:  # pragma: no cover - optional dependency guard
    raise SystemExit(
        "netCDF4 is required to read Argo profiles. Install it with `pip install netCDF4`."
    ) from exc


# ==========================
# Editable configuration
# ==========================

DEFAULT_CONFIG = {
    "argo_dir": "argo_downloads",    # directory containing flattened NetCDF files
    "pattern": "*.nc",              # glob pattern for NetCDF selection
    "output": None,                  # optional output path for figure
    "dpi": 150,                      # figure DPI when saving
    "title": "Argo Profile Locations",
    "show": False,                   # display figure interactively
    "figure_size": [11.0, 6.5],      # inches, width x height
    "dot_size": 5,                   # marker size for scatter plot
    "dot_alpha": 0.6,                # transparency for scatter points
    "dot_color": "tab:red",         # marker color
    "failures_to_show": 5,           # number of failing files to print
}


#: Candidate variable names that may contain latitude / longitude
LAT_NAMES = ("LATITUDE", "latitude", "lat", "profile_latitude")
LON_NAMES = ("LONGITUDE", "longitude", "lon", "profile_longitude")


def _load_json_config(path: Optional[Path]) -> MutableMapping[str, object]:
    if not path:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _apply_overrides(base: MutableMapping[str, object], overrides: MutableMapping[str, object]) -> None:
    for key, value in overrides.items():
        if value is None:
            continue
        base[key] = value


def _as_bool(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "t", "yes", "y", "1"}:
            return True
        if lowered in {"false", "f", "no", "n", "0"}:
            return False
    return bool(value)


def _get_first_variable(ds: Dataset, names: Sequence[str]) -> np.ndarray:
    for name in names:
        if name in ds.variables:
            data = ds.variables[name][:]
            return np.asarray(data, dtype=float).ravel()
    raise KeyError(f"None of {names} found in {ds.filepath()}")


def extract_profile_positions(nc_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Return latitude/longitude arrays for a single NetCDF file."""

    try:
        with Dataset(nc_path, mode="r") as ds:
            lats = _get_first_variable(ds, LAT_NAMES)
            lons = _get_first_variable(ds, LON_NAMES)
    except Exception as exc:
        raise RuntimeError(f"Failed to read coordinates from {nc_path}") from exc

    if lats.size != lons.size:
        raise ValueError(
            f"Latitude/longitude size mismatch in {nc_path}: {lats.size} vs {lons.size}"
        )

    return lats, lons


def gather_positions(files: Sequence[Path]) -> Tuple[np.ndarray, np.ndarray, List[Path]]:
    """Aggregate lat/lon arrays from multiple NetCDF files.

    Returns arrays of latitudes, longitudes, and a list of files that failed to
    parse.
    """

    collected_lats: List[np.ndarray] = []
    collected_lons: List[np.ndarray] = []
    failures: List[Path] = []

    for path in files:
        try:
            lats, lons = extract_profile_positions(path)
        except Exception:
            failures.append(path)
            continue
        if lats.size == 0:
            continue
        collected_lats.append(lats)
        collected_lons.append(lons)

    if not collected_lats:
        return np.empty(0), np.empty(0), failures

    all_lats = np.concatenate(collected_lats)
    all_lons = np.concatenate(collected_lons)
    return all_lats, all_lons, failures


def make_plot(
    lats: np.ndarray,
    lons: np.ndarray,
    output: Path | None,
    title: str,
    dpi: int,
    show: bool,
    figure_size: Tuple[float, float],
    dot_size: float,
    dot_alpha: float,
    dot_color: str,
) -> None:
    """Render the scatter plot on a world map."""

    fig = plt.figure(figsize=figure_size)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_global()
    ax.coastlines(resolution="110m", linewidth=0.8)
    ax.stock_img()

    if lats.size:
        ax.scatter(
            lons,
            lats,
            s=dot_size,
            c=dot_color,
            alpha=dot_alpha,
            transform=ccrs.PlateCarree(),
            linewidths=0,
        )
    else:
        ax.text(
            0.5,
            0.5,
            "No profile positions available",
            transform=ax.transAxes,
            ha="center",
            va="center",
            fontsize=12,
        )

    ax.set_title(title)
    plt.tight_layout()

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=dpi)
        print(f"Saved plot to {output}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def find_netcdf_files(argo_dir: Path, pattern: str) -> List[Path]:
    return sorted(argo_dir.glob(pattern))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot locations of downloaded Argo profiles",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", type=Path, help="JSON file with configuration overrides")
    parser.add_argument("--argo-dir", type=Path, help="Directory containing flattened Argo NetCDF profiles")
    parser.add_argument("--pattern", help="Glob pattern for NetCDF files inside argo-dir")
    parser.add_argument("--output", type=Path, help="File path to save the figure (PNG, PDF, etc.)")
    parser.add_argument("--dpi", type=int, help="Figure DPI when saving to file")
    parser.add_argument("--title", help="Title for the plot")
    parser.add_argument("--show", dest="show", action="store_true", help="Display the plot interactively")
    parser.add_argument("--no-show", dest="show", action="store_false", help="Disable interactive display")
    parser.add_argument("--fig-width", type=float, help="Figure width in inches")
    parser.add_argument("--fig-height", type=float, help="Figure height in inches")
    parser.add_argument("--dot-size", type=float, help="Marker size for profile locations")
    parser.add_argument("--dot-alpha", type=float, help="Marker transparency (0-1)")
    parser.add_argument("--dot-color", help="Marker color")
    parser.add_argument(
        "--failures-to-show",
        type=int,
        help="How many failing files to list in warnings",
    )
    parser.set_defaults(show=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = dict(DEFAULT_CONFIG)
    file_overrides = _load_json_config(args.config)
    _apply_overrides(config, file_overrides)

    cli_overrides: MutableMapping[str, object] = {}
    if args.argo_dir is not None:
        cli_overrides["argo_dir"] = str(args.argo_dir)
    if args.pattern is not None:
        cli_overrides["pattern"] = args.pattern
    if args.output is not None:
        cli_overrides["output"] = str(args.output)
    if args.dpi is not None:
        cli_overrides["dpi"] = max(1, args.dpi)
    if args.title is not None:
        cli_overrides["title"] = args.title
    if args.show is not None:
        cli_overrides["show"] = args.show
    if args.dot_size is not None:
        cli_overrides["dot_size"] = max(0.1, args.dot_size)
    if args.dot_alpha is not None:
        cli_overrides["dot_alpha"] = min(max(0.0, args.dot_alpha), 1.0)
    if args.dot_color is not None:
        cli_overrides["dot_color"] = args.dot_color
    if args.failures_to_show is not None:
        cli_overrides["failures_to_show"] = max(0, args.failures_to_show)
    if args.fig_width is not None or args.fig_height is not None:
        current_width, current_height = config.get("figure_size", [11.0, 6.5])
        width = args.fig_width if args.fig_width is not None else current_width
        height = args.fig_height if args.fig_height is not None else current_height
        cli_overrides["figure_size"] = [max(1e-3, width), max(1e-3, height)]

    _apply_overrides(config, cli_overrides)

    try:
        figure_size_values = tuple(float(x) for x in config.get("figure_size", [11.0, 6.5]))
        if len(figure_size_values) != 2:
            raise ValueError
    except Exception as exc:
        raise SystemExit(f"Invalid figure_size configuration: {config.get('figure_size')}") from exc

    argo_dir = Path(str(config["argo_dir"])).expanduser()
    if not argo_dir.exists() or not argo_dir.is_dir():
        print(f"Error: directory {argo_dir} does not exist or is not a directory", file=sys.stderr)
        return 1

    pattern = str(config.get("pattern", "*.nc"))
    files = find_netcdf_files(argo_dir, pattern)
    if not files:
        print(f"No NetCDF files matching pattern '{pattern}' found in {argo_dir}")

    lats, lons, failures = gather_positions(files)

    failures_to_show = int(config.get("failures_to_show", 5))
    if failures:
        print(f"Warning: failed to parse {len(failures)} file(s)")
        for path in failures[:failures_to_show]:
            print(f"  - {path}")
        if len(failures) > failures_to_show:
            print("  ...")

    output_path = config.get("output")
    output: Optional[Path]
    if output_path:
        output = Path(str(output_path)).expanduser()
    else:
        output = None

    show_flag = _as_bool(config.get("show"))

    figure_size = (float(figure_size_values[0]), float(figure_size_values[1]))

    make_plot(
        lats,
        lons,
        output,
        str(config.get("title", "Argo Profile Locations")),
        int(config.get("dpi", 150)),
        show_flag,
        figure_size,
        float(config.get("dot_size", 5.0)),
        float(config.get("dot_alpha", 0.6)),
        str(config.get("dot_color", "tab:red")),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
