"""Compare Western Boundary Current (Kuroshio) paths from SCHISM and AVISO data.

The script extracts contour-derived paths from a SCHISM mean elevation field and an
AVISO mean dynamic topography (or similar) product, optionally overlays mean
current speeds, and visualises the two paths within a configurable plotting
extent. Both a configuration section and CLI flags are provided for rapid
experimentation without modifying code.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
import os

os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt
import matplotlib.cm as mcm
import numpy as np
from cartopy.mpl.gridliner import LatitudeFormatter, LongitudeFormatter
from matplotlib.tri import Triangulation

from pylib import loadz, read, read_schism_hgrid


# -----------------------------------------------------------------------------
# Configuration data structures
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class MaskBox:
    lon_bounds: Tuple[float, float]
    lat_bounds: Tuple[float, float]


@dataclass(frozen=True)
class PlotConfig:
    schism_hgrid: Path
    schism_mean_elev: Path
    schism_variable: str
    aviso_stats: Path
    aviso_variable: str
    canonical_lon_mode: str
    aviso_lon_bounds: Tuple[float, float]
    aviso_lat_bounds: Tuple[float, float]
    aviso_mask_boxes: Sequence[MaskBox]
    aviso_contour_level: float
    schism_contour_level: float
    depth_mask_threshold: float
    include_currents: bool
    mean_velocity_path: Optional[Path]
    extent: Tuple[float, float, float, float]
    speed_clim: Tuple[float, float]
    figure_size: Tuple[float, float]
    output: Optional[Path]
    show_plot: bool
    draw_bathy: bool
    bathy_levels: Tuple[float, ...]
    show_maps: bool
    map_clim: Optional[Tuple[float, float]]
    map_cmap: str


DEFAULT_CONFIG = PlotConfig(
    schism_hgrid=Path("../run/RUN01c/hgrid.gr3"),
    schism_mean_elev=Path("npz/RUN01c_elev_mean.npz"),
    schism_variable="elev_mean",
    aviso_stats=Path("npz/DUACS-global-adt-20220201-20220401_global_stats.npz"),
    aviso_variable="adt_mean",
    canonical_lon_mode="180",
    aviso_lon_bounds=(80, 170),
    aviso_lat_bounds=(00.0, 70.0),
    aviso_mask_boxes=(),
    aviso_contour_level=0.25,
    schism_contour_level=-0.1,
    depth_mask_threshold=20.0,
    include_currents=False,
    mean_velocity_path=None,
    extent=(80, 170, 0.0, 70.0),
    speed_clim=(0.0, 2.0),
    figure_size=(7.0, 6.0),
    output=Path("./test.png"),
    show_plot=False,
    draw_bathy=False,
    bathy_levels=(2000.0,),
    show_maps=True,
    map_clim=(0,2),
    map_cmap="jet",
)


# -----------------------------------------------------------------------------
# Utility helpers
# -----------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--schism-hgrid", type=Path, help="Path to SCHISM hgrid file")
    parser.add_argument("--schism-elev", type=Path, help="Path to SCHISM mean elevation NPZ")
    parser.add_argument("--schism-var", default=None, help="Variable name inside SCHISM NPZ (default: mean)")
    parser.add_argument("--aviso", type=Path, help="Path to AVISO statistics NPZ")
    parser.add_argument("--aviso-var", default=None, help="Variable name in AVISO NPZ (e.g., adt_mean)")
    parser.add_argument(
        "--lon-mode",
        choices=["180", "360"],
        help="Canonical longitude mode (default: 180 for [-180, 180])",
    )
    parser.add_argument("--obs-level", type=float, help="Contour level for AVISO path")
    parser.add_argument("--model-level", type=float, help="Contour level for SCHISM path")
    parser.add_argument(
        "--extent",
        type=float,
        nargs=4,
        metavar=("LON_MIN", "LON_MAX", "LAT_MIN", "LAT_MAX"),
        help="Plot extent and AVISO subset (lon_min lon_max lat_min lat_max)",
    )
    parser.add_argument("--currents", type=Path, help="Path to mean velocity NPZ for overlay")
    parser.add_argument("--include-currents", dest="include_currents", action="store_true")
    parser.add_argument("--no-currents", dest="include_currents", action="store_false")
    parser.add_argument(
        "--draw-bathy",
        dest="draw_bathy",
        action="store_true",
        help="Draw bathymetry contour(s)",
    )
    parser.add_argument(
        "--no-bathy",
        dest="draw_bathy",
        action="store_false",
        help="Disable bathymetry contour(s)",
    )
    parser.add_argument(
        "--bathy-levels",
        type=float,
        nargs='+',
        help="Bathymetry levels to plot (meters)",
    )
    parser.add_argument(
        "--show-maps",
        dest="show_maps",
        action="store_true",
        help="Also render AVISO/SCHISM mean elevation maps",
    )
    parser.add_argument(
        "--no-maps",
        dest="show_maps",
        action="store_false",
        help="Skip AVISO/SCHISM mean elevation maps",
    )
    parser.set_defaults(include_currents=None, draw_bathy=None, show_maps=None)
    parser.add_argument(
        "--map-clim",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        help="Colorbar range for mean-elevation maps",
    )
    parser.add_argument(
        "--map-cmap",
        help="Colormap name for mean-elevation maps (e.g., viridis, cmo.balance)",
    )
    parser.add_argument("--output", type=Path, help="Figure output path")
    parser.add_argument("--no-show", action="store_true", help="Skip displaying the figure")
    return parser.parse_args()


def apply_cli_overrides(cfg: PlotConfig, args: argparse.Namespace) -> PlotConfig:
    updated = cfg
    if args.schism_hgrid:
        updated = replace(updated, schism_hgrid=args.schism_hgrid)
    if args.schism_elev:
        updated = replace(updated, schism_mean_elev=args.schism_elev)
    if args.schism_var:
        updated = replace(updated, schism_variable=args.schism_var)
    if args.aviso:
        updated = replace(updated, aviso_stats=args.aviso)
    if args.aviso_var:
        updated = replace(updated, aviso_variable=args.aviso_var)
    if args.lon_mode:
        updated = replace(updated, canonical_lon_mode=args.lon_mode)
    if args.obs_level is not None:
        updated = replace(updated, aviso_contour_level=args.obs_level)
    if args.model_level is not None:
        updated = replace(updated, schism_contour_level=args.model_level)
    if args.extent:
        lon_bounds = (args.extent[0], args.extent[1])
        lat_bounds = (args.extent[2], args.extent[3])
        updated = replace(
            updated,
            extent=tuple(args.extent),
            aviso_lon_bounds=lon_bounds,
            aviso_lat_bounds=lat_bounds,
        )
    if args.include_currents is not None:
        updated = replace(updated, include_currents=args.include_currents)
    if args.currents:
        updated = replace(updated, mean_velocity_path=args.currents, include_currents=True)
    if args.draw_bathy is not None:
        updated = replace(updated, draw_bathy=args.draw_bathy)
    if args.bathy_levels:
        updated = replace(updated, bathy_levels=tuple(args.bathy_levels))
    if args.show_maps is not None:
        updated = replace(updated, show_maps=args.show_maps)
    if args.map_clim:
        updated = replace(updated, map_clim=(args.map_clim[0], args.map_clim[1]))
    if args.map_cmap:
        updated = replace(updated, map_cmap=args.map_cmap)
    if args.output:
        updated = replace(updated, output=args.output)
    if args.no_show:
        updated = replace(updated, show_plot=False)
    elif args.output:
        updated = replace(updated, show_plot=False)

    # Ensure longitude-dependent configuration is expressed in the canonical mode
    lon_bounds = normalize_bounds(updated.aviso_lon_bounds, updated.canonical_lon_mode)
    mask_boxes = tuple(
        MaskBox(
            lon_bounds=normalize_bounds(box.lon_bounds, updated.canonical_lon_mode),
            lat_bounds=box.lat_bounds,
        )
        for box in updated.aviso_mask_boxes
    )
    extent_lon = normalize_bounds((updated.extent[0], updated.extent[1]), updated.canonical_lon_mode)
    updated = replace(
        updated,
        aviso_lon_bounds=lon_bounds,
        aviso_mask_boxes=mask_boxes,
        extent=(extent_lon[0], extent_lon[1], updated.extent[2], updated.extent[3]),
    )
    return updated


def configure_rc_params() -> None:
    plt.rc("font", size=8)
    plt.rc("axes", titlesize=8, labelsize=8)
    plt.rc("xtick", labelsize=8)
    plt.rc("ytick", labelsize=8)
    plt.rc("legend", fontsize=8)
    plt.rc("figure", titlesize=8)


def normalize_longitudes(lon: np.ndarray, mode: str) -> np.ndarray:
    arr = np.asarray(lon, dtype=float)
    mode = mode.strip().lower()
    if mode in {"180", "-180"}:
        arr = (arr + 180.0) % 360.0 - 180.0
    elif mode in {"360", "0-360"}:
        arr = np.mod(arr, 360.0)
    return arr


def normalize_bounds(bounds: Tuple[float, float], mode: str) -> Tuple[float, float]:
    normalized = normalize_longitudes(np.array(bounds), mode)
    return float(normalized[0]), float(normalized[1])


def longitude_mask(lon: np.ndarray, bounds: Tuple[float, float]) -> np.ndarray:
    lon_min, lon_max = bounds
    if lon_min <= lon_max:
        return (lon >= lon_min) & (lon <= lon_max)
    # dateline crossing
    return (lon >= lon_min) | (lon <= lon_max)


def resolve_cmap(name: str):
    name = name.strip()
    if name.startswith("cmo."):
        attr = name.split(".", 1)[1]
        return getattr(cmocean.cm, attr)
    return mcm.get_cmap(name)


def load_aviso_mean(cfg: PlotConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    obs = loadz(str(cfg.aviso_stats))
    field = getattr(obs, cfg.aviso_variable)
    lon = np.asarray(obs.lon)
    lat = np.asarray(obs.lat)
    lon = normalize_longitudes(lon, cfg.canonical_lon_mode)
    lon_bounds = normalize_bounds(cfg.aviso_lon_bounds, cfg.canonical_lon_mode)
    lon_mask = longitude_mask(lon, lon_bounds)
    lat_mask = (lat >= cfg.aviso_lat_bounds[0]) & (lat <= cfg.aviso_lat_bounds[1])
    sub_field = field[np.ix_(lat_mask, lon_mask)]
    return lon[lon_mask], lat[lat_mask], sub_field


def build_mask(
    field_shape: Tuple[int, int],
    lon: np.ndarray,
    lat: np.ndarray,
    boxes: Sequence[MaskBox],
    lon_mode: str,
) -> np.ndarray:
    mask = np.zeros(field_shape, dtype=bool)
    for box in boxes:
        lon_bounds = normalize_bounds(box.lon_bounds, lon_mode)
        lon_mask = longitude_mask(lon, lon_bounds)
        lat_mask = (lat >= box.lat_bounds[0]) & (lat <= box.lat_bounds[1])
        mask |= lat_mask[:, None] & lon_mask[None, :]
    return mask


def extract_lines_from_collections(collections: Iterable) -> np.ndarray:
    vertices: List[np.ndarray] = []
    for coll in collections:
        for path in coll.get_paths():
            vertices.append(path.vertices.copy())
    if not vertices:
        return np.empty((0, 2))
    merged = np.vstack(vertices)
    order = np.argsort(merged[:, 0])
    return merged[order]


def compute_observation_path(
    lon: np.ndarray,
    lat: np.ndarray,
    field: np.ndarray,
    level: float,
    mask: np.ndarray,
) -> Tuple[np.ndarray, np.ma.MaskedArray]:
    fig, ax = plt.subplots()
    masked = np.ma.masked_where(mask, field)
    cs = ax.contour(lon, lat, masked, levels=[level])
    points = extract_lines_from_collections(cs.collections)
    plt.close(fig)
    return points, masked


def compute_model_path(
    grid,
    mean_field: np.ndarray,
    level: float,
    depth_threshold: float,
) -> np.ndarray:
    triang = Triangulation(grid.x, grid.y)
    mask = np.any((grid.dp <= depth_threshold)[triang.triangles], axis=1)
    triang.set_mask(mask)
    fig, ax = plt.subplots()
    cs = ax.tricontour(triang, mean_field, levels=[level])
    points = extract_lines_from_collections(cs.collections)
    plt.close(fig)
    return points


def prepare_current_overlay(grid, mvel, clim: Tuple[float, float]) -> Dict[str, object]:
    return {
        "fmt": 1,
        "value": mvel.mean,
        "clim": list(clim),
        "cb": False,
        "cmap": cmocean.cm.speed,
        "zorder": 0,
    }


def plot_paths(
    cfg: PlotConfig,
    grid,
    model_path: np.ndarray,
    obs_path: np.ndarray,
    current_field: Optional[Dict[str, object]],
) -> plt.Figure:
    lon_bounds = normalize_bounds((cfg.extent[0], cfg.extent[1]), cfg.canonical_lon_mode)

    fig, ax = plt.subplots(
        figsize=cfg.figure_size,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )

    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "physical", "land", "110m", edgecolor="face", facecolor="grey"
        ),
        zorder=12,
    )

    xticks = np.linspace(lon_bounds[0], lon_bounds[1], num=5)
    yticks = np.linspace(cfg.extent[2], cfg.extent[3], num=6)
    ax.set_xticks(xticks, crs=ccrs.PlateCarree())
    ax.set_yticks(yticks, crs=ccrs.PlateCarree())
    ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".1f", zero_direction_label=True))
    ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".1f"))

    if cfg.draw_bathy and cfg.bathy_levels:
        grid.plot(
            fmt=2,
            levels=list(cfg.bathy_levels),
            colors=["k"] + ["w"] * (len(cfg.bathy_levels) - 1),
            linestyles=["solid"] * len(cfg.bathy_levels),
            cb=False,
            linewidths=1,
            cmap=None,
        )

    if cfg.include_currents and current_field:
        grid.plot(**current_field)
        cbar = plt.colorbar(
            ticks=np.linspace(cfg.speed_clim[0], cfg.speed_clim[1], num=6),
            orientation="vertical",
        )
        cbar.ax.set_ylabel("Speed (m/s)")

    if model_path.size:
        ax.plot(model_path[:, 0], model_path[:, 1], color="b", linewidth=1.0, label="SCHISM")
    if obs_path.size:
        ax.plot(obs_path[:, 0], obs_path[:, 1], color="r", linewidth=1.0, label="AVISO")

    ax.set_xlim(lon_bounds[0], lon_bounds[1])
    ax.set_ylim(cfg.extent[2], cfg.extent[3])
    ax.legend(loc="upper left")
    ax.set_title("Kuroshio Path Comparison")
    return fig


def plot_mean_maps(
    cfg: PlotConfig,
    lon: np.ndarray,
    lat: np.ndarray,
    obs_field: np.ma.MaskedArray,
    grid,
    schism_field: np.ndarray,
    depth_threshold: float,
) -> Optional[plt.Figure]:
    lon_bounds = normalize_bounds((cfg.extent[0], cfg.extent[1]), cfg.canonical_lon_mode)
    lat_bounds = (cfg.extent[2], cfg.extent[3])

    obs_valid = obs_field.compressed()

    grid_lon = grid.x
    grid_lat = np.asarray(grid.y)
    node_mask = (
        longitude_mask(grid_lon, lon_bounds)
        & (grid_lat >= lat_bounds[0])
        & (grid_lat <= lat_bounds[1])
        & (grid.dp > depth_threshold)
    )

    model_vals = np.full_like(schism_field, np.nan, dtype=float)
    model_vals[node_mask] = schism_field[node_mask]
    model_valid = model_vals[~np.isnan(model_vals)]

    combined = np.concatenate([obs_valid, model_valid]) if model_valid.size else obs_valid
    if combined.size == 0:
        return None

    if cfg.map_clim is not None:
        vmin, vmax = cfg.map_clim
    else:
        vmin = float(np.nanmin(combined))
        vmax = float(np.nanmax(combined))
        if np.isclose(vmin, vmax):
            vmax = vmin + 1e-3

    cmap = resolve_cmap(cfg.map_cmap)

    fig, axes = plt.subplots(
        1, 2, figsize=(12, 5), subplot_kw={"projection": ccrs.PlateCarree()}
    )
    lon_mesh, lat_mesh = np.meshgrid(lon, lat)

    for ax in axes:
        ax.add_feature(
            cfeature.NaturalEarthFeature(
                "physical", "land", "50m", edgecolor="face", facecolor="grey"
            ),
            zorder=2,
        )
        ax.set_extent([lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]], crs=ccrs.PlateCarree())
        ax.xaxis.set_major_formatter(LongitudeFormatter(number_format=".1f", zero_direction_label=True))
        ax.yaxis.set_major_formatter(LatitudeFormatter(number_format=".1f"))
        xticks = np.linspace(lon_bounds[0], lon_bounds[1], num=5)
        yticks = np.linspace(lat_bounds[0], lat_bounds[1], num=6)
        ax.set_xticks(xticks, crs=ccrs.PlateCarree())
        ax.set_yticks(yticks, crs=ccrs.PlateCarree())
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

    m0 = axes[0].pcolormesh(
        lon_mesh,
        lat_mesh,
        obs_field,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        shading="auto",
    )
    axes[0].set_title("AVISO Mean")

    triang = Triangulation(grid_lon, grid_lat)
    tri_mask = np.isnan(model_vals)[triang.triangles].any(axis=1)
    triang.set_mask(tri_mask)
    axes[1].tricontourf(
        triang,
        np.ma.masked_invalid(model_vals),
        levels=30,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
    )
    axes[1].set_title("SCHISM Mean")

    cbar = fig.colorbar(m0, ax=axes.ravel().tolist(), orientation="vertical", fraction=0.046, pad=0.04)
    cbar.set_label("Elevation (m)")

    return fig


# -----------------------------------------------------------------------------
# Main workflow
# -----------------------------------------------------------------------------


def main() -> None:
    configure_rc_params()
    args = parse_args()
    cfg = apply_cli_overrides(DEFAULT_CONFIG, args)
    if cfg.output and not isinstance(cfg.output, Path):
        cfg = replace(cfg, output=Path(cfg.output))

    grid = read_schism_hgrid(str(cfg.schism_hgrid))
    grid_x = normalize_longitudes(np.asarray(grid.x), cfg.canonical_lon_mode)
    grid.x = grid_x
    if hasattr(grid, "lon"):
        grid.lon = normalize_longitudes(np.asarray(grid.lon), cfg.canonical_lon_mode)
    schism_npz = loadz(str(cfg.schism_mean_elev))
    schism_mean = getattr(schism_npz, cfg.schism_variable)

    lon, lat, obs_field = load_aviso_mean(cfg)
    mask = build_mask(obs_field.shape, lon, lat, cfg.aviso_mask_boxes, cfg.canonical_lon_mode)
    obs_path, masked_obs = compute_observation_path(lon, lat, obs_field, cfg.aviso_contour_level, mask)

    model_path = compute_model_path(grid, schism_mean, cfg.schism_contour_level, cfg.depth_mask_threshold)
    if model_path.size:
        model_path[:, 0] = normalize_longitudes(model_path[:, 0], cfg.canonical_lon_mode)
    if obs_path.size:
        obs_path[:, 0] = normalize_longitudes(obs_path[:, 0], cfg.canonical_lon_mode)

    current_overlay = None
    if cfg.include_currents and cfg.mean_velocity_path:
        mvel = read(str(cfg.mean_velocity_path))
        current_overlay = prepare_current_overlay(grid, mvel, cfg.speed_clim)

    fig = plot_paths(cfg, grid, model_path, obs_path, current_overlay)

    map_fig = None
    if cfg.show_maps:
        map_fig = plot_mean_maps(cfg, lon, lat, masked_obs, grid, schism_mean, cfg.depth_mask_threshold)
        if map_fig is None:
            print("Mean-map figure skipped: no valid data within requested bounds.")

    if cfg.output:
        cfg.output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(cfg.output, dpi=300, bbox_inches="tight")
        print(f"Saved figure to {cfg.output}")
        if map_fig is not None:
            map_output = cfg.output.with_name(cfg.output.stem + "_maps" + cfg.output.suffix)
            map_output.parent.mkdir(parents=True, exist_ok=True)
            map_fig.savefig(map_output, dpi=300, bbox_inches="tight")
            print(f"Saved mean maps to {map_output}")
    if cfg.show_plot:
        plt.show()
    plt.close(fig)
    if map_fig is not None:
        plt.close(map_fig)


if __name__ == "__main__":
    main()
