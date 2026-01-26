"""Merge DUACS AVISO chunk outputs over a specified time window.

This utility stitches the chunked products produced by `pextract_AVISO_slab.py`
into a smaller subset covering the requested period and computes per-grid-point
statistics (min/mean/max) for `adt`, `ugos`, and `vgos`.
"""

from __future__ import annotations

import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

from pylib import datenum


@dataclass
class Config:
    base_path: Path
    start_date: Tuple[int, int, int]
    end_date: Tuple[int, int, int]
    variables: Tuple[str, ...] = ("adt", "ugos", "vgos")
    reuse_manifest: bool = True
    output_name: Optional[str] = None  # default: derived from dates


def load_manifest(base: Path) -> Optional[List[Dict[str, object]]]:
    manifest_path = base.parent / f"{base.name}_chunks_manifest.json"
    if not manifest_path.exists():
        return None
    with open(manifest_path, "r", encoding="utf-8") as fp:
        data = json.load(fp)
    chunks = data.get("chunks", []) if isinstance(data, dict) else []
    chunks.sort(key=lambda item: item.get("chunk", 0))
    return chunks


def iter_chunk_files(base: Path, reuse_manifest: bool) -> Iterable[Path]:
    manifest = load_manifest(base) if reuse_manifest else None
    if manifest:
        for item in manifest:
            fname = item.get("file")
            if fname:
                yield base.parent / str(fname)
    else:
        yield from sorted(base.parent.glob(f"{base.name}_chunk*.npz"))


def subset_merge(config: Config) -> None:
    base = config.base_path
    chunk_paths = list(iter_chunk_files(base, config.reuse_manifest))
    if not chunk_paths:
        raise SystemExit("No chunk files found. Ensure pextract_AVISO_slab.py has been run.")

    start_t = datenum(*config.start_date)
    end_t = datenum(*config.end_date)

    subset_times: List[np.ndarray] = []
    subset_fields: Dict[str, List[np.ndarray]] = {var: [] for var in config.variables}
    lon = lat = None

    for path in chunk_paths:
        with np.load(path) as data:
            times = data["time"]
            if times.size == 0:
                continue
            mask = (times >= start_t) & (times <= end_t)
            if not np.any(mask):
                continue

            if lon is None:
                lon = data["lon"]
                lat = data["lat"]

            subset_times.append(times[mask])
            for var in config.variables:
                frames = data[var][mask].astype(np.float32)
                subset_fields[var].append(frames)

    if not subset_times:
        raise SystemExit("No data intersecting the requested time window.")

    lon = np.asarray(lon)
    lat = np.asarray(lat)
    times_concat = np.concatenate(subset_times)
    order = np.argsort(times_concat)
    times_concat = times_concat[order]

    merged_fields = {}
    for var, parts in subset_fields.items():
        if not parts:
            continue
        merged = np.concatenate(parts, axis=0)
        merged_fields[var] = merged[order].astype(np.float32)

    out_suffix = (
        config.output_name
        or f"{config.start_date[0]:04d}{config.start_date[1]:02d}{config.start_date[2]:02d}-"
        f"{config.end_date[0]:04d}{config.end_date[1]:02d}{config.end_date[2]:02d}"
    )
    merged_path = base.parent / f"{base.name}_{out_suffix}.npz"
    np.savez_compressed(
        merged_path,
        time=times_concat,
        lon=lon,
        lat=lat,
        **merged_fields,
    )

    stats_arrays: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    if lat is None or lon is None:
        raise SystemExit("Longitude/latitude metadata missing in input chunks.")
    grid_shape = (lat.size, lon.size)
    for var in config.variables:
        merged = merged_fields.get(var)
        if merged is None or merged.size == 0:
            fill = np.full(grid_shape, np.nan, dtype=np.float32)
            stats_arrays[var] = (fill, fill.copy(), fill.copy())
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            var_min = np.nanmin(merged, axis=0).astype(np.float32)
            var_max = np.nanmax(merged, axis=0).astype(np.float32)
            var_mean = np.nanmean(merged, axis=0).astype(np.float32)
        stats_arrays[var] = (var_min, var_max, var_mean)

    stats_path = base.parent / f"{base.name}_{out_suffix}_stats.npz"
    np.savez_compressed(
        stats_path,
        time_start=float(times_concat.min()),
        time_end=float(times_concat.max()),
        lon=lon,
        lat=lat,
        **{f"{var}_min": stats_arrays[var][0] for var in stats_arrays},
        **{f"{var}_max": stats_arrays[var][1] for var in stats_arrays},
        **{f"{var}_mean": stats_arrays[var][2] for var in stats_arrays},
    )

    print(f"Saved merged subset to {merged_path}")
    print(f"Saved subset statistics to {stats_path}")


if __name__ == "__main__":
    CONFIG = Config(
        base_path=Path("./npz/DUACS-global-adt-19930401-20221231"),
        start_date=(2022, 2, 1),
        end_date=(2022, 4, 1),
        reuse_manifest=True,
    )

    subset_merge(CONFIG)
