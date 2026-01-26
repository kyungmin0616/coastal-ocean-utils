import json
import os
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
from netCDF4 import Dataset

from pylib import datenum

MPI = None
COMM = None
RANK = 0
SIZE = 1

USE_MPI = "--mpi" in sys.argv or os.environ.get("ENABLE_MPI", "0") == "1"
if "--mpi" in sys.argv:
    sys.argv.remove("--mpi")

if USE_MPI:
    try:
        from mpi4py import MPI as _MPI

        MPI = _MPI
        COMM = MPI.COMM_WORLD
        RANK = COMM.Get_rank()
        SIZE = COMM.Get_size()
    except (ImportError, Exception) as exc:
        print(f"[WARN] MPI requested but initialization failed: {exc}. Falling back to serial mode.")
        MPI = None
        COMM = None
        RANK = 0
        SIZE = 1


CONFIG = {
    "data_dir": "/storage/coda1/p-ed70/0/kpark350/dataset/AVISO/DUACS_global",
    "start_date": (1993, 4, 1),
    "end_date": (2023, 1, 1),
    "output_base": Path("./npz/DUACS-global-adt-19930401-20221231"),
    "chunk_size": 30,  # number of files per chunk; tweak based on memory budget
    "coordinates": {
        "canonical": "180",  # "180" for [-180,180], "360" for [0,360)
    },
    "variables": {
        "adt": "adt",
        "ugos": "ugos",
        "vgos": "vgos",
    },
}


class RunningStats:
    """Streaming min/max/mean calculation over time for 2D grids."""

    def __init__(self, shape: tuple[int, int]) -> None:
        self.count = np.zeros(shape, dtype=np.int64)
        self.total = np.zeros(shape, dtype=np.float64)
        self.min_val = np.full(shape, np.inf, dtype=np.float64)
        self.max_val = np.full(shape, -np.inf, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        data = values.astype(np.float64, copy=False)
        valid_mask = ~np.isnan(data)
        if not np.any(valid_mask):
            return

        self.count[valid_mask] += 1
        self.total[valid_mask] += data[valid_mask]
        self.min_val[valid_mask] = np.minimum(self.min_val[valid_mask], data[valid_mask])
        self.max_val[valid_mask] = np.maximum(self.max_val[valid_mask], data[valid_mask])

    def finalize(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        mean = np.where(self.count > 0, self.total / self.count, np.nan)
        min_val = np.where(self.count > 0, self.min_val, np.nan)
        max_val = np.where(self.count > 0, self.max_val, np.nan)
        return min_val.astype(np.float32), max_val.astype(np.float32), mean.astype(np.float32)


def pack_stats(stats_dict: Dict[str, RunningStats]) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    packed = {}
    for field, rs in stats_dict.items():
        count = rs.count.astype(np.int64)
        total = rs.total.astype(np.float64)
        min_field = np.where(rs.count > 0, rs.min_val, np.inf)
        max_field = np.where(rs.count > 0, rs.max_val, -np.inf)
        packed[field] = (count, total, min_field, max_field)
    return packed


def gather_and_combine_stats(stats_dict: Dict[str, RunningStats]) -> Optional[Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    packed = pack_stats(stats_dict)
    if MPI:
        gathered = COMM.gather(packed, root=0)
        if RANK != 0:
            return None
    else:
        gathered = [packed]

    combined: Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for field in stats_dict:
        global_count = None
        global_total = None
        global_min = None
        global_max = None
        for entry in gathered:
            if field not in entry:
                continue
            count, total, min_field, max_field = entry[field]
            if global_count is None:
                global_count = count.copy()
                global_total = total.copy()
                global_min = min_field.copy()
                global_max = max_field.copy()
            else:
                global_count += count
                global_total += total
                global_min = np.minimum(global_min, min_field)
                global_max = np.maximum(global_max, max_field)
        if global_count is None:
            continue
        mean = np.where(global_count > 0, global_total / global_count, np.nan)
        min_grid = np.where(global_count > 0, global_min, np.nan)
        max_grid = np.where(global_count > 0, global_max, np.nan)
        combined[field] = (
            min_grid.astype(np.float32),
            max_grid.astype(np.float32),
            mean.astype(np.float32),
        )
    return combined


def normalize_longitudes(lon: Union[np.ndarray, float], mode: Optional[str]) -> Union[np.ndarray, float]:
    if mode is None:
        return lon
    mode = str(mode).strip().lower()
    arr = np.asarray(lon, dtype=float)
    if mode in {"180", "-180", "[-180,180]", "[-180, 180]"}:
        arr = (arr + 180.0) % 360.0 - 180.0
    elif mode in {"360", "[0,360]", "[0,360)", "0-360"}:
        arr = np.mod(arr, 360.0)
    else:
        return lon
    if np.isscalar(lon):
        return float(arr)
    return arr


def parse_time_from_filename(filename: str) -> float:
    parts = filename.replace(".", "_").split("_")
    time_token = parts[5]
    return datenum(int(time_token[:4]), int(time_token[4:6]), int(time_token[6:]))


def get_nc_files(data_dir: Path, start: float, end: float) -> tuple[list[Path], np.ndarray]:
    nc_files = [p for p in data_dir.iterdir() if p.suffix == ".nc"]
    times = np.array([parse_time_from_filename(p.name) for p in nc_files])
    mask = (times >= start) & (times <= end)
    filtered_files = [p for p, keep in zip(nc_files, mask) if keep]
    filtered_times = times[mask]
    sort_idx = np.argsort(filtered_times)
    return [filtered_files[i] for i in sort_idx], filtered_times[sort_idx]


def read_variable(ds: Dataset, name: str) -> np.ndarray:
    var = ds.variables[name]
    raw = var[:].astype(np.float32)
    scale = getattr(var, "scale_factor", 1.0)
    offset = getattr(var, "add_offset", 0.0)
    fill_value = getattr(var, "_FillValue", None)

    if raw.ndim == 3:
        raw = raw[0]

    if fill_value is not None:
        mask = raw == fill_value
    else:
        mask = None

    data = raw * scale + offset
    if mask is not None:
        data = np.where(mask, np.nan, data)
    return data.astype(np.float32)


def main() -> None:
    config = CONFIG
    data_dir = Path(config["data_dir"]).expanduser()
    start = datenum(*config["start_date"])
    end = datenum(*config["end_date"])

    files, times = get_nc_files(data_dir, start, end)
    if not files:
        raise SystemExit("No NetCDF files found within the requested time range.")

    total_files = len(files)
    chunk_size = int(config.get("chunk_size") or total_files)
    canonical_mode = config["coordinates"].get("canonical", "180")

    longitude = None
    latitude = None
    grid_shape: Optional[tuple[int, int]] = None
    global_stats: Optional[dict[str, RunningStats]] = None
    chunk_manifest = []

    output_base = config["output_base"]
    output_base.parent.mkdir(parents=True, exist_ok=True)

    chunk_count = int(np.ceil(total_files / chunk_size))

    for chunk_id, chunk_start in enumerate(range(0, total_files, chunk_size)):
        chunk_end = min(chunk_start + chunk_size, total_files)
        chunk_indices = np.arange(chunk_start, chunk_end)
        chunk_times = times[chunk_indices]

        if MPI:
            mask = (np.arange(chunk_indices.size) % SIZE) == RANK
        else:
            mask = np.ones(chunk_indices.size, dtype=bool)

        local_indices = chunk_indices[mask]
        local_files = [files[i] for i in local_indices]
        local_times = times[local_indices]

        chunk_adt_frames: list[np.ndarray] = []
        chunk_ugos_frames: list[np.ndarray] = []
        chunk_vgos_frames: list[np.ndarray] = []
        chunk_stats: Optional[dict[str, RunningStats]] = None

        for local_pos, idx in enumerate(local_indices, 1):
            path = files[idx]
            msg_prefix = f"[Rank {RANK}] " if MPI else ""
            print(f"{msg_prefix}Chunk {chunk_id + 1}/{chunk_count} - file {local_pos}/{len(local_indices)}: {path.name}")
            with Dataset(path) as ds:
                if longitude is None:
                    longitude = normalize_longitudes(np.array(ds.variables["longitude"][:]), canonical_mode)
                    latitude = np.array(ds.variables["latitude"][:])
                    grid_shape = (latitude.size, longitude.size)

                if chunk_stats is None and grid_shape is not None:
                    chunk_stats = {var: RunningStats(grid_shape) for var in config["variables"]}
                if global_stats is None and grid_shape is not None:
                    global_stats = {var: RunningStats(grid_shape) for var in config["variables"]}

                adt = read_variable(ds, config["variables"]["adt"])
                ugos = read_variable(ds, config["variables"]["ugos"])
                vgos = read_variable(ds, config["variables"]["vgos"])

            chunk_adt_frames.append(adt)
            chunk_ugos_frames.append(ugos)
            chunk_vgos_frames.append(vgos)

            if chunk_stats is not None:
                chunk_stats["adt"].update(adt)
                chunk_stats["ugos"].update(ugos)
                chunk_stats["vgos"].update(vgos)
            if global_stats is not None:
                global_stats["adt"].update(adt)
                global_stats["ugos"].update(ugos)
                global_stats["vgos"].update(vgos)

        if MPI:
            lon_candidates = COMM.allgather(longitude)
            lat_candidates = COMM.allgather(latitude)
            longitude = next((lon for lon in lon_candidates if lon is not None), longitude)
            latitude = next((lat for lat in lat_candidates if lat is not None), latitude)

        if grid_shape is None and longitude is not None and latitude is not None:
            grid_shape = (latitude.size, longitude.size)
        if chunk_stats is None and grid_shape is not None:
            chunk_stats = {var: RunningStats(grid_shape) for var in config["variables"]}
        if global_stats is None and grid_shape is not None:
            global_stats = {var: RunningStats(grid_shape) for var in config["variables"]}

        if longitude is None or latitude is None or grid_shape is None:
            raise SystemExit("Failed to determine longitude/latitude grid from input files.")

        if chunk_adt_frames:
            local_adt = np.stack(chunk_adt_frames).astype(np.float32)
            local_ugos = np.stack(chunk_ugos_frames).astype(np.float32)
            local_vgos = np.stack(chunk_vgos_frames).astype(np.float32)
        else:
            local_adt = np.empty((0, *grid_shape), dtype=np.float32)
            local_ugos = np.empty((0, *grid_shape), dtype=np.float32)
            local_vgos = np.empty((0, *grid_shape), dtype=np.float32)

        payload = (local_times, local_adt, local_ugos, local_vgos)
        if MPI:
            gathered = COMM.gather(payload, root=0)
        else:
            gathered = [payload]

        chunk_combined = gather_and_combine_stats(chunk_stats or {})

        if not MPI or RANK == 0:
            combined_entries = []
            for t_arr, adt_arr, ugos_arr, vgos_arr in gathered:
                if t_arr.size == 0:
                    continue
                for idx in range(t_arr.size):
                    combined_entries.append(
                        (t_arr[idx], adt_arr[idx], ugos_arr[idx], vgos_arr[idx])
                    )

            if combined_entries:
                combined_entries.sort(key=lambda item: item[0])
                all_times_chunk = np.array([item[0] for item in combined_entries], dtype=np.float64)
                adt_chunk = np.stack([item[1] for item in combined_entries]).astype(np.float32)
                ugos_chunk = np.stack([item[2] for item in combined_entries]).astype(np.float32)
                vgos_chunk = np.stack([item[3] for item in combined_entries]).astype(np.float32)
            else:
                all_times_chunk = np.array([], dtype=np.float64)
                adt_chunk = np.empty((0, *grid_shape), dtype=np.float32)
                ugos_chunk = np.empty((0, *grid_shape), dtype=np.float32)
                vgos_chunk = np.empty((0, *grid_shape), dtype=np.float32)

            chunk_base = output_base.parent / f"{output_base.name}_chunk{chunk_id:04d}"
            np.savez_compressed(
                chunk_base,
                time=all_times_chunk,
                lon=longitude,
                lat=latitude,
                adt=adt_chunk,
                ugos=ugos_chunk,
                vgos=vgos_chunk,
            )

            if chunk_combined:
                adt_min, adt_max, adt_mean = chunk_combined["adt"]
                ugos_min, ugos_max, ugos_mean = chunk_combined["ugos"]
                vgos_min, vgos_max, vgos_mean = chunk_combined["vgos"]
            else:
                adt_min = adt_max = adt_mean = np.full(grid_shape, np.nan, dtype=np.float32)
                ugos_min = ugos_max = ugos_mean = np.full(grid_shape, np.nan, dtype=np.float32)
                vgos_min = vgos_max = vgos_mean = np.full(grid_shape, np.nan, dtype=np.float32)

            stats_path = output_base.parent / f"{output_base.name}_chunk{chunk_id:04d}_stats.npz"
            np.savez_compressed(
                stats_path,
                time_start=float(all_times_chunk.min()) if all_times_chunk.size else np.nan,
                time_end=float(all_times_chunk.max()) if all_times_chunk.size else np.nan,
                adt_min=adt_min,
                adt_max=adt_max,
                adt_mean=adt_mean,
                ugos_min=ugos_min,
                ugos_max=ugos_max,
                ugos_mean=ugos_mean,
                vgos_min=vgos_min,
                vgos_max=vgos_max,
                vgos_mean=vgos_mean,
                lon=longitude,
                lat=latitude,
            )

            chunk_manifest.append(
                {
                    "chunk": int(chunk_id),
                    "file": f"{chunk_base.name}.npz",
                    "stats": f"{stats_path.name}",
                    "count": int(all_times_chunk.size),
                    "time_start": float(all_times_chunk.min()) if all_times_chunk.size else np.nan,
                    "time_end": float(all_times_chunk.max()) if all_times_chunk.size else np.nan,
                }
            )

            print(f"Saved chunk data to {chunk_base}.npz")
            print(f"Saved chunk statistics to {stats_path}")

    if global_stats is None:
        raise SystemExit("No data processed; check input range.")

    global_combined = gather_and_combine_stats(global_stats)

    if not MPI or RANK == 0:
        adt_min, adt_max, adt_mean = global_combined["adt"]
        ugos_min, ugos_max, ugos_mean = global_combined["ugos"]
        vgos_min, vgos_max, vgos_mean = global_combined["vgos"]

        global_stats_path = output_base.parent / f"{output_base.name}_global_stats.npz"
        np.savez_compressed(
            global_stats_path,
            adt_min=adt_min,
            adt_max=adt_max,
            adt_mean=adt_mean,
            ugos_min=ugos_min,
            ugos_max=ugos_max,
            ugos_mean=ugos_mean,
            vgos_min=vgos_min,
            vgos_max=vgos_max,
            vgos_mean=vgos_mean,
            lon=longitude,
            lat=latitude,
        )

        manifest_path = output_base.parent / f"{output_base.name}_chunks_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as fp:
            json.dump({"chunks": chunk_manifest}, fp, indent=2)

        print(f"Saved global statistics to {global_stats_path}")
        print(f"Saved chunk manifest to {manifest_path}")


if __name__ == "__main__":
    main()
