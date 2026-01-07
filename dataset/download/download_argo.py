#!/usr/bin/env python3
"""Download Argo profile data from the USGODAE selection service.

The USGODAE interface exposes two endpoints:
  * ``argo_select.pl`` (GET) returns an HTML page with all matching profile IDs
  * ``argo_download.tar.gz`` (POST) packages the selected profiles into a tarball

This script automates both steps. Edit ``DEFAULT_CONFIG`` or override values on
the command line to control the selection bounding box, date window, whether to
limit to delayed-mode profiles, and which data products to download. When run
with ``--mpi`` (and ``mpi4py`` installed) the work is divided across ranks so
large requests complete more quickly.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import shutil
import sys
import tarfile
import time
from pathlib import Path
from typing import Any, Dict, Iterator, List, MutableMapping, Optional, Sequence, Tuple, Union

import requests

try:
    from mpi4py import MPI  # type: ignore[import]

    _MPI_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    MPI = None  # type: ignore[assignment]
    _MPI_AVAILABLE = False

SELECT_URL = "https://nrlgodae1.nrlmry.navy.mil/cgi-bin/argo_select.pl"
DOWNLOAD_URL = "https://nrlgodae1.nrlmry.navy.mil/cgi-bin/argo_download.tar.gz"

# Maps form button names to the text expected by the CGI script
DOWNLOAD_BUTTON_LABELS = {
    "PROF": "Profile Dload",
    "TRAJ": "Trajectory Dload",
    "TECH": "Tech Data Dload",
    "ALLDATA": "All Data Dload",
}

# ==========================
# Editable configuration
# ==========================
DEFAULT_CONFIG: Dict[str, object] = {
    "start_date": "2022-01-01",      # inclusive (YYYY-MM-DD)
    "end_date": "2022-12-31",        # inclusive (YYYY-MM-DD)
    "north_lat": 70.0,                # degrees north
    "south_lat": -40.0,               # degrees south
    "west_lon": 80.0,               # degrees east/west (-180 to 360)
    "east_lon": -80.0,
    "dac": "ALL",                    # e.g., ALL, aoml, coriolis, bodc, meds, ...
    "float_id": "ALL",               # specific float id or ALL
    "gentype": "plt",                # txt, plt, idplt (controls plotting on HTML page)
    "delayed_only": True,             # True -> delayed mode profiles only
    "download_types": ["PROF"],       # list drawn from DOWNLOAD_BUTTON_LABELS keys
    "output_dir": "NP/2022/",  # where to place downloaded tarballs
    "chunk_size": 500,                # number of profiles per tar request
    "max_profiles": None,             # optional cap for debugging
    "list_only": False,               # True -> just print list, no download
    "write_manifest": None,           # optional path to write matching list (.txt)
    "timeout": 120,                   # seconds for each HTTP request
    "verify_tls": False,              # set to False or a CA bundle path acceptable by requests
    "user_agent": "coastal-ocean-utils/argo-downloader",
    "download_retries": 3,            # attempts per tarball chunk
    "download_retry_sleep": 5.0,      # seconds between retry attempts
    "extract_archives": True,         # extract tarballs after download
    "keep_archives": False,           # keep original tar.gz files on disk
    "overwrite_existing": False,      # overwrite files when extracting (else dedupe)
    "use_mpi": False,                # distribute work across MPI ranks
}

_TOTAL_REGEX = re.compile(r"Total Matching Profiles\\s+(\\d+)", re.IGNORECASE)
_SELECT_BLOCK_REGEX = re.compile(r'<select name="float_list"[^>]*>(.*?)</select>', re.IGNORECASE | re.DOTALL)
_OPTION_VALUE_REGEX = re.compile(r'<option value="([^"<>]+)"')


def _load_json_config(path: Optional[str]) -> Dict[str, object]:
    if not path:
        return {}
    with open(os.fspath(path), "r", encoding="utf-8") as f:
        return json.load(f)


def _coerce_bool(value: Optional[Union[str, bool]]) -> Optional[bool]:
    if value is None or isinstance(value, bool):
        return value
    val = str(value).strip().lower()
    if val in {"true", "t", "yes", "y", "1"}:
        return True
    if val in {"false", "f", "no", "n", "0"}:
        return False
    return None


def _bool_from_cfg(value: Optional[Union[str, bool]], default: bool = False) -> bool:
    coerced = _coerce_bool(value)
    if coerced is None:
        if value is None:
            return default
        return bool(value)
    return coerced


def _ensure_download_types(types: Sequence[str]) -> List[str]:
    result: List[str] = []
    for item in types:
        key = item.strip().upper()
        if key not in DOWNLOAD_BUTTON_LABELS:
            raise ValueError(f"Unknown download type '{item}'. Choose from {sorted(DOWNLOAD_BUTTON_LABELS)}")
        if key not in result:
            result.append(key)
    return result


def _parse_date(value: Union[str, dt.date, dt.datetime]) -> dt.date:
    if isinstance(value, dt.datetime):
        return value.date()
    if isinstance(value, dt.date):
        return value
    text = str(value).strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y%m%d"):
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"Cannot parse date '{value}'")


def _chunked(seq: Sequence[str], size: int) -> Iterator[List[str]]:
    if size <= 0:
        yield list(seq)
        return
    for idx in range(0, len(seq), size):
        yield list(seq[idx : idx + size])


def _init_mpi(enabled: bool) -> Tuple[Optional[Any], int, int, bool]:
    if not enabled:
        return None, 0, 1, False
    if not _MPI_AVAILABLE:
        print(
            "Warning: mpi4py not available; continuing in serial mode despite --mpi flag",
            file=sys.stderr,
        )
        return None, 0, 1, False
    comm = MPI.COMM_WORLD
    return comm, comm.Get_rank(), comm.Get_size(), True


def _slice_for_rank(seq: Sequence[str], rank: int, size: int) -> List[str]:
    n = len(seq)
    if size <= 1 or n == 0:
        return list(seq)
    base = n // size
    remainder = n % size
    start = rank * base + min(rank, remainder)
    length = base + (1 if rank < remainder else 0)
    end = start + length
    if start >= n:
        return []
    return list(seq[start:end])


class ArgoDownloader:
    """Encapsulates calls to the USGODAE Argo selection and download services."""

    def __init__(self, config: MutableMapping[str, object]):
        self.cfg = config
        self.session = requests.Session()
        headers = self.session.headers
        headers.setdefault("User-Agent", str(self.cfg.get("user_agent") or "argo-downloader"))
        self.verify = self._coerce_verify_option(self.cfg.get("verify_tls", True))
        self.timeout = float(self.cfg.get("timeout", 60))
        self.rank = int(self.cfg.get("mpi_rank", 0))
        self.world_size = max(1, int(self.cfg.get("mpi_size", 1)))
        if self.verify is False:
            try:
                from urllib3 import disable_warnings
                from urllib3.exceptions import InsecureRequestWarning

                disable_warnings(InsecureRequestWarning)
            except Exception:
                pass

    @staticmethod
    def _coerce_verify_option(value: object) -> Union[bool, str]:
        if isinstance(value, bool):
            return value
        if value is None:
            return True
        text = str(value).strip()
        lowered = text.lower()
        if lowered in {"false", "0", "no", "off"}:
            return False
        if lowered in {"true", "1", "yes", "on"}:
            return True
        return text  # assume path to CA bundle

    def _build_query_params(self) -> Dict[str, str]:
        start = _parse_date(self.cfg["start_date"])
        end = _parse_date(self.cfg["end_date"])
        if start > end:
            raise ValueError("start_date must be on or before end_date")
        params: Dict[str, str] = {
            "startyear": f"{start.year:04d}",
            "startmonth": f"{start.month:02d}",
            "startday": f"{start.day:02d}",
            "endyear": f"{end.year:04d}",
            "endmonth": f"{end.month:02d}",
            "endday": f"{end.day:02d}",
            "Nlat": str(self.cfg["north_lat"]),
            "Slat": str(self.cfg["south_lat"]),
            "Wlon": str(self.cfg["west_lon"]),
            "Elon": str(self.cfg["east_lon"]),
            "dac": str(self.cfg.get("dac", "ALL")),
            "floatid": str(self.cfg.get("float_id", "ALL")),
            "gentype": str(self.cfg.get("gentype", "plt")),
        }
        if _coerce_bool(self.cfg.get("delayed_only")):
            params["delayed"] = "true"
        return params

    def fetch_float_list(self) -> Tuple[List[str], Optional[int]]:
        params = self._build_query_params()
        resp = self.session.get(SELECT_URL, params=params, timeout=self.timeout, verify=self.verify)
        resp.raise_for_status()
        html = resp.text
        options = self._extract_float_list(html)
        match = _TOTAL_REGEX.search(html)
        total = int(match.group(1)) if match else None
        return options, total

    @staticmethod
    def _extract_float_list(html: str) -> List[str]:
        block_match = _SELECT_BLOCK_REGEX.search(html)
        if not block_match:
            return []
        block = block_match.group(1)
        return _OPTION_VALUE_REGEX.findall(block)

    def download_packages(self, profile_ids: Sequence[str], download_types: Sequence[str]) -> None:
        if not profile_ids:
            if self.world_size == 1 or self.rank == 0:
                print("No matching profiles to download.")
            return
        out_dir = Path(self.cfg.get("output_dir", "argo_downloads")).expanduser()
        out_dir.mkdir(parents=True, exist_ok=True)
        chunk_size = int(self.cfg.get("chunk_size") or 0)
        chunk_size = max(chunk_size, 0)
        total = len(profile_ids)
        retries = max(1, int(self.cfg.get("download_retries", 1)))
        retry_sleep = max(0.0, float(self.cfg.get("download_retry_sleep", 0.0)))
        rank_tag = f"R{self.rank:03d}" if self.world_size > 1 else ""
        for dtype in download_types:
            label = DOWNLOAD_BUTTON_LABELS[dtype]
            chunks = list(_chunked(profile_ids, chunk_size or total))
            chunk_count = len(chunks)
            for idx, subset in enumerate(chunks, start=1):
                if self.world_size > 1:
                    if chunk_count > 1:
                        name_stub = f"{dtype.lower()}_{rank_tag}_{idx:04d}.tar.gz"
                    else:
                        name_stub = f"{dtype.lower()}_{rank_tag}.tar.gz"
                else:
                    name_stub = f"{dtype.lower()}_{idx:04d}.tar.gz" if chunk_count > 1 else f"{dtype.lower()}.tar.gz"
                outfile = out_dir / name_stub
                prefix = f"[{rank_tag}] " if rank_tag else ""
                print(f"{prefix}[{dtype}] downloading chunk {idx}/{chunk_count} with {len(subset)} profiles -> {outfile}")
                form = [("float_list", value) for value in subset]
                form.append((dtype, label))
                attempt = 0
                while True:
                    attempt += 1
                    resp = None
                    try:
                        resp = self.session.post(
                            DOWNLOAD_URL,
                            data=form,
                            timeout=self.timeout,
                            verify=self.verify,
                            stream=True,
                        )
                        resp.raise_for_status()
                        ctype = resp.headers.get("content-type", "")
                        if "gzip" not in ctype and "tar" not in ctype:
                            # consume to surface the server's message
                            data = resp.content
                            snippet = data[:500].decode("latin-1", errors="replace")
                            raise RuntimeError(
                                f"Unexpected response content-type '{ctype}'. Server said: {snippet}"
                            )
                        with outfile.open("wb") as fh:
                            for chunk in resp.iter_content(chunk_size=65536):
                                if chunk:
                                    fh.write(chunk)
                        break
                    except (requests.exceptions.RequestException, OSError) as exc:
                        if outfile.exists():
                            try:
                                outfile.unlink()
                            except OSError:
                                pass
                        if attempt >= retries:
                            raise RuntimeError(
                                f"Failed to download chunk {idx}/{chunk_count} after {retries} attempts"
                            ) from exc
                        remaining = retries - attempt
                        prefix_err = f"[{rank_tag}] " if rank_tag else ""
                        print(
                            f"  {prefix_err}chunk {idx}/{chunk_count} failed ({exc}); {remaining} retries left...",
                            file=sys.stderr,
                        )
                        if retry_sleep:
                            print(
                                f"    {prefix_err}sleeping {retry_sleep:.1f}s before retry",
                                file=sys.stderr,
                            )
                            time.sleep(retry_sleep)
                    finally:
                        if resp is not None:
                            resp.close()
                self._postprocess_archive(outfile)

    @staticmethod
    def _dedupe_path(path: Path) -> Path:
        if not path.exists():
            return path
        stem = path.stem
        suffix = path.suffix
        counter = 1
        while True:
            candidate = path.with_name(f"{stem}_{counter}{suffix}")
            if not candidate.exists():
                return candidate
            counter += 1

    def _extract_archive(self, archive_path: Path) -> int:
        overwrite = bool(self.cfg.get("overwrite_existing"))
        out_dir = archive_path.parent
        extracted = 0
        try:
            with tarfile.open(archive_path, "r:*") as tar:
                for member in tar.getmembers():
                    if not member.isfile():
                        continue
                    target_name = Path(member.name).name
                    target_path = out_dir / target_name
                    if overwrite and target_path.exists():
                        target_path.unlink()
                    if not overwrite:
                        target_path = self._dedupe_path(target_path)
                    fileobj = tar.extractfile(member)
                    if fileobj is None:
                        continue
                    with fileobj:
                        with target_path.open("wb") as handle:
                            shutil.copyfileobj(fileobj, handle)
                    extracted += 1
        except tarfile.TarError as exc:
            raise RuntimeError(f"Failed to extract {archive_path}: {exc}") from exc
        return extracted

    def _postprocess_archive(self, archive_path: Path) -> None:
        if not archive_path.exists():
            return
        if not self.cfg.get("extract_archives", True):
            return
        extracted = self._extract_archive(archive_path)
        prefix = f"[R{self.rank:03d}] " if self.world_size > 1 else ""
        print(f"    {prefix}extracted {extracted} file(s) to {archive_path.parent}")
        if not self.cfg.get("keep_archives", False):
            try:
                archive_path.unlink()
            except OSError:
                pass


def _apply_overrides(base: MutableMapping[str, object], overrides: MutableMapping[str, object]) -> None:
    for key, value in overrides.items():
        if value is None:
            continue
        base[key] = value


def _parse_cli(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Argo profile data using the USGODAE selection service",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", help="JSON file with configuration overrides")
    parser.add_argument("--start", help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end", help="Override end date (YYYY-MM-DD)")
    parser.add_argument("--north", type=float, help="North latitude")
    parser.add_argument("--south", type=float, help="South latitude")
    parser.add_argument("--west", type=float, help="West longitude")
    parser.add_argument("--east", type=float, help="East longitude")
    parser.add_argument("--dac", help="Limit to a single DAC (e.g., aoml)")
    parser.add_argument("--float-id", help="Limit to a single float ID")
    parser.add_argument(
        "--download-type",
        action="append",
        dest="download_types",
        help="Download type to request (PROF, TRAJ, TECH, ALLDATA). Repeat to request multiple.",
    )
    parser.add_argument("--chunk-size", type=int, help="Profiles per download request")
    parser.add_argument("--max-profiles", type=int, help="Limit the number of profiles downloaded")
    parser.add_argument("--list-only", action="store_true", help="Only list matching profiles; skip downloads")
    parser.add_argument("--write-manifest", help="Path to write matching profile list")
    parser.add_argument("--download-retries", type=int, help="Retries per tarball chunk (>=1)")
    parser.add_argument("--download-retry-sleep", type=float, help="Seconds to sleep between retries")
    parser.add_argument("--keep-archives", action="store_true", help="Keep downloaded tar.gz archives after extraction")
    parser.add_argument(
        "--no-keep-archives",
        dest="keep_archives",
        action="store_false",
        help="Remove tar.gz archives after extraction (default)",
    )
    parser.add_argument(
        "--mpi",
        dest="use_mpi",
        action="store_true",
        help="Enable MPI parallel downloads (requires mpi4py)",
    )
    parser.add_argument(
        "--no-mpi",
        dest="use_mpi",
        action="store_false",
        help="Disable MPI even if configured",
    )
    parser.add_argument(
        "--no-extract",
        dest="extract_archives",
        action="store_false",
        help="Skip extracting downloaded archives",
    )
    parser.add_argument(
        "--extract",
        dest="extract_archives",
        action="store_true",
        help="Force extraction of downloaded archives",
    )
    parser.add_argument(
        "--overwrite-existing",
        dest="overwrite_existing",
        action="store_true",
        help="Overwrite files when extracting instead of creating numbered copies",
    )
    parser.add_argument(
        "--delayed-only",
        dest="delayed_only",
        action="store_true",
        help="Request delayed mode profiles only",
    )
    parser.add_argument(
        "--include-realtime",
        dest="delayed_only",
        action="store_false",
        help="Include real-time profiles (delayed_only = False)",
    )
    parser.add_argument("--verify", dest="verify_tls", help="TLS verify mode (true/false or CA bundle path)")
    parser.add_argument("--timeout", type=float, help="HTTP timeout in seconds")
    parser.add_argument("--output", dest="output_dir", help="Output directory for tar files")
    parser.set_defaults(
        delayed_only=None,
        extract_archives=None,
        keep_archives=None,
        overwrite_existing=None,
        use_mpi=None,
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_cli(argv)
    config = dict(DEFAULT_CONFIG)
    file_overrides = _load_json_config(args.config)
    _apply_overrides(config, file_overrides)

    cli_overrides: Dict[str, object] = {}
    if args.start:
        cli_overrides["start_date"] = args.start
    if args.end:
        cli_overrides["end_date"] = args.end
    if args.north is not None:
        cli_overrides["north_lat"] = args.north
    if args.south is not None:
        cli_overrides["south_lat"] = args.south
    if args.west is not None:
        cli_overrides["west_lon"] = args.west
    if args.east is not None:
        cli_overrides["east_lon"] = args.east
    if args.dac:
        cli_overrides["dac"] = args.dac
    if args.float_id:
        cli_overrides["float_id"] = args.float_id
    if args.chunk_size is not None:
        cli_overrides["chunk_size"] = max(0, args.chunk_size)
    if args.max_profiles is not None:
        cli_overrides["max_profiles"] = max(0, args.max_profiles)
    if args.download_retries is not None:
        cli_overrides["download_retries"] = max(1, args.download_retries)
    if args.download_retry_sleep is not None:
        cli_overrides["download_retry_sleep"] = max(0.0, args.download_retry_sleep)
    if args.keep_archives is not None:
        cli_overrides["keep_archives"] = args.keep_archives
    if args.extract_archives is not None:
        cli_overrides["extract_archives"] = args.extract_archives
    if args.overwrite_existing is not None:
        cli_overrides["overwrite_existing"] = args.overwrite_existing
    if args.use_mpi is not None:
        cli_overrides["use_mpi"] = args.use_mpi
    if args.list_only:
        cli_overrides["list_only"] = True
    if args.write_manifest:
        cli_overrides["write_manifest"] = args.write_manifest
    if args.delayed_only is not None:
        cli_overrides["delayed_only"] = args.delayed_only
    if args.download_types:
        cli_overrides["download_types"] = args.download_types
    if args.verify_tls is not None:
        cli_overrides["verify_tls"] = args.verify_tls
    if args.timeout is not None:
        cli_overrides["timeout"] = max(1.0, args.timeout)
    if args.output_dir:
        cli_overrides["output_dir"] = args.output_dir

    _apply_overrides(config, cli_overrides)

    try:
        config["download_types"] = _ensure_download_types(config.get("download_types", ["PROF"]))
    except ValueError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    mpi_requested = _bool_from_cfg(config.get("use_mpi"), default=False)
    comm, rank, size, mpi_active = _init_mpi(mpi_requested)
    config["mpi_rank"] = rank
    config["mpi_size"] = size

    max_profiles = config.get("max_profiles")
    downloader = ArgoDownloader(config)

    if mpi_active:
        if rank == 0:
            profiles, reported_total = downloader.fetch_float_list()
            if max_profiles:
                profiles = profiles[: int(max_profiles)]
        else:
            profiles = []
            reported_total = None
        profiles = comm.bcast(profiles, root=0)
        reported_total = comm.bcast(reported_total, root=0)
    else:
        profiles, reported_total = downloader.fetch_float_list()
        if max_profiles:
            profiles = profiles[: int(max_profiles)]

    if rank == 0:
        print(f"Found {len(profiles)} profiles", end="")
        if reported_total is not None and reported_total != len(profiles):
            print(f" (server reported {reported_total} total)")
        else:
            print()

    manifest_path = config.get("write_manifest")
    if manifest_path and (not mpi_active or rank == 0):
        manifest_file = Path(manifest_path).expanduser()
        manifest_file.parent.mkdir(parents=True, exist_ok=True)
        manifest_file.write_text("\n".join(profiles) + "\n", encoding="utf-8")
        print(f"Wrote profile list to {manifest_file}")

    if config.get("list_only"):
        if not mpi_active or rank == 0:
            for idx, item in enumerate(profiles, start=1):
                print(f"{idx:5d} {item}")
        if mpi_active and comm is not None:
            comm.Barrier()
        return 0

    local_profiles = _slice_for_rank(profiles, rank, size) if mpi_active else profiles
    if mpi_active and size > 1:
        print(f"[R{rank:03d}] assigned {len(local_profiles)} profiles")

    downloader.download_packages(local_profiles, config["download_types"])

    if mpi_active and comm is not None:
        comm.Barrier()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
