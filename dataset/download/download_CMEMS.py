import os
import copernicusmarine
from pathlib import Path

# --- Configuration Section ---
data_type = "monthly"  # Options: "monthly", "daily", "climatology"

start_year = 1993
end_year   = 2024

datasets = {
    "monthly": {
        "id": "cmems_mod_glo_phy_my_0.083deg_P1M-m",
        "filter": "mercatorglorys12v1_gl12_mean_{year}*.nc",
        "dir": "./CMEMS_monthly"
    },
    "daily": {
        "id": "cmems_mod_glo_phy_my_0.083deg_P1D-m",
        "filter": "mercatorglorys12v1_gl12_mean_{year}*.nc",
        "dir": "./CMEMS_daily"
    },
    "climatology": {
        "id": "cmems_mod_glo_phy_my_0.083deg_P1M-m_clim", # Placeholder ID
        "filter": "*{year}*.nc",
        "dir": "./CMEMS_climatology"
    }
}

cfg = datasets[data_type]
dataset_id = cfg["id"]
base_dir = Path(cfg["dir"])
# -----------------------------

base_dir.mkdir(parents=True, exist_ok=True)

all_files = []

for year in range(start_year, end_year + 1):
    year_dir = base_dir / f"{year}"
    year_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading year {year} into {year_dir}")

    files = copernicusmarine.get(
        dataset_id=dataset_id,
        username="kpark",
        password="KyungminPark0616",
        filter=cfg["filter"].format(year=year),
        output_directory=str(year_dir),
        no_directories=True,
        skip_existing=True,      # resumable
        overwrite=False,         # explicit
        max_concurrent_requests=1,
        disable_progress_bar=True
    )

    all_files.extend(files)

print(f"Total files downloaded: {len(all_files)}")
