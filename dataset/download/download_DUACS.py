import os
import re
from pathlib import Path
import copernicusmarine

# --- Configuration ---
start_year = 2024
end_year = 2024

dataset_id = "c3s_obs-sl_glo_phy-ssh_my_twosat-l4-duacs-0.25deg_P1D"
output_dir = Path("test")
output_dir.mkdir(parents=True, exist_ok=True)

# --- Credentials (recommended: environment variables) ---
# export CMEMS_USERNAME="..."
# export CMEMS_PASSWORD="..."
username="kpark"
password="KyungminPark0616"

if not username or not password:
    raise RuntimeError("Set CMEMS_USERNAME and CMEMS_PASSWORD environment variables.")

print(f"Downloading DUACS data to: {output_dir}")
print(f"Dataset: {dataset_id}")

# IMPORTANT:
# Use a regex that matches the DATE PART, not the DT2024 version string.
# Example filename: dt_global_twosat_phy_l4_20240115_vDT2024.nc
# Regex below matches: _YYYYMMDD_ for the target year.
for year in range(start_year, end_year + 1):
    print(f"\n=== Processing year {year} ===")

    year_regex = rf".*_({year})(0[1-9]|1[0-2])([0-2][0-9]|3[0-1])_vDT2024.*\.nc$"

    # Optional: sanity check regex
    re.compile(year_regex)

    copernicusmarine.get(
        dataset_id=dataset_id,
        username=username,
        password=password,
        output_directory=str(output_dir),
        no_directories=True,

        # Key fixes:
        regex=year_regex,      # precise year selection
        #skip_existing=True,    # avoids downloading again & avoids _(1)/(2) duplicates  [oai_citation:1‡help.marine.copernicus.eu](https://help.marine.copernicus.eu/en/articles/8286883-copernicus-marine-toolbox-api-get-original-files)

        # Choose ONE of these policies:
        overwrite=False,       # keep existing canonical files
        # overwrite=True,      # if you prefer always re-downloading (no duplicates but will re-fetch)
    )

print("\n--------------Done-------------")
