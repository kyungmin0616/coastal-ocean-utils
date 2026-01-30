#!/usr/bin/env python3

import sys
from pathlib import Path
from datetime import datetime

from pylib import datenum

# -------------------------------------------------
# Check input arguments
# -------------------------------------------------
if len(sys.argv) not in (5, 6):
    print("Usage:")
    print("  python filter_nc_by_date.py NC_DIR START_DATE END_DATE OUTPUT_FILE [DATES_FILE]")
    print("Example:")
    print("  python filter_nc_by_date.py ./cmems 2008-05-30 2013-07-20 files.out")
    print("  python filter_nc_by_date.py ./cmems 2008-05-30 2013-07-20 files.out dates.out")
    sys.exit(1)

nc_dir       = Path(sys.argv[1])
start_str    = sys.argv[2]
end_str      = sys.argv[3]
output_file  = sys.argv[4]
dates_file   = sys.argv[5] if len(sys.argv) == 6 else str(Path(output_file).with_name("dates.out"))

# -------------------------------------------------
# Validate directory
# -------------------------------------------------
if not nc_dir.is_dir():
    raise NotADirectoryError(f"{nc_dir} is not a valid directory")

# -------------------------------------------------
# Parse date range
# -------------------------------------------------
start_dt = datetime.strptime(start_str, "%Y-%m-%d")
end_dt   = datetime.strptime(end_str, "%Y-%m-%d")
end_dt   = end_dt.replace(hour=23, minute=59, second=59)

# -------------------------------------------------
# Scan files
# -------------------------------------------------
selected = []
dates = []

for f in sorted(nc_dir.glob("cmems_*.nc")):
    try:
        # Extract datetime from filename
        # cmems_YYYY_MM_DD_HH.nc
        dt_str = f.stem.replace("cmems_", "")
        file_dt = datetime.strptime(dt_str, "%Y_%m_%d_%H")

        if start_dt <= file_dt <= end_dt:
            selected.append(f.name)   # or f.as_posix() if full path needed
            dates.append(datenum(file_dt.year, file_dt.month, file_dt.day, file_dt.hour))

    except ValueError:
        # Skip files with unexpected naming
        continue

# -------------------------------------------------
# Write output
# -------------------------------------------------
with open(output_file, "w") as fout:
    for name in selected:
        fout.write(f"{name}\n")

with open(dates_file, "w") as fout:
    for val in dates:
        fout.write(f"{val:.8f}\n")

print(f"Directory      : {nc_dir}")
print(f"Date range     : {start_dt} → {end_dt}")
print(f"Files selected : {len(selected)}")
print(f"Output written : {output_file}")
print(f"Dates written  : {dates_file}")
