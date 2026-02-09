from PIL import Image
import os
import re
from datetime import datetime

duration = 60  # hourly

img_dir = os.path.expanduser('./2Dmaps/RUN01d/')
sname = os.path.expanduser('./RUN01d.gif')
startT, endT = '2017-1-2 00:00:00', '2017-12-31 00:00:00'  # inclusive
frame_mode = "hourly"  # hourly, daily, monthly, yearly
fast_year_filter = True  # skip files outside year range before parsing
progress_every = 50  # set to 0 to only print first/last
keep_unparsed = False  # keep files with unparsed timestamps at the end


def _parse_user_time(val):
    if val is None:
        return None
    s = str(val).strip()
    if not s:
        return None
    try:
        return datetime.fromisoformat(s)
    except Exception:
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%Y%m%d%H%M%S", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt)
        except Exception:
            continue
    return None


def _push_candidate(candidates, dt, spec, pos):
    if dt is None:
        return
    candidates.append((spec, pos, dt))


def _parse_time_from_name(name):
    stem = os.path.splitext(os.path.basename(name))[0]
    candidates = []

    # Flexible separated date/time with 1-2 digit fields
    pattern = (
        r"(19|20)\d{2}[-_./]?(\d{1,2})[-_./]?(\d{1,2})"
        r"(?:[T _-]?(\d{1,2})(?:[:_-]?(\d{1,2}))?(?:[:_-]?(\d{1,2}))?)?"
    )
    for m in re.finditer(pattern, stem):
        year = int(m.group(0)[0:4])
        month = int(m.group(2))
        day = int(m.group(3))
        hour = int(m.group(4) or 0)
        minute = int(m.group(5) or 0)
        second = int(m.group(6) or 0)
        try:
            dt = datetime(year, month, day, hour, minute, second)
        except Exception:
            dt = None
        spec = 3 + int(m.group(4) is not None) + int(m.group(5) is not None) + int(m.group(6) is not None)
        _push_candidate(candidates, dt, spec, m.start())

    # Contiguous digit formats (take all matches, pick the most specific, last)
    for length, fmt, spec in (
        (14, "%Y%m%d%H%M%S", 6),
        (12, "%Y%m%d%H%M", 5),
        (10, "%Y%m%d%H", 4),
        (8, "%Y%m%d", 3),
    ):
        for m in re.finditer(rf"\d{{{length}}}", stem):
            try:
                dt = datetime.strptime(m.group(0), fmt)
            except Exception:
                dt = None
            _push_candidate(candidates, dt, spec, m.start())

    # Fallback: join all digit parts
    parts = re.findall(r"\d+", stem)
    if parts:
        joined = "".join(parts)
        for length, fmt, spec in (
            (14, "%Y%m%d%H%M%S", 6),
            (12, "%Y%m%d%H%M", 5),
            (10, "%Y%m%d%H", 4),
            (8, "%Y%m%d", 3),
        ):
            if len(joined) >= length:
                try:
                    dt = datetime.strptime(joined[:length], fmt)
                except Exception:
                    dt = None
                _push_candidate(candidates, dt, spec, len(stem))
                break

    if not candidates:
        return None
    # Prefer most specific, then right-most occurrence
    candidates.sort(key=lambda x: (x[0], x[1]), reverse=True)
    return candidates[0][2]


def _bucket_key(dt, mode):
    if mode == "hourly":
        return (dt.year, dt.month, dt.day, dt.hour)
    if mode == "daily":
        return (dt.year, dt.month, dt.day)
    if mode == "monthly":
        return (dt.year, dt.month)
    if mode == "yearly":
        return (dt.year,)
    raise ValueError(f"Unknown frame_mode: {mode}")


def _year_tokens(start_dt, end_dt):
    if start_dt is None or end_dt is None:
        return set()
    return {str(y) for y in range(start_dt.year, end_dt.year + 1)}


frame_mode = frame_mode.strip().lower()
start_dt = _parse_user_time(startT)
end_dt = _parse_user_time(endT)
allowed_years = _year_tokens(start_dt, end_dt)

if start_dt is None or end_dt is None:
    print("WARNING: startT/endT could not be parsed; time filtering may be skipped.")

# Scan only files in time window
count_png = 0
count_skipped_year = 0
count_unparsed = 0
parsed = []
unparsed = []
for entry in os.scandir(img_dir):
    if not entry.is_file() or not entry.name.lower().endswith('.png'):
        continue
    count_png += 1
    if fast_year_filter and allowed_years:
        m = re.search(r"(19|20)\d{2}", entry.name)
        if m and m.group(0) not in allowed_years:
            count_skipped_year += 1
            continue
    dt = _parse_time_from_name(entry.name)
    if dt is None:
        count_unparsed += 1
        if keep_unparsed:
            unparsed.append(entry.path)
        continue
    if start_dt and dt < start_dt:
        continue
    if end_dt and dt > end_dt:
        continue
    parsed.append((dt, entry.path))

print(f"Scanned {count_png} PNG files in {img_dir}")
if count_skipped_year:
    print(f"Skipped {count_skipped_year} PNG files outside year range")
if count_unparsed:
    print(f"Skipped {count_unparsed} PNG files with unparsed timestamps")

# Sort + downsample
parsed.sort(key=lambda x: (x[0], os.path.basename(x[1])))
if frame_mode != "hourly":
    grouped = []
    seen = set()
    for dt, fname in parsed:
        key = _bucket_key(dt, frame_mode)
        if key in seen:
            continue
        seen.add(key)
        grouped.append((dt, fname))
    parsed = grouped

fnames = [f for _, f in parsed]
if keep_unparsed and unparsed:
    fnames.extend(sorted(unparsed))

if not fnames:
    raise RuntimeError("No frames found in the selected time range.")
print(f"Selected {len(fnames)} frames between {startT} and {endT} (mode={frame_mode})")

# Create the frames (SAFE: no open-file leaks)
frames = []
total = len(fnames)
for i, fname in enumerate(fnames, start=1):
    if i == 1 or i == total or (progress_every and i % progress_every == 0):
        print(f"Loading frame {i}/{total}: {os.path.basename(fname)}")
    with Image.open(fname) as im:
        frames.append(im.copy())  # detach from file handle

# Save GIF
print(f"Writing GIF to {sname} ({total} frames)")
frames[0].save(
    sname,
    format='GIF',
    append_images=frames[1:],
    save_all=True,
    duration=duration,
    loop=0
)
