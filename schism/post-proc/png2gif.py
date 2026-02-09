from PIL import Image
import os
import re
import shutil
import subprocess
import tempfile
from datetime import datetime

base_duration = 60  # milliseconds for one nominal time step

img_dir = os.path.expanduser('./RUN02a/')
sname = os.path.expanduser('./RUN02a.gif')
output_format = "mp4"  # gif, mp4
startT, endT = '2017-1-2 00:00:00', '2017-4-1 00:00:00'  # inclusive
frame_mode = "hourly"  # hourly, daily, monthly, yearly
fast_year_filter = True  # skip files outside year range before parsing
progress_every = 50  # set to 0 to only print first/last
keep_unparsed = False  # keep files with unparsed timestamps at the end
duration_mode = "proportional"  # constant, proportional
min_duration = 30  # ms, used only for proportional mode
max_duration = 1200  # ms, used only for proportional mode
mp4_codec = "libx264"
mp4_crf = 20
mp4_preset = "medium"
mp4_pix_fmt = "yuv420p"


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


def _format_gap(seconds):
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def _nominal_step_seconds(mode, positive_gaps):
    if mode == "hourly":
        return 3600
    if mode == "daily":
        return 86400
    if mode == "monthly":
        return 86400
    if mode == "yearly":
        return 86400
    if positive_gaps:
        return min(positive_gaps)
    return 1


def _build_durations(dts, mode, base_ms, min_ms, max_ms):
    if not dts:
        return []
    if len(dts) == 1:
        return [base_ms]

    deltas = []
    for i in range(len(dts) - 1):
        delta = int((dts[i + 1] - dts[i]).total_seconds())
        deltas.append(delta)

    positive_gaps = [d for d in deltas if d > 0]
    if not positive_gaps:
        return [base_ms] * len(dts)

    step_seconds = _nominal_step_seconds(mode, positive_gaps)
    durations = []
    for delta in deltas:
        if delta <= 0:
            ms = base_ms
        else:
            scaled = int(round(base_ms * (delta / step_seconds)))
            ms = max(min_ms, min(max_ms, scaled))
        durations.append(ms)

    durations.append(durations[-1] if durations else base_ms)
    return durations


def _duration_list(frame_durations, total):
    if isinstance(frame_durations, list):
        return frame_durations
    return [frame_durations] * total


def _ffmpeg_quote_path(path):
    return path.replace("'", r"'\''")


def _find_ffmpeg_executable():
    # 1) Prefer system ffmpeg on PATH.
    exe = shutil.which("ffmpeg")
    if exe:
        return exe

    # 2) Fallback: imageio-ffmpeg ships an ffmpeg binary.
    try:
        import imageio_ffmpeg  # type: ignore
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


def _save_mp4_with_ffmpeg(
    frame_paths,
    frame_durations_ms,
    output_path,
    codec="libx264",
    crf=20,
    preset="medium",
    pix_fmt="yuv420p",
):
    ffmpeg_exe = _find_ffmpeg_executable()
    if ffmpeg_exe is None:
        raise RuntimeError(
            "No ffmpeg executable found. The pip package 'ffmpeg' is not the "
            "video encoder binary. Install system ffmpeg (e.g., 'brew install ffmpeg' "
            "or 'conda install -c conda-forge ffmpeg') or install 'imageio-ffmpeg', "
            "or switch output_format='gif'."
        )

    abs_frame_paths = [os.path.abspath(path) for path in frame_paths]
    abs_output_path = os.path.abspath(output_path)
    concat_file = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".txt",
            delete=False,
            encoding="utf-8",
        ) as tf:
            concat_file = tf.name
            for path, dur_ms in zip(abs_frame_paths, frame_durations_ms):
                tf.write(f"file '{_ffmpeg_quote_path(path)}'\n")
                tf.write(f"duration {max(0.001, dur_ms / 1000.0):.6f}\n")
            # Repeat last frame so ffmpeg keeps the final duration.
            tf.write(f"file '{_ffmpeg_quote_path(abs_frame_paths[-1])}'\n")

        cmd = [
            ffmpeg_exe,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_file,
            "-vsync",
            "vfr",
            "-c:v",
            codec,
            "-crf",
            str(crf),
            "-preset",
            preset,
            "-pix_fmt",
            pix_fmt,
            "-movflags",
            "+faststart",
            abs_output_path,
        ]
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise RuntimeError(f"ffmpeg failed while writing {abs_output_path}") from exc
    finally:
        if concat_file and os.path.exists(concat_file):
            os.remove(concat_file)


frame_mode = frame_mode.strip().lower()
output_format = output_format.strip().lower()
if output_format not in {"gif", "mp4"}:
    raise ValueError(f"Unknown output_format: {output_format}")

if output_format == "gif" and not sname.lower().endswith(".gif"):
    sname = os.path.splitext(sname)[0] + ".gif"
if output_format == "mp4" and not sname.lower().endswith(".mp4"):
    sname = os.path.splitext(sname)[0] + ".mp4"

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

selected_dts = [dt for dt, _ in parsed]
if len(selected_dts) > 1:
    gaps = [
        int((selected_dts[i + 1] - selected_dts[i]).total_seconds())
        for i in range(len(selected_dts) - 1)
        if selected_dts[i + 1] > selected_dts[i]
    ]
    if gaps:
        min_gap = min(gaps)
        max_gap = max(gaps)
        if min_gap != max_gap:
            print(
                f"Detected irregular timeline: min gap={_format_gap(min_gap)}, "
                f"max gap={_format_gap(max_gap)}"
            )
        else:
            print(f"Detected regular timeline gap={_format_gap(min_gap)}")

# Build per-frame durations (used for GIF and MP4 timeline pacing)
if duration_mode == "proportional" and len(selected_dts) == len(fnames):
    frame_durations = _build_durations(
        selected_dts, frame_mode, base_duration, min_duration, max_duration
    )
elif duration_mode == "proportional":
    print("WARNING: Unparsed frames included; falling back to constant frame duration.")
    frame_durations = base_duration
else:
    frame_durations = base_duration

if output_format == "gif":
    # Create the frames (SAFE: no open-file leaks)
    frames = []
    total = len(fnames)
    for i, fname in enumerate(fnames, start=1):
        if i == 1 or i == total or (progress_every and i % progress_every == 0):
            print(f"Loading frame {i}/{total}: {os.path.basename(fname)}")
        with Image.open(fname) as im:
            frames.append(im.copy())  # detach from file handle

    print(f"Writing GIF to {sname} ({total} frames)")
    frames[0].save(
        sname,
        format='GIF',
        append_images=frames[1:],
        save_all=True,
        duration=frame_durations,
        loop=0
    )
else:
    total = len(fnames)
    dur_list = _duration_list(frame_durations, total)
    print(f"Writing MP4 to {sname} ({total} frames)")
    _save_mp4_with_ffmpeg(
        fnames,
        dur_list,
        sname,
        codec=mp4_codec,
        crf=mp4_crf,
        preset=mp4_preset,
        pix_fmt=mp4_pix_fmt,
    )
