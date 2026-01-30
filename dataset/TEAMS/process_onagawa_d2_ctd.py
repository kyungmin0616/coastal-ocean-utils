#!/usr/bin/env python3
"""
Process Onagawa Bay D2 CTD xlsx files (edit CONFIG below):
- read CTD profiles
- map station coordinates from *_en.xml
- plot selected variables by station/date
- optionally save npz for downstream use
- optionally write station.in for SCHISM extraction

Options:
- CONFIG controls defaults (paths, vars to plot/save, whether to plot/save).
- CLI flags can override paths and enable/disable plotting or npz output.

Examples:
- Plot with CONFIG defaults:
  python process_onagawa_d2_ctd.py
- Plot to a custom directory:
  python process_onagawa_d2_ctd.py --plot-dir /tmp/ctd_plots
- Save npz (uses CONFIG['SAVE_VARS']):
  python process_onagawa_d2_ctd.py --save-npz
- Limit number of profiles plotted:
  python process_onagawa_d2_ctd.py --max-profiles 10
"""
import argparse
import os
import re
import zipfile
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

try:
    import numpy as np
except Exception:
    np = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional plotting
    plt = None


CONFIG = {  # Edit this block to change defaults.
    'BASE_DIR': '/Users/kpark/Codes/D26-017-selected/OnagawaBayData/D2',
    'PLOT_DIR': './OnagawaD2Plots/',
    'PLOT_VARS': ['temp', 'sal', 'chl_flu','chl_a'],
    'SAVE_VARS': [
        'temp',
        'sal',
        'cond',
        'ec25',
        'density',
        'sigma_t',
        'chl_flu',
        'chl_a',
        'turbidity_mid',
        'do_pct',
        'do_mg',
    ],
    'MAX_PROFILES': None,
    'SAVE_NPZ': True,
    'NPZ_PATH': './npz/onagawa_d2_ctd.npz',
    'LOCAL_TIME_UTC_OFFSET_HOURS': 9.0,
    'PLOT_TIME_UTC': True,
    'USE_XML_COORDS': True,
    'SKIP_PLOT': False,
    'WRITE_STATION_IN': True,
    'STATION_IN_PATH': './station_onagawa_d2.in',
    'STATION_IN_FLAGS': '1 0 0 0 1 1 1 1 1',
    'STATION_IN_NAME_STYLE': 'station',
    'STATION_IN_DEPTH': 0.0,
    'LON_WRAP_360': False,
}

DEFAULT_BASE = CONFIG['BASE_DIR']

HEADER_ALIASES = {
    'year': ['年'],
    'month': ['月'],
    'day': ['日'],
    'station': ['地点'],
    'depth': ['深度[m]'],
    'temp': ['水温[C]'],
    'sal': ['塩分', '塩分[]'],
    'cond': ['電導度[mS/cm]'],
    'ec25': ['EC25[uS/cm]'],
    'density': ['Density[kg/m3]'],
    'sigma_t': ['σT', 'σT[]'],
    'chl_flu': ['Chl-Flu.[ppb]'],
    'chl_a': ['Chl-a[ug/L]', 'Chl-a[ug/l]'],
    'turbidity_mid': ['濁度中レンジ'],
    'do_pct': ['DO[%]'],
    'do_mg': ['DO[mg/L]'],
}

VAR_LABELS = {
    'temp': 'Temperature (C)',
    'sal': 'Salinity',
    'cond': 'Conductivity (mS/cm)',
    'ec25': 'EC25 (uS/cm)',
    'density': 'Density (kg/m3)',
    'sigma_t': 'Sigma-T',
    'chl_flu': 'Chl-Flu (ppb)',
    'chl_a': 'Chl-a (ug/L)',
    'turbidity_mid': 'Turbidity (mid)',
    'do_pct': 'DO (%)',
    'do_mg': 'DO (mg/L)',
}


def _wrap_lon(lon, wrap_360):
    if lon is None:
        return None
    if not wrap_360:
        return lon
    if lon < 0:
        return lon + 360.0
    if lon <= 180.0:
        return lon + 360.0
    return lon


def _shift_time(dt, hours):
    if dt is None or hours is None:
        return dt
    return dt + timedelta(hours=hours)


def _local_to_utc(dt, offset_hours):
    return _shift_time(dt, -offset_hours) if offset_hours is not None else dt


def _plot_time(dt, use_utc):
    if not use_utc:
        return dt
    offset_hours = CONFIG.get('LOCAL_TIME_UTC_OFFSET_HOURS', 0.0)
    return _local_to_utc(dt, offset_hours)


def _station_label(dataset, station_id, style):
    station_id = station_id or ''
    if style == 'station':
        return station_id or dataset
    if style == 'dataset_station':
        if station_id:
            return f'{dataset}_{station_id}'
        return dataset
    return station_id or dataset

FIXED_ORDER = [
    'year',
    'month',
    'day',
    'station',
    'depth',
    'temp',
    'sal',
    'cond',
    'ec25',
    'density',
    'sigma_t',
    'chl_flu',
    'chl_a',
    'turbidity_mid',
    'do_pct',
    'do_mg',
]


def _cell_str(val):
    if val is None:
        return ''
    s = str(val).strip()
    if s.startswith('\ufeff'):
        s = s.lstrip('\ufeff')
    return s


def _to_float(val):
    try:
        return float(str(val))
    except Exception:
        return None


def _parse_excel_serial(val):
    try:
        f = float(val)
    except Exception:
        return None
    if f < 30000 or f > 60000:
        return None
    base = datetime(1899, 12, 30)
    return base + timedelta(days=f)


def _parse_date(val):
    d = _parse_excel_serial(val)
    if d is not None:
        return d
    s = _cell_str(val)
    if not s:
        return None
    s = s.replace('.', '/').replace('-', '/')
    if '/' in s:
        parts = s.split('/')
        if len(parts) >= 3 and all(p.strip().isdigit() for p in parts[:3]):
            y = int(parts[0])
            if y < 100:
                y += 2000
            m = int(parts[1])
            d = int(parts[2])
            try:
                return datetime(y, m, d)
            except Exception:
                return None
    if s.isdigit():
        if len(s) == 8:
            return datetime(int(s[0:4]), int(s[4:6]), int(s[6:8]))
        if len(s) == 6:
            return datetime(int(s[0:2]) + 2000, int(s[2:4]), int(s[4:6]))
    return None


def _normalize_header(name):
    s = _cell_str(name)
    s = s.replace('µ', 'u').replace('μ', 'u').replace('℃', 'C').replace('³', '3')
    for ch in ['‑', '–', '—', '−', '‐', '－']:
        s = s.replace(ch, '-')
    s = re.sub(r'\s+', '', s)
    return s.lower()


def _build_header_index(header):
    alias_map = {}
    for key, aliases in HEADER_ALIASES.items():
        for alias in aliases:
            alias_map[_normalize_header(alias)] = key
    col_map = {}
    for idx, name in enumerate(header):
        norm = _normalize_header(name)
        key = alias_map.get(norm)
        if key and key not in col_map:
            col_map[key] = idx
    return col_map


def _looks_like_furigana_header(header):
    furigana = {'ネン', 'ツキ', 'ヒ', 'チテン'}
    return any(_cell_str(h) in furigana for h in header)


def _build_column_map(header):
    if _looks_like_furigana_header(header):
        return {key: idx for idx, key in enumerate(FIXED_ORDER) if idx < len(header)}
    return _build_header_index(header)

def xlsx_shared_strings(zf):
    if 'xl/sharedStrings.xml' not in zf.namelist():
        return []
    root = ET.fromstring(zf.read('xl/sharedStrings.xml'))
    ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    return [t.text or '' for t in root.iter(f'{ns}t')]


def xlsx_sheet_map(zf):
    wb = ET.fromstring(zf.read('xl/workbook.xml'))
    ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    rid_ns = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
    rels = ET.fromstring(zf.read('xl/_rels/workbook.xml.rels'))
    rel_map = {r.get('Id'): r.get('Target') for r in rels}
    sheets = []
    for s in wb.iter(f'{ns}sheet'):
        name = s.get('name')
        rid = s.get(f'{rid_ns}id')
        target = rel_map.get(rid, '')
        if target.startswith('/'):
            target = target[1:]
        if not target.startswith('xl/'):
            target = f'xl/{target}'
        sheets.append((name, target))
    return sheets


def read_sheet_rows(zf, sheet_path, shared_strings, max_rows=None):
    ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    root = ET.fromstring(zf.read(sheet_path))
    rows = []
    for row in root.iter(f'{ns}row'):
        cells = []
        for c in row.iter(f'{ns}c'):
            r = c.get('r')
            if r:
                m = re.match(r'[A-Z]+', r)
                idx = 0
                for ch in m.group(0):
                    idx = idx * 26 + (ord(ch) - 64)
                idx -= 1
            else:
                idx = len(cells)
            while len(cells) <= idx:
                cells.append('')
            v = c.find(f'{ns}v')
            if c.get('t') == 's' and v is not None:
                sval = shared_strings[int(v.text)] if v.text and v.text.isdigit() else ''
                cells[idx] = sval
            elif c.get('t') == 'inlineStr':
                t = c.find(f'{ns}is/{ns}t')
                cells[idx] = t.text if t is not None else ''
            else:
                cells[idx] = v.text if v is not None else ''
        if cells:
            rows.append(cells)
        if max_rows and len(rows) >= max_rows:
            break
    return rows


def load_station_coords(xml_path):
    coords = {}
    if not xml_path or not os.path.exists(xml_path):
        return coords
    try:
        root = ET.parse(xml_path).getroot()
    except Exception:
        return coords
    for point in root.iter('Point'):
        name = None
        coord = None
        for child in list(point):
            if child.tag == 'Point_Name':
                name = _cell_str(child.text)
            elif child.tag == 'Point_Coordinates':
                coord = _cell_str(child.text)
        if not coord:
            continue
        parts = [p.strip() for p in coord.split(',')]
        if len(parts) < 2:
            continue
        lat = _to_float(parts[0])
        lon = _to_float(parts[1])
        if lat is None or lon is None:
            continue
        if name:
            coords[name] = (lat, lon)
    return coords


def find_station_latlon(coords, station_id):
    if station_id is None:
        return None, None, None
    sid = str(station_id).strip()
    if not sid:
        return None, None, None
    key = sid
    if sid.isdigit():
        key = f"St.{int(sid)}"
    if key in coords:
        lat, lon = coords[key]
        return lat, lon, key
    if sid in coords:
        lat, lon = coords[sid]
        return lat, lon, sid
    return None, None, key


def parse_ctd_xlsx(path):
    try:
        with zipfile.ZipFile(path) as zf:
            shared = xlsx_shared_strings(zf)
            sheets = xlsx_sheet_map(zf)
            if not sheets:
                return [], []
            _, sheet = sheets[0]
            rows = read_sheet_rows(zf, sheet, shared)
    except zipfile.BadZipFile:
        return [], []
    if not rows:
        return [], []
    header = rows[0]
    data_rows = rows[1:]
    return header, data_rows


def _parse_obs_time(row, col_map):
    day_idx = col_map.get('day')
    if day_idx is None or day_idx >= len(row):
        return None
    day_val = row[day_idx]
    d = _parse_date(day_val)
    if d is not None:
        return d
    y = _to_float(row[col_map.get('year', -1)])
    m = _to_float(row[col_map.get('month', -1)])
    dd = _to_float(day_val)
    if y is None or m is None or dd is None:
        return None
    try:
        return datetime(int(y), int(m), int(dd))
    except Exception:
        return None


def collect_records(base_dir, plot_vars, save_vars, use_xml_coords=True):
    records = []
    var_keys = sorted(set(plot_vars + save_vars + ['depth']))
    datasets = sorted([d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))])
    for d in datasets:
        folder = os.path.join(base_dir, d)
        xml_path = None
        for name in os.listdir(folder):
            if name.endswith('_en.xml'):
                xml_path = os.path.join(folder, name)
                break
        coords = load_station_coords(xml_path) if use_xml_coords else {}
        for name in os.listdir(folder):
            if name.startswith('~$') or name.startswith('._') or name.startswith('.'):
                continue
            if 'CTD' not in name or not name.lower().endswith('.xlsx'):
                continue
            path = os.path.join(folder, name)
            header, rows = parse_ctd_xlsx(path)
            if not rows:
                continue
            col_map = _build_column_map(header)
            required = ['year', 'month', 'day', 'station', 'depth']
            missing = [k for k in required if k not in col_map]
            if missing:
                print(f'Skipping {path}: missing columns {missing}')
                continue
            for row in rows:
                row = list(row)
                if len(row) < len(header):
                    row.extend([''] * (len(header) - len(row)))
                date_val = _parse_obs_time(row, col_map)
                if date_val is None:
                    continue
                station_id = _cell_str(row[col_map['station']])
                depth = _to_float(row[col_map['depth']])
                if depth is None:
                    continue
                lat, lon, station_name = find_station_latlon(coords, station_id)
                rec = {
                    'dataset': d,
                    'source': name,
                    'time': date_val,
                    'station_id': station_id,
                    'station_name': station_name,
                    'lat': lat,
                    'lon': lon,
                    'depth': depth,
                }
                for key in var_keys:
                    if key == 'depth':
                        continue
                    idx = col_map.get(key)
                    if idx is None or idx >= len(row):
                        rec[key] = None
                    else:
                        rec[key] = _to_float(row[idx])
                records.append(rec)
    return records


def plot_profiles(records, plot_vars, out_dir, max_profiles=None):
    if plt is None:
        raise RuntimeError('matplotlib is not available for plotting')
    if not plot_vars:
        return 0
    os.makedirs(out_dir, exist_ok=True)
    use_utc = CONFIG.get('PLOT_TIME_UTC', False)
    tz_label = 'UTC' if use_utc else 'local'
    groups = {}
    for rec in records:
        plot_dt = _plot_time(rec['time'], use_utc)
        key = (plot_dt.date(), rec['station_name'] or rec['station_id'])
        groups.setdefault(key, []).append(rec)
    count = 0
    for (date, station), recs in sorted(groups.items()):
        if max_profiles is not None and count >= max_profiles:
            break
        plot_keys = []
        for key in plot_vars:
            vals = [r.get(key) for r in recs if r.get(key) is not None and r.get('depth') is not None]
            if vals:
                plot_keys.append(key)
        if not plot_keys:
            continue
        fig, axes = plt.subplots(1, len(plot_keys), figsize=(4 * len(plot_keys), 6), sharey=True)
        if len(plot_keys) == 1:
            axes = [axes]
        for ax, key in zip(axes, plot_keys):
            pairs = [(r['depth'], r.get(key)) for r in recs if r.get(key) is not None and r.get('depth') is not None]
            pairs.sort(key=lambda x: x[0])
            depths = [p[0] for p in pairs]
            vals = [p[1] for p in pairs]
            ax.plot(vals, depths, '-o', ms=3)
            ax.set_xlabel(VAR_LABELS.get(key, key))
            ax.grid(True, alpha=0.3)
        axes[0].set_ylabel('Depth (m)')
        axes[0].invert_yaxis()
        title = f'{station} {date.isoformat()} ({tz_label})'
        fig.suptitle(title)
        fig.tight_layout()
        out_name = f'ctd_{date.strftime("%Y%m%d")}_{station}.png'.replace(' ', '_')
        out_path = os.path.join(out_dir, out_name)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        count += 1
    return count


def save_npz(records, out_path, save_vars):
    if np is None:
        raise RuntimeError('numpy is required to save npz output')
    offset_hours = CONFIG.get('LOCAL_TIME_UTC_OFFSET_HOURS', 0.0)
    time_local = [r['time'] for r in records]
    time_utc = [_local_to_utc(t, offset_hours) for t in time_local]
    times = np.array(time_utc, dtype='datetime64[s]')
    stations = np.array([r['station_id'] for r in records], dtype='U')
    station_names = np.array([r['station_name'] or '' for r in records], dtype='U')
    lats = np.array([np.nan if r['lat'] is None else r['lat'] for r in records], dtype=float)
    lons = np.array([np.nan if r['lon'] is None else r['lon'] for r in records], dtype=float)
    depth = np.array([r['depth'] for r in records], dtype=float)
    sources = np.array([r['source'] for r in records], dtype='U')
    datasets = np.array([r['dataset'] for r in records], dtype='U')

    payload = {
        'time': times,
        'time_local': np.array(time_local, dtype='datetime64[s]'),
        'station_id': stations,
        'station_name': station_names,
        'lat': lats,
        'lon': lons,
        'depth': depth,
        'source': sources,
        'dataset': datasets,
    }
    for key in save_vars:
        payload[key] = np.array(
            [np.nan if r.get(key) is None else r.get(key) for r in records],
            dtype=float,
        )

    np.savez(out_path, **payload)


def write_station_in(records, out_path, flags, name_style, depth, wrap_360):
    stations = []
    seen = set()
    for rec in records:
        lat = rec.get('lat')
        lon = rec.get('lon')
        if lat is None or lon is None:
            continue
        name = _station_label(rec.get('dataset', ''), rec.get('station_id', ''), name_style)
        lon_val = _wrap_lon(lon, wrap_360)
        key = (name, round(lat, 6), round(lon_val, 6))
        if key in seen:
            continue
        seen.add(key)
        stations.append((name, lon_val, lat))
    if not stations:
        return 0
    stations.sort(key=lambda x: x[0])
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write(f'{flags}\n')
        f.write(f'{len(stations)}\n')
        for idx, (name, lon, lat) in enumerate(stations, start=1):
            f.write(f'{idx} {lon:.6f} {lat:.6f} {depth:.6f} # {name}\n')
    return len(stations)


def main():
    parser = argparse.ArgumentParser(description='Process Onagawa D2 CTD xlsx files.')
    parser.add_argument('--base', default=CONFIG['BASE_DIR'], help='Base D2 directory')
    parser.add_argument('--plot-dir', default=None, help='Output plot directory')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    parser.add_argument('--save-npz', action='store_true', help='Save npz output')
    parser.add_argument('--npz-path', default=None, help='Path to write npz')
    parser.add_argument('--max-profiles', type=int, default=None, help='Limit number of plots')
    args = parser.parse_args()

    base_dir = args.base or CONFIG['BASE_DIR']
    plot_vars = CONFIG['PLOT_VARS']
    save_vars = CONFIG['SAVE_VARS']
    use_xml_coords = CONFIG['USE_XML_COORDS']
    skip_plot = CONFIG['SKIP_PLOT'] or args.no_plot
    max_profiles = args.max_profiles if args.max_profiles is not None else CONFIG['MAX_PROFILES']
    plot_dir = args.plot_dir or CONFIG['PLOT_DIR'] or os.path.join(base_dir, 'plots_ctd')
    save_npz_flag = args.save_npz or CONFIG['SAVE_NPZ']
    npz_path = args.npz_path or CONFIG['NPZ_PATH'] or os.path.join(base_dir, 'onagawa_d2_ctd.npz')
    write_station_in_flag = CONFIG['WRITE_STATION_IN']
    station_in_path = CONFIG['STATION_IN_PATH'] or os.path.join(base_dir, 'station.in')
    station_flags = CONFIG['STATION_IN_FLAGS']
    station_name_style = CONFIG['STATION_IN_NAME_STYLE']
    station_depth = CONFIG['STATION_IN_DEPTH']
    wrap_360 = CONFIG['LON_WRAP_360']

    records = collect_records(base_dir, plot_vars, save_vars, use_xml_coords=use_xml_coords)
    if not records:
        print('No CTD records found.')
        return 1
    print(f'Loaded {len(records)} CTD samples from {base_dir}')

    if not skip_plot:
        count = plot_profiles(records, plot_vars, plot_dir, max_profiles=max_profiles)
        print(f'Saved {count} profile plots to {plot_dir}')

    if save_npz_flag:
        if np is None:
            print('numpy is required for --save-npz')
            return 1
        save_npz(records, npz_path, save_vars)
        print(f'Saved npz to {npz_path}')
    if write_station_in_flag:
        count = write_station_in(records, station_in_path, station_flags, station_name_style, station_depth, wrap_360)
        print(f'Saved station.in with {count} stations to {station_in_path}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
