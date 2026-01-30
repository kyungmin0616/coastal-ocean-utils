#!/usr/bin/env python3
"""
Process Sendai Bay D2 time-series logger data (edit CONFIG below):
- read multiple time-series formats (xlsx/csv)
- use station coordinates from *_en.xml
- plot selected variables by station
- optionally save npz for downstream use
- optionally write station.in for SCHISM extraction

Examples:
- Plot with CONFIG defaults:
  python process_sendai_d2_timeseries.py
- Save npz (uses CONFIG['SAVE_VARS']):
  python process_sendai_d2_timeseries.py --save-npz
- Plot to a custom directory:
  python process_sendai_d2_timeseries.py --plot-dir /tmp/sendai_ts_plots
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
    'BASE_DIR': '/Users/kpark/Codes/D26-017-selected/SendaiBayData/D2',
    'PLOT_DIR': './SendaiD2Plots/',
    'PLOT_VARS': ['temp', 'sal'],
    'SAVE_VARS': ['temp', 'sal', 'cond', 'ec25', 'depth', 'battery'],
    'MAX_PLOTS': None,
    'PLOT_EVERY_N': 1,
    'SAVE_NPZ': True,
    'NPZ_PATH': './npz/sendai_d2_timeseries.npz',
    'LOCAL_TIME_UTC_OFFSET_HOURS': 9.0,
    'PLOT_TIME_UTC': True,
    'USE_XML_COORDS': True,
    'SKIP_PLOT': False,
    'SKIP_FILES': ['2012.xlsx', '2013.xlsx', 'Logger_Point.xlsx'],
    'STATION_OVERRIDE': {},
    'CHANNEL_STATIONS': {},
    'WRITE_STATION_IN': True,
    'STATION_IN_PATH': None,
    'STATION_IN_FLAGS': '1 0 0 0 1 1 1 1 1',
    'STATION_IN_NAME_STYLE': 'dataset_station',
    'STATION_IN_DEPTH': 0.0,
    'LON_WRAP_360': False,
}


VAR_LABELS = {
    'temp': 'Temperature (C)',
    'sal': 'Salinity',
    'cond': 'Conductivity (mS/cm)',
    'ec25': 'EC25 (uS/cm)',
    'depth': 'Depth (m)',
    'battery': 'Battery (V)',
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


def _parse_excel_datetime(date_val, time_val=None):
    d = _to_float(date_val)
    if d is None:
        return None
    if time_val is not None:
        t = _to_float(time_val)
        if t is not None and t < 1.0:
            d = d + t
    base = datetime(1899, 12, 30)
    return base + timedelta(days=d)


def _parse_datetime_str(val):
    s = _cell_str(val)
    if not s:
        return None
    s = s.replace('-', '/')
    parts = s.split()
    if len(parts) == 2:
        date_s, time_s = parts
    elif len(parts) == 1:
        date_s, time_s = parts[0], '00:00'
    else:
        return None
    date_bits = date_s.split('/')
    if len(date_bits) != 3:
        return None
    try:
        y = int(date_bits[0])
        m = int(date_bits[1])
        d = int(date_bits[2])
    except Exception:
        return None
    time_bits = time_s.split(':')
    try:
        hh = int(time_bits[0])
        mm = int(time_bits[1]) if len(time_bits) > 1 else 0
        ss = int(time_bits[2]) if len(time_bits) > 2 else 0
    except Exception:
        return None
    try:
        return datetime(y, m, d, hh, mm, ss)
    except Exception:
        return None


def xlsx_shared_strings(zf):
    if 'xl/sharedStrings.xml' not in zf.namelist():
        return []
    root = ET.fromstring(zf.read('xl/sharedStrings.xml'))
    ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    return [t.text or '' for t in root.iter(f'{ns}t')]


def xlsx_first_sheet(zf):
    wb = ET.fromstring(zf.read('xl/workbook.xml'))
    ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
    rid_ns = '{http://schemas.openxmlformats.org/officeDocument/2006/relationships}'
    rels = ET.fromstring(zf.read('xl/_rels/workbook.xml.rels'))
    rel_map = {r.get('Id'): r.get('Target') for r in rels}
    for s in wb.iter(f'{ns}sheet'):
        rid = s.get(f'{rid_ns}id')
        target = rel_map.get(rid, '')
        if target.startswith('/'):
            target = target[1:]
        if not target.startswith('xl/'):
            target = f'xl/{target}'
        return target
    return None


def iter_xlsx_rows(path):
    with zipfile.ZipFile(path) as zf:
        shared = xlsx_shared_strings(zf)
        sheet = xlsx_first_sheet(zf)
        if not sheet or sheet not in zf.namelist():
            return
        ns = '{http://schemas.openxmlformats.org/spreadsheetml/2006/main}'
        root = ET.fromstring(zf.read(sheet))
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
                    sval = shared[int(v.text)] if v.text and v.text.isdigit() else ''
                    cells[idx] = sval
                elif c.get('t') == 'inlineStr':
                    t = c.find(f'{ns}is/{ns}t')
                    cells[idx] = t.text if t is not None else ''
                else:
                    cells[idx] = v.text if v is not None else ''
            yield cells


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


def get_station_coords(coords, station_id):
    if station_id in coords:
        return coords[station_id]
    return None, None


def choose_station_id(coords, dataset, channel_idx=None):
    override = CONFIG['STATION_OVERRIDE'].get(dataset)
    if override:
        return override, coords.get(override, (None, None))
    if coords:
        keys = sorted(coords.keys())
        if len(keys) == 1:
            return keys[0], coords[keys[0]]
        if channel_idx is not None:
            mapping = CONFIG['CHANNEL_STATIONS'].get(dataset, [])
            if channel_idx < len(mapping):
                key = mapping[channel_idx]
                return key, coords.get(key, (None, None))
        if 'St.1' in coords:
            return 'St.1', coords['St.1']
        return keys[0], coords[keys[0]]
    return dataset, (None, None)


def parse_2014_xlsx(path, dataset, coords):
    records = []
    rows = list(iter_xlsx_rows(path))
    if not rows:
        return records
    for row in rows[2:]:
        date_val = row[0] if len(row) > 0 else ''
        time_val = row[1] if len(row) > 1 else ''
        temp = _to_float(row[2]) if len(row) > 2 else None
        depth = _to_float(row[3]) if len(row) > 3 else None
        sal = _to_float(row[4]) if len(row) > 4 else None
        dt = _parse_excel_datetime(date_val, time_val)
        if dt is None or temp is None:
            continue
        station_id, (lat, lon) = choose_station_id(coords, dataset)
        records.append({
            'time': dt,
            'station_id': station_id,
            'lat': lat,
            'lon': lon,
            'temp': temp,
            'sal': sal,
            'depth': depth,
        })
    return records


def parse_2015_xlsx(path, dataset, coords):
    records = []
    for row in iter_xlsx_rows(path):
        dt = _parse_excel_datetime(row[0]) if len(row) > 0 else None
        temp = _to_float(row[6]) if len(row) > 6 else None
        sal = _to_float(row[7]) if len(row) > 7 else None
        if dt is None or temp is None:
            continue
        station_id, (lat, lon) = choose_station_id(coords, dataset)
        records.append({
            'time': dt,
            'station_id': station_id,
            'lat': lat,
            'lon': lon,
            'temp': temp,
            'sal': sal,
        })
    return records


def parse_2016_xlsx(path, dataset, coords):
    records = []
    for row in iter_xlsx_rows(path):
        dt = _parse_excel_datetime(row[0]) if len(row) > 0 else None
        temp1 = _to_float(row[1]) if len(row) > 1 else None
        sal1 = _to_float(row[2]) if len(row) > 2 else None
        if dt is None or temp1 is None:
            continue
        station_id, (lat, lon) = choose_station_id(coords, dataset, channel_idx=0)
        records.append({
            'time': dt,
            'station_id': station_id,
            'lat': lat,
            'lon': lon,
            'temp': temp1,
            'sal': sal1,
        })
        temp2 = _to_float(row[3]) if len(row) > 3 else None
        sal2 = _to_float(row[4]) if len(row) > 4 else None
        if temp2 is not None and sal2 is not None:
            station_id2, (lat2, lon2) = choose_station_id(coords, dataset, channel_idx=1)
            records.append({
                'time': dt,
                'station_id': station_id2,
                'lat': lat2,
                'lon': lon2,
                'temp': temp2,
                'sal': sal2,
            })
    return records


def parse_2017_xlsx(path, dataset, coords):
    records = []
    for row in iter_xlsx_rows(path):
        dt = _parse_excel_datetime(row[0]) if len(row) > 0 else None
        temp = _to_float(row[1]) if len(row) > 1 else None
        sal = _to_float(row[2]) if len(row) > 2 else None
        if dt is None or temp is None:
            continue
        station_id, (lat, lon) = choose_station_id(coords, dataset)
        records.append({
            'time': dt,
            'station_id': station_id,
            'lat': lat,
            'lon': lon,
            'temp': temp,
            'sal': sal,
        })
    return records


def parse_2013_fix_xlsx(path, dataset, coords):
    records = []
    rows = list(iter_xlsx_rows(path))
    if not rows:
        return records
    for row in rows[3:]:
        if len(row) < 2:
            continue
        dt = _parse_excel_datetime(row[0], row[1])
        temp = _to_float(row[2]) if len(row) > 2 else None
        sal = _to_float(row[3]) if len(row) > 3 else None
        depth = _to_float(row[5]) if len(row) > 5 else None
        if depth is None and len(row) > 4:
            depth = _to_float(row[4])
        cond = _to_float(row[6]) if len(row) > 6 else None
        if dt is None or temp is None:
            continue
        station_id, (lat, lon) = choose_station_id(coords, dataset)
        records.append({
            'time': dt,
            'station_id': station_id,
            'lat': lat,
            'lon': lon,
            'temp': temp,
            'sal': sal,
            'cond': cond,
            'depth': depth,
        })
    return records


def parse_2012_st_xlsx(path, dataset, coords):
    records = []
    m = re.search(r'2012_ST(\d+)', os.path.basename(path))
    station_id = f"St.{m.group(1)}" if m else None
    lat, lon = get_station_coords(coords, station_id) if station_id else (None, None)
    rows = list(iter_xlsx_rows(path))
    if not rows:
        return records
    for row in rows[2:]:
        if not row:
            continue
        for offset in range(0, len(row), 8):
            block = row[offset : offset + 7]
            if len(block) < 3:
                continue
            dt = _parse_excel_datetime(block[0], block[1])
            temp = _to_float(block[2]) if len(block) > 2 else None
            cond = _to_float(block[3]) if len(block) > 3 else None
            depth = _to_float(block[5]) if len(block) > 5 else None
            if depth is None and len(block) > 4:
                depth = _to_float(block[4])
            sal = _to_float(block[6]) if len(block) > 6 else None
            if dt is None or temp is None:
                continue
            records.append({
                'time': dt,
                'station_id': station_id or dataset,
                'lat': lat,
                'lon': lon,
                'temp': temp,
                'sal': sal,
                'cond': cond,
                'depth': depth,
            })
    return records


def parse_2018_csv(path, dataset, coords):
    records = []
    rows = None
    for enc in ['utf-8-sig', 'utf-8', 'cp932', 'shift_jis']:
        try:
            with open(path, encoding=enc, errors='strict') as f:
                rows = [line.rstrip('\n') for line in f]
            if rows:
                break
        except Exception:
            continue
    if not rows:
        return records
    header_idx = None
    for i, line in enumerate(rows[:200]):
        if line.startswith('日時') or '水温' in line:
            header_idx = i
            break
    if header_idx is None:
        header_idx = 0
    header = [h.strip() for h in rows[header_idx].split(',')]
    col_map = {}
    for idx, name in enumerate(header):
        if '日時' in name:
            col_map['time'] = idx
        elif '水温' in name:
            col_map['temp'] = idx
        elif '塩分' in name:
            col_map['sal'] = idx
        elif '電導度' in name and '25' not in name:
            col_map['cond'] = idx
        elif 'EC' in name or '25' in name:
            col_map['ec25'] = idx
        elif '電池' in name:
            col_map['battery'] = idx
    for line in rows[header_idx + 1:]:
        if not line or line.startswith('//') or line.startswith('['):
            continue
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 2:
            continue
        time_val = parts[col_map.get('time', 0)] if 'time' in col_map else parts[0]
        dt = _parse_datetime_str(time_val)
        if dt is None:
            continue
        temp = _to_float(parts[col_map['temp']]) if 'temp' in col_map and col_map['temp'] < len(parts) else None
        sal = _to_float(parts[col_map['sal']]) if 'sal' in col_map and col_map['sal'] < len(parts) else None
        cond = _to_float(parts[col_map['cond']]) if 'cond' in col_map and col_map['cond'] < len(parts) else None
        ec25 = _to_float(parts[col_map['ec25']]) if 'ec25' in col_map and col_map['ec25'] < len(parts) else None
        battery = _to_float(parts[col_map['battery']]) if 'battery' in col_map and col_map['battery'] < len(parts) else None
        if temp is None:
            continue
        station_id, (lat, lon) = choose_station_id(coords, dataset)
        records.append({
            'time': dt,
            'station_id': station_id,
            'lat': lat,
            'lon': lon,
            'temp': temp,
            'sal': sal,
            'cond': cond,
            'ec25': ec25,
            'battery': battery,
        })
    return records


def iter_data_files(base_dir):
    for root, _, names in os.walk(base_dir):
        for name in names:
            if name.startswith('~$') or name.startswith('._') or name.startswith('.'):
                continue
            if name in CONFIG['SKIP_FILES']:
                continue
            lower = name.lower()
            if not (lower.endswith('.xlsx') or lower.endswith('.csv')):
                continue
            if lower.endswith('_d01.xlsx') or lower.endswith('_d01_ja.xml') or lower.endswith('_d01_en.xml'):
                continue
            yield os.path.join(root, name)


def collect_records(base_dir, use_xml_coords=True):
    records = []
    for path in sorted(iter_data_files(base_dir)):
        dataset = os.path.basename(os.path.dirname(path))
        coords = {}
        if use_xml_coords:
            xml_path = None
            for name in os.listdir(os.path.dirname(path)):
                if name.endswith('_en.xml'):
                    xml_path = os.path.join(os.path.dirname(path), name)
                    break
            coords = load_station_coords(xml_path)
        name = os.path.basename(path)
        if name == '2014.xlsx':
            recs = parse_2014_xlsx(path, dataset, coords)
        elif name == '2013_FIX.xlsx':
            recs = parse_2013_fix_xlsx(path, dataset, coords)
        elif name == '2015.xlsx':
            recs = parse_2015_xlsx(path, dataset, coords)
        elif name.startswith('2012_ST') and name.lower().endswith('.xlsx'):
            recs = parse_2012_st_xlsx(path, dataset, coords)
        elif name == '2016.xlsx':
            recs = parse_2016_xlsx(path, dataset, coords)
        elif name == '2017.xlsx':
            recs = parse_2017_xlsx(path, dataset, coords)
        elif name == '2018.csv':
            recs = parse_2018_csv(path, dataset, coords)
        else:
            continue
        for rec in recs:
            rec['dataset'] = dataset
            rec['source'] = name
        records.extend(recs)
    return records


def plot_timeseries(records, plot_vars, out_dir, max_plots=None, every_n=1):
    if plt is None:
        raise RuntimeError('matplotlib is not available for plotting')
    if not plot_vars:
        return 0
    os.makedirs(out_dir, exist_ok=True)
    use_utc = CONFIG.get('PLOT_TIME_UTC', False)
    time_label = 'Time (UTC)' if use_utc else 'Time (local)'
    groups = {}
    for rec in records:
        key = (rec.get('dataset'), rec.get('station_id'))
        groups.setdefault(key, []).append(rec)
    count = 0
    for (dataset, station), recs in sorted(groups.items()):
        if max_plots is not None and count >= max_plots:
            break
        recs = sorted(recs, key=lambda r: _plot_time(r['time'], use_utc))
        fig, axes = plt.subplots(len(plot_vars), 1, figsize=(8, 3 * len(plot_vars)), sharex=True)
        if len(plot_vars) == 1:
            axes = [axes]
        for ax, key in zip(axes, plot_vars):
            times = [_plot_time(r['time'], use_utc) for r in recs if r.get(key) is not None]
            vals = [r.get(key) for r in recs if r.get(key) is not None]
            if every_n > 1:
                times = times[::every_n]
                vals = vals[::every_n]
            ax.plot(times, vals, '-', lw=0.7)
            ax.set_ylabel(VAR_LABELS.get(key, key))
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel(time_label)
        title = f'{dataset} {station}'
        fig.suptitle(title)
        fig.tight_layout()
        out_name = f'timeseries_{dataset}_{station}.png'.replace(' ', '_')
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
    payload = {
        'time': np.array(time_utc, dtype='datetime64[s]'),
        'time_local': np.array(time_local, dtype='datetime64[s]'),
        'station_id': np.array([r.get('station_id', '') for r in records], dtype='U'),
        'lat': np.array([np.nan if r.get('lat') is None else r.get('lat') for r in records], dtype=float),
        'lon': np.array([np.nan if r.get('lon') is None else r.get('lon') for r in records], dtype=float),
        'dataset': np.array([r.get('dataset', '') for r in records], dtype='U'),
        'source': np.array([r.get('source', '') for r in records], dtype='U'),
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
    parser = argparse.ArgumentParser(description='Process Sendai D2 time-series data.')
    parser.add_argument('--base', default=CONFIG['BASE_DIR'], help='Base D2 directory')
    parser.add_argument('--plot-dir', default=None, help='Output plot directory')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    parser.add_argument('--save-npz', action='store_true', help='Save npz output')
    parser.add_argument('--npz-path', default=None, help='Path to write npz')
    parser.add_argument('--max-plots', type=int, default=None, help='Limit number of plots')
    parser.add_argument('--plot-every', type=int, default=None, help='Plot every Nth sample')
    args = parser.parse_args()

    base_dir = args.base or CONFIG['BASE_DIR']
    plot_vars = CONFIG['PLOT_VARS']
    save_vars = CONFIG['SAVE_VARS']
    use_xml_coords = CONFIG['USE_XML_COORDS']
    skip_plot = CONFIG['SKIP_PLOT'] or args.no_plot
    max_plots = args.max_plots if args.max_plots is not None else CONFIG['MAX_PLOTS']
    plot_dir = args.plot_dir or CONFIG['PLOT_DIR'] or os.path.join(base_dir, 'plots_ts')
    save_npz_flag = args.save_npz or CONFIG['SAVE_NPZ']
    npz_path = args.npz_path or CONFIG['NPZ_PATH'] or os.path.join(base_dir, 'sendai_d2_timeseries.npz')
    every_n = args.plot_every if args.plot_every is not None else CONFIG['PLOT_EVERY_N']
    write_station_in_flag = CONFIG['WRITE_STATION_IN']
    station_in_path = CONFIG['STATION_IN_PATH'] or os.path.join(base_dir, 'station.in')
    station_flags = CONFIG['STATION_IN_FLAGS']
    station_name_style = CONFIG['STATION_IN_NAME_STYLE']
    station_depth = CONFIG['STATION_IN_DEPTH']
    wrap_360 = CONFIG['LON_WRAP_360']

    records = collect_records(base_dir, use_xml_coords=use_xml_coords)
    if not records:
        print('No time-series records found.')
        return 1
    print(f'Loaded {len(records)} samples from {base_dir}')

    if not skip_plot:
        count = plot_timeseries(records, plot_vars, plot_dir, max_plots=max_plots, every_n=every_n)
        print(f'Saved {count} time-series plots to {plot_dir}')

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
