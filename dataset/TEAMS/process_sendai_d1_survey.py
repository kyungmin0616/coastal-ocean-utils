#!/usr/bin/env python3
"""
Process Sendai Bay D1 Survey_outline xlsx files (edit CONFIG below):
- read station-based survey snapshots (two rows per station: surface, bottom)
- map station coordinates from xlsx; fallback to *_en.xml if needed
- plot selected variables by station and layer
- optionally save npz and station.in for SCHISM extraction
- optional T-S diagrams by survey file or dataset
- optional map plots of surface/bottom values at stations

Examples:
- Plot with CONFIG defaults:
  python process_sendai_d1_survey.py
- Only T-S diagrams:
  # set CONFIG['PLOT_TS']=True and CONFIG['PLOT_TIMESERIES']=False
  python process_sendai_d1_survey.py
- Save npz:
  python process_sendai_d1_survey.py --save-npz
"""
import argparse
import math
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
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except Exception:  # pragma: no cover - optional plotting
    ccrs = None
    cfeature = None


CONFIG = {  # Edit this block to change defaults.
    'BASE_DIR': '/Users/kpark/Codes/D26-017-selected/SendaiBayData/D1',
    'PLOT_DIR': './SendaiD1Plots/',
    'PLOT_VARS': ['temp', 'sal'],
    'SAVE_VARS': ['temp', 'sal', 'depth'],
    'MAX_PLOTS': None,
    'PLOT_TIMESERIES': False,
    'PLOT_TS': False,
    'TS_DIR': None,
    'TS_GROUP_BY': 'source',  # "source" (survey file), "dataset" (folder)
    'TS_COLOR_BY': 'layer',  # "layer" or "depth"
    'TS_MAX_PLOTS': None,
    'PLOT_MAPS': True,
    'MAP_DIR': './SendaiD1Plots/',
    'MAP_VARS': ['temp', 'sal'],
    'MAP_GROUP_BY': 'source',  # "source" or "dataset"
    'MAP_LAYER': 'both',  # "surface", "bottom", or "both"
    'MAP_MAX_PLOTS': None,
    'MAP_CMAP': 'viridis',
    'MAP_EXTENT': [140.78, 141.23, 37.73, 38.5],  # [lon_min, lon_max, lat_min, lat_max]
    'MAP_PAD_DEG': 0.03,
    'MAP_COAST_RES': '10m',
    'MAP_FIGSIZE': None,  # None -> auto-size by extent
    'SAVE_NPZ': True,
    'NPZ_PATH': './npz/sendai_d1_survey.npz',
    'LOCAL_TIME_UTC_OFFSET_HOURS': 9.0,
    'PLOT_TIME_UTC': True,
    'USE_XML_COORDS': True,
    'SKIP_PLOT': False,
    'SURVEY_FILES': None,
    'DATE_COL': 5,
    'SURVEY_COLUMN_MAP': {
        'Survey_outline_Jul2013.xlsx': {'temp': 10, 'sal': 12, 'depth': 6},
        'Survey_outline_Nov2013.xlsx': {'temp': 10, 'sal': 13, 'depth': 6},
        'Survey_outline_Jul2014.xlsx': {'temp': 10, 'sal': 12, 'depth': 6},
        'Survey_outline_Oct-Nov2014.xlsx': {'temp': 10, 'sal': 12, 'depth': 6},
        'Survey_outline_Mar2015.xlsx': {'temp': 10, 'sal': 12, 'depth': 6},
        'Survey_outline_Sep2015.xlsx': {'temp': 14, 'sal': 16, 'depth': 6},
        'Survey_outline_Jun-Jul2017.xlsx': {'temp': 9, 'sal': 16, 'depth': 6},
        'Survey_outline_Jul2019.xlsx': {'temp': 9, 'sal': 16, 'depth': 6},
    },
    'STATION_MATCH_MAX_DEG': 0.01,
    'WRITE_STATION_IN': True,
    'STATION_IN_PATH': './station_sendai_d1.in',
    'STATION_IN_FLAGS': '1 0 0 0 1 1 1 1 1',
    'STATION_IN_NAME_STYLE': 'dataset_station',
    'STATION_IN_DEPTH': 0.0,
    'LON_WRAP_360': False,
}


VAR_LABELS = {
    'temp': 'Temperature (C)',
    'sal': 'Salinity',
    'depth': 'Depth (m)',
}


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


def _parse_excel_datetime(date_val, frac_val=None):
    base = _to_float(date_val)
    if base is None:
        return None
    if frac_val is not None:
        frac = _to_float(frac_val)
        if frac is not None and 0 <= frac < 1.0:
            base = base + frac
    base_dt = datetime(1899, 12, 30)
    return base_dt + timedelta(days=base)


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


def _safe_name(text):
    if not text:
        return 'unknown'
    name = str(text)
    name = name.replace(os.sep, '_')
    if os.altsep:
        name = name.replace(os.altsep, '_')
    name = re.sub(r'[^A-Za-z0-9_.-]+', '_', name)
    return name.strip('._') or 'unknown'


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


def find_lat_lon(row):
    lat = None
    lon = None
    for v in row:
        f = _to_float(v)
        if f is None:
            continue
        if lat is None and 36.0 <= f <= 40.0:
            lat = f
            continue
        if lon is None and 139.0 <= f <= 143.0:
            lon = f
    return lat, lon


def station_id_from_row(row):
    if not row:
        return None
    s = _cell_str(row[0])
    if not s:
        return None
    if re.match(r'^[A-Za-z]+\\d[\\w-]*$', s):
        return s
    if s.startswith('St.'):
        return s
    return None


def match_station_by_coords(lat, lon, coords, max_deg):
    if lat is None or lon is None or not coords:
        return None
    best = None
    best_dist = None
    for name, (clat, clon) in coords.items():
        if clat is None or clon is None:
            continue
        dlat = clat - lat
        dlon = clon - lon
        dist = dlat * dlat + dlon * dlon
        if best_dist is None or dist < best_dist:
            best = name
            best_dist = dist
    if best_dist is None:
        return None
    if (best_dist ** 0.5) > max_deg:
        return None
    return best


def parse_survey_file(path, dataset, coords):
    basename = os.path.basename(path)
    col_map = CONFIG['SURVEY_COLUMN_MAP'].get(basename)
    if not col_map:
        return []
    date_col = CONFIG['DATE_COL']
    rows = list(iter_xlsx_rows(path))
    records = []
    data_rows = []
    for row in rows:
        lat, lon = find_lat_lon(row)
        if lat is None or lon is None:
            continue
        data_rows.append((row, lat, lon))
    for idx, (row, lat, lon) in enumerate(data_rows):
        if idx + 1 >= len(data_rows):
            break
        next_row = data_rows[idx + 1][0]
        date_val = row[date_col] if len(row) > date_col else None
        time_frac = next_row[date_col] if len(next_row) > date_col else None
        dt = _parse_excel_datetime(date_val, time_frac)
        if dt is None:
            continue
        station_id = station_id_from_row(row)
        if not station_id:
            station_id = match_station_by_coords(lat, lon, coords, CONFIG['STATION_MATCH_MAX_DEG'])
        if not station_id:
            station_id = dataset
        temp = _to_float(row[col_map['temp']]) if col_map.get('temp') is not None and len(row) > col_map['temp'] else None
        sal = _to_float(row[col_map['sal']]) if col_map.get('sal') is not None and len(row) > col_map['sal'] else None
        depth = _to_float(row[col_map['depth']]) if col_map.get('depth') is not None and len(row) > col_map['depth'] else None
        records.append({
            'time': dt,
            'station_id': station_id,
            'lat': lat,
            'lon': lon,
            'temp': temp,
            'sal': sal,
            'depth': depth,
            'layer': 'surface',
        })
        temp_b = _to_float(next_row[col_map['temp']]) if col_map.get('temp') is not None and len(next_row) > col_map['temp'] else None
        sal_b = _to_float(next_row[col_map['sal']]) if col_map.get('sal') is not None and len(next_row) > col_map['sal'] else None
        depth_b = _to_float(next_row[col_map['depth']]) if col_map.get('depth') is not None and len(next_row) > col_map['depth'] else None
        records.append({
            'time': dt,
            'station_id': station_id,
            'lat': lat,
            'lon': lon,
            'temp': temp_b,
            'sal': sal_b,
            'depth': depth_b,
            'layer': 'bottom',
        })
    return records


def iter_survey_files(base_dir):
    files = []
    for root, _, names in os.walk(base_dir):
        for name in names:
            if not name.startswith('Survey') or not name.endswith('.xlsx'):
                continue
            files.append(os.path.join(root, name))
    return sorted(files)


def collect_records(base_dir, use_xml_coords=True):
    records = []
    files = CONFIG['SURVEY_FILES'] or iter_survey_files(base_dir)
    for path in files:
        dataset = os.path.basename(os.path.dirname(path))
        coords = {}
        if use_xml_coords:
            xml_path = None
            for name in os.listdir(os.path.dirname(path)):
                if name.endswith('_en.xml'):
                    xml_path = os.path.join(os.path.dirname(path), name)
                    break
            coords = load_station_coords(xml_path)
        recs = parse_survey_file(path, dataset, coords)
        for rec in recs:
            rec['dataset'] = dataset
            rec['source'] = os.path.basename(path)
        records.extend(recs)
    return records


def plot_timeseries(records, plot_vars, out_dir, max_plots=None):
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
        recs = sorted(recs, key=lambda r: (_plot_time(r['time'], use_utc), r.get('layer', '')))
        fig, axes = plt.subplots(len(plot_vars), 1, figsize=(8, 3 * len(plot_vars)), sharex=True)
        if len(plot_vars) == 1:
            axes = [axes]
        for ax, key in zip(axes, plot_vars):
            for layer, style in [('surface', '-'), ('bottom', '--')]:
                times = [
                    _plot_time(r['time'], use_utc)
                    for r in recs
                    if r.get(key) is not None and r.get('layer') == layer
                ]
                vals = [r.get(key) for r in recs if r.get(key) is not None and r.get('layer') == layer]
                if not times:
                    continue
                ax.plot(times, vals, style, lw=0.8, label=layer)
            ax.set_ylabel(VAR_LABELS.get(key, key))
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel(time_label)
        axes[0].legend(loc='best', fontsize=8)
        title = f'{dataset} {station}'
        fig.suptitle(title)
        fig.tight_layout()
        safe_dataset = _safe_name(dataset)
        safe_station = _safe_name(station)
        out_name = f'survey_{safe_dataset}_{safe_station}.png'
        out_path = os.path.join(out_dir, out_name)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        count += 1
    return count


def plot_ts_diagrams(records, out_dir, group_by='source', color_by='layer', max_plots=None):
    if plt is None:
        raise RuntimeError('matplotlib is not available for plotting')
    os.makedirs(out_dir, exist_ok=True)
    groups = {}
    for rec in records:
        temp = rec.get('temp')
        sal = rec.get('sal')
        if temp is None or sal is None:
            continue
        dataset = rec.get('dataset')
        source = rec.get('source')
        if group_by == 'dataset':
            key = (dataset, None)
        else:
            key = (dataset, source)
        groups.setdefault(key, []).append(rec)
    count = 0
    for (dataset, source), recs in sorted(groups.items()):
        if max_plots is not None and count >= max_plots:
            break
        fig, ax = plt.subplots(figsize=(5.5, 5))
        if color_by == 'depth':
            depths = [r.get('depth') for r in recs if r.get('depth') is not None]
            temps = [r.get('temp') for r in recs if r.get('depth') is not None]
            sals = [r.get('sal') for r in recs if r.get('depth') is not None]
            if depths:
                sc = ax.scatter(sals, temps, c=depths, s=16, cmap='viridis', edgecolors='none')
                cbar = fig.colorbar(sc, ax=ax)
                cbar.set_label('Depth (m)')
            else:
                ax.scatter([r.get('sal') for r in recs], [r.get('temp') for r in recs], s=16)
        else:
            for layer, style, color in [('surface', 'o', 'tab:blue'), ('bottom', 's', 'tab:orange')]:
                sals = [r.get('sal') for r in recs if r.get('layer') == layer]
                temps = [r.get('temp') for r in recs if r.get('layer') == layer]
                if not sals:
                    continue
                ax.scatter(sals, temps, s=18, marker=style, color=color, label=layer, alpha=0.8)
            ax.legend(loc='best', fontsize=8)
        ax.set_xlabel('Salinity')
        ax.set_ylabel('Temperature (C)')
        ax.grid(True, alpha=0.3)
        title = dataset or ''
        if source:
            title = f'{title} {os.path.splitext(source)[0]}'.strip()
        ax.set_title(title)
        fig.tight_layout()
        safe_dataset = _safe_name(dataset)
        safe_source = _safe_name(os.path.splitext(source or '')[0])
        if group_by == 'dataset':
            out_name = f'ts_{safe_dataset}.png'
        else:
            out_name = f'ts_{safe_dataset}_{safe_source}.png'
        out_path = os.path.join(out_dir, out_name)
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        count += 1
    return count


def _compute_extent(lons, lats, pad_deg):
    if not lons or not lats:
        return None
    lon_min = min(lons)
    lon_max = max(lons)
    lat_min = min(lats)
    lat_max = max(lats)
    pad_lon = max((lon_max - lon_min) * 0.05, pad_deg)
    pad_lat = max((lat_max - lat_min) * 0.05, pad_deg)
    return [lon_min - pad_lon, lon_max + pad_lon, lat_min - pad_lat, lat_max + pad_lat]


def _auto_figsize(extent, layer_mode):
    if extent is None:
        return (10, 5) if layer_mode == 'both' else (6, 5)
    lon_min, lon_max, lat_min, lat_max = extent
    lat_span = max(1e-6, lat_max - lat_min)
    mean_lat = 0.5 * (lat_min + lat_max)
    lon_span = max(1e-6, (lon_max - lon_min) * math.cos(math.radians(mean_lat)))
    aspect = max(0.45, min(2.5, lon_span / lat_span))
    panel_height = 5.0
    panel_width = panel_height * aspect
    if layer_mode == 'both':
        return (max(9.0, panel_width * 2 + 1.5), panel_height)
    return (max(5.5, panel_width + 1.0), panel_height)


def plot_spatial_maps(
    records,
    out_dir,
    map_vars,
    group_by='source',
    layer_mode='both',
    max_plots=None,
    cmap='viridis',
    extent=None,
    pad_deg=0.03,
    coast_res='10m',
    figsize=(10, 5),
):
    if plt is None:
        raise RuntimeError('matplotlib is not available for plotting')
    if ccrs is None or cfeature is None:
        raise RuntimeError('cartopy is required for map plotting')
    if layer_mode not in ('surface', 'bottom', 'both'):
        raise ValueError(f'Invalid layer_mode: {layer_mode}')
    os.makedirs(out_dir, exist_ok=True)
    groups = {}
    for rec in records:
        if rec.get('lat') is None or rec.get('lon') is None:
            continue
        dataset = rec.get('dataset')
        source = rec.get('source')
        if group_by == 'dataset':
            key = (dataset, None)
        else:
            key = (dataset, source)
        groups.setdefault(key, []).append(rec)
    count = 0
    for (dataset, source), recs in sorted(groups.items()):
        if max_plots is not None and count >= max_plots:
            break
        for var in map_vars:
            layers = ['surface', 'bottom'] if layer_mode == 'both' else [layer_mode]
            vals_all = [r.get(var) for r in recs if r.get(var) is not None and r.get('layer') in layers]
            if not vals_all:
                continue
            vmin = min(vals_all)
            vmax = max(vals_all)
            lons = [r.get('lon') for r in recs if r.get('lon') is not None]
            lats = [r.get('lat') for r in recs if r.get('lat') is not None]
            plot_extent = extent or _compute_extent(lons, lats, pad_deg)
            fig_size = figsize or _auto_figsize(plot_extent, layer_mode)
            if layer_mode == 'both':
                fig, axes = plt.subplots(
                    1,
                    2,
                    figsize=fig_size,
                    subplot_kw={'projection': ccrs.PlateCarree()},
                    constrained_layout=True,
                )
                if hasattr(axes, 'ravel'):
                    axes = axes.ravel().tolist()
                else:
                    axes = [axes]
            else:
                fig, ax = plt.subplots(
                    figsize=fig_size,
                    subplot_kw={'projection': ccrs.PlateCarree()},
                    constrained_layout=True,
                )
                axes = [ax]
            mappable = None
            for ax, layer in zip(axes, layers):
                ax.add_feature(cfeature.LAND, facecolor='0.9')
                ax.add_feature(cfeature.COASTLINE.with_scale(coast_res), linewidth=0.6)
                ax.add_feature(cfeature.BORDERS.with_scale(coast_res), linewidth=0.4, linestyle=':')
                if plot_extent is not None:
                    ax.set_extent(plot_extent, crs=ccrs.PlateCarree())
                vals = [r.get(var) for r in recs if r.get('layer') == layer and r.get(var) is not None]
                lons_layer = [r.get('lon') for r in recs if r.get('layer') == layer and r.get(var) is not None]
                lats_layer = [r.get('lat') for r in recs if r.get('layer') == layer and r.get(var) is not None]
                if not vals:
                    ax.set_title(f'{layer} (no data)')
                    continue
                mappable = ax.scatter(
                    lons_layer,
                    lats_layer,
                    c=vals,
                    s=28,
                    cmap=cmap,
                    vmin=vmin,
                    vmax=vmax,
                    transform=ccrs.PlateCarree(),
                    edgecolors='k',
                    linewidths=0.3,
                )
                ax.set_title(layer)
            title = dataset or ''
            if source:
                title = f'{title} {os.path.splitext(source)[0]}'.strip()
            fig.suptitle(f'{title} {VAR_LABELS.get(var, var)}')
            if mappable is not None:
                if len(axes) > 1:
                    fig.colorbar(
                        mappable,
                        ax=axes,
                        orientation='horizontal',
                        shrink=0.85,
                        pad=0.08,
                        label=VAR_LABELS.get(var, var),
                    )
                else:
                    fig.colorbar(mappable, ax=axes, shrink=0.85, pad=0.02, label=VAR_LABELS.get(var, var))
            safe_dataset = _safe_name(dataset)
            safe_source = _safe_name(os.path.splitext(source or '')[0])
            safe_var = _safe_name(var)
            if group_by == 'dataset':
                out_name = f'map_{safe_dataset}_{safe_var}_{layer_mode}.png'
            else:
                out_name = f'map_{safe_dataset}_{safe_source}_{safe_var}_{layer_mode}.png'
            out_path = os.path.join(out_dir, out_name)
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            count += 1
            if max_plots is not None and count >= max_plots:
                break
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
        'layer': np.array([r.get('layer', '') for r in records], dtype='U'),
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
    parser = argparse.ArgumentParser(description='Process Sendai D1 Survey_outline xlsx files.')
    parser.add_argument('--base', default=CONFIG['BASE_DIR'], help='Base D1 directory')
    parser.add_argument('--plot-dir', default=None, help='Output plot directory')
    parser.add_argument('--no-plot', action='store_true', help='Disable plotting')
    parser.add_argument('--save-npz', action='store_true', help='Save npz output')
    parser.add_argument('--npz-path', default=None, help='Path to write npz')
    parser.add_argument('--max-plots', type=int, default=None, help='Limit number of plots')
    args = parser.parse_args()

    base_dir = args.base or CONFIG['BASE_DIR']
    plot_vars = CONFIG['PLOT_VARS']
    save_vars = CONFIG['SAVE_VARS']
    use_xml_coords = CONFIG['USE_XML_COORDS']
    skip_plot = CONFIG['SKIP_PLOT'] or args.no_plot
    max_plots = args.max_plots if args.max_plots is not None else CONFIG['MAX_PLOTS']
    plot_dir = args.plot_dir or CONFIG['PLOT_DIR'] or os.path.join(base_dir, 'plots_survey')
    plot_timeseries_flag = CONFIG['PLOT_TIMESERIES']
    plot_ts_flag = CONFIG['PLOT_TS']
    ts_dir = CONFIG['TS_DIR'] or os.path.join(base_dir, 'plots_ts')
    ts_group_by = CONFIG['TS_GROUP_BY']
    ts_color_by = CONFIG['TS_COLOR_BY']
    ts_max_plots = CONFIG['TS_MAX_PLOTS']
    plot_maps_flag = CONFIG['PLOT_MAPS']
    map_dir = CONFIG['MAP_DIR'] or os.path.join(base_dir, 'plots_map')
    map_vars = CONFIG['MAP_VARS']
    map_group_by = CONFIG['MAP_GROUP_BY']
    map_layer = CONFIG['MAP_LAYER']
    map_max_plots = CONFIG['MAP_MAX_PLOTS']
    map_cmap = CONFIG['MAP_CMAP']
    map_extent = CONFIG['MAP_EXTENT']
    map_pad_deg = CONFIG['MAP_PAD_DEG']
    map_coast_res = CONFIG['MAP_COAST_RES']
    map_figsize = CONFIG['MAP_FIGSIZE']
    save_npz_flag = args.save_npz or CONFIG['SAVE_NPZ']
    npz_path = args.npz_path or CONFIG['NPZ_PATH'] or os.path.join(base_dir, 'sendai_d1_survey.npz')
    write_station_in_flag = CONFIG['WRITE_STATION_IN']
    station_in_path = CONFIG['STATION_IN_PATH'] or os.path.join(base_dir, 'station.in')
    station_flags = CONFIG['STATION_IN_FLAGS']
    station_name_style = CONFIG['STATION_IN_NAME_STYLE']
    station_depth = CONFIG['STATION_IN_DEPTH']
    wrap_360 = CONFIG['LON_WRAP_360']

    records = collect_records(base_dir, use_xml_coords=use_xml_coords)
    if not records:
        print('No survey records found.')
        return 1
    print(f'Loaded {len(records)} survey samples from {base_dir}')

    if not skip_plot and plot_timeseries_flag:
        count = plot_timeseries(records, plot_vars, plot_dir, max_plots=max_plots)
        print(f'Saved {count} survey plots to {plot_dir}')
    if not skip_plot and plot_ts_flag:
        count = plot_ts_diagrams(records, ts_dir, group_by=ts_group_by, color_by=ts_color_by, max_plots=ts_max_plots)
        print(f'Saved {count} T-S plots to {ts_dir}')
    if not skip_plot and plot_maps_flag:
        count = plot_spatial_maps(
            records,
            map_dir,
            map_vars,
            group_by=map_group_by,
            layer_mode=map_layer,
            max_plots=map_max_plots,
            cmap=map_cmap,
            extent=map_extent,
            pad_deg=map_pad_deg,
            coast_res=map_coast_res,
            figsize=map_figsize,
        )
        print(f'Saved {count} map plots to {map_dir}')

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
