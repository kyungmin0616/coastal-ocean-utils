#!/usr/bin/env python3
import json
import os
import sys
import time

import matplotlib
matplotlib.use('Agg')
from pylib import *
import numpy as np
import pandas as pd
from mpi4py import MPI
from scipy import interpolate, signal

close("all")

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
StartT_model = [
    datenum(2021, 6, 1), datenum(2021, 6, 1), datenum(2021, 6, 1),
    datenum(2021, 6, 1), datenum(2021, 6, 1), datenum(2021, 6, 1),
    datenum(2021, 6, 1), datenum(2021, 6, 1), datenum(2021, 6, 1),
    datenum(2021, 6, 1), datenum(2021, 6, 1), datenum(2021, 6, 1),
    datenum(2021, 6, 1), datenum(2021, 6, 1), datenum(2021, 6, 1),
    datenum(2021, 6, 1), datenum(2021, 6, 1), datenum(2021, 6, 1)
]  # plot start time, model start time

ontr = 0
mntr = 0
omean = 1  # 1: mean obs data
ofreq = 'H'  # if omean=1, define frequency. T:minutely, H:hourly, D:daily, W:weekly, M:monthly
mmean = 1  # 1: mean model data
mfreq = 'H'  # if mmean=1, define frequency. T:minutely, H:hourly, D:daily, W:weekly, M:monthly

compare_velocity_mode = 'pcd'  # options: 'magnitude', 'pcd'

# choose which variables to compare
compare_variables = ['WL', 'VEL', 'TEMP', 'SALT']
#compare_variables = ['WL']
# y-axis limits (set to None to use automatic scaling)
axis_limits = {
    'WL': [-1, 1],
    'VEL_magnitude': [0, 1.4],
    'VEL_pcd': [-1.5, 1.5],
    'TEMP': [14, 30],
    'SALT': [0, 28],
}

# Butterworth low-pass filter settings for non-tidal residuals
cutoff_period_hours = 34  # set to None to disable filtering when ontr/mntr are enabled
butterworth_order = 4
filter_pad_days = 60

runs = ['npz/RUN12l_schism.npz']
lw = 2

tags = ['RUN12l']
bpfile = './stations/stationExp'
stts = [datenum('2021-7-1'), datenum('2021-6-24'), datenum('2021-7-1'), datenum('2021-7-1')]
edts = [datenum('2021-10-1'), datenum('2021-7-18'), datenum('2021-10-1'), datenum('2021-10-1')]
sname = os.path.expanduser('images/all_RUN12l_mpi')

_obs_dir = os.getenv('SCHISM_OBS_DIR')
if _obs_dir:
    _obs_dir = os.path.expanduser(_obs_dir)
else:
    _obs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'npz')

obs_paths = {
    'WL': os.path.join(_obs_dir, 'twl-ufs-2021.npz'),
    'VEL': os.path.join(_obs_dir, 'current-ufs-2021.npz'),
    'TEMP': os.path.join(_obs_dir, 'temp-ufs.npz'),
    'SALT': os.path.join(_obs_dir, 'salt-ufs.npz'),
}

_CFG_ENV = 'COMP_SCHISM_CONFIG'
_cfg_raw = os.getenv(_CFG_ENV)
if _cfg_raw:
    try:
        _cfg = json.loads(_cfg_raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f'Invalid JSON in {_CFG_ENV}: {exc}') from exc

    def _coerce_times(values, fallback):
        if values is None:
            return fallback
        coerced = []
        for val in values:
            if isinstance(val, (float, int)):
                coerced.append(float(val))
            elif val:
                coerced.append(datenum(val))
        return coerced or fallback

    runs = _cfg.get('runs', runs)
    tags = _cfg.get('tags', tags)
    bpfile = _cfg.get('bpfile', bpfile)
    stts = _coerce_times(_cfg.get('stts'), stts)
    edts = _coerce_times(_cfg.get('edts'), edts)
    sname = os.path.expanduser(_cfg.get('sname', sname))

    lw = _cfg.get('lw', lw)

    if 'obs_paths' in _cfg:
        obs_paths.update({k: v for k, v in _cfg['obs_paths'].items() if v})


#colors = 'kgbcmo'
colors = [
    "black", "blue", "green", "red", "cyan", "magenta", "yellow",
    "orange", "purple", "brown", "pink", "gray", "olive", "teal", "navy"
]
lstyle = ['-'] * len(colors)
markers = ['None'] * len(colors)

# --------------------------------------------------------------------------
# MPI / job submission setup (adapted from run_mpi_template.py)
# --------------------------------------------------------------------------
walltime = '00:30:00'
nnode = 1
ppn = 8

add_var(
    ['ibatch', 'qnode', 'qname', 'account', 'reservation', 'jname', 'scrout'],
    [0, None, None, None, None, 'comp_obs_mpi', 'screen.out'],
    locals(),
)

mpi_env_flags = (
    'OMPI_COMM_WORLD_SIZE',
    'OMPI_COMM_WORLD_RANK',
    'PMI_SIZE',
    'PMI_RANK',
    'PMIX_RANK',
    'MPI_LOCALNRANKS',
    'SLURM_PROCID',
)
mpi_env = any(name in os.environ for name in mpi_env_flags)
auto_enabled_mpi = False
if mpi_env and ibatch == 0:
    ibatch = 1
    auto_enabled_mpi = True

current_dir = os.path.abspath(os.path.curdir)
if ibatch == 0:
    os.environ['job_on_node'] = '1'
    os.environ['bdir'] = current_dir
elif mpi_env:
    os.environ.setdefault('job_on_node', '1')
    os.environ.setdefault('bdir', current_dir)

if ibatch == 1 and os.getenv('job_on_node') is None:
    if os.getenv('param') is None:
        fmt = 0
        bcode = sys.argv[0]
        os.environ['qnode'] = get_qnode(qnode)
    else:
        fmt = 1
        bdir, bcode = os.getenv('param').split()
        os.chdir(bdir)
    scode = get_hpc_command(
        bcode,
        os.path.abspath(os.path.curdir),
        jname,
        qnode,
        nnode,
        ppn,
        walltime,
        scrout,
        fmt,
        'param',
        qname,
        account,
        reservation,
    )
    print(scode)
    os.system(scode)
    os._exit(0)

bdir = os.getenv('bdir', os.path.abspath(os.path.curdir))
os.chdir(bdir)

if ibatch == 0:
    nproc = 1
    myrank = 0
    comm = None
else:
    comm = MPI.COMM_WORLD
    nproc = comm.Get_size()
    myrank = comm.Get_rank()

if myrank == 0:
    t0 = time.time()

print(f'myrank={myrank}, nproc={nproc}, host={os.getenv("HOST") or "local"}')
if auto_enabled_mpi and myrank == 0:
    print('Detected MPI launcher; enabling ibatch=1 for parallel execution')
sys.stdout.flush()

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def ensure_dir(path):
    if not fexist(path):
        print(f'making folder {path}')
        os.makedirs(path, exist_ok=True)


def resample_timeseries(times, values, freq):
    if len(times) == 0:
        return times, values
    stamps = [num2date(t).strftime('%Y-%m-%d %H:%M:%S') for t in times]
    df = pd.DataFrame(data=values, index=pd.to_datetime(stamps))
    grouped = df.groupby(df.index.floor(freq)).mean()
    return array(datenum(grouped.index.astype(str))), squeeze(grouped.values)


def resolve_obs_path(var_key):
    path = obs_paths.get(var_key)
    if path is None:
        raise KeyError(f'Observation path not configured for {var_key}')
    resolved = os.path.abspath(os.path.expanduser(path))
    if not fexist(resolved):
        raise FileNotFoundError(f'Observation file not found for {var_key}: {resolved}')
    return resolved


def interpolate_model_to_obs(model_times, model_values, obs_times):
    if len(model_times) < 2 or len(obs_times) == 0:
        return None
    try:
        return interpolate.interp1d(
            model_times,
            model_values,
            kind='linear',
            fill_value='extrapolate'
        )(obs_times)
    except ValueError:
        return None


def apply_butterworth_lowpass(values, times, cutoff_hours, order=4):
    """Apply zero-phase Butterworth low-pass filter; assumes times in datenum."""
    if cutoff_hours is None:
        return values
    if len(values) < max(order * 3, 8):
        return values
    times = np.asarray(times, float)
    values = np.asarray(values, float)

    dt = np.diff(times)
    if not len(dt):
        return values
    dt_hours = np.median(dt) * 24.0
    if not np.isfinite(dt_hours) or dt_hours <= 0:
        return values

    cutoff_frequency = 1.0 / cutoff_hours  # cycles per hour
    nyquist_frequency = 0.5 / dt_hours
    if nyquist_frequency <= 0:
        return values

    normalized_cutoff = cutoff_frequency / nyquist_frequency
    if not np.isfinite(normalized_cutoff) or normalized_cutoff <= 0:
        return values
    if normalized_cutoff >= 1:
        normalized_cutoff = min(0.999, normalized_cutoff)
        if normalized_cutoff >= 1:
            return values

    try:
        b, a = signal.butter(order, normalized_cutoff, btype='low')
        return signal.filtfilt(b, a, values)
    except ValueError:
        return values


use_pcd = compare_velocity_mode.lower() == 'pcd'

SMALL_SIZE = 14
MEDIUM_SIZE = 14
BIGGER_SIZE = 14
rc('font', size=SMALL_SIZE)
rc('axes', titlesize=SMALL_SIZE)
rc('xtick', labelsize=SMALL_SIZE)
rc('ytick', labelsize=SMALL_SIZE)
rc('legend', fontsize=SMALL_SIZE)
rc('axes', labelsize=MEDIUM_SIZE)
rc('figure', titlesize=BIGGER_SIZE)

ensure_dir(sname)

# --------------------------------------------------------------------------
# Station metadata
# --------------------------------------------------------------------------
bp = read_schism_bpfile(bpfile)
bp.staid = array([entry.split(' ')[0] for entry in bp.station])
bp.var = array([entry.split(' ')[1] for entry in bp.station])

# --------------------------------------------------------------------------
# Model data
# --------------------------------------------------------------------------
Model = []
for m, run in enumerate(runs):
    if run.endswith('staout'):
        dataset = npz_data()
        data = loadtxt(run + '_1')
        dataset.time = data[:, 0] / 86400 + StartT_model[m]
        dataset.dth = (data[:, 0][1] - data[:, 0][0]) / 3600
        dataset.elev = data[:, 1:].transpose()
        data = loadtxt(run + '_5')
        dataset.temp = data[:, 1:].transpose()
        data = loadtxt(run + '_6')
        dataset.salt = data[:, 1:].transpose()
        dataset.hvel = []
        data = loadtxt(run + '_7')
        dataset.hvel.append(data[:, 1:].transpose())
        data = loadtxt(run + '_8')
        dataset.hvel.append(data[:, 1:].transpose())
        dataset.hvel = array(dataset.hvel).transpose(1, 2, 0)
    elif run.endswith('.npz'):
        dataset = loadz(run)
        dataset.time = dataset.time + StartT_model[m]
        dataset.dth = (dataset.time[1] - dataset.time[0]) * 24
    else:
        raise ValueError(f'Unsupported run data type for {run}')
    Model.append(dataset)

del dataset

# --------------------------------------------------------------------------
# Initialize stats containers
# --------------------------------------------------------------------------
base_fields = {
    'WL': ['ms', 'lon', 'lat', 'name', 'R', 'RMSE', 'Bias'],
    'VEL': ['ms', 'lon', 'lat', 'name', 'R', 'RMSE', 'Bias'],
    'TEMP': ['ms', 'lon', 'lat', 'name', 'R', 'RMSE', 'Bias'],
    'SALT': ['ms', 'lon', 'lat', 'name', 'R', 'RMSE', 'Bias'],
}

if use_pcd:
    base_fields['VEL'].extend(['theta', 'mean_proj'])

run_stats = {}
for tag in tags:
    stats = zdata()
    for var_name, fields in base_fields.items():
        for field in fields:
            setattr(stats, f'{var_name}_{field}', [])
    run_stats[tag] = stats

# --------------------------------------------------------------------------
# Comparison loop
# --------------------------------------------------------------------------
wlc = cc = tc = sc = 0
pcd_obs_means = []
pcd_obs_angles = []
pcd_depths = []
pcd_station_names = []

for idx in arange(bp.nsta):
    figure(1, figsize=[15, 9])
    clf()
    var = bp.var[idx]
    station_id = bp.staid[idx]
    if var not in compare_variables:
        continue
    if idx % nproc != myrank:
        continue
    print(f'[rank {myrank}/{nproc}] plotting {bp.var[idx]} at {bp.staid[idx]}')
    sys.stdout.flush()

    if var == 'WL':
        yname = 'Water level (m)'
        ym = axis_limits.get('WL', [-1.5, 1.5])
        stan = station_id
        wlc += 1
        obs = loadz(resolve_obs_path('WL'))
        stt, edt = stts[0], edts[0]
        buffer_days = filter_pad_days if (ontr == 1 or mntr == 1) else 0
        filter_start = stt - buffer_days
        filter_end = edt + buffer_days
        selector = (
            (obs.station == station_id)
            * (obs.time >= filter_start)
            * (obs.time <= filter_end)
        )
        oti = obs.time[selector]
        oyi = obs.elev[selector]
    elif var == 'VEL':
        stan = station_id.split('_')[1] + '_' + station_id.split('_')[0]
        cc += 1
        obs = loadz(resolve_obs_path('VEL'))
        sid = station_id.split('_')[1]
        if sid == 'n03020':
            stt = datenum(2021, 7, 3)
            edt = datenum(2021, 10, 1)
        elif sid == 'n06010':
            stt = stts[1]
            edt = edts[1]
        elif sid == 'n07010':
            stt = datenum(2021, 7, 1)
            edt = datenum(2021, 10, 1)
        else:
            stt, edt = stts[1], edts[1]
        filter_start, filter_end = stt, edt
        selector = (obs.station == station_id) * (obs.time > stt) * (obs.time < edt)
        oti = obs.time[selector]
        osi = obs.spd[selector]
        odi = obs.dir[selector]
        if use_pcd:
            yname = 'Velocity along PCD (m/s)'
            ym = axis_limits.get('VEL_pcd', [-1.5, 1.5])
            u = osi * sin(deg2rad(odi))
            v = osi * cos(deg2rad(odi))
            u_p = u - nanmean(u)
            v_p = v - nanmean(v)
            mean_u2 = nanmean(u_p ** 2)
            mean_v2 = nanmean(v_p ** 2)
            mean_uv = nanmean(u_p * v_p)
            theta_o = 0.5 * arctan2(2 * mean_uv, (mean_u2 - mean_v2))
            theta_obs_deg = rad2deg(theta_o)
            oyi = u * cos(theta_o) + v * sin(theta_o)
            pcd_obs_means.append(nanmean(oyi))
            pcd_obs_angles.append(theta_obs_deg)
            try:
                depth_value = float(stan.split('_')[1])
            except (IndexError, ValueError):
                depth_value = float('nan')
            pcd_depths.append(depth_value)
            pcd_station_names.append(stan)
        else:
            yname = 'Current speed (m/s)'
            ym = axis_limits.get('VEL_magnitude', [0, 1.4])
            theta_obs_deg = None
            oyi = osi
    elif var == 'TEMP':
        yname = 'Water temperature (deg)'
        ym = axis_limits.get('TEMP', [14, 30])
        stan = station_id
        tc += 1
        obs = loadz(resolve_obs_path('TEMP'))
        stt, edt = stts[2], edts[2]
        filter_start, filter_end = stt, edt
        selector = (obs.station == station_id) * (obs.time > stt) * (obs.time < edt)
        oti = obs.time[selector]
        oyi = obs.temp[selector]
    elif var == 'SALT':
        yname = 'Salinity (PSU)'
        ym = axis_limits.get('SALT', [0, 28])
        stan = station_id
        sc += 1
        obs = loadz(resolve_obs_path('SALT'))
        stt, edt = stts[3], edts[3]
        filter_start, filter_end = stt, edt
        selector = (obs.station == station_id) * (obs.time > stt) * (obs.time < edt)
        oti = obs.time[selector]
        oyi = obs.salt[selector]
    else:
        print(f'Wrong variable name in {idx + 1} line. Check station.in')
        continue

    if len(oyi) == 0:
        print(f'No data at {station_id}')
        continue

    if len(oti) > 100:
        ts = find_continuous_sections(oti, 1.0)
        eoti = array([section[-1] + 1 / 24 for section in ts.sections])
        eoyi = ones(len(eoti)) * nan
        oti = r_[oti, eoti]
        oyi = r_[oyi, eoyi]
        sind = argsort(oti)
        oti = oti[sind]
        oyi = oyi[sind]

    if omean == 1:
        oti, oyi = resample_timeseries(oti, oyi, ofreq)

    if var == 'WL' and ontr == 1:
        valid_obs = ~isnan(oyi)
        oti = oti[valid_obs]
        oyi = oyi[valid_obs]
        if len(oyi) == 0:
            continue
        oyi = apply_butterworth_lowpass(oyi, oti, cutoff_period_hours, butterworth_order)

    plot_mask = (oti >= stt) & (oti <= edt)
    oti = oti[plot_mask]
    oyi = oyi[plot_mask]
    if len(oyi) == 0:
        print(f'No observational data in target window after filtering at {station_id}')
        continue

    ax = gca()
    ax.plot(oti, oyi, linestyle=lstyle[0], color='r', marker=markers[0], ms=3, alpha=0.85, lw=lw)

    for nn, run in enumerate(runs):
        tag = tags[nn]
        stats_container = run_stats[tag]
        mti = Model[nn].time

        if var == 'WL':
            myi = Model[nn].elev[idx, :]
            sno = wlc
            if (ontr == 1 or mntr == 1):
                filter_mask = (mti >= filter_start) & (mti <= filter_end)
                mti = mti[filter_mask]
                myi = myi[filter_mask]
        elif var == 'VEL':
            u_model = Model[nn].hvel[idx, :, 0]
            v_model = Model[nn].hvel[idx, :, 1]
            if use_pcd:
                u_p = u_model - nanmean(u_model)
                v_p = v_model - nanmean(v_model)
                mean_u2 = nanmean(u_p ** 2)
                mean_v2 = nanmean(v_p ** 2)
                mean_uv = nanmean(u_p * v_p)
                theta_m = 0.5 * arctan2(2 * mean_uv, (mean_u2 - mean_v2))
                theta_mod_deg = rad2deg(theta_m)
                myi = u_model * cos(theta_m) + v_model * sin(theta_m)
            else:
                theta_mod_deg = None
                myi = sqrt(u_model ** 2 + v_model ** 2)
            sno = cc
        elif var == 'TEMP':
            myi = Model[nn].temp[idx, :]
            sno = tc
        elif var == 'SALT':
            myi = Model[nn].salt[idx, :]
            sno = sc
        else:
            continue

        if mmean == 1:
            mti, myi = resample_timeseries(mti, myi, mfreq)

        if var == 'WL' and mntr == 1:
            valid_model = ~isnan(myi)
            mti = mti[valid_model]
            myi = myi[valid_model]
            if len(myi) == 0:
                continue
            myi = apply_butterworth_lowpass(myi, mti, cutoff_period_hours, butterworth_order)

        plot_mask_model = (mti >= stt) & (mti <= edt)
        mti = mti[plot_mask_model]
        myi = myi[plot_mask_model]
        if len(mti) == 0:
            continue

        valid_obs = ~isnan(oyi)
        oti_valid = oti[valid_obs]
        oyi_valid = oyi[valid_obs]
        valid_model = ~isnan(myi)
        mti_valid = mti[valid_model]
        myi_valid = myi[valid_model]

        if len(mti_valid) < 2:
            continue
        range_mask = (oti_valid >= mti_valid.min()) * (oti_valid <= mti_valid.max())
        otii = oti_valid[range_mask]
        if len(otii) == 0:
            continue
        oyii = oyi_valid[range_mask]
        myii = interpolate_model_to_obs(mti_valid, myi_valid, otii)
        if myii is None:
            continue

        st = get_stat(myii, oyii)

        prefix = f'{var}_'
        getattr(stats_container, prefix + 'ms').append(st.ms)
        getattr(stats_container, prefix + 'lon').append(bp.x[idx])
        getattr(stats_container, prefix + 'lat').append(bp.y[idx])
        getattr(stats_container, prefix + 'name').append(stan)
        getattr(stats_container, prefix + 'R').append(st.R)
        getattr(stats_container, prefix + 'RMSE').append(st.RMSD)
        getattr(stats_container, prefix + 'Bias').append(st.ME)

        if var == 'VEL' and use_pcd:
            getattr(stats_container, 'VEL_theta').append(theta_mod_deg)
            getattr(stats_container, 'VEL_mean_proj').append(nanmean(myii))

        ax.plot(
            mti,
            myi,
            linestyle=lstyle[nn],
            color=colors[nn],
            marker=markers[nn],
            ms=3,
            alpha=0.85,
            lw=lw
        )

        xts, xls = get_xtick(fmt=2, xts=linspace(stt, edt, 5), str='%b-%d')
        ax.set_xticks(xts)
        ax.set_xticklabels(xls)
        ax.set_xlim([stt, edt])
        if ym is not None:
            ax.set_ylim(ym)

        if var == 'VEL' and use_pcd:
            text_str = (
                f'No.{sno} ({stan}-{var}), {tag}---> R: {st.R:.2f}, RMSE: {st.RMSD:.2f}, '
                f'Bias: {st.ME:.2f}, theta_obs: {theta_obs_deg:.2f}, theta_mod: {theta_mod_deg:.2f}'
            )
        else:
            text_str = (
                f'No.{sno} ({stan}-{var}), {tag}---> R: {st.R:.2f}, RMSE: {st.RMSD:.2f}, Bias: {st.ME:.2f}'
            )

        ax.text(
            ax.get_xlim()[0] + 0.00 * diff(ax.get_xlim()),
            ax.get_ylim()[0] + (1.1 + nn * 0.1) * diff(ax.get_ylim()),
            text_str,
            color='k',
            fontweight='bold'
        )

    legend(['Obs.', *tags])
    xlabel('Date (2021)')
    ylabel(yname)
    ax.xaxis.grid('on')
    ax.yaxis.grid('on')
    gcf().tight_layout()

    savefig(f'{sname}/{var}_station_{stan}.png', bbox_inches='tight')
    close()

local_stats_payload = {}
for tag in tags:
    stats_container = run_stats[tag]
    payload = {}
    for var_name, fields in base_fields.items():
        for field in fields:
            attr = f'{var_name}_{field}'
            payload[attr] = list(getattr(stats_container, attr))
    local_stats_payload[tag] = payload

local_pcd_payload = {
    'means': list(pcd_obs_means),
    'angles': list(pcd_obs_angles),
    'depths': list(pcd_depths),
    'names': list(pcd_station_names),
}

if comm is not None:
    gathered_stats = comm.gather(local_stats_payload, root=0)
    gathered_pcd = comm.gather(local_pcd_payload, root=0)
else:
    gathered_stats = [local_stats_payload]
    gathered_pcd = [local_pcd_payload]

if myrank == 0:
    merged_stats = {
        tag: {f'{var_name}_{field}': [] for var_name, fields in base_fields.items() for field in fields}
        for tag in tags
    }
    for payload in gathered_stats:
        for tag, attr_map in payload.items():
            for attr, values in attr_map.items():
                merged_stats[tag][attr].extend(values)

    for tag in tags:
        stats_container = run_stats[tag]
        for var_name, fields in base_fields.items():
            for field in fields:
                attr = f'{var_name}_{field}'
                setattr(stats_container, attr, array(merged_stats[tag][attr]))

    pcd_obs_means = []
    pcd_obs_angles = []
    pcd_depths = []
    pcd_station_names = []
    for payload in gathered_pcd:
        pcd_obs_means.extend(payload['means'])
        pcd_obs_angles.extend(payload['angles'])
        pcd_depths.extend(payload['depths'])
        pcd_station_names.extend(payload['names'])

if myrank == 0:
    for tag in tags:
        stats_container = run_stats[tag]
        print(f'---- {tag} mean stat ----')
        if len(stats_container.WL_R):
            print(
                'WL-->',
                'R:', round(stats_container.WL_R.mean(), 2), '/',
                'RMSE:', round(stats_container.WL_RMSE.mean(), 2), '/',
                'Bias:', round(stats_container.WL_Bias.mean(), 2)
            )
        if len(stats_container.VEL_R):
            msg = (
                'VEL-->',
                'R:', round(stats_container.VEL_R.mean(), 2), '/',
                'RMSE:', round(stats_container.VEL_RMSE.mean(), 2), '/',
                'Bias:', round(stats_container.VEL_Bias.mean(), 2)
            )
            print(*msg, end='')
            if use_pcd and len(stats_container.VEL_theta):
                print('/', 'PCD:', round(stats_container.VEL_theta.mean(), 2))
            else:
                print()
        if len(stats_container.TEMP_R):
            print(
                'TEMP-->',
                'R:', round(stats_container.TEMP_R.mean(), 2), '/',
                'RMSE:', round(stats_container.TEMP_RMSE.mean(), 2), '/',
                'Bias:', round(stats_container.TEMP_Bias.mean(), 2)
            )
        if len(stats_container.SALT_R):
            print(
                'SALT-->',
                'R:', round(stats_container.SALT_R.mean(), 2), '/',
                'RMSE:', round(stats_container.SALT_RMSE.mean(), 2), '/',
                'Bias:', round(stats_container.SALT_Bias.mean(), 2)
            )

        savez(f'{tag}_stats', stats_container)

if myrank == 0 and use_pcd and len(pcd_obs_means):
    close()
    figure(2, figsize=[12, 7])
    clf()
    obs_records = [
        (name, depth, mean)
        for name, depth, mean in zip(pcd_station_names, pcd_depths, pcd_obs_means)
        if np.isfinite(depth) and np.isfinite(mean)
    ]
    if not obs_records:
        print('No valid PCD observations available for velocity profile plot')
    else:
        obs_records.sort(key=lambda rec: rec[1])
        depth_arr = array([rec[1] for rec in obs_records])
        obs_mean_arr = array([rec[2] for rec in obs_records])
        plot(obs_mean_arr, depth_arr, 'r')
        mean_obs_theta = mean(array(pcd_obs_angles)) if len(pcd_obs_angles) else nan
        for nn, tag in enumerate(tags):
            stats_container = run_stats[tag]
            if not len(stats_container.VEL_mean_proj) or not len(stats_container.VEL_name):
                continue
            model_lookup = {name: value for name, value in zip(stats_container.VEL_name, stats_container.VEL_mean_proj)}
            theta_lookup = {name: value for name, value in zip(stats_container.VEL_name, stats_container.VEL_theta)} if len(stats_container.VEL_theta) else {}
            model_series = []
            obs_series = []
            depth_series = []
            theta_series = []
            for name, depth, obs_mean in obs_records:
                model_val = model_lookup.get(name)
                if model_val is None or not np.isfinite(model_val):
                    continue
                model_series.append(model_val)
                obs_series.append(obs_mean)
                depth_series.append(depth)
                theta_val = theta_lookup.get(name)
                if theta_val is not None and np.isfinite(theta_val):
                    theta_series.append(theta_val)
            if len(model_series) < 2:
                continue
            model_arr = array(model_series)
            depth_used = array(depth_series)
            obs_used = array(obs_series)
            plot(
                model_arr,
                depth_used,
                linestyle=lstyle[nn],
                color=colors[nn],
                marker=markers[nn],
                ms=3,
                alpha=0.85,
                lw=lw
            )
            st = get_stat(model_arr, obs_used)
            theta_mod_mean = nanmean(array(theta_series)) if len(theta_series) else nan
            text(
                xlim()[0] + 0.00 * diff(xlim()),
                ylim()[0] + (1.1 + nn * 0.1) * diff(ylim()),
                f'{tag}---> R: {st.R:.2f}, RMSE: {st.RMSD:.2f}, Bias: {st.ME:.2f}, '
                f'theta_obs: {mean_obs_theta:.2f}, theta_mod: {theta_mod_mean:.2f}',
                color='k',
                fontweight='bold'
            )
        if len(depth_arr):
            ylim([nanmin(depth_arr), nanmax(depth_arr)])
        gca().invert_yaxis()
        tight_layout()
        legend(['Obs.', *tags])
        savefig(f'{sname}/v_profile.png', bbox_inches='tight')

if comm is not None:
    comm.Barrier()

if myrank == 0:
    dt = time.time() - t0
    print(f'total time used: {dt} s')
    sys.stdout.flush()

if ibatch == 1:
    sys.exit(0) if qnode in ['bora', 'levante'] else os._exit(0)
else:
    sys.exit(0)
