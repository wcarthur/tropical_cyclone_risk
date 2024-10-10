"""
Plot lifetime maximum intensity

@author: Craig Arthur
"""
import warnings
import sys
import os
import glob
import xarray as xr
import numpy as np
import datetime
import seaborn as sns

import matplotlib.pyplot as plt

sns.set_context("notebook")

warnings.filterwarnings('ignore', category=RuntimeWarning)

cwd = os.path.dirname(os.path.realpath("."))
parent = os.path.dirname(cwd)
sys.path.append(cwd)

import namelist

def plotLMI():
    exp_name = namelist.exp_name
    startdate = f'{namelist.start_year}{namelist.start_month:02d}'
    enddate = f'{namelist.end_year}{namelist.end_month:02d}'

    filename = f"tracks_*_{namelist.exp_prefix}_{startdate}_{enddate}*.nc"
    fn_tracks = sorted(glob.glob(f"{namelist.base_directory}/{exp_name}/{filename}"))
    ds = xr.open_mfdataset(fn_tracks, concat_dim="n_trk", combine="nested",
                           data_vars="minimal", drop_variables="seeds_per_month")
    drop_vars = ["lon_trks", "lat_trks",
                 "u250_trks", "v250_trks",
                 "u850_trks", "v850_trks",
                 "v_trks", "m_trks", "vmax_trks", "rmax_trks", "r34_trks",
                 "tc_month", "tc_years", "tc_basins"]
    ds_seeds = xr.open_mfdataset(fn_tracks, concat_dim="year", combine="nested",
                                 data_vars="minimal", drop_variables=drop_vars)
    yearS = namelist.start_year
    yearE = namelist.end_year
    n_sim = len(fn_tracks)
    ntrks_per_year = namelist.tracks_per_year
    ntrks_per_sim = ntrks_per_year * (yearE - yearS + 1)
    seeds_per_month_basin = ds_seeds['seeds_per_month']
    seeds_per_month = np.nansum(seeds_per_month_basin, axis=1)

    vmax = ds['vmax_trks'].load().data
    # ds['lon_trks'][mask, :].data
    lon_filt = np.full(ds['lon_trks'].shape, np.nan)
    # ds['lat_trks'][mask, :].data
    lat_filt = np.full(ds['lat_trks'].shape, np.nan)
    # ds['vmax_trks'][mask, :].data
    vmax_filt = np.full(ds['vmax_trks'].shape, np.nan)
    # ds['vmax_trks'][mask, :].data
    m_filt = np.full(ds['m_trks'].shape, np.nan)
    lon_trks = ds['lon_trks'].load().data
    lat_trks = ds['lat_trks'].load().data
    m_trks = ds['m_trks'].load().data
    basin_trks = ds['tc_basins'].load().data
    yr_trks = ds['tc_years'].load().data
    mnth_trks = ds['tc_month'].load().data

    lon_genesis = np.full(lon_filt.shape[0], np.nan)
    lat_genesis = np.full(lon_filt.shape[0], np.nan)
    for i in range(lon_filt.shape[0]):
        if len(np.argwhere(vmax[i, :] >= 15).flatten()) > 0:
            # Genesis occurs when the TC first achieves 30 knots (15 m/s).
            gen_idxs = np.argwhere(vmax[i, :] < 15).flatten()
            idx_gen = np.argwhere(vmax[i, :] >= 15).flatten()[0]
            lon_genesis[i] = lon_trks[i, idx_gen]
            lat_genesis[i] = lat_trks[i, idx_gen]

            # TC decays after it has reached 15 m/s
            decay_idxs = np.argwhere(vmax[i, :] < 15).flatten()
            idxs_lmi = np.argwhere(
                decay_idxs >= np.nanargmax(vmax[i, :])).flatten()
            if len(decay_idxs) > 0 and len(idxs_lmi) > 0:
                idx_decay = decay_idxs[idxs_lmi[0]]
            else:
                idx_decay = vmax.shape[1]

            nt = idx_decay - idx_gen
            vmax_filt[i, 0:nt] = vmax[i, idx_gen:idx_decay]
            lon_filt[i, 0:nt] = lon_trks[i, idx_gen:idx_decay]
            lat_filt[i, 0:nt] = lat_trks[i, idx_gen:idx_decay]
            m_filt[i, 0:nt] = m_trks[i, idx_gen:idx_decay]

    lon_min = 20
    lon_max = 250
    lat_min = -45
    lat_max = 0

    vmax_idxs = np.full(lon_filt.shape[0], np.nan, dtype=int)
    lat_vmaxs = np.full(lon_filt.shape[0], np.nan)

    for i in range(lon_filt.shape[0]):
        vmax_idxs[i] = np.nanargmax(vmax_filt[i, :])
        lat_vmaxs[i] = lat_filt[i, np.nanargmax(vmax_filt[i, :])]

    # Load IBTrACS data:
    fn_ib = r"C:\WorkSpace\data\IBTrACS.ALL.v04r00.nc"
    yearS_ib = namelist.start_year
    yearE_ib = namelist.end_year
    ds_ib = xr.open_dataset(fn_ib)
    dt_ib = np.array(ds_ib['time'][:, 0].values,
                     dtype="datetime64[ms]").astype(object)
    date_mask = np.logical_and(dt_ib >= datetime.datetime(
        yearS_ib, 1, 1), dt_ib <= datetime.datetime(yearE_ib+1, 1, 1))
    ib_lon = ds_ib['lon'].data
    ib_lon[ib_lon < 0] += 360
    ib_lat = ds_ib['lat'].load()
    usa_wind = ds_ib['usa_wind'].load()
    n_tc_per_year = np.zeros(yearE_ib - yearS_ib + 1)
    n_mtc_per_year = np.zeros(yearE_ib - yearS_ib + 1)
    gen_lon_hist = np.full(dt_ib.shape, np.nan)
    gen_lat_hist = np.full(dt_ib.shape, np.nan)
    track_lon_hist = np.full(ib_lon.shape, np.nan)
    track_lat_hist = np.full(ib_lon.shape, np.nan)
    basin_hist = ds_ib['basin'].load().data[:, 0].astype(str)
    basin_names = np.array(sorted(np.unique(basin_hist)))
    n_tc_per_year_basin = np.zeros((len(basin_names), yearE_ib - yearS_ib + 1))
    pdi_per_year_basin = np.zeros((len(basin_names), yearE_ib - yearS_ib + 1))
    mpd_per_year_basin = np.zeros((len(basin_names), yearE_ib - yearS_ib + 1))
    vmax_tc = np.nanmax(usa_wind, axis=1)
    tc_lmi = np.full(dt_ib.shape, np.nan)
    tc_lat_lmi = np.full(dt_ib.shape, np.nan)
    for i in range(yearS_ib, yearE_ib + 1, 1):
        mask = ((dt_ib >= datetime.datetime(i, 1, 1)) &
                (dt_ib <= datetime.datetime(i, 12, 31)) &
                (~np.all(np.isnan(usa_wind.data), axis=1)) &
                (ib_lon[:, 0] >= lon_min) &
                (ib_lon[:, 0] <= lon_max))
        mask_f = np.argwhere(mask).flatten()

        n_tc_per_year[i - yearS_ib] = np.sum(mask)
        for (b_idx, b_name) in enumerate(basin_names):
            if b_name[0] == 'S':
                b_mask = ((dt_ib >= datetime.datetime(i-1, 6, 1)) &
                          (dt_ib <= datetime.datetime(i, 6, 30)) &
                          (~np.all(np.isnan(usa_wind.data), axis=1)))
            else:
                b_mask = ((dt_ib >= datetime.datetime(i, 1, 1)) &
                          (dt_ib <= datetime.datetime(i, 12, 31)) &
                          (~np.all(np.isnan(usa_wind.data), axis=1)))
            n_tc_per_year_basin[b_idx, i - yearS_ib] = np.sum(
                b_mask & (vmax_tc >= 34) & (basin_hist == b_name))
            pdi_mask = b_mask & (basin_hist == b_name)
            mpd_per_year_basin[b_idx, i -
                               yearS_ib] = np.sum(np.power(vmax_tc[pdi_mask] / 1.94384, 3))

        # n_mtc_per_year[i - yearS_ib] = np.sum(vmax_tc_yr >= 35)
        vmax_time = usa_wind[mask, :]['time'].load().data
        int_obs = usa_wind[mask, :].data / 1.94384

        for j in range(int(np.sum(mask))):
            if not bool(np.all(np.isnan(usa_wind[mask][j]))):
                lmi_idx = np.nanargmax(usa_wind[mask][j], axis=0)
                tc_lat_lmi[mask_f[j]] = float(ib_lat[mask][j, lmi_idx])
                tc_lmi[mask_f[j]] = float(usa_wind[mask][j, lmi_idx])

                gen_idx = np.nanargmin(
                    np.abs(usa_wind[mask][j, :] - 35), axis=0)
                gen_lon_hist[mask_f[j]] = ib_lon[mask][j, gen_idx]
                gen_lat_hist[mask_f[j]] = ib_lat[mask][j, gen_idx]
                track_lon_hist[mask_f[j], gen_idx:] = ib_lon[mask][j, gen_idx:]
                track_lat_hist[mask_f[j], gen_idx:] = ib_lat[mask][j, gen_idx:]

    # Move longitudes to the eastern hemisphere (0 < lon <= 360)
    gen_lon_hist[gen_lon_hist < 0] += 360

    # Number of resamplings
    n_ss = 1000

    basin_id = 'GL'
    basin_mask = (basin_trks == basin_id) if (
        basin_id != 'GL') else (basin_trks != 'GL')
    lmi_bins = np.arange(35, 206, 10)
    lmi_cbins = np.arange(40, 201, 10)
    vmax_lmi = np.nanmax(vmax_filt * 1.94384, axis=1)
    tc_lmi_subsample = np.zeros((int(np.sum(n_tc_per_year)), n_ss))
    for i in range(n_ss):
        tc_lmi_subsample[:, i] = np.random.choice(
            vmax_lmi[basin_mask], int(np.sum(n_tc_per_year)))

    h_lmi = np.apply_along_axis(lambda a: np.histogram(
        a, bins=lmi_bins, density='pdf')[0], 1, tc_lmi_subsample)
    mn_lmi_errP = np.nanquantile(
        h_lmi, 0.95, axis=0) - np.nanmean(h_lmi, axis=0)
    mn_lmi_errN = np.nanmean(h_lmi, axis=0) - \
        np.nanquantile(h_lmi, 0.05, axis=0)
    mn_lmi_errN = np.where(mn_lmi_errN < 0, 0, mn_lmi_errN)
    mn_lmi_errP = np.where(mn_lmi_errP < 0, 0, mn_lmi_errP)

    h_lmi_obs = np.histogram(tc_lmi[date_mask & (basin_hist == basin_id) if (basin_id != 'GL') else (basin_hist != 'GL')],
                             density=True, bins=lmi_bins)[0]

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.plot(lmi_cbins, h_lmi_obs, 'r*', markersize=10)
    ax.bar(lmi_cbins, np.nanmean(h_lmi, axis=0), width=8,
            yerr=np.stack((mn_lmi_errN, mn_lmi_errP)), error_kw={'elinewidth': 2, 'capsize': 6})
    plt.xlabel('Lifetime Maximum Intensity (kts)')
    plt.ylabel('Density')
    plt.grid()
    plt.legend(['Observed', 'Downscaling'])
    plt.xlim([35, 205])
    plt.tight_layout()
    plt.savefig(f"{namelist.base_directory}/{exp_name}/LMI_distribution.png", bbox_inches='tight')

    llmi_bins = np.arange(-35, 0.1, 2.5)
    llmi_cbins = np.arange(-33.5, 0.1, 2.5)
    tc_llmi_subsample = np.zeros((int(np.sum(n_tc_per_year)), n_ss))
    for i in range(n_ss):
        tc_llmi_subsample[:, i] = np.random.choice(
            lat_vmaxs, int(np.sum(n_tc_per_year)))

    h_llmi = np.apply_along_axis(lambda a: np.histogram(
        a, bins=llmi_bins, density='pdf')[0], 1, tc_llmi_subsample)
    h_llmi_obs = np.histogram(tc_lat_lmi[date_mask & (basin_hist == basin_id) if (basin_id != 'GL') else (basin_hist != 'GL')],
                              density=True, bins=llmi_bins)[0]

    mn_llmi_errP = np.nanquantile(
        h_llmi, 0.95, axis=0) - np.nanmean(h_llmi, axis=0)
    mn_llmi_errN = np.nanmean(h_llmi, axis=0) - \
        np.nanquantile(h_llmi, 0.05, axis=0)
    mn_llmi_errN = np.where(mn_llmi_errN < 0, 0, mn_llmi_errN)
    mn_llmi_errP = np.where(mn_llmi_errP < 0, 0, mn_llmi_errP)
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.plot(llmi_cbins, h_llmi_obs, 'r*', markersize=10)
    ax.bar(llmi_cbins, np.nanmean(h_llmi, axis=0), width=2.,
            yerr=np.stack((mn_llmi_errN, mn_llmi_errP)), error_kw={'elinewidth': 2, 'capsize': 6})
    ax.set_xlabel("Latitude of lifetime maximum intensity")
    plt.legend(['Observed', 'Downscaling'])
    plt.grid()
    plt.savefig(f"{namelist.base_directory}/{exp_name}/LLMI_distribution.png", bbox_inches='tight')

