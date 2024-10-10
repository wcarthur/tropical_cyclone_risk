import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
import geopandas as gpd
from shapely.geometry import LineString, Point

sns.set_context("notebook")

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

import sys
cwd = os.path.dirname(os.path.realpath("."))
parent = os.path.dirname(cwd)
sys.path.append(cwd)

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature


import namelist

def savefig(filename, *args, **kwargs):
    """
    Add a timestamp to each figure when saving

    :param str filename: Path to store the figure at
    :param args: Additional arguments to pass to `plt.savefig`
    :param kwargs: Additional keyword arguments to pass to `plt.savefig`
    """
    fig = plt.gcf()
    plt.text(0.99, 0.01, f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
            transform=fig.transFigure, ha='right', va='bottom',
            fontsize='xx-small')
    plt.savefig(filename, *args, **kwargs)

def isLeft(line, point):
    """
    Test whether a point is to the left of a (directed) line segment.

    :param line: :class:`Shapely.geometry.LineString` of the line feature being tested
    :param point: :class:`Shapely.geometry.Point` being tested
    """
    start = Point(line.coords[0])
    end = Point(line.coords[1])

    det = (end.x - start.x) * (point.y - start.y) - (end.y - start.y) * (point.x - start.x)

    if det > 0: return True
    if det <= 0: return False

def isLandfall(gate, tracks):
    try:
        crossings = tracks.crosses(gate.geometry)
    except IllegalArgumentException:
        return 0
    landfall = []
    for t in tracks[crossings].itertuples():
        if isLeft(gate.geometry, Point(t.geometry.coords[0])):
            landfall.append(True)
        else:
            landfall.append(False)

    return tracks[crossings][landfall]

def countCrossings(gates, tracks):
    columns=['gate', 'label', 'count', 'maxlfintensity']
    agggates = pd.DataFrame(columns=columns)
    aggbp = [6, 12, 17, 24, 31, 36, 40, 43, 48]
    agglabels = ['West Coast', 'Pilbara', 'Kimberley', 'Top End',
                 'Gulf', 'NQLD', 'CQLD', 'SQLD', 'NSW']
    aggcount = 0
    aggvmax = 0.0
    aidx = 0
    for i, gate in enumerate(gates.itertuples(index=False)):
        ncrossings = 0
        l = isLandfall(gate, tracks)
        ncrossings = len(l)

        if ncrossings > 0:
            gates.loc[i, 'count'] = ncrossings
            gates.loc[i, 'meanlfintensity'] = l['vmax'].mean()
            gates.loc[i, 'maxlfintensity'] = l['vmax'].max()
            aggvmax = np.maximum(aggvmax, l['vmax'].max())

        else:
            gates.loc[i, 'count'] = 0
            gates.loc[i, 'meanlfintensity'] = np.nan
            gates.loc[i, 'maxlfintensity'] = np.nan

        aggcount += ncrossings
        if i == aggbp[aidx]:
            agggates.loc[aidx, 'gate'] = aidx
            agggates.loc[aidx, 'label'] = agglabels[aidx]
            agggates.loc[aidx, 'count'] = aggcount
            agggates.loc[aidx, 'maxlfintensity'] = aggvmax
            aggcount = 0
            aggvmax = 0
            aidx += 1
    gates['prob'] = gates['count'] / gates['count'].sum()
    agggates['prob'] = agggates['count'] / agggates['count'].sum()


    return gates, agggates

def load_obs_tracks():
    # Load IBTrACS data:
    fn_ib = r"C:\WorkSpace\data\IBTrACS.ALL.v04r00.nc"
    yearS_ib = namelist.start_year
    yearE_ib = namelist.end_year
    ds_ib = xr.open_dataset(fn_ib)
    dt_ib = np.array(ds_ib['time'][:, 0].values,
                        dtype="datetime64[ms]").astype(object)
    date_mask = np.logical_and(
        dt_ib >= datetime(yearS_ib, 1, 1),
        dt_ib <= datetime(yearE_ib+1, 1, 1)
        )
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
        mask = ((dt_ib >= datetime(i, 1, 1)) &
                (dt_ib <= datetime(i, 12, 31)) &
                (~np.all(np.isnan(usa_wind.data), axis=1)) &
                (ib_lon[:, 0] >= lon_min) &
                (ib_lon[:, 0] <= lon_max))
        mask_f = np.argwhere(mask).flatten()

        n_tc_per_year[i - yearS_ib] = np.sum(mask)
        for (b_idx, b_name) in enumerate(basin_names):
            if b_name[0] == 'S':
                b_mask = ((dt_ib >= datetime(i-1, 6, 1)) &
                            (dt_ib <= datetime(i, 6, 30)) &
                            (~np.all(np.isnan(usa_wind.data), axis=1)))
            else:
                b_mask = ((dt_ib >= datetime(i, 1, 1)) &
                            (dt_ib <= datetime(i, 12, 31)) &
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
    obstcs = []
    for i in range(track_lon_hist.shape[0]):
        segments = []
        lons = track_lon_hist[i, ~np.isnan(track_lon_hist[i])]
        lats = track_lat_hist[i, ~np.isnan(track_lon_hist[i])]
        vmaxs = usa_wind[i, ~np.isnan(track_lon_hist[i])]
        for n in range(len(lons) - 1):
            segment = LineString([[lons[n], lats[n]],
                                [lons[n+1], lats[n+1]]])
            segments.append(segment)
        df = gpd.GeoDataFrame(
            data=np.hstack((lons[:-1], lats[:-1], vmaxs[:-1])).reshape(3, -1).T,
            columns=["Longitude", "Latitude", "vmax"], geometry=segments)
        df['tcnum'] = i
        df['category'] = pd.cut(df['vmax'],
                                bins=[0, 17, 32, 42, 49, 58, 70, 1000],
                                labels=["TD", "TS", "1", "2", "3", "4", "5"])
        obstcs.append(df)

    obstcdf = pd.concat(obstcs)
    return obstcdf

######################################################

gates = gpd.read_file("C:/WorkSpace/tcha/data/gates.shp")

exp_name = namelist.exp_name
exp_prefix = namelist.exp_prefix
startdate = f'{namelist.start_year}{namelist.start_month:02d}'
enddate= f'{namelist.end_year}{namelist.end_month:02d}'

filename = f"tracks_*_era5_{startdate}_{enddate}*.nc"

fn_tracks = sorted(glob.glob(f"{namelist.base_directory}/{exp_name}/{filename}"))

ds = xr.open_mfdataset(fn_tracks, concat_dim = "n_trk", combine = "nested",
                        data_vars="minimal", drop_variables = "seeds_per_month")

drop_vars = ["lon_trks", "lat_trks", "u250_trks", "v250_trks", "u850_trks", "v850_trks",
                 "v_trks", "m_trks", "vmax_trks", "rmax_trks", "r34_trks",
                 "tc_month", "tc_years", "tc_basins"]
ds_seeds = xr.open_mfdataset(fn_tracks, concat_dim = "year", combine = "nested",
                            data_vars="minimal", drop_variables = drop_vars)
yearS = namelist.start_year
yearE = namelist.end_year
n_sim = len(fn_tracks)
nyears = yearE - yearS + 1
ntrks_per_year = namelist.tracks_per_year
ntrks_per_sim = ntrks_per_year * (yearE - yearS + 1)
seeds_per_month_basin = ds_seeds['seeds_per_month']
seeds_per_month = np.nansum(seeds_per_month_basin, axis = 1)

vmax = ds['vmax_trks'].load().data
lon_filt = np.full(ds['lon_trks'].shape, np.nan) #ds['lon_trks'][mask, :].data
lat_filt = np.full(ds['lat_trks'].shape, np.nan) #ds['lat_trks'][mask, :].data
vmax_filt = np.full(ds['vmax_trks'].shape, np.nan) #ds['vmax_trks'][mask, :].data
m_filt = np.full(ds['m_trks'].shape, np.nan) #ds['vmax_trks'][mask, :].data
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
        idxs_lmi = np.argwhere(decay_idxs >= np.nanargmax(vmax[i, :])).flatten()
        if len(decay_idxs) > 0 and len(idxs_lmi) > 0:
            idx_decay = decay_idxs[idxs_lmi[0]]
        else:
            idx_decay = vmax.shape[1]

        nt = idx_decay - idx_gen
        vmax_filt[i, 0:nt] = vmax[i, idx_gen:idx_decay]
        lon_filt[i, 0:nt] = lon_trks[i, idx_gen:idx_decay]
        lat_filt[i, 0:nt] = lat_trks[i, idx_gen:idx_decay]
        m_filt[i, 0:nt] = m_trks[i, idx_gen:idx_decay]

lon_min = 20; lon_max = 250;
lat_min = -45; lat_max = 0;


freqsamples = pd.read_csv(r"C:\WorkSpace\tropical_cyclone_risk\data\tccount.samples.csv")
freqsamples['year'] = freqsamples['year']+1981

print("Loading observed landfalls")
obstcdf = load_obs_tracks()
obslf = gates.copy()
obslf, obsagglf = countCrossings(obslf, obstcdf)
print("Sampling synthetic catalogue")
nsamples = 200  # Number of samples to generate
lfsamples = np.zeros((nsamples, len(gates)))
agglfsamples = np.zeros((nsamples, len(obsagglf)))
for n in range(nsamples):
    tracks = []
    # For each year, we sample from the randomly generated time series of
    # annual TC counts
    for year in range(yearS, yearE+1):
        yidx = np.where(yr_trks==year)[0]
        ntcs = np.random.choice(freqsamples[freqsamples['year']==year]['ntcs'])
        r_idxs = np.random.choice(yidx, size=ntcs)

        # Random selection of events for the selected year
        for ridx in r_idxs:
            lons = lon_filt[ridx, ~np.isnan(lon_filt[ridx, :])]
            lats = lat_filt[ridx, ~np.isnan(lon_filt[ridx, :])]
            vmaxs = vmax_filt[ridx, ~np.isnan(lon_filt[ridx, :])]
            year = yr_trks[ridx] * np.ones(len(lons), dtype=int)
            segments = []
            for k in range(lons.shape[0]-1):
                if np.isnan(lons[k+1]):
                    continue
                segment = LineString([[lons[k], lats[k]],
                                  [lons[k+1], lats[k+1]]])
                segments.append(segment)
            df = gpd.GeoDataFrame(data=np.vstack((lons[:-1], lats[:-1], vmaxs[:-1], year[:-1])).T,
                                  columns=["Longitude", "Latitude", "vmax", "year"], geometry=segments)
            df['tcnum'] = ridx
            df['category'] = pd.cut(df['vmax'], bins=[0, 17, 32, 42, 49, 58, 70, 1000],
                                labels=["TD", "TS", "1", "2", "3", "4", "5"])
            tracks.append(df)
    trackdf = pd.concat(tracks)
    gatedf = gates.copy()
    gatedf, gateaggdf = countCrossings(gatedf, trackdf)
    lfsamples[n, :] = gatedf['count']
    agglfsamples[n, :] = gateaggdf['count']

#lfsamples = lfsamples / lfsamples.sum(axis=1)[:, None]
lfmean = lfsamples.mean(axis=0)
lfmed = np.percentile(lfsamples, 0.5, axis=0)
lfstd = lfsamples.std(axis=0)
lf10 = np.percentile(lfsamples, 10, axis=0)
lf90 = np.percentile(lfsamples, 90, axis=0)
lferr = np.vstack((lf10, lf90))

agglfsamples = agglfsamples / agglfsamples.sum(axis=1)[:, None]

alfmean = agglfsamples.mean(axis=0)
alfmed = np.percentile(agglfsamples, 0.5, axis=0)
alfstd = agglfsamples.std(axis=0)
alf10 = np.percentile(agglfsamples, 10, axis=0)
alf90 = np.percentile(agglfsamples, 90, axis=0)
alferr = np.vstack((alf10, alf90))

# Plot boxplot of the landfall rates:
fig, ax = plt.subplots(1, 1, figsize=(12, 6), sharex=True)
cor = np.corrcoef(obslf['prob'], lfmed)[0, 1]
ax.boxplot(lfsamples / nyears, patch_artist=True,
           medianprops=dict(color='k'),
           flierprops=dict(marker='.', markersize=5))
ax.plot(obslf.gate, obslf['count'] / nyears,'r*', markersize=10, label="Obs (1980 - 2021)")
ax.set_xlim((0, 48))
ax.set_xticks(np.arange(0, 49,2))
ymin=0; ymax=np.ceil(np.max(obslf['count'] / nyears))
ax.set_yticks(np.linspace(0, ymax+.1, 5))
ax.set_ylim((0, ymax))
ax.set_ylabel('Landfall count')
ax.set_xticklabels(gatedf['label'][::2], rotation='vertical')
ax.text(0.05, 0.93, 'r = %0.3f' % cor, transform = ax.transAxes,)
ax.legend()
savefig("%s/%s/landfall.png" % (namelist.base_directory, exp_name), bbox_inches='tight')

# Plot boxplot of the aggregated landfall rates:
fig, ax = plt.subplots(1, 1, figsize=(12, 6), sharex=True)
#cor = np.corrcoef(obsagglf['prob'], alfmed)[0, 1]
ax.boxplot(agglfsamples, patch_artist=True,
           medianprops=dict(color='k'),
           flierprops=dict(marker='.', markersize=5))
ax.plot(obsagglf.gate, obsagglf['prob'],'r*', markersize=10, label="Obs (1980 - 2021)")
ax.set_xlim((0, 8))
ax.set_xticks(np.arange(0, 9))
#ax.set_yticks(np.arange(0, 0.15, .01))
#ax.set_ylim((0, 0.15))
ax.set_ylabel('Probability of landfall')
ax.set_xticklabels(gateaggdf['label'], rotation='vertical')
#ax.text(0.05, 0.93, 'r = %0.3f' % cor, transform = ax.transAxes,)
ax.legend()
savefig("%s/%s/agglandfall.png" % (namelist.base_directory, exp_name), bbox_inches='tight')