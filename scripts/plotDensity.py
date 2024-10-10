import os
import glob
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime
import seaborn as sns
import geopandas as gpd
from itertools import product
from shapely.geometry import LineString, Polygon

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
from util import basins

colorseq=['#FFFFFF', '#ceebfd', '#87CEFA', '#4969E1', '#228B22',
          '#90EE90', '#FFDD66', '#FFCC00', '#FF9933',
          '#FF6600', '#FF0000', '#B30000', '#73264d']
cmap = sns.blend_palette(colorseq, as_cmap=True)

DX = 1.0
DY = 1.0

storm_id_field = "tcnum"
grid_id_field = "gridid"


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

def plot_density(dataArray: xr.DataArray, source: str, outputFile: str):
    """
    Plot track density and save figure to file
    """

    lon = dataArray.lon
    lat = dataArray.lat
    xx, yy = np.meshgrid(lon, lat)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.figure.set_size_inches(10,6)
    cb = plt.contourf(xx, yy, dataArray.T, cmap=cmap,
                      transform=ccrs.PlateCarree(),
                      levels=np.arange(0.05, 1.01, 0.05),
                      extend='both')
    plt.colorbar(cb, label="TCs/year", shrink=0.9)
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True, linestyle=":")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 'small'}
    gl.ylabel_style = {'size': 'small'}

    plt.text(1.0, -0.1, f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
             transform=ax.transAxes, ha='right', fontsize='xx-small',)
    plt.text(0.0, -0.1, f"Source: {source}",
             transform=ax.transAxes,
             fontsize='xx-small', ha='left',)
    savefig(outputFile, bbox_inches='tight')
    plt.close()


def plot_density_difference(dataArray: xr.DataArray, source: str, outputFile: str):
    """
    Plot track density and save figure to file
    """
    lon = dataArray.lon
    lat = dataArray.lat
    xx, yy = np.meshgrid(lon, lat)
    ax = plt.axes(projection=ccrs.PlateCarree(central_longitude=180))
    ax.figure.set_size_inches(10,6)
    cb = plt.contourf(xx, yy, dataArray.T, cmap="RdBu",
                      transform=ccrs.PlateCarree(),
                      levels=np.arange(-0.5, .51, 0.05),
                      extend='both')
    plt.colorbar(cb, label=r"$\Delta$TCs/year", shrink=0.9)
    ax.coastlines()
    gl = ax.gridlines(draw_labels=True, linestyle=":")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 'small'}
    gl.ylabel_style = {'size': 'small'}

    plt.text(1.0, -0.1, f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
             transform=ax.transAxes, ha='right', fontsize='xx-small',)
    plt.text(0.0, -0.1, f"Source: {source}", transform=ax.transAxes, fontsize='xx-small', ha='left',)
    savefig(outputFile, bbox_inches='tight')
    plt.close()


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
                (ib_lon[:, 0] >= minlon) &
                (ib_lon[:, 0] <= maxlon))
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

def createGrid(xmin, xmax, ymin, ymax, wide, length):
    """
    Create a uniform grid across a specified extent, returning a
    `gpd.GeoDataFrame` of the grid to facilitate a spatial join process.

    :param float xmin: minimum longitude of the grid
    :param float xmax: maximum longitude of the grid
    :param float ymin: minimum latitude of the grid
    :param float ymax: maximum latitude of the grid
    :param float wide: longitudinal extent of each grid cell
    :param float length: latitudinal extent of each grid cell

    :returns: `gpd.GeoDataFrame` of collection of polygons representing the
    grid.
    """
    cols = list(np.arange(xmin, xmax + wide, wide))
    rows = list(np.arange(ymin, ymax + length, length))
    polygons = []
    for x, y in product(cols[:-1], rows[:-1]):
        polygons.append(
            Polygon([(x, y), (x + wide, y),
                     (x + wide, y + length),
                     (x, y + length)]))
    gridid = np.arange(len(polygons))
    grid = gpd.GeoDataFrame({'gridid': gridid,
                             'geometry': polygons})
    return grid

def gridDensity(tracks: gpd.GeoDataFrame, grid: gpd.GeoDataFrame,
                grid_id_field: str, storm_id_field: str):
    """
    Calculate the count of events passing across each grid cell


    """
    dfjoin = gpd.sjoin(grid, tracks)
    df2 = dfjoin.groupby(grid_id_field).agg({storm_id_field:'nunique'})
    dfcount = grid.merge(df2, how='left', left_on=grid_id_field, right_index=True)
    dfcount[storm_id_field] = dfcount[storm_id_field].fillna(0)
    dfcount.rename(columns = {storm_id_field:'storm_count'}, inplace = True)
    dfcount['storm_count'] = dfcount['storm_count'].fillna(0)
    return dfcount


def obs_track_density(basin="AU"):
    b = basins.TC_Basin(basin)
    minlon, minlat, maxlon, maxlat = b.get_bounds()

    lon = np.arange(minlon, maxlon, DX)
    lat = np.arange(minlat, maxlat, DY)

    dfgrid = createGrid(minlon, maxlon, minlat, maxlat, DX, DY)
    dims = (int((maxlon - minlon)/DX), int((maxlat - minlat)/DY))
    print("Loading observed track density")
    obstcdf = load_obs_tracks()
    obstd = gridDensity(obstcdf, dfgrid, grid_id_field, storm_id_field)
    tdarray = obstd['storm_count'].values.reshape(dims) / (namelist.end_year - namelist.start_year + 1)
    obsda = xr.DataArray(tdarray, coords=[lon, lat], dims=['lon', 'lat'],
                    attrs=dict(long_name='Mean annual TC frequency',
                                units='TCs/year'))
    plot_density(obsda, "IBTrACS",
                 os.path.join(f"{namelist.base_directory}/{exp_name}",
                              f"observed_track_density.{basin}.png"))

    return obsda

def loadTracks():
    exp_name = namelist.exp_name
    exp_prefix = namelist.exp_prefix
    startdate = f'{namelist.start_year}{namelist.start_month:02d}'
    enddate= f'{namelist.end_year}{namelist.end_month:02d}'

    filename = f"tracks_*_era5_{startdate}_{enddate}*.nc"

    fn_tracks = sorted(glob.glob(f"{namelist.base_directory}/{exp_name}/{filename}"))

    ds = xr.open_mfdataset(fn_tracks, concat_dim = "n_trk", combine = "nested",
                            data_vars="minimal", drop_variables = "seeds_per_month")

    yearS = namelist.start_year
    yearE = namelist.end_year
    n_sim = len(fn_tracks)
    ntrks_per_year = namelist.tracks_per_year
    ntrks_per_sim = ntrks_per_year * (yearE - yearS + 1)

    vmax = ds['vmax_trks'].load().data
    lon_filt = np.full(ds['lon_trks'].shape, np.nan) #ds['lon_trks'][mask, :].data
    lat_filt = np.full(ds['lat_trks'].shape, np.nan) #ds['lat_trks'][mask, :].data
    vmax_filt = np.full(ds['vmax_trks'].shape, np.nan) #ds['vmax_trks'][mask, :].data
    m_filt = np.full(ds['m_trks'].shape, np.nan) #ds['vmax_trks'][mask, :].data
    lon_trks = ds['lon_trks'].load().data
    lat_trks = ds['lat_trks'].load().data
    m_trks = ds['m_trks'].load().data
    yr_trks = ds['tc_years'].load().data

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

    return lon_filt, lat_filt, vmax_filt, m_filt, yr_trks

######################################################


yearS = namelist.start_year
yearE = namelist.end_year
exp_name = namelist.exp_name
lon_trks, lat_trks, vmax_trks, m_trks, yr_trks = loadTracks()

b = basins.TC_Basin("AU")
minlon, minlat, maxlon, maxlat = b.get_bounds()

lon = np.arange(minlon, maxlon, DX)
lat = np.arange(minlat, maxlat, DY)
xx, yy = np.meshgrid(lon, lat)

dfgrid = createGrid(minlon, maxlon, minlat, maxlat, DX, DY)
dims = (int((maxlon - minlon)/DX), int((maxlat - minlat)/DY))

freqsamples = pd.read_csv(r"..\data\tccount.samples.csv")
freqsamples['year'] = freqsamples['year']+1981

obsda = obs_track_density(basin="AU")

print("Sampling synthetic catalogue")
nsamples = 10  # Number of samples to generate
tdsamples = np.zeros((nsamples, len(dfgrid)))
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
            lons = lon_trks[ridx, ~np.isnan(lon_trks[ridx, :])]
            lats = lat_trks[ridx, ~np.isnan(lon_trks[ridx, :])]
            vmaxs = vmax_trks[ridx, ~np.isnan(lon_trks[ridx, :])]
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
    std = gridDensity(trackdf, dfgrid, grid_id_field, storm_id_field)
    tdsamples[n, :] = std['storm_count'].fillna(0) / (yearE - yearS + 1)

norm_factor = (namelist.tracks_per_year / 10)
tdmean = tdsamples.mean(axis=0) * norm_factor
tdmed = np.percentile(tdsamples, 0.5, axis=0) * norm_factor

tdarray = tdmean.reshape(dims) # tc = tropical cyclone *count*
da = xr.DataArray(tdarray, coords=[lon, lat], dims=['lon', 'lat'],
                  attrs=dict(long_name='Mean annual TC frequency',
                             units='TCs/year'))
plot_density(da, f"BAM - {exp_name}", os.path.join(f"{namelist.base_directory}/{exp_name}",
                                                   "mean_track_density.png"))

tdarray = tdmed.reshape(dims) # tc = tropical cyclone *count*
da = xr.DataArray(tdarray, coords=[lon, lat], dims=['lon', 'lat'],
                  attrs=dict(long_name='Mean annual TC frequency',
                             units='TCs/year'))
plot_density(da, f"BAM - {exp_name}", os.path.join(f"{namelist.base_directory}/{exp_name}",
                                                   "median_track_density.png"))

plot_density_difference(da - obsda, f"BAM ({exp_name}) - obs",
                        os.path.join(f"{namelist.base_directory}/{exp_name}",
                        "mean_track_density_difference.png"))

td10 = np.quantile(tdsamples, 0.1, axis=0).reshape(dims) * norm_factor
td90 = np.quantile(tdsamples, 0.9, axis=0,).reshape(dims) * norm_factor
tderr = np.vstack((td10, td90))

da = xr.DataArray(td90, coords=[lon, lat], dims=['lon', 'lat'],
                  attrs=dict(long_name='Mean annual TC frequency',
                             units='TCs/year'))
plot_density(da, "Lin TC model 90th percentile", os.path.join(f"{namelist.base_directory}/{exp_name}", "p90_track_density.png"))

da = xr.DataArray(td10, coords=[lon, lat], dims=['lon', 'lat'],
                  attrs=dict(long_name='Mean annual TC frequency',
                             units='TCs/year'))
plot_density(da, "Lin TC model 10th percentile", os.path.join(f"{namelist.base_directory}/{exp_name}", "p10_track_density.png"))