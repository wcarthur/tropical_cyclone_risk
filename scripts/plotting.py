import os
import sys
import glob
from copy import copy
from itertools import product

import xarray as xr
import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
import geopandas as gpd
from shapely.geometry import LineString, Polygon

from scipy.ndimage import gaussian_filter

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

cwd = os.path.dirname(os.path.realpath("."))
parent = os.path.dirname(cwd)
sys.path.append(cwd)

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER

from util import basins, sphere
import namelist

print(f"Plotting data for {namelist.exp_name}")
BASEDIR = f"{namelist.base_directory}/{namelist.exp_name}"
proj = ccrs.PlateCarree(central_longitude=180)
trans = ccrs.PlateCarree()


def savefig(filename, *args, **kwargs):
    """
    Add a timestamp to each figure when saving

    :param str filename: Path to store the figure at
    :param args: Additional arguments to pass to `plt.savefig`
    :param kwargs: Additional keyword arguments to pass to `plt.savefig`
    """
    fig = plt.gcf()
    plt.text(
        0.99,
        0.01,
        f"Created: {datetime.now():%Y-%m-%d %H:%M %z}",
        transform=fig.transFigure,
        ha="right",
        va="bottom",
        fontsize="xx-small",
    )
    plt.savefig(filename, *args, **kwargs)


def load_tracks(**kwargs):
    """
    Load a set of tracks from a set of files. Further manipulation
    of the data is needed (e.g. filtering positions where vmax < 15 m/s).

    :return: `xr.Dataset` containing synthetic tracks
    """
    exp_name = namelist.exp_name
    startdate = f"{namelist.start_year}{namelist.start_month:02d}"
    enddate = f"{namelist.end_year}{namelist.end_month:02d}"

    filename = f"tracks_*_era5_{startdate}_{enddate}*.nc"
    fn_tracks = sorted(glob.glob(f"{namelist.base_directory}/{exp_name}/{filename}"))
    ds = xr.open_mfdataset(
        fn_tracks,
        concat_dim="n_trk",
        combine="nested",
        compat="override",
        coords="all",
        data_vars="minimal",
        **kwargs,
    )
    return ds


def load_obs_tracks(extents):
    fn_ib = r"C:\WorkSpace\data\IBTrACS.ALL.v04r00.nc"
    lon_min, lon_max, lat_min, lat_max = extents
    yearS_ib = namelist.start_year
    yearE_ib = namelist.end_year
    ds_ib = xr.open_dataset(fn_ib)
    dt_ib = np.array(ds_ib["time"][:, 0].values, dtype="datetime64[ms]").astype(object)
    date_mask = np.logical_and(
        dt_ib >= datetime(yearS_ib, 1, 1), dt_ib <= datetime(yearE_ib + 1, 1, 1)
    )
    ib_lon = ds_ib["lon"].data
    ib_lon[ib_lon < 0] += 360
    ib_lat = ds_ib["lat"].load()
    usa_wind = ds_ib["usa_wind"].load()
    n_tc_per_year = np.zeros(yearE_ib - yearS_ib + 1)
    n_mtc_per_year = np.zeros(yearE_ib - yearS_ib + 1)
    gen_lon_hist = np.full(dt_ib.shape, np.nan)
    gen_lat_hist = np.full(dt_ib.shape, np.nan)
    track_lon_hist = np.full(ib_lon.shape, np.nan)
    track_lat_hist = np.full(ib_lon.shape, np.nan)
    basin_hist = ds_ib["basin"].load().data[:, 0].astype(str)
    basin_names = np.array(sorted(np.unique(basin_hist)))
    n_tc_per_year_basin = np.zeros((len(basin_names), yearE_ib - yearS_ib + 1))
    pdi_per_year_basin = np.zeros((len(basin_names), yearE_ib - yearS_ib + 1))
    mpd_per_year_basin = np.zeros((len(basin_names), yearE_ib - yearS_ib + 1))
    vmax_tc = np.nanmax(usa_wind, axis=1)
    tc_lmi = np.full(dt_ib.shape, np.nan)
    tc_lat_lmi = np.full(dt_ib.shape, np.nan)
    for i in range(yearS_ib, yearE_ib + 1, 1):
        mask = (
            (dt_ib >= datetime(i, 1, 1))
            & (dt_ib <= datetime(i, 12, 31))
            & (~np.all(np.isnan(usa_wind.data), axis=1))
            & (ib_lon[:, 0] >= lon_min)
            & (ib_lon[:, 0] <= lon_max)
        )
        mask_f = np.argwhere(mask).flatten()

        n_tc_per_year[i - yearS_ib] = np.sum(mask)
        for b_idx, b_name in enumerate(basin_names):
            if b_name[0] == "S":
                b_mask = (
                    (dt_ib >= datetime(i - 1, 6, 1))
                    & (dt_ib <= datetime(i, 6, 30))
                    & (~np.all(np.isnan(usa_wind.data), axis=1))
                )
            else:
                b_mask = (
                    (dt_ib >= datetime(i, 1, 1))
                    & (dt_ib <= datetime(i, 12, 31))
                    & (~np.all(np.isnan(usa_wind.data), axis=1))
                )
            n_tc_per_year_basin[b_idx, i - yearS_ib] = np.sum(
                b_mask & (vmax_tc >= 34) & (basin_hist == b_name)
            )
            pdi_mask = b_mask & (basin_hist == b_name)
            mpd_per_year_basin[b_idx, i - yearS_ib] = np.sum(
                np.power(vmax_tc[pdi_mask] / 1.94384, 3)
            )

        # n_mtc_per_year[i - yearS_ib] = np.sum(vmax_tc_yr >= 35)
        vmax_time = usa_wind[mask, :]["time"].load().data
        int_obs = usa_wind[mask, :].data / 1.94384

        for j in range(int(np.sum(mask))):
            if not bool(np.all(np.isnan(usa_wind[mask][j]))):
                lmi_idx = np.nanargmax(usa_wind[mask][j], axis=0)
                tc_lat_lmi[mask_f[j]] = float(ib_lat[mask][j, lmi_idx])
                tc_lmi[mask_f[j]] = float(usa_wind[mask][j, lmi_idx])

                gen_idx = np.nanargmin(np.abs(usa_wind[mask][j, :] - 35), axis=0)
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

def createGrid(wide, length, basin="GL"):
    """
    Create a uniform grid across a specified extent, returning a
    `gpd.GeoDataFrame` of the grid to facilitate a spatial join process.

    :param float wide: longitudinal extent of each grid cell
    :param float length: latitudinal extent of each grid cell
    :param str basin: Name of the basin. Default="GL"

    :returns: `gpd.GeoDataFrame` of collection of polygons representing the
    grid.
    """
    b = basins.TC_Basin(basin)
    lon_min, lat_min, lon_max, lat_max = b.get_bounds()
    cols = list(np.arange(lon_min, lon_max + wide, wide))
    rows = list(np.arange(lat_min, lat_max + length, length))
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

def grid_density(tracks: gpd.GeoDataFrame, grid: gpd.GeoDataFrame,
                grid_id_field: str, storm_id_field: str):
    """
    Calculate the count of events passing across each grid cell

    This counts the number of *unique* events in each grid cell.
    It is not the number of TC records in each grid cell, which may be a
    function of the temporal resolution of the track data (3-hourly tracks
    have twice as many points as 6-hourly track data)
    """
    dfjoin = gpd.sjoin(grid, tracks)
    df2 = dfjoin.groupby(grid_id_field).agg({storm_id_field:'nunique'})
    dfcount = grid.merge(df2, how='left', left_on=grid_id_field, right_index=True)
    dfcount[storm_id_field] = dfcount[storm_id_field].fillna(0)
    dfcount.rename(columns = {storm_id_field:'storm_count'}, inplace = True)
    dfcount['storm_count'] = dfcount['storm_count'].fillna(0)
    return dfcount


def create_grid(basin):
    b = basins.TC_Basin(basin)
    lon_min, lat_min, lon_max, lat_max = b.get_bounds()
    # 1x1 degree grid:
    x_binEdges = np.arange(lon_min, lon_max + 0.1, 1.0)
    y_binEdges = np.arange(lat_min, lat_max + 0.1, 1.0)
    x_binCenter = (x_binEdges[1:] + x_binEdges[0:-1]) / 2
    y_binCenter = (y_binEdges[1:] + y_binEdges[0:-1]) / 2

    return x_binEdges, y_binEdges, x_binCenter, y_binCenter

def plot_genesis(filename: str, basin="GL"):
    """
    Plot genesis and track density

    :param str filename: File path to save the figure to
    :param extents: Extent of map to plot, defaults to 'GL'

    """

    xEdge, yEdge, xCentre, yCentre = create_grid(basin)
    b = basins.TC_Basin(basin)
    lon_min, lat_min, lon_max, lat_max = b.get_bounds()


    gen_pdf = np.zeros((len(xCentre), len(yCentre)))
    den_pdf = np.zeros((len(xCentre), len(yCentre)))

    ds = load_tracks(drop_variables = "seeds_per_month")
    nyears = namelist.end_year - namelist.start_year + 1
    nsim = len(ds.n_trk) / namelist.tracks_per_year / nyears

    vmax = ds["vmax_trks"].load().data
    lon_trks = ds["lon_trks"].load().data
    lat_trks = ds["lat_trks"].load().data
    m_trks = ds["m_trks"].load().data
    lon_filt = np.full(ds["lon_trks"].shape, np.nan)  # ds['lon_trks'][mask, :].data
    lat_filt = np.full(ds["lat_trks"].shape, np.nan)  # ds['lat_trks'][mask, :].data
    vmax_filt = np.full(ds["vmax_trks"].shape, np.nan)  # ds['vmax_trks'][mask, :].data
    m_filt = np.full(ds["m_trks"].shape, np.nan)  # ds['vmax_trks'][mask, :].data

    # Here, we only consider a TC from the first point where it exceeds
    # the threshold, to the point it decays to 10 m/s (after it has
    # reached its peak intensity).
    lon_genesis = np.full(lon_filt.shape[0], np.nan)
    lat_genesis = np.full(lon_filt.shape[0], np.nan)

    for i in range(lon_filt.shape[0]):
        if len(np.argwhere(vmax[i, :] >= 15).flatten()) > 0:
            # Genesis occurs when the TC first achieves 30 knots (15 m/s).
            idx_gen = np.argwhere(vmax[i, :] >= 15).flatten()[0]
            lon_genesis[i] = ds["lon_trks"][i, idx_gen]
            lat_genesis[i] = ds["lat_trks"][i, idx_gen]

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

    gen_pdf = (
        np.histogram2d(lon_genesis, lat_genesis, bins=[xEdge, yEdge])[0]
    )
    den_pdf = (
        np.histogram2d(
            lon_filt.flatten(), lat_filt.flatten(), bins=[xEdge, yEdge]
        )[0]
    )

    # plt.rcParams.update({'font.size': 14})
    dlon_label = 10
    lon_lmin = np.floor(lon_min / dlon_label) * dlon_label
    lon_lmax = np.ceil(lon_max / dlon_label) * dlon_label

    xlocs = np.arange(lon_lmin, lon_lmax + dlon_label, dlon_label)
    fig, axes = plt.subplots(
        2, 1, facecolor="w", edgecolor="k", subplot_kw={"projection": proj}
    )
    fig.set_size_inches(15, 8)
    # ax = fig.add_subplot(111, projection=proj)
    axes[0].coastlines(resolution="50m")
    axes[0].set_extent(
        [lon_min, lon_max, lat_min, lat_max], crs=trans
    )
    axes[0].gridlines(
        draw_labels=False, crs=trans, xlocs=xlocs, color="gray", alpha=0.3
    )
    gl = axes[0].gridlines(
        draw_labels=True,
        crs=trans,
        xlocs=xlocs[1:-1],
        color="gray",
        alpha=0.3,
    )
    gl.bottom_labels = True
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    levels = np.arange(3, 30, 3)

    axes[1].coastlines(resolution="50m")
    axes[1].set_extent(
        [lon_min, lon_max, lat_min, lat_max], crs=trans
    )
    axes[1].gridlines(
        draw_labels=False, crs=trans, xlocs=xlocs, color="gray", alpha=0.3
    )
    gl = axes[1].gridlines(
        draw_labels=True,
        crs=trans,
        xlocs=xlocs[1:-1],
        color="gray",
        alpha=0.3,
    )
    gl.bottom_labels = True
    gl.top_labels = False
    gl.right_labels = False
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    levels = np.arange(3, 30, 3)

    from copy import copy

    palette = copy(plt.get_cmap("viridis_r"))
    palette.set_under("white", 1.0)
    # gen_pdf[gen_pdf == 0] = 1e-6                # so we can take the log
    # cmin = np.quantile(np.log(gen_pdf[gen_pdf > 1]), 0.1)
    # cmax = np.quantile(np.log(gen_pdf), 1)
    cmax = np.quantile(gen_pdf, 0.999)
    cmin = 1  # cmax / 10
    levels = np.linspace(cmin, cmax, 11)
    cs = axes[0].contourf(
        xCentre,
        yCentre,
        gen_pdf.T,
        levels=levels,
        extend="max",
        cmap=palette,
        transform=trans,
    )
    plt.colorbar(cs, orientation="horizontal", ax=axes[0], aspect=50)
    axes[0].set_title("Genesis PDF")

    cmax = np.quantile(den_pdf / 20, 0.999)
    cmin = cmax / 10
    levels = np.linspace(cmin, cmax, 11)
    cs = axes[1].contourf(
        xCentre,
        yCentre,
        den_pdf.T / 20,
        levels=levels,
        extend="max",
        cmap=palette,
        transform=trans,
    )
    plt.colorbar(cs, orientation="horizontal", ax=axes[1], aspect=50)
    axes[1].set_title("Track density")
    savefig(filename, bbox_inches="tight")


plot_genesis(os.path.join(BASEDIR, "track_density.png"))

sim_name = namelist.exp_name
ds = load_tracks(drop_variables="seeds_per_month")
drop_vars = [
    "lon_trks",
    "lat_trks",
    "u250_trks",
    "v250_trks",
    "u850_trks",
    "v850_trks",
    "v_trks",
    "m_trks",
    "vmax_trks",
    "tc_month",
    "tc_years",
    "tc_basins",
]
ds_seeds = load_tracks(drop_variables=drop_vars)
yearS = namelist.start_year
yearE = namelist.end_year
n_sim = int(len(ds.n_trk) / len(np.unique(ds.n_trk)))
ntrks_per_year = namelist.tracks_per_year
ntrks_per_sim = ntrks_per_year * (yearE - yearS + 1)
seeds_per_month_basin = ds_seeds["seeds_per_month"]
seeds_per_month = np.nansum(seeds_per_month_basin, axis=1)

month_hist = np.full((ntrks_per_year, 12), np.nan)
storm_samples = ds.tc_month.values.reshape(ntrks_per_year, -1)
for i in range(ntrks_per_year):
    month_hist[i, :], _ = np.histogram(
        storm_samples[i, :], bins=np.arange(0.5, 12.51, 1)
    )
month_hist = month_hist / month_hist.sum(axis=1)[:, None]
fig, ax = plt.subplots(1, 1)
ax.boxplot(
    month_hist,
    patch_artist=True,
    medianprops=dict(color="k"),
    flierprops=dict(marker=".", markersize=5),
)
ax.set_xlabel("Month")
ax.set_ylabel("Proportion of TCs")
ax.set_title("Proportion of TCs by month")
ax.grid()
plt.gca().set_axisbelow(True)
savefig(os.path.join(BASEDIR, "monthly_prop_TCs.png"))

vmax = ds["vmax_trks"].load().data
lon_filt = np.full(ds["lon_trks"].shape, np.nan)  # ds['lon_trks'][mask, :].data
lat_filt = np.full(ds["lat_trks"].shape, np.nan)  # ds['lat_trks'][mask, :].data
vmax_filt = np.full(ds["vmax_trks"].shape, np.nan)  # ds['vmax_trks'][mask, :].data
m_filt = np.full(ds["m_trks"].shape, np.nan)  # ds['vmax_trks'][mask, :].data
lon_trks = ds["lon_trks"].load().data
lat_trks = ds["lat_trks"].load().data
m_trks = ds["m_trks"].load().data
basin_trks = ds["tc_basins"].load().data
yr_trks = ds["tc_years"].load().data
mnth_trks = ds["tc_month"].load().data

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


lon_min = 20
lon_max = 250
lat_min = -45
lat_max = 0
lon_cen = 180
dlon_label = 20

lon_lmin = np.floor(lon_min / dlon_label) * dlon_label
lon_lmax = np.ceil(lon_max / dlon_label) * dlon_label
xlocs = np.arange(lon_lmin, lon_lmax + dlon_label, dlon_label)
xlocs_shift = np.copy(xlocs)
xlocs_shift[xlocs > lon_cen] -= 360

fn_ib = r"C:\WorkSpace\data\IBTrACS.ALL.v04r00.nc"
yearS_ib = 1981
yearE_ib = 2021
ds_ib = xr.open_dataset(fn_ib)
dt_ib = np.array(ds_ib["time"][:, 0].values, dtype="datetime64[ms]").astype(object)
date_mask = np.logical_and(
    dt_ib >= datetime(yearS_ib, 1, 1), dt_ib <= datetime(yearE_ib + 1, 1, 1)
)
ib_lon = ds_ib["lon"].data
ib_lon[ib_lon < 0] += 360
ib_lat = ds_ib["lat"].load()
usa_wind = ds_ib["usa_wind"].load()
n_tc_per_year = np.zeros(yearE_ib - yearS_ib + 1)
n_mtc_per_year = np.zeros(yearE_ib - yearS_ib + 1)
gen_lon_hist = np.full(dt_ib.shape, np.nan)
gen_lat_hist = np.full(dt_ib.shape, np.nan)
track_lon_hist = np.full(ib_lon.shape, np.nan)
track_lat_hist = np.full(ib_lon.shape, np.nan)
basin_hist = ds_ib["basin"].load().data[:, 0].astype(str)
basin_names = np.array(sorted(np.unique(basin_hist)))
n_tc_per_year_basin = np.zeros((len(basin_names), yearE_ib - yearS_ib + 1))
pdi_per_year_basin = np.zeros((len(basin_names), yearE_ib - yearS_ib + 1))
mpd_per_year_basin = np.zeros((len(basin_names), yearE_ib - yearS_ib + 1))
vmax_tc = np.nanmax(usa_wind, axis=1)
tc_lmi = np.full(dt_ib.shape, np.nan)
tc_lat_lmi = np.full(dt_ib.shape, np.nan)
for i in range(yearS_ib, yearE_ib + 1, 1):
    mask = (
        (dt_ib >= datetime(i, 1, 1))
        & (dt_ib <= datetime(i, 12, 31))
        & (~np.all(np.isnan(usa_wind.data), axis=1))
        & (ib_lon[:, 0] >= lon_min)
        & (ib_lon[:, 0] <= lon_max)
    )
    mask_f = np.argwhere(mask).flatten()

    n_tc_per_year[i - yearS_ib] = np.sum(mask)
    for b_idx, b_name in enumerate(basin_names):
        if b_name[0] == "S" or b_name[0] == "A":
            b_mask = (
                (dt_ib >= datetime(i - 1, 6, 1))
                & (dt_ib <= datetime(i, 6, 30))
                & (~np.all(np.isnan(usa_wind.data), axis=1))
            )
        else:
            b_mask = (
                (dt_ib >= datetime(i, 1, 1))
                & (dt_ib <= datetime(i, 12, 31))
                & (~np.all(np.isnan(usa_wind.data), axis=1))
            )
        n_tc_per_year_basin[b_idx, i - yearS_ib] = np.sum(
            b_mask & (vmax_tc >= 34) & (basin_hist == b_name)
        )
        pdi_mask = b_mask & (basin_hist == b_name)
        mpd_per_year_basin[b_idx, i - yearS_ib] = np.sum(
            np.power(vmax_tc[pdi_mask] / 1.94384, 3)
        )

    # n_mtc_per_year[i - yearS_ib] = np.sum(vmax_tc_yr >= 35)
    vmax_time = usa_wind[mask, :]["time"].load().data
    int_obs = usa_wind[mask, :].data / 1.94384

    for j in range(int(np.sum(mask))):
        if not bool(np.all(np.isnan(usa_wind[mask][j]))):
            lmi_idx = np.nanargmax(usa_wind[mask][j], axis=0)
            tc_lat_lmi[mask_f[j]] = float(ib_lat[mask][j, lmi_idx])
            tc_lmi[mask_f[j]] = float(usa_wind[mask][j, lmi_idx])

            gen_idx = np.nanargmin(np.abs(usa_wind[mask][j, :] - 35), axis=0)
            gen_lon_hist[mask_f[j]] = ib_lon[mask][j, gen_idx]
            gen_lat_hist[mask_f[j]] = ib_lat[mask][j, gen_idx]
            track_lon_hist[mask_f[j], gen_idx:] = ib_lon[mask][j, gen_idx:]
            track_lat_hist[mask_f[j], gen_idx:] = ib_lat[mask][j, gen_idx:]
gen_lon_hist[gen_lon_hist < 0] += 360

AUS_mask = (gen_lon_hist >= 100) & (gen_lon_hist <= 165) & (gen_lat_hist <= 0)
basin_hist[AUS_mask] = "AU"
basin_names = np.array(sorted(np.unique(basin_hist)))
basin_names = np.array(["SP", "SI", "AU"])
n_tc_per_year_basin = np.zeros((len(basin_names), yearE_ib - yearS_ib + 1))
pdi_per_year_basin = np.zeros((len(basin_names), yearE_ib - yearS_ib + 1))
mpd_per_year_basin = np.zeros((len(basin_names), yearE_ib - yearS_ib + 1))
for i in range(yearS_ib, yearE_ib + 1, 1):
    mask = (
        (dt_ib >= datetime(i, 1, 1))
        & (dt_ib <= datetime(i, 12, 31))
        & (~np.all(np.isnan(usa_wind.data), axis=1))
        & (ib_lon[:, 0] >= lon_min)
        & (ib_lon[:, 0] <= lon_max)
    )

    n_tc_per_year[i - yearS_ib] = np.sum(mask)
    for b_idx, b_name in enumerate(["SP", "SI", "AU"]):
        if b_name[0] == "S" or (b_name == "SH") or (b_name == "AU"):
            b_mask = (
                (dt_ib >= datetime(i - 1, 6, 1))
                & (dt_ib <= datetime(i, 5, 31))
                & (~np.all(np.isnan(usa_wind.data), axis=1))
            )
        else:
            b_mask = (
                (dt_ib >= datetime(i, 1, 1))
                & (dt_ib <= datetime(i, 12, 31))
                & (~np.all(np.isnan(usa_wind.data), axis=1))
            )
        n_tc_per_year_basin[b_idx, i - yearS_ib] = np.sum(
            b_mask & (vmax_tc >= 34) & (basin_hist == b_name)
        )
        pdi_mask = b_mask & (basin_hist == b_name)
        mpd_per_year_basin[b_idx, i - yearS_ib] = np.sum(
            np.power(vmax_tc[pdi_mask] / 1.94384, 3)
        )

basin_ids = np.array([k for k in namelist.basin_bounds if k != "GL"])
basin_ids = np.array(["SI", "AU", "SP"], dtype="<U2")

# Plot seed ratio:
plt.figure(figsize=(8, 6))
for i, basin_id in enumerate(basin_ids):
    seeds_per_month = ds_seeds["seeds_per_month"][
        :, ds["basin"] == basin_id, :
    ].squeeze()
    # Compute probability of seed to TCs
    nTCs_per_month = np.zeros(12)
    nSeeds_per_month = np.zeros(12)
    for m_idx in range(12):
        mask = np.sum((mnth_trks == (m_idx + 1)) & (basin_trks == basin_id))
        nTCs_per_month[m_idx] = np.sum(mask)
        nSeeds_per_month[m_idx] = np.sum(seeds_per_month[:, m_idx])
    plt.plot(range(1, 13), nTCs_per_month / nSeeds_per_month)
plt.legend(basin_ids)
plt.grid()
plt.xlabel("Month")
plt.ylabel("Seed Genesis Probability")
plt.xticks(range(1, 13, 1))
plt.xlim([1, 12])
# plt.ylim([0, 0.07])
savefig(os.path.join(BASEDIR, "seed_genesis_prob.png"))

n_tc_per_year_downscaling_basin = np.zeros((len(basin_ids), yearE - yearS + 1))
mpd_per_year_downscaling_basin = np.zeros((len(basin_ids), yearE - yearS + 1))
dt_seeds = np.array(
    [[datetime(x, y, 1) for y in range(1, 13)] for x in seeds_per_month["year"].values]
)
dt_trks = np.array(
    [datetime(int(yr_trks[i]), int(mnth_trks[i]), 1) for i in range(len(yr_trks))]
)

fig, axs = plt.subplots(figsize=(15, 9), ncols=2, nrows=2, sharex=True)
for i, basin_id in enumerate(basin_ids):
    seeds_per_month = ds_seeds["seeds_per_month"][
        :, ds["basin"] == basin_id, :
    ].squeeze()

    # Compute probability of seed to TCs
    nTCs_per_year = np.zeros((yearE - yearS + 1))
    nSeeds_per_year = np.zeros((yearE - yearS + 1))
    for yr in np.unique(ds_seeds["year"]):
        if basin_id[0] == "X":
            mask = (
                (dt_trks >= datetime(yr - 1, 6, 1))
                & (dt_trks <= datetime(yr, 5, 31))
                & (basin_trks == basin_id)
            )
            seed_mask = (dt_seeds >= datetime(yr - 1, 6, 1)) & (
                dt_seeds <= datetime(yr, 5, 31)
            )
        else:
            mask = np.sum((yr_trks == yr) & (basin_trks == basin_id))
            seed_mask = (dt_seeds >= datetime(yr, 1, 1)) & (
                dt_seeds <= datetime(yr, 12, 31)
            )
        nTCs_per_year[yr - yearS] = np.sum(mask)
        nSeeds_per_year[yr - yearS] = np.sum(seeds_per_month.data[seed_mask])

    n_tc_per_year = np.sum(
        n_tc_per_year_basin[(basin_names == basin_id), :], axis=0
    ).flatten()
    dt_year = ds["year"].sel(year=slice(yearS, yearE))
    if not np.any(nTCs_per_year):
        continue
    n_tc_per_year_downscaling = nTCs_per_year / nSeeds_per_year
    seed_ratio = np.nanmean(n_tc_per_year_downscaling) / np.nanmean(n_tc_per_year)
    n_tc_per_year_downscaling /= seed_ratio
    n_tc_per_year_downscaling_basin[i, :] = n_tc_per_year_downscaling

    n_ss = 1000
    pdi_per_year_downscaling = np.zeros((n_ss, yearE - yearS + 1))
    max_pdi_per_year_downscaling_ss = np.zeros((n_ss, yearE - yearS + 1))

    for yr in np.unique(ds_seeds["year"]):
        mask = (yr_trks == yr) & (basin_trks == basin_id)
        max_pdi_trks = np.power(np.nanmax(vmax_filt[mask, :], axis=1), 3)
        # vmax_pdi_trks = np.power(vmax_filt[mask, :], 3)
        # vmax_pdi_trks[np.isnan(vmax_pdi_trks)] = 0
        for n_idx in range(n_ss):
            mpd_subsample = np.random.choice(
                max_pdi_trks, int(np.round(n_tc_per_year_downscaling[yr - yearS]))
            )
            max_pdi_per_year_downscaling_ss[n_idx, yr - yearS] = np.nansum(
                mpd_subsample
            )
            # for j in range(vmax_pdi_trks.shape[0]):
            #    pdi_per_year_downscaling[yr-yearS] += np.trapz(vmax_pdi_trks[j, :], x=ds['time'].data)

    max_pdi_per_year_downscaling = np.nanmean(max_pdi_per_year_downscaling_ss, axis=0)
    mpd_per_year = mpd_per_year_basin[basin_names == basin_id, :].squeeze()
    mpd_per_year_downscaling_basin[i, :] = max_pdi_per_year_downscaling
    # print(np.corrcoef(mpd_per_year_downscaling_basin[i, :], mpd_per_year)[0,1])

    if basin_id[0] == "S":
        ax = axs.flatten()[i]
        (h1,) = ax.plot(dt_year[1:], n_tc_per_year[1:], "k", linewidth=4)
        (h2,) = ax.plot(
            dt_year[1:], n_tc_per_year_downscaling[1:], color="r", linewidth=3
        )
        mask_yrs = (ds["year"] >= yearS) & (ds["year"] <= yearE)
        ax.grid()
        ax.set_xlim([yearS + 1, yearE])
        # ax.set_xlabel('Year'); ax.set_ylabel('Number of TCs');
        ax.text(
            0.02,
            0.97,
            basin_id,
            transform=ax.transAxes,
            verticalalignment="top",
            weight="bold",
            fontsize=24,
        )
        ax.text(
            0.78,
            0.9,
            "r = %0.2f"
            % float(
                np.corrcoef(n_tc_per_year[1:], n_tc_per_year_downscaling[1:])[0, 1]
            ),
            transform=ax.transAxes,
            bbox=dict(facecolor="gray", edgecolor="black"),
        )
        ax.set_xticks(range(yearS + 1, yearE + 1, 5))
        ax.set_xticklabels(range(yearS + 1, yearE + 1, 5))
        ax.set_ylim([0, np.ceil(np.nanmax(n_tc_per_year) * 1.15 / 5) * 5])
        yLim = ax.get_ylim()
        ax.set_yticks(np.arange(0, yLim[1] + 1, 5))
    else:
        ax = axs.flatten()[i]
        (h1,) = ax.plot(dt_year, n_tc_per_year, "k", linewidth=4)
        (h2,) = ax.plot(dt_year, n_tc_per_year_downscaling, color="r", linewidth=3)
        mask_yrs = (ds["year"] >= yearS) & (ds["year"] <= yearE)
        ax.grid()
        ax.set_xlim([yearS, yearE])
        ax.set_xticks(range(yearS + 1, yearE + 1, 5))
        ax.set_ylim([0, np.ceil(np.nanmax(n_tc_per_year) * 1.15 / 5) * 5])
        # ax.set_xlabel('Year'); ax.set_ylabel('Number of TCs');
        ax.text(
            0.02,
            0.97,
            basin_id,
            transform=ax.transAxes,
            verticalalignment="top",
            weight="bold",
            fontsize=18,
        )
        ax.text(
            0.78,
            0.9,
            "r = %0.2f"
            % float(np.corrcoef(n_tc_per_year, n_tc_per_year_downscaling)[0, 1]),
            transform=ax.transAxes,
            bbox=dict(facecolor="gray", edgecolor="black"),
        )
        yLim = ax.get_ylim()
        ax.set_yticks(np.arange(0, yLim[1] + 1, 5))
        ax.set_xticks(range(yearS, yearE + 1, 5))
        ax.set_xticklabels(range(yearS, yearE + 1, 5))

ax = axs.flatten()[-1]
gl_hist_tc = np.sum(n_tc_per_year_basin, axis=0)
gl_downscaling_tc = np.sum(n_tc_per_year_downscaling_basin, axis=0)
(h1,) = ax.plot(dt_year, gl_hist_tc, "k", linewidth=4)
(h2,) = ax.plot(dt_year, gl_downscaling_tc, color="r", linewidth=3)
mask_yrs = (ds["year"] >= yearS) & (ds["year"] <= yearE)
ax.grid()
ax.set_xlim([yearS + 1, yearE])
# ax.set_ylim([60, 115])
# ax.set_xlabel('Year'); ax.set_ylabel('Number of TCs');
ax.text(
    0.02,
    0.98,
    "SH",
    transform=ax.transAxes,
    verticalalignment="top",
    weight="bold",
    fontsize=18,
)
ax.text(
    0.78,
    0.9,
    "r = %0.2f" % float(np.corrcoef(gl_hist_tc[1:], gl_downscaling_tc[1:])[0, 1]),
    transform=ax.transAxes,
    bbox=dict(facecolor="gray", edgecolor="black"),
)
ax.set_xticks(range(yearS + 1, yearE + 1, 5))
ax.set_xticklabels(range(yearS + 1, yearE + 1, 5))
fig.suptitle("Seasonal TC numbers")
fig.supxlabel("Year")
fig.supylabel("Number of TCs")
savefig(os.path.join(BASEDIR, "seasonal_tc_numbers.png"))

fig, axs = plt.subplots(figsize=(15, 9), ncols=2, nrows=2, sharex=True)
for i, basin_id in enumerate(basin_ids):
    # if basin_id == 'SH':
    #    continue
    idx = np.argwhere(basin_names == basin_id).flatten()[0]
    if basin_id[0] == "S" or basin_id[0] == "A":
        ax = axs.flatten()[i]
        (h1,) = ax.plot(dt_year[1:], mpd_per_year_basin[idx, 1:], "k", linewidth=4)
        (h2,) = ax.plot(
            dt_year[1:], mpd_per_year_downscaling_basin[i, 1:], color="r", linewidth=3
        )
        mask_yrs = (ds["year"] >= yearS) & (ds["year"] <= yearE)
        ax.grid()
        # ax.set_xlim([yearS+1, yearE]);
        ax.set_xlabel("Year")
        ax.set_ylabel("Storm Max. PDI ($m^3 / s^3$)")
        ax.text(
            0.02,
            0.97,
            basin_id,
            transform=ax.transAxes,
            verticalalignment="top",
            weight="bold",
            fontsize=24,
        )
        r2 = float(
            np.corrcoef(
                mpd_per_year_basin[idx, 1:], mpd_per_year_downscaling_basin[i, 1:]
            )[0, 1]
        )
        bias = float(
            mpd_per_year_downscaling_basin[i, 1:].mean()
            - mpd_per_year_basin[idx, 1:].mean()
        )
        ax.text(
            0.78,
            0.93,
            f"r = {r2:0.2f}",
            transform=ax.transAxes,
            bbox=dict(facecolor="gray", edgecolor="black"),
        )
        ax.text(
            0.78,
            0.83,
            f"Bias = {bias/10e6:0.2f}",
            transform=ax.transAxes,
            bbox=dict(facecolor="gray", edgecolor="black"),
        )

        ax.set_xticks(range(yearS + 1, yearE + 1, 5))
        ax.set_xticklabels(range(yearS + 1, yearE + 1, 5))
        ax.ticklabel_format(axis="y", style="sci", scilimits=(6, 6))
        # ax.set_ylim([0, np.ceil(np.nanmax(n_tc_per_year)*1.15 / 5) * 5])
        # yLim = ax.get_ylim(); ax.set_yticks(np.arange(0, yLim[1]+1, 5))
    else:
        ax = axs.flatten()[i]
        (h1,) = ax.plot(dt_year, mpd_per_year_basin[idx, :], "k", linewidth=4)
        (h2,) = ax.plot(
            dt_year, mpd_per_year_downscaling_basin[i, :], color="r", linewidth=3
        )
        mask_yrs = (ds["year"] >= yearS) & (ds["year"] <= yearE)
        ax.grid()
        ax.set_xlim([yearS, yearE])
        ax.set_xticks(range(yearS + 1, yearE + 1, 5))
        # ax.set_ylim([0, np.ceil(np.nanmax(n_tc_per_year)*1.15 / 5) * 5])
        ax.set_xlabel("Year")
        ax.set_ylabel("Storm Max. PDI ($m^3 / s^3$)")
        ax.text(
            0.02,
            0.97,
            basin_id,
            transform=ax.transAxes,
            verticalalignment="top",
            weight="bold",
            fontsize=24,
        )

        r2 = float(
            np.corrcoef(
                mpd_per_year_basin[idx, :], mpd_per_year_downscaling_basin[i, :]
            )[0, 1]
        )
        bias = float(
            mpd_per_year_downscaling_basin[i, :].mean()
            - mpd_per_year_basin[idx, :].mean()
        )
        ax.text(
            0.78,
            0.93,
            f"r = {r2:0.2f}",
            transform=ax.transAxes,
            bbox=dict(facecolor="gray", edgecolor="black"),
        )
        ax.text(
            0.78,
            0.83,
            f"Bias = {bias/10e6:0.2f}",
            transform=ax.transAxes,
            bbox=dict(facecolor="gray", edgecolor="black"),
        )
        ax.ticklabel_format(axis="y", style="sci", scilimits=(6, 6))
        # yLim = ax.get_ylim(); ax.set_yticks(np.arange(0, yLim[1]+1, 5))
        # ax.set_xticks(range(yearS, yearE+1, 5));
        # ax.set_xticklabels(range(yearS, yearE+1, 5));

ax = axs.flatten()[-1]
gl_hist_mpdi = np.sum(mpd_per_year_basin, axis=0)
gl_downscaling_mpdi = np.sum(mpd_per_year_downscaling_basin, axis=0)
ax = plt.gca()
ax.set_axisbelow(True)
ax.bar(dt_year - 0.2, gl_hist_mpdi, 0.4)
ax.bar(dt_year + 0.2, gl_downscaling_mpdi, 0.4, fc=[1, 0, 0, 0.7])
# plt.ylim([0, 1.5e7])
plt.grid()
plt.xlim([1980 - 0.4, 2021 + 0.4])
ax.set_xticks(np.arange(yearS + 1, 2022, 5))
r2 = float(np.corrcoef(gl_hist_mpdi[1:], gl_downscaling_mpdi[1:])[0, 1])
bias = float(gl_downscaling_mpdi[1:].mean() - gl_hist_mpdi[1:].mean())
ax.text(
    0.02,
    0.93,
    f"r = {r2:0.2f}",
    transform=ax.transAxes,
    bbox=dict(facecolor="gray", edgecolor="black"),
)
ax.text(
    0.02,
    0.83,
    f"Bias = {bias/10e6:0.2f}",
    transform=ax.transAxes,
    bbox=dict(facecolor="gray", edgecolor="black"),
)
# yr_labels = np.array([str(x) for x in np.arange(1979, 2022, 1)])
# yr_labels[0::5] = ''; yr_labels[2::5] = ''; yr_labels[3::5] = ''; yr_labels[4::5] = ''
# ax.set_xticklabels(yr_labels)
plt.legend(["Observed", "Downscaling"], ncol=1, loc="upper center")
fig.supylabel("Storm Max. PDI ($m^3 / s^3$)")
fig.supxlabel("Year")
fig.suptitle("Seasonal Max. PDI ($m^3 / s^3$)")
fig.tight_layout()
savefig(os.path.join(BASEDIR, "seasonal_basin_pdi.png"))

plt.figure(figsize=(8, 4))
gl_hist_mpdi = np.sum(mpd_per_year_basin, axis=0)
gl_downscaling_mpdi = np.sum(mpd_per_year_downscaling_basin, axis=0)
ax = plt.gca()
ax.set_axisbelow(True)
ax.bar(dt_year - 0.2, gl_hist_mpdi, 0.4)
ax.bar(dt_year + 0.2, gl_downscaling_mpdi, 0.4, fc=[1, 0, 0, 0.7])
# plt.ylim([0, 1.5e7]);
plt.grid()
plt.xlim([1980 - 0.4, 2021 + 0.4])
ax.set_xticks(np.arange(yearS + 1, 2022, 5))
ax.text(
    0.02,
    0.93,
    "r = %0.2f" % float(np.corrcoef(gl_hist_mpdi[1:], gl_downscaling_mpdi[1:])[0, 1]),
    transform=ax.transAxes,
    bbox=dict(facecolor="gray", edgecolor="black"),
)
# yr_labels = np.array([str(x) for x in np.arange(1979, 2022, 1)])
# yr_labels[0::5] = ''; yr_labels[2::5] = ''; yr_labels[3::5] = ''; yr_labels[4::5] = ''
# ax.set_xticklabels(yr_labels)
plt.legend(["Observed", "Downscaling"], ncol=1, loc="upper right")
plt.ylabel("Storm Max. PDI ($m^3 / s^3$)")
plt.xlabel("Year")
savefig(os.path.join(BASEDIR, "seasonal_pdi.png"))

basin_id = "AU"
basin_mask = basin_hist == basin_id if (basin_id != "GL") else (basin_hist != "GL")
basin_trks_mask = basin_trks == basin_id if (basin_id != "GL") else (basin_trks != "GL")
tc_sample_mask = (dt_ib >= datetime(1979, 1, 1)) & basin_mask
mask = np.nanmax(ds_ib["usa_wind"][tc_sample_mask, :], axis=1) >= 34
seasonal_hist_counts = np.array(
    [dt_ib[i].month for i in range(dt_ib.shape[0]) if tc_sample_mask[i]]
)
hist_season = np.histogram(
    seasonal_hist_counts, bins=np.arange(0.5, 12.6, 1), density="pdf"
)
N_per_year = len(seasonal_hist_counts) / (yearE - yearS + 1)
sept_count = np.sum(hist_season[0][0:]) * N_per_year

tc_months = ds["tc_month"].load().data
n_ss = 300  # int(np.sum(n_tc_per_year))
n_hist = int(N_per_year * (yearE - yearS + 1))
tc_months_subsample = np.zeros((n_hist, n_ss))
for i in range(n_ss):
    tc_months_subsample[:, i] = np.random.choice(tc_months[basin_trks_mask], n_hist)
m_bins = np.arange(0.5, 12.6, 1)
h_season = np.apply_along_axis(
    lambda x: np.histogram(x, bins=m_bins, density="pdf")[0], 0, tc_months_subsample
)
downscaling_fac = sept_count / np.sum(np.nanmean(h_season, axis=1)[0:])
mn_season_pdf = np.nanmean(h_season, axis=1) * downscaling_fac
mn_season_errP = (
    np.nanquantile(downscaling_fac * h_season, 0.975, axis=1) - mn_season_pdf
)
mn_season_errN = mn_season_pdf - np.nanquantile(
    downscaling_fac * h_season, 0.025, axis=1
)

plt.figure(figsize=(8, 6))
plt.bar(
    range(1, 13, 1),
    mn_season_pdf,
    yerr=np.stack((mn_season_errN, mn_season_errP)),
    error_kw={"elinewidth": 2, "capsize": 6},
)
plt.ylabel("Storms per Month")
plt.xlabel("Month")
# plt.bar(range(1, 13, 1), hist_season[0]*np.nanmean(n_tc_per_year), fc=(1, 0, 0, 0.5));
plt.plot(range(1, 13, 1), hist_season[0] * N_per_year, "r*", linewidth=5)
plt.ylabel("Storms per Month")
plt.xlabel("Month")
plt.legend(["Observed", "Downscaling"])
plt.xlim([0.5, 12.5])
plt.xticks(range(1, 13, 1))
plt.grid()
plt.gca().set_axisbelow(True)
savefig(os.path.join(BASEDIR, "monthly_TC_count.png"))

fig, axs = plt.subplots(1, 3, sharex=True, sharey=True, figsize=(15, 6))

for n, basin_id in enumerate(basin_ids):
    ax = axs.flatten()[n]
    basin_mask = basin_trks == basin_id if basin_id != "GL" else basin_trks != "GL"
    basin_hist_mask = basin_hist == basin_id if basin_id != "GL" else basin_hist != "GL"
    y_binEdges = np.arange(lat_min, lat_max + 0.1, 2.5)
    y_binCenter = (y_binEdges[1:] + y_binEdges[0:-1]) / 2

    hist_gen_lat = gen_lat_hist[~np.isnan(gen_lat_hist) & date_mask & basin_hist_mask]
    tc_lat_subsample = np.zeros((int(np.sum(n_tc_per_year)), n_ss))
    for i in range(n_ss):
        tc_lat_subsample[:, i] = np.random.choice(
            lat_genesis[basin_mask], int(np.sum(n_tc_per_year))
        )
    h_lat = np.apply_along_axis(
        lambda x: np.histogram(x, bins=y_binEdges)[0], 0, tc_lat_subsample
    )
    h_lat_hist = np.histogram(hist_gen_lat, bins=y_binEdges)
    # NH Hemisphere
    # idx_et = np.argwhere(y_binCenter >= 30).flatten()[0]
    # downscaling_fac = np.sum(h_lat_hist[0][0:idx_et]) / np.sum(np.nanmean(h_lat, axis = 1)[0:idx_et])
    # SH Hemisphere
    idx_et = np.argwhere(y_binCenter <= 90).flatten()[0]
    downscaling_fac = np.sum(h_lat_hist[0][idx_et:]) / np.sum(
        np.nanmean(h_lat, axis=1)[idx_et:]
    )

    mn_lat_pdf = np.nanmean(h_lat, axis=1) * downscaling_fac
    mn_lat_errP = np.nanquantile(downscaling_fac * h_lat, 0.975, axis=1) - mn_lat_pdf
    mn_lat_errN = mn_lat_pdf - np.nanquantile(downscaling_fac * h_lat, 0.025, axis=1)

    mn_lat_errN = np.where(mn_lat_errN < 0, 0, mn_lat_errN)
    mn_lat_errP = np.where(mn_lat_errP < 0, 0, mn_lat_errP)

    # plt.figure(figsize=(8, 6));
    ax.bar(
        y_binCenter,
        mn_lat_pdf,
        width=2.0,
        yerr=np.stack((mn_lat_errN, mn_lat_errP)),
        error_kw={"elinewidth": 2, "capsize": 6},
    )
    ax.plot(y_binCenter, h_lat_hist[0], "r*", linewidth=5)
    ax.set_xlabel("Latitude")
    # plt.ylabel('Number of Events')
    ax.grid()
    ax.set_xticks(np.arange(-40, 41, 10))
    ax.set_xlim([-45, 0])
    plt.gca().set_axisbelow(True)
    ax.text(0.05, 0.95, basin_id, transform=ax.transAxes, fontweight="bold", va="top")
axs[-1].legend(["Observations", "Downscaling"], loc="upper right")
axs[0].set_ylabel("Number of storms")
fig.suptitle("Genesis latitude")
fig.tight_layout()
savefig(os.path.join(BASEDIR, "genesis_latitude_by_basin.png"))

# Plot return period of severe TCs

x_binEdges = np.arange(lon_min, lon_max + 0.1, 1)
y_binEdges = np.arange(lat_min, lat_max + 0.1, 1)
x_binCenter = (x_binEdges[1:] + x_binEdges[0:-1]) / 2
y_binCenter = (y_binEdges[1:] + y_binEdges[0:-1]) / 2
nX = len(x_binEdges)
nY = len(y_binEdges)

track_arr_hist = np.zeros((180 * track_lon_hist.shape[0], 3))
track_arr_hist[:, 0] = track_lon_hist[:, ::2].flatten()
track_arr_hist[:, 1] = track_lat_hist[:, ::2].flatten()
track_arr_hist[:, 2] = np.tile(
    np.array(range(0, 180)), (track_lon_hist.shape[0], 1)
).flatten()
H_td, _ = np.histogramdd(track_arr_hist, bins=[x_binEdges, y_binEdges, range(0, 180)])

H_count = np.zeros((nX - 1, nY - 1))
H_count_hist = np.zeros((nX - 1, nY - 1))
dt_mask = (dt_ib >= datetime(1979, 1, 1, 0)) & (dt_ib <= datetime(2021, 12, 31, 23))
for i in range(track_lon_hist.shape[0]):
    if ~np.all(np.isnan(track_lon_hist[i, :])) and dt_mask[i]:
        H, _, _ = np.histogram2d(
            track_lon_hist[i, usa_wind[i, :] >= 37],
            track_lat_hist[i, usa_wind[i, :] >= 37],
            bins=[x_binEdges, y_binEdges],
        )
        H_count_hist += np.minimum(H, 1)

for i in range(lon_filt.shape[0]):
    H, _, _ = np.histogram2d(
        lon_filt[i, (vmax_filt[i, :] * 1.94384) >= 37],
        lat_filt[i, (vmax_filt[i, :] * 1.94384) >= 37],
        bins=[x_binEdges, y_binEdges],
    )
    H_count += np.minimum(H, 1)

fig, ax = plt.subplots(3, 1, figsize=(20, 13), subplot_kw={"projection": proj})
ax[0].coastlines(resolution="50m")
ax[0].set_extent([lon_min, lon_max, lat_min, lat_max], crs=trans)
ax[0].gridlines(
    draw_labels=False, crs=trans, xlocs=xlocs, color="gray", alpha=0.3
)
gl = ax[0].gridlines(
    draw_labels=True,
    crs=trans,
    xlocs=xlocs_shift[1:-1],
    color="gray",
    alpha=0.3,
)
gl.bottom_labels = True
gl.top_labels = False
gl.right_labels = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

return_period_fac = np.nanmean(np.sum(n_tc_per_year_basin, axis=0))
return_period = (20 * n_sim) / (H_count.T * 20 / return_period_fac)
return_period_hist = 20 / gaussian_filter(H_count_hist.T, sigma=1)

levels = np.linspace(0, 1.7, 101)
palette = copy(plt.get_cmap("viridis"))
palette.set_over("white", 1.0)
ax[0].contourf(
    x_binCenter,
    y_binCenter,
    np.log10(return_period),
    levels=levels,
    extend="both",
    cmap=palette,
    transform=trans,
)
ax[0].text(
    0.015,
    0.88,
    "Downscaling",
    transform=ax[0].transAxes,
    bbox=dict(facecolor="gray", edgecolor="black"),
)

# ax = fig.add_subplot(312, projection=proj)
ax[1].coastlines(resolution="50m")
ax[1].set_extent([lon_min, lon_max, lat_min, lat_max], crs=trans)
ax[1].gridlines(
    draw_labels=False, crs=trans, xlocs=xlocs, color="gray", alpha=0.3
)
gl = ax[1].gridlines(
    draw_labels=True,
    crs=trans,
    xlocs=xlocs_shift[1:-1],
    color="gray",
    alpha=0.3,
)
gl.bottom_labels = True
gl.top_labels = False
gl.right_labels = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER
im = ax[1].contourf(
    x_binCenter,
    y_binCenter,
    np.log10(return_period_hist),
    levels=levels,
    extend="both",
    cmap=palette,
    transform=trans,
)
ax[1].text(
    0.015,
    0.88,
    "Observations",
    transform=ax[1].transAxes,
    bbox=dict(facecolor="gray", edgecolor="black"),
)

cbar_ax = fig.add_axes([0.25, 0.95, 0.5, 0.015])
cbar = fig.colorbar(
    im,
    cax=cbar_ax,
    ticks=np.log10(np.array([1, 2, 5, 10, 25, 50, 100])),
    orientation="horizontal",
)
cbar.ax.set_xticklabels([1, 2, 5, 10, 25, 50, 100])
cbar.ax.set_xlabel("Return Period (Years)")

diff_ret_pd = return_period_hist - return_period
diff_ret_pd[np.isneginf(diff_ret_pd) | np.isposinf(diff_ret_pd)] = np.nan
log_diff_ret_pd = np.zeros(diff_ret_pd.shape)
lin_thresh = 2
lin_scale = 0.5
log_diff_ret_pd[diff_ret_pd > lin_thresh] = (
    np.log10(diff_ret_pd[diff_ret_pd > lin_thresh]) + lin_scale - np.log10(lin_thresh)
)
log_diff_ret_pd[diff_ret_pd < -lin_thresh] = (
    -np.log10(-diff_ret_pd[diff_ret_pd < -lin_thresh])
    - lin_scale
    + np.log10(lin_thresh)
)
log_diff_ret_pd[np.abs(diff_ret_pd) <= lin_thresh] = (
    diff_ret_pd[np.abs(diff_ret_pd) <= lin_thresh] * (lin_scale * 2) / lin_thresh
)
log_diff_ret_pd[(return_period >= 1e5) | (return_period_hist >= 1e5)] = np.nan
log_diff_ret_pd[log_diff_ret_pd <= -4.25] = np.nan
log_diff_ret_pd[log_diff_ret_pd >= 4.25] = np.nan
log_diff_ret_pd[return_period_hist > 43] = np.nan

# ax = fig.add_subplot(313, projection=proj)
ax[2].coastlines(resolution="50m")
ax[2].set_extent([lon_min, lon_max, lat_min, lat_max], crs=trans)
ax[2].gridlines(
    draw_labels=False, crs=trans, xlocs=xlocs, color="gray", alpha=0.3
)
gl = ax[2].gridlines(
    draw_labels=True,
    crs=trans,
    xlocs=xlocs_shift[1:-1],
    color="gray",
    alpha=0.3,
)
gl.bottom_labels = True
gl.top_labels = False
gl.right_labels = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

palette = copy(plt.get_cmap("jet"))
palette.set_under("white", 1.0)
levels = np.linspace(-1.7 - lin_scale / 2, 1.7 + lin_scale / 2, 101)

im = ax[2].contourf(
    x_binCenter,
    y_binCenter,
    log_diff_ret_pd,
    levels=levels,
    cmap="RdBu_r",
    transform=trans,
)
cbar_ax = fig.add_axes([0.25, 0.05, 0.5, 0.015])
yr_labels = np.array([5, 10, 25, 50])
lin_labels = np.array([-2, -1, 0, 1, 2])
lin_ticks = lin_labels * lin_scale / lin_thresh
yr_ticks = np.concatenate(
    (
        np.flip(-np.log10(yr_labels) - lin_scale + np.log10(lin_thresh)),
        lin_ticks,
        np.log10(yr_labels) + lin_scale - np.log10(lin_thresh),
    )
)
yr_tick_labels = np.concatenate((np.flip(-yr_labels), lin_labels, yr_labels))
cbar = fig.colorbar(im, cax=cbar_ax, ticks=yr_ticks, orientation="horizontal")
cbar.ax.set_xticklabels(yr_tick_labels)
cbar.ax.set_xlabel("Return Period Difference (Years)")
savefig(os.path.join(BASEDIR, "return_period_diff.png"))
