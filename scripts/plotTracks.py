"""
Plot sample tracks from downscaling
@author: Craig Arthur
"""

import os
import glob
import xarray as xr
import numpy as np


import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

import sys

cwd = os.path.dirname(os.path.realpath("."))
parent = os.path.dirname(cwd)
sys.path.append(cwd)

import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import seaborn as sns
from calendar import month_abbr

import namelist

colours=['0.75', '#0FABF6', '#0000FF',
         '#00FF00', '#FF8100', '#ff0000']
intervals=[0, 17.5, 24.5, 32.5, 44.2, 55.5, 1000]
cmap = ListedColormap(colours)
norm = BoundaryNorm(intervals, cmap.N)

def fn_plot_duplicates(fn_plot):
    f_int = 0
    fn_plot_out = fn_plot
    while os.path.exists(fn_plot_out):
        fn_plot_out = fn_plot.rstrip('.png') + '_e%d.png' % f_int
        f_int += 1
    return fn_plot_out

def plotTracks(ntracks: int):
    exp_name = namelist.exp_name
    startdate = f"{namelist.start_year}{namelist.start_month:02d}"
    enddate = f"{namelist.end_year}{namelist.end_month:02d}"

    filename = f"tracks_*_era5_{startdate}_{enddate}*.nc"
    fn_tracks = sorted(glob.glob(f"{namelist.base_directory}/{exp_name}/{filename}"))
    ds = xr.open_mfdataset(
        fn_tracks,
        concat_dim="n_trk",
        combine="nested",
        data_vars="minimal",
        drop_variables="seeds_per_month",
    )

    yearS = namelist.start_year
    yearE = namelist.end_year
    n_sim = len(fn_tracks)
    ntrks_per_year = namelist.tracks_per_year

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

    lon_min = 90
    lon_max = 180
    lat_min = -45
    lat_max = 0

    proj = ccrs.PlateCarree(central_longitude=180)

    fig, ax = plt.subplots(1, 1, figsize=(12, 8), subplot_kw={"projection": proj})
    ax.coastlines(resolution="50m")
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    gl = ax.gridlines(
        draw_labels=True,
        crs=ccrs.PlateCarree(),
        xlocs=np.arange(20, 240, 20),
        color="gray",
        alpha=0.3,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.add_feature(cfeature.LAND)
    r_idxs = np.random.randint(0, lon_filt.shape[0], size=ntracks)

    for idx in r_idxs:
        lats = lat_filt[idx, :].T
        lons = lon_filt[idx, :].T
        vmax = vmax_filt[idx, :].T
        for i in range(len(lats) - 1):
            plt.plot([lons[i], lons[i+1]],
                     [lats[i], lats[i+1]],
                     color=cmap(norm(vmax[i])),
                     transform=ccrs.PlateCarree(),
                     alpha=0.9,
                     linewidth=1,
                     zorder=max(vmax))

        plt.scatter(
            lon_filt[idx, 0],
            lat_filt[idx, 0],
            color="k",
            s=5,
            transform=ccrs.PlateCarree(),
        )
    labels = ['TL', 'TC1', 'TC2', 'TC3', 'TC4', 'TC5']
    handles = []
    for c, l in zip(cmap.colors, labels):
        handles.append(Line2D([0], [0], color=c, label=l))
    ax = plt.gca()
    ax.legend(handles, labels, loc=3, ncols=2, frameon=True, fontsize='xx-small')
    fname = fn_plot_duplicates("%s/%s/tracks.png" % (namelist.base_directory, exp_name))
    plt.savefig(
        fname,
        bbox_inches="tight",
        dpi=600,
    )

def plotGenesis(ntracks: int, colfield='month'):
    exp_name = namelist.exp_name
    startdate = f"{namelist.start_year}{namelist.start_month:02d}"
    enddate = f"{namelist.end_year}{namelist.end_month:02d}"

    filename = f"tracks_*_era5_{startdate}_{enddate}*.nc"
    fn_tracks = sorted(glob.glob(f"{namelist.base_directory}/{exp_name}/{filename}"))
    ds = xr.open_mfdataset(
        fn_tracks,
        concat_dim="n_trk",
        combine="nested",
        data_vars="minimal",
        drop_variables="seeds_per_month",
    )

    yearS = namelist.start_year
    yearE = namelist.end_year
    n_sim = len(fn_tracks)
    ntrks_per_year = namelist.tracks_per_year

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

    lon_min = 90
    lon_max = 180
    lat_min = -45
    lat_max = 0

    proj = ccrs.PlateCarree(central_longitude=180)

    fig, ax = plt.subplots(1, 1, figsize=(12, 6), subplot_kw={"projection": proj})
    ax.coastlines(resolution="50m")
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=ccrs.PlateCarree())

    gl = ax.gridlines(
        draw_labels=True,
        crs=ccrs.PlateCarree(),
        xlocs=np.arange(20, 240, 20),
        color="gray",
        alpha=0.3,
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'size': 10}
    gl.ylabel_style = {'size': 10}
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    ax.add_feature(cfeature.LAND)
    r_idxs = np.random.randint(0, lon_filt.shape[0], size=ntracks)

    cs = ax.scatter(lon_filt[r_idxs, 0],
                    lat_filt[r_idxs, 0],
                    c=mnth_trks[r_idxs],
                    s=5,
                    transform=ccrs.PlateCarree(),
                    cmap=sns.color_palette("Paired", 12, as_cmap=True),
                    vmin=1, vmax=12
    )


    handles, labels = cs.legend_elements()
    labels = [month_abbr[i] for i in range(1, 13)]
    plt.legend(handles, labels, title="Month", bbox_to_anchor=(0.5, 0),
           loc="lower center", ncols=6, bbox_transform=fig.transFigure)
    fname = fn_plot_duplicates("%s/%s/genesis.png" % (namelist.base_directory, exp_name))
    plt.savefig(
        fname,
        bbox_inches="tight",
        dpi=600,
    )