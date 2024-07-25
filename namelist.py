import os
import numpy as np

"""
Namelist file that serves as the configuration file for the TC-risk model.
"""

########################## File System Parameters ###########################
src_directory = os.path.dirname(os.path.abspath(__file__))
base_directory = '%s/data/era5' % src_directory
output_directory = '%s/data/era5' % src_directory
exp_name = "beta"
# For now, we support either 'GCM' or 'ERA5'. Different file types and variable
# names can be added by modifying the "input.py" file and adding the appropriate
# variable key words in the structure var_keys.
dataset_type = 'ERA5' #'GCM'
exp_prefix = 'era5' #GFDL-CM4_ssp585_r1i1p1f1'

# Variable naming based on dataset_type.
# 'sst' is sea-surface temperature (monthly-averaged)
# 'mslp' is mean sea-level pressure (monthly-averaged)
# 'temp' is temperature (monthly-averaged)
# 'sp_hum' is specific humidity (monthly-averaged)
# 'u' is zonal wind (daily)
# 'v' is meridional wind (daily)
var_keys = {'ERA5': {'sst': 'sst', 'mslp': 'sp', 'temp': 't',
                     'sp_hum': 'q', 'u': 'u', 'v': 'v',
                     'lvl': 'level', 'lon': 'longitude', 'lat': 'latitude'},
            'GCM': {'sst': 'tas', 'mslp': 'psl', 'temp': 'ta',
                    'sp_hum': 'hus', 'u': 'ua', 'v': 'va',
                    'lvl': 'plev', 'lon': 'lon', 'lat': 'lat'}}

########################### Data Input Parameters ###########################
data_ts = 'monthly'      # timestep of input data, 'monthly' or '6-hourly'
                         # resolutions are supported.
########################### Parallelism Parameters ##########################
n_procs = 96              # number of processes to use in dask
scheduler = 'processes'   # specify the dask scheduler to use
############################ TC Risk Parameters #############################
"""
These parameters configure the dates for the TC-risk model.
"""
start_year = 1981                     # year to start downscaling
start_month = 1                       # month of start_year to start downscaling
end_year = 2021                       # year to stop downscaling
end_month = 12                        # month of end_year to stop downscaling

"""
These parameters configure the output.
"""
output_interval_s = 3600              # output interval of tracks, seconds (does not change time integration)
total_track_time_days = 15            # total time to integrate tracks, days
tracks_per_year = 20                  # total number of tracks to simulate per year

"""
These parameters configure thermodynamics and thermodynamic constants.
"""
p_midlevel = 60000
PI_reduc = 0.80
Ck = 1.2e-3
Cd = 1.2e-3
Cd_land = 4.7e-3
select_thermo = 1   # 1 for pseudoadiabatic, 2 for reversible thermodynamics
select_interp = 2   # 1 for computation, 2 for interpolation

"""
These parameters configure track and intensity constants.
"""
# Defines the steering levels (hPa) of the storm (see paper)
# If 250- and 850-hPa, uses two steering levels.
# If 250-, 500-, and 850-hPa, uses three steering levels.
# The steering_coefficients ('steering_coefs') should have the same
# length as the number of levels.
steering_levels = [250, 850]
steering_coefs = [0.2, 0.8]           # constant steering coefficients if not coupled

# Coefficients for climatological gradients.
# Intercept, dudy, dvdx, dzdy, dudp, dvdp.
# These coefficients are for SH storms only - WCA 2024-06-28
gradient_coefs = [[-0.603, -0.113, -0.169, 0.0, -0.099, -0.062],
                  [-0.075, -0.041, 0.0, -0.029, -0.033, -0.093]]
coupled_track = True                  # track coupled to intensity; overrides alpha
y_alpha = [0.1464, 0.8536]            # value of steering coefficient at 0 knots
m_alpha = [0.00231, -0.00231]         # change of each coefficient per unit storm intensity, 1 / kts
alpha_max = [0.3716, 0.8086]          # maximum value of each steering coefficient (coupled track only)
alpha_min = [0.1914, 0.6284]          # minimum value of each steering coefficient (coupled track only)
u_beta = -1.0                         # zonal beta drift, m/s
v_beta = 2.5                          # meridional beta drift, m/s
T_days = 20                           # period of the fourier series, days
seed_v_init_ms = 5                    # initial seed v intensity, m/s
seed_v_2d_threshold_ms = 6.5          # seed v threshold after 2 days, m/s
seed_v_threshold_ms = 15              # seed v threshold over entire lifetime, m/s
seed_vmax_threshold_ms = 18           # seed vmax threshold over entire lifetime, m/s
# Atmospheric boundary layer depth (FAST), m
atm_bl_depth = {'NA': 1400.0, 'EP': 1400.0, 'WP': 1800.0, 'AU': 1800.0,
                'SI': 1600.0, 'SP': 2000.0, 'NI': 1500.0}
# (WCA) Number of sine waves for the Fourier Series:
N_sine_waves = {'NA': 15, 'EP': 15, 'WP': 15, 'AU': 25,
                'SI': 15, 'SP': 25, 'NI': 15}
log_chi_fac = 0.5                     # addition to chi in log space
chi_fac = 1.3                         # addition to chi
lat_vort_fac = 2                      # sets where vorticity threshold decays toward equator
lat_vort_power = {'NA': 6, 'EP': 6,   # power decay towards the equator
                  'WP': 3.5, 'AU': 8,
                  'SI': 6, 'SP': 10, 'NI': 2.5}
# Initial m based on large-scale relative humidity
f_mInit = lambda rh: 0.20 / (1 + np.exp(-(rh - 0.55) * 10)) + 0.125

"""
Basins for which the model is enabled.
The basin bounds dictionary maps a basin identifier to the basin boundaries.
The bounds are ordered as (LL - Lower Left, UR - Upper Right):
[LL Longitude, LL Latitude, UR Longitude, UR Latitude].
Note the basins bounds are extended slightly to allow tracks to
extend slightly beyond the bounds, as this may be the case when a TC starts
in one basin and goes to another.
Identifiers: EP - Eastern Pacific
             NA - North Atlantic
             NI - North Indian
             SI - South Indian
             SP - South Pacific
             WP - Western Pacific
             GL - Global (no basin)
"""
basin_bounds = {'EP': ['180E', '0N', '290E', '60N'],
                'NA': ['260E', '0N', '360E', '60N'],
                'NI': ['30E', '0N', '100E', '50N'],
                'SI': ['20E', '45S', '100E', '0S'],
                'AU': ['100E', '45S', '180E', '0S'],
                'SP': ['180E', '45S', '250E', '0S'],
                'WP': ['100E', '0N', '180E', '60N'],
                'GL': ['0E', '45S', '360E', '0S']}  # WCA: Change to Southern Hemisphere

