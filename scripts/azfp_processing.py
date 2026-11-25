
import echopype as ep
import echopype.colormap
from echopype.qc import exist_reversed_time
import matplotlib.pyplot as plt
import pvlib
import pandas as pd
import xarray as xr
import numpy as np
import os


"""
By Ali Stein-Gagnon for CSC 475 Fall 2026

small edits: olivia whitelaw

Much of this code was adapted from the Echopype documentation:
https://echopype.readthedocs.io/en/latest/convert.html

Mostly from the raw data processing and visualization sections.

Purpose and KPIs:
1. Calculate solar elevation using ping time and mooring coordinates.
2. Calculate dB differencing between the two AZFP channels.
3. Visualize calibrated backscatter and frequency differences.
4. Seperates daytime hours.

TODO: Masking for fishes, saving to netcdf files
"""

# --------------------------------------------------
# MANUAL CONFIGURATION VARIABLES
# --------------------------------------------------

# folder made by sort_raw_by_week.py, containing week1, week2, ... week54
# input your own path to the week folder
# Ex. "Users/oliviawhitelaw/Documents/CSC 475/where-has-my-fish-gone/2025-03-01_MalaspinaStrait/2025-03-01_MalaspinaStrait/AZFP/55158/Weeks"
WEEKS_DIR = r"/Users/oliviawhitelaw/Documents/CSC 475/where-has-my-fish-gone/2025-03-01_MalaspinaStrait/2025-03-01_MalaspinaStrait/AZFP/55158/Weeks"

# one Sv file saved per week: week1_Sv.zarr, week2_Sv.zarr, ..
OUTPUT_DIR = r"data/processed/Sv"

# which weeks to run: = None means run all the weeks and Ex. [1] to test the first week
WEEKS_TO_RUN = [1]

# skip weeks that have a saved Sv file 
SKIP_EXISTING = True

# plots: echograms, freq diff, and solar elevation
# Note: turn off when running all files 
MAKE_PLOTS = True

# also open the plots in a window False = only save them
SHOW_PLOTS = False

PLOTS_DIR = r"data/processed/Sv/plots"
 
# DFO's labelled regions (made by parse_cropped_regions.py) drawn on the echograms
REGIONS_PATH = r"data/processed/split_7030/regions_manifest.csv"

# calibration table: sound speed (SS_m_s) and absorption (AC_dB_m) per month and frequency
CALIBRATION_PATH = r"data/calibration/azfp_calibration.csv"

# The month is not set by hand: each week uses the calibration of the month with the most
# days of data in that week folder (read from the .01A file names, YYMMDDHH.01A).
 
# Instrument depth (m), used to turn distance from the transducer into depth below the surface:
# depth = INSTRUMENT_DEPTH - echo_range. DFO's region depths line up with this (week 1 check plots);
# the RBR logger's monthly mean depth is 116.4-116.7 m
INSTRUMENT_DEPTH = 116.5

# enough to see whether schools line up with DFO's boxes without making dozens of plots per week
N_REGION_PLOTS = 3
 
# colour scale for echograms (dB) Slonimer tested for his images (−100 to −20 dB)
SV_MIN, SV_MAX = -100, -30

# edit: month is no longer set by hand: it is read from each .01A file name (YYMMDDHH.01A)
# YEAR_MONTH = "2026-02"

# Deployment coordinates
# Best these remain hardcoded as I took them from the R file and they shouldnt change for all of the Malaspina data
LATITUDE = 49.5348
LONGITUDE = -124.0110

# AZFP channel identifiers
CHANNEL_67 = "55158-67-1"
CHANNEL_125 = "55158-125-2"

# Environmental parameters
# salinity and pressure are only used by Echopype to calculate sound speed and absorption.
# We give it those directly from the .ECS values (see get_environmental_parameters),
# so these values do not affect Sv. They are only here because Echopype requires them for AZFP data.
SALINITY = 35  
PRESSURE = 117 # roughly 


# --------------------------------------------------
# LOAD AND COMBINE RAW AZFP FILES
# --------------------------------------------------

# edit: added week_calibration_month and find_files

# the month with the most days of data in this week decides the calibration

    # Ex. week 5 (Sat Mar 29 to Fri Apr 4) has 3 March days and 4 April days, so it uses April's values
    # days are counted from the files that exist, so week 53 (Feb 28 to Mar 6, 2026) uses february:
    # recording stopped on Mar 1, 2026, so there are no March days with data.
 
    # AZFP files are named YYMMDDHH.01A, so the first 6 characters are the date, Ex: 250329 = Mar 29, 2025
def week_calibration_month(echo_files):
    
    days = pd.Series([os.path.basename(f)[0:6] for f in echo_files]).drop_duplicates()
    months = pd.to_datetime(days, format="%y%m%d").dt.strftime("%Y-%m")
    return months.value_counts().idxmax()


# All files in a folder with this extension, .01A or .XML
def find_files(folder, extension):
    return sorted(
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.upper().endswith(extension.upper())
    )


def load_echo_files(echo_files, xml_path):


    ed_list = []

    for echo_file in echo_files:
        print(f"Opening: {echo_file}")

        ed = ep.open_raw(
            echo_file,
            sonar_model="AZFP",
            xml_path=xml_path
        )

        ed_list.append(ed)

    print("Files opened:", len(ed_list))

    print("Combining EchoData files...")

    combined_ed = ep.combine_echodata(
        echodata_list=ed_list
    )

    return combined_ed, ed_list


# --------------------------------------------------
# GET ENVIRONMENTAL PARAMETERS
# --------------------------------------------------

def get_environmental_parameters(year_month):
 
    # sound speed and absorption from the .ECS values for this month
    # Translates 2025-03 to Mar 2025 because the month format used in azfp_calibration.csv is different
    cal_df = pd.read_csv(CALIBRATION_PATH)
    month_label = pd.to_datetime(year_month).strftime("%b %Y")
    month_rows = cal_df[cal_df["month"] == month_label].set_index("frequency_khz")
 
    if len(month_rows) != 2:
        raise ValueError(f"Expected 2 calibration rows for {month_label}, found {len(month_rows)}")

    # only grab from 67 kHz because sound speed is the same for both frequencies 
    sound_speed = float(month_rows.loc[67, "SS_m_s"])
 
    # absorption differs by frequency so Echopype needs one value per channel
    sound_absorption = xr.DataArray(
        [float(month_rows.loc[67, "AC_dB_m"]), float(month_rows.loc[125, "AC_dB_m"])],
        dims=["channel"],
        coords={"channel": [CHANNEL_67, CHANNEL_125]},
    )
 
    # echopype still requires salinity and pressure for AZFP,
    # but uses the sound speed and absorption given here instead of calculating its own
    env_params = {
        "salinity": SALINITY,
        "pressure": PRESSURE,
        "sound_speed": sound_speed,
        "sound_absorption": sound_absorption,
    }
 
    print(f"{month_label}: sound speed {sound_speed} m/s, "
          f"absorption {sound_absorption.values.tolist()} dB/m (67, 125 kHz)")
 
    return env_params

# --------------------------------------------------
#  BACKSCATTER
# --------------------------------------------------

def calculate_backscatter(combined_ed, env_params):
 
    print("Calculating volume backscattering strength (Sv)...")
 
    ds_Sv = ep.calibrate.compute_Sv(
        combined_ed,
        env_params=env_params,

        # Echopype adds ASL's AZFP Sv offset by default (1.1 dB for our settings).
        # Echoview's CalibrationOffsetSv defaults to 0 and DFO's .ECS does not set it so we manually set ours to 0 to match
        cal_params={"Sv_offset": 0}
    )

    # compute_Sv only gives echo_range: the distance from the transducer.
    # DFO's region depths are depth below the surface, so add depth = INSTRUMENT_DEPTH - echo_range
    # ex. Herring Week1_r4 is 5-37 m deep = 80-112 m from the transducer.
    ds_Sv = ep.consolidate.add_depth(ds_Sv, depth_offset=INSTRUMENT_DEPTH, downward=False)
 
    return ds_Sv


# prints the calibration values Echopype actually used to make the Sv
# expect: this month's sound speed and absorption, Sv_offset 0, and the XML values for the rest
def print_calibration_used(ds_Sv):
    
    print("Calibration values used (67 kHz, 125 kHz):")
    for name in ["sound_speed", "sound_absorption", "Sv_offset",
                 "EL", "DS", "TVR", "VTX0", "equivalent_beam_angle"]:
        if name in ds_Sv:
            values = ds_Sv[name]

            # sound speed can be stored once per ping; all pings have the same value
            if "ping_time" in values.dims:
                values = values.isel(ping_time=0)
            print(f"  {name}: {values.values}")


# temp comment out - will move to other file 
# # --------------------------------------------------
# # CALCULATE SOLAR ELEVATION
# # --------------------------------------------------

# def calculate_solar_elevation(ds_Sv):

#     ping_times = pd.DatetimeIndex(ds_Sv["ping_time"].values)

#     if ping_times.tz is None:
#         ping_times = ping_times.tz_localize("UTC")

#     # Calculate solar position
#     solar_position = pvlib.solarposition.get_solarposition(
#         time=ping_times,
#         latitude=LATITUDE,
#         longitude=LONGITUDE
#     )

#     # Plot solar elevation
#     plt.figure(figsize=(12, 5))

#     plt.plot(
#         ping_times,
#         solar_position["apparent_elevation"],
#         marker=".",
#         markersize=2,
#         linestyle="-"
#     )

#     plt.xlabel("Ping Time (UTC)")
#     plt.ylabel("Solar Elevation (degrees)")
#     plt.title("Solar Elevation at Each AZFP Ping")
#     plt.xticks(rotation=45)
#     plt.grid()
#     plt.tight_layout()
#     plt.show()

#     return solar_position


# # --------------------------------------------------
# # CALCULATE FREQUENCY DIFFERENCE
# # --------------------------------------------------

# def calculate_frequency_difference(ds_Sv):

#     Sv_67 = ds_Sv["Sv"].sel(channel=CHANNEL_67)
#     Sv_125 = ds_Sv["Sv"].sel(channel=CHANNEL_125)

#     Sv_difference = Sv_67 - Sv_125

#     # Plot dB difference
#     Sv_difference.plot(
#         x="ping_time",
#         figsize=(10, 7),
#         vmin=-15,
#         vmax=15,
#         cmap="RdBu_r"
#     )

#     plt.title("AZFP Frequency Difference (125 - 67 kHz)")
#     plt.show()

#     return Sv_difference


# --------------------------------------------------
# CHECKS AND PLOTS Note will need adjusting 
# --------------------------------------------------
 
CHANNELS = {CHANNEL_67: "67 kHz", CHANNEL_125: "125 kHz"}
REGION_COLOURS = {"Herring": "magenta", "Salmon": "lime"}
 
 
def print_week_summary(ds_Sv):
    # A quick overview of the week's Sv: how much data, and whether the values look sensible
    times = pd.DatetimeIndex(ds_Sv["ping_time"].values)
    gaps = times.to_series().diff().dt.total_seconds()
 
    print(f"Pings: {len(times)}, from {times[0]} to {times[-1]}")
    print(f"Ping interval: usually {gaps.median():.0f} s; gaps longer than 60 s: {(gaps > 60).sum()}"
          f" (longest {gaps.max() / 60:.1f} min)")
 
    for channel, name in CHANNELS.items():
        sv = ds_Sv["Sv"].sel(channel=channel).isel(ping_time=slice(None, None, 10)).values
        rng = ds_Sv["echo_range"].sel(channel=channel).isel(ping_time=0).values
        print(f"{name}: range {np.nanmin(rng):.2f} to {np.nanmax(rng):.2f} m, "
              f"Sv median {np.nanmedian(sv):.1f} dB (5-95%: {np.nanpercentile(sv, 5):.1f} to "
              f"{np.nanpercentile(sv, 95):.1f} dB), empty values {np.isnan(sv).mean() * 100:.1f}%")
 
 
def load_week_regions(week):
    # DFO's herring and salmon regions for this week.
    # The .EVR depths are depth below the surface, so they are compared with the Sv "depth".
    if not os.path.exists(REGIONS_PATH):
        print(f"No region file at {REGIONS_PATH}, plotting without regions")
        return pd.DataFrame()

    regions = pd.read_csv(REGIONS_PATH)
    regions = regions[regions["week"] == week].copy()

    # Column names: depth_min_m/depth_max_m (new parser), or older names
    regions = regions.rename(columns={"range_min_m": "depth_min_m", "range_max_m": "depth_max_m",
                                      "depth_top": "depth_min_m", "depth_bottom": "depth_max_m"})

    

    # region times are the instrument clock as written in the .EVR, the same clock as ping_time
    # time zone is unconfirmed (probably local standard time, UTC-8), so no time zone is attached
    regions["start"] = pd.to_datetime(regions["start_time"], format="ISO8601")
    regions["end"] = pd.to_datetime(regions["end_time"], format="ISO8601")

    return regions
 
 
def get_depth(ds_Sv, channel):
    # Depth below the surface for one channel (same for every ping)
    rng = ds_Sv["echo_range"].sel(channel=channel).isel(ping_time=0).values
    return INSTRUMENT_DEPTH - rng


def draw_echogram(ax, ds_Sv, channel, title):
    # Echogram of one channel: time across, depth down (surface at the top, like Echoview)
    sv = ds_Sv["Sv"].sel(channel=channel)
    depth = get_depth(ds_Sv, channel)
    mesh = ax.pcolormesh(sv["ping_time"].values, depth, sv.values.T,
                         vmin=SV_MIN, vmax=SV_MAX, cmap="ep.ek500", shading="auto")
    ax.set_ylim(INSTRUMENT_DEPTH, depth.min())  # deepest at the bottom
    ax.set_ylabel("Depth (m)")
    ax.set_title(title)
    return mesh
 
 
def draw_regions(ax, regions):
    # DFO's region boxes (start-end time, min-max depth)
    for r in regions.itertuples():
        colour = REGION_COLOURS.get(r.label, "white")
        ax.add_patch(plt.Rectangle((r.start, r.depth_min_m), r.end - r.start,
                                   r.depth_max_m - r.depth_min_m,
                                   fill=False, edgecolor=colour, linewidth=1.5))
 
 
def plot_week_echogram(ds_Sv, regions, week):
    # The whole week for both channels, with DFO's regions.
    # Every 10th ping is shown so the plot stays light; the saved Sv keeps every ping.
    ds_small = ds_Sv.isel(ping_time=slice(None, None, 10))
 
    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    for ax, (channel, name) in zip(axes, CHANNELS.items()):
        mesh = draw_echogram(ax, ds_small, channel, f"Week {week}, {name}")
        draw_regions(ax, regions)
    fig.colorbar(mesh, ax=axes, label="Sv (dB)")
    axes[-1].set_xlabel("Ping time (instrument clock)")
    return fig
 
 
def plot_region_zoom(ds_Sv, region, week):
    # One labelled region close up, at full resolution, with 10 minutes either side.
    # If the box sits on the school, the times and ranges line up with DFO's.
    margin = pd.Timedelta(minutes=10)
    ds_zoom = ds_Sv.sel(ping_time=slice(region.start - margin, region.end + margin))
    if ds_zoom.sizes["ping_time"] == 0:
        print(f"Region {region.key}: no pings at {region.start}, cannot plot")
        return None
 
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for ax, (channel, name) in zip(axes, CHANNELS.items()):
        mesh = draw_echogram(ax, ds_zoom, channel, f"{region.key} ({region.label}), {name}")
        draw_regions(ax, pd.DataFrame([region._asdict()]))
        ax.tick_params(axis="x", rotation=45)
    fig.colorbar(mesh, ax=axes, label="Sv (dB)")
    return fig
 
 
def plot_sv_distribution(ds_Sv, week):
    # Left: how common each Sv value is. Right: average Sv at each range.
    # The right panel should show a strong peak near the surface (about 110-120 m from the transducer).
    ds_small = ds_Sv.isel(ping_time=slice(None, None, 10))
 
    fig, (ax_hist, ax_prof) = plt.subplots(1, 2, figsize=(14, 5))
    for channel, name in CHANNELS.items():
        sv = ds_small["Sv"].sel(channel=channel)
        values = sv.values.ravel()
        ax_hist.hist(values[np.isfinite(values)], bins=200, range=(-140, 0), histtype="step", label=name)
 
        # Average in linear units, then back to dB
        mean_profile = 10 * np.log10(np.nanmean(10 ** (sv.values / 10), axis=0))
        ax_prof.plot(mean_profile, get_depth(ds_small, channel), label=name)
 
    ax_hist.set_xlabel("Sv (dB)")
    ax_hist.set_ylabel("Number of pixels")
    ax_hist.set_title(f"Week {week}: Sv values")
    ax_hist.legend()
    ax_prof.set_xlabel("Mean Sv (dB)")
    ax_prof.set_ylabel("Depth (m)")
    ax_prof.invert_yaxis()
    ax_prof.set_title(f"Week {week}: mean Sv by depth")
    ax_prof.legend()
    return fig
 
 
def save_figure(fig, name):
    if fig is None:
        return
    path = os.path.join(PLOTS_DIR, name)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    print(f"Plot saved: {path}")
    if not SHOW_PLOTS:
        plt.close(fig)
 
 
def make_week_plots(ds_Sv, week):
    os.makedirs(PLOTS_DIR, exist_ok=True)
    regions = load_week_regions(week)
    print(f"DFO regions this week: {len(regions)}")
 
    save_figure(plot_week_echogram(ds_Sv, regions, week), f"week{week}_echogram.png")
    save_figure(plot_sv_distribution(ds_Sv, week), f"week{week}_sv_distribution.png")
    for region in regions.head(N_REGION_PLOTS).itertuples():
        save_figure(plot_region_zoom(ds_Sv, region, week), f"week{week}_region_{region.key}.png")
 
    if SHOW_PLOTS:
        plt.show()
        plt.close("all")


# --------------------------------------------------
# MAIN PROCESSING
# --------------------------------------------------


def process_week(week, folder):
 
    # .01A files in this week's folder (sorted Sat-Fri by sort_raw_by_week.py)
    echo_files = find_files(folder, ".01A")
    xml_files = find_files(folder, ".XML")
 
 
    if not echo_files:
        print("No .01A files in this folder, skipping")
        return None
    if not xml_files:
        raise FileNotFoundError(f"No .XML file in {folder}")
 
    # the settings are identical in every month's XML so any XML in the folder works
    xml_path = xml_files[0]
 
    # one calibration for the whole week: the month with the most days of data
    year_month = week_calibration_month(echo_files)
 
    
    combined_ed, ed_list = load_echo_files(echo_files, xml_path)
    env_params = get_environmental_parameters(year_month)
    ds_Sv = calculate_backscatter(combined_ed, env_params)
 
    # release the ed objects or there will be 100 errors
    del combined_ed
    del ed_list
 
    # check the values Echopype actually used
    print_calibration_used(ds_Sv)
 
    # Check that ping times only go forward. A jump back in time would put DFO's regions in the wrong place
    # only warns of error doesn't change the data 
    if exist_reversed_time(ds_Sv, "ping_time"):
        print(f"WARNING: week {week} has ping times that go backwards. Check before using it.")
    else:
        print("Ping times OK: always increasing")
 
    # record which month's calibration was used
    ds_Sv.attrs["week"] = week
    ds_Sv.attrs["calibration_month"] = year_month
 
    return ds_Sv



def main():


    os.makedirs(OUTPUT_DIR, exist_ok=True)
 
    # Week folders made by sort_raw_by_week.py: WEEKS_DIR/week1 ... WEEKS_DIR/week54
    weeks = range(1, 55) if WEEKS_TO_RUN is None else WEEKS_TO_RUN
    print(f"Weeks to process: {list(weeks)}")
 
    for week in weeks:
 
        folder = os.path.join(WEEKS_DIR, f"week{week}")
        if not os.path.isdir(folder):
            print(f"Week {week}: no folder at {folder}, skipping")
            continue
 
        out_path = os.path.join(OUTPUT_DIR, f"week{week}_Sv.zarr")
        if SKIP_EXISTING and os.path.exists(out_path):
            print(f"Week {week} already saved, loading it for the summary and plots: {out_path}")
            ds_Sv = xr.open_zarr(out_path)
            print_week_summary(ds_Sv)
            if MAKE_PLOTS:
                make_week_plots(ds_Sv, week)
            continue
 
        ds_Sv = process_week(week, folder)
        if ds_Sv is None:
            continue
 
        # Save this week's Sv
        ds_Sv.to_zarr(out_path, mode="w")
        print(f"Saved: {out_path}")
 
        # Overview of the week, then the check plots
        print_week_summary(ds_Sv)
        if MAKE_PLOTS:
            make_week_plots(ds_Sv, week)
 
        del ds_Sv
 
    print("\nAZFP processing complete")
   



    # # Load and combine the 01A files
    # combined_ed, ed_list = load_echo_files()

    # env_params = get_environmental_parameters()

    # ds_Sv = calculate_backscatter(combined_ed, env_params)

    # #TODO: Save the calibrated backscatter netcdf file(s) to outputs. Named per day or per week

    # # IMPORTANT release the ed objects or there will be 100 errors
    # del combined_ed
    # del ed_list

    # solar_position = calculate_solar_elevation(ds_Sv)

    # Sv_difference = calculate_frequency_difference(ds_Sv)

    # plot_backscatter(ds_Sv)

    # print("AZFP processing complete.")


if __name__ == "__main__":
    main()
