"""
solar elevation and daytime at the mooring

    python pipeline.py solar

what it does
  writes data/processed/daytime/sun_times.csv with sunrise and sunset for every day of the deployment,
  in the instrument clock (clock_utc_offset in config.yaml)
  checks clock_utc_offset against the daytime intervals in dfo's cell exports (WeekN_KrillSv.csv),
  when those files are in dfo_cell_exports
  checks that every labelled region is in daylight

daytime means the sun is more than daytime_min_elevation degrees above the horizon (config.yaml)

other scripts can import
  solar_elevation(times)   degrees above the horizon for instrument clock times
  is_daytime(times)        true where the sun is above daytime_min_elevation
"""

import argparse
import os

import numpy as np
import pandas as pd
import pvlib

from paths import path, setting

LATITUDE = setting("latitude")
LONGITUDE = setting("longitude")
CLOCK_UTC_OFFSET = setting("clock_utc_offset")      # hours, the instrument clock is utc plus this
MIN_ELEVATION = setting("daytime_min_elevation")    # degrees

# offsets compared in the clock check: local standard time, local daylight time, utc
OFFSETS_TO_TEST = [-8, -7, 0]


def to_utc(times, offset_hours=None):
    # instrument clock times have no time zone, so shift them to utc with the clock offset
    offset = CLOCK_UTC_OFFSET if offset_hours is None else offset_hours
    return (pd.DatetimeIndex(times) - pd.Timedelta(hours=offset)).tz_localize("UTC")


def solar_elevation(times, offset_hours=None):
    # apparent elevation includes refraction, so the sun counts as up a few minutes before it rises geometrically
    position = pvlib.solarposition.get_solarposition(to_utc(times, offset_hours), LATITUDE, LONGITUDE,
                                                     method="nrel_numpy")
    return position["apparent_elevation"].to_numpy()


def is_daytime(times, offset_hours=None):
    return solar_elevation(times, offset_hours) > MIN_ELEVATION


def sun_times(first_day, last_day, offset_hours=None):
    # sunrise and sunset for each day in the instrument clock, found on a one minute grid
    minutes = pd.date_range(pd.Timestamp(first_day).normalize(),
                            pd.Timestamp(last_day).normalize() + pd.Timedelta(days=1), freq="1min", inclusive="left")
    up = pd.Series(is_daytime(minutes, offset_hours), index=minutes)
    days = []
    for day, flags in up.groupby(up.index.normalize()):
        daylight = flags.index[flags.values]
        days.append({
            "date": day.date(),
            "sunrise": daylight.min() if len(daylight) else pd.NaT,
            "sunset": daylight.max() if len(daylight) else pd.NaT,
            "day_length_h": round(len(daylight) / 60, 2),
        })
    return pd.DataFrame(days)


def dfo_day_edges(export_dir):
    # first and last ping of each day in dfo's krill cell exports, which only cover daytime
    t = lambda d, h: pd.to_datetime(d.astype(str) + " " + h.str.strip().str[:8])
    edges = []
    for week in range(1, 55):
        f = os.path.join(export_dir, f"Week{week}_KrillSv.csv")
        if not os.path.exists(f):
            continue
        cells = pd.read_csv(f, encoding="utf-8-sig", usecols=["Date_S", "Time_S", "Date_E", "Time_E"])
        start, end = t(cells.Date_S, cells.Time_S), t(cells.Date_E, cells.Time_E)
        days = pd.DataFrame({"start": start, "end": end, "date": start.dt.date})
        per_day = days.groupby("date").agg(first=("start", "min"), last=("end", "max")).reset_index()
        # the first and last day of each file are cut by dfo's weekly window, not by the sun
        edges.append(per_day.iloc[1:-1].assign(week=week))
    return pd.concat(edges, ignore_index=True) if edges else pd.DataFrame()


def check_clock_offset(edges):
    # how far dfo's daytime edges sit from sunrise and sunset for each candidate clock offset
    rows = []
    for offset in OFFSETS_TO_TEST:
        sun = sun_times(edges["date"].min(), edges["date"].max(), offset).set_index("date")
        joined = edges.join(sun, on="date")
        start_gap = (joined["first"] - joined["sunrise"]).dt.total_seconds() / 60
        end_gap = (joined["last"] - joined["sunset"]).dt.total_seconds() / 60
        rows.append({
            "clock_utc_offset": offset,
            "days": len(joined),
            "dfo_start_minus_sunrise_median_min": round(start_gap.median(), 1),
            "dfo_end_minus_sunset_median_min": round(end_gap.median(), 1),
            "median_abs_gap_min": round(pd.concat([start_gap, end_gap]).abs().median(), 1),
        })
    return pd.DataFrame(rows)


def check_regions(split):
    # sun elevation at the start and end of every labelled region
    # a region is kept when it starts in daylight, so the few that end minutes after sunset stay in
    regions = pd.read_csv(path("processed") / split / "regions_manifest.csv", parse_dates=["start_time", "end_time"])
    regions["elevation_start"] = solar_elevation(regions["start_time"]).round(1)
    regions["elevation_end"] = solar_elevation(regions["end_time"]).round(1)
    regions["daylight"] = regions["elevation_start"] > MIN_ELEVATION
    regions["ends_after_sunset"] = regions["daylight"] & (regions["elevation_end"] <= MIN_ELEVATION)
    return regions


def main():
    parser = argparse.ArgumentParser(description="Sunrise and sunset, clock check, and daylight check of the labelled regions.")
    parser.add_argument("--split", default=setting("split"))
    args = parser.parse_args()

    print(f"mooring {LATITUDE}, {LONGITUDE}; clock offset {CLOCK_UTC_OFFSET} h from utc; "
          f"daytime = sun above {MIN_ELEVATION} degrees")

    # sunrise and sunset for the whole deployment
    out_dir = path("daytime")
    os.makedirs(out_dir, exist_ok=True)
    sun = sun_times("2025-03-01", "2026-03-13")
    sun.to_csv(out_dir / "sun_times.csv", index=False)
    print(f"Saved: {out_dir / 'sun_times.csv'} ({len(sun)} days, day length "
          f"{sun.day_length_h.min()} to {sun.day_length_h.max()} h)")

    # clock check against dfo's daytime cell exports
    export_dir = path("dfo_cell_exports")
    if export_dir.exists():
        edges = dfo_day_edges(export_dir)
        result = check_clock_offset(edges)
        best = result.loc[result["median_abs_gap_min"].idxmin(), "clock_utc_offset"]
        print(f"\nClock check against DFO's daytime intervals ({len(edges)} days):")
        print(result.to_string(index=False))
        print(f"Best match: clock_utc_offset = {best} "
              f"({'matches' if best == CLOCK_UTC_OFFSET else 'DIFFERS FROM'} config.yaml)")
    else:
        print(f"\nNo DFO cell exports at {export_dir}, skipping the clock check")

    # labelled regions in daylight
    regions = check_regions(args.split)
    show = ["key", "label", "start_time", "end_time", "elevation_start", "elevation_end", "split"]
    dark = regions[~regions["daylight"]]
    late = regions[regions["ends_after_sunset"]]
    print(f"\nLabelled regions kept (start in daylight): {regions['daylight'].sum()} of {len(regions)}")
    if len(dark):
        print("Not kept (start before sunrise or after sunset):")
        print(dark[show].to_string(index=False))
    if len(late):
        print(f"Kept, but end after sunset ({len(late)}):")
        print(late[show].to_string(index=False))


if __name__ == "__main__":
    main()
