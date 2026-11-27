"""
data cleaning: mark bad data with a mask for each week's Sv, without changing the Sv

    python pipeline.py clean --weeks 3 53     # some weeks
    python pipeline.py clean --all            # every week with a saved Sv file

reason codes (powers of 2, so a pixel can carry several reasons and its value is their sum)
  1   dfo bad data region: every region of type 0 "bad (no data)" in the .EVR files
      regions with depths of 9999.99 cover the whole ping (echoview's full depth), and so do rectangles
      drawn from above the surface to the transducer (within EDGE_TOLERANCE)
      rectangles are masked as boxes and drawn polygons by their outline
      the region type decides, its classification (for example herring on a boat ping) is ignored
  2   end of deployment: everything from bad_data_after in config.yaml (signal lost on 1 mar 2026)
  4   above dfo's surface exclusion line, from the cell exports (WeekN_KrillSv.csv, one line per 10 min
      interval, interpolated between interval centres)
      during a labelled herring or salmon region the line is raised to the region's top, so the labelled
      school is never cut by the surface mask (dfo drew some near surface herring boxes above their own line)
  8   below dfo's near transducer exclusion line (110 m depth, about 6 m above the transducer)
  16  daytime ping with no dfo surface line within dfo_line_max_gap_min, so the surface cannot be removed
      (night pings are not coded here, only daytime data are used and solar.py decides what is daytime)

outputs
  data/processed/Sv/weekN_mask.zarr             uint8 mask (channel, ping_time, range_sample) on the Sv grid,
                                                0 is good, use it as Sv.where(mask == 0)
  data/processed/bad_data/bad_data_list.csv     one row per bad region or rule, with how many pings it hit
  data/processed/bad_data/mask_summary.csv      one row per week, % of pings and pixels masked per reason
  data/processed/bad_data/dfo_lines.csv         dfo's surface and near transducer lines, one row per interval
  data/processed/bad_data/regions_masked.csv    % of each labelled region masked, per reason

dfo drew the regions and lines on one echogram, so they are applied to both channels (67 and 125 kHz)
region and line depths are depth below the surface, compared with the Sv depth (INSTRUMENT_DEPTH minus
echo_range), the same comparison the check plots use
"""

import argparse
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.path import Path as Polygon

from paths import path, setting
from parse_cropped_regions import parse_evr_file, get_week_number
from solar import is_daytime

# zarr warns about consolidated metadata on every write, it does not affect the files
warnings.filterwarnings("ignore", message=".*Consolidated metadata.*")

CODES = {
    1: "DFO bad data region (.EVR type 0)",
    2: "end of deployment (after bad_data_after in config.yaml)",
    4: "above DFO surface exclusion line",
    8: "below DFO near transducer exclusion line",
    16: "daytime ping with no DFO surface line",
}
FULL_DEPTH = 9000          # a depth beyond this in an .EVR means the whole ping (echoview writes 9999.99)

# dfo's whole echogram rectangles end at 115.24 m but the Sv's first sample is at 115.25 m, slightly deeper,
# so a one sample strip just above the transducer was left unmasked, this allowance (m) closes it
EDGE_TOLERANCE = 0.5
BAD_TYPES = {"bad (no data)"}

# pings further than this from any dfo interval get no surface line (code 16 in daytime)
MAX_GAP_S = setting("dfo_line_max_gap_min") * 60


def epoch_seconds(times):
    # ping_time has no time zone (instrument clock), parse_evr_file uses the same convention
    # times read from csv can come back in seconds or microseconds, so convert to nanoseconds first
    return pd.DatetimeIndex(times).as_unit("ns").asi8 / 1e9


# dfo bad data regions (code 1)

def load_bad_regions(evr_dir):
    # every bad data region in every WeekN.EVR, with its outline in (epoch seconds, depth)
    rows = []
    for f in sorted(Path(evr_dir).glob("*.[Ee][Vv][Rr]")):
        for reg in parse_evr_file(f):
            if reg["region_type"] not in BAD_TYPES:
                continue
            t, z = reg["poly"][:, 0], reg["poly"][:, 1]
            rows.append({
                "file": f.name,
                "evr_week": get_week_number(f.name, t.min()),
                "region_id": int(reg["region_id"]),
                "classification": reg["classification"],
                "n_points": reg["n_points"],
                "t0": t.min(), "t1": t.max(),
                "depth_top": z.min(), "depth_bottom": z.max(),
                "whole_ping": bool((np.abs(z) > FULL_DEPTH).all()),
                "rectangle": len(np.unique(t)) == 2 and len(np.unique(z)) == 2,
                "poly": reg["poly"],
            })
    return pd.DataFrame(rows)


def covers_whole_ping(reg, depth):
    # true for 9999.99 regions and for rectangles drawn over the whole echogram
    if reg.whole_ping:
        return True
    return bool(reg.rectangle and reg.depth_top <= np.nanmin(depth) + EDGE_TOLERANCE
                and reg.depth_bottom >= np.nanmax(depth) - EDGE_TOLERANCE)


def mask_region(mask, reg, ping_s, depth, whole_ping):
    # set code 1 inside one region and return the number of pings it touches
    # ping times are sorted, so the region's pings are one block from i0 to i1
    i0 = np.searchsorted(ping_s, reg.t0, side="left")
    i1 = np.searchsorted(ping_s, reg.t1, side="right")
    if i1 <= i0:
        return 0
    if whole_ping:
        mask[:, i0:i1, :] |= 1
        return int(i1 - i0)

    for ch in range(depth.shape[0]):
        rows = np.where((depth[ch] >= reg.depth_top) & (depth[ch] <= reg.depth_bottom))[0]
        if len(rows) == 0:
            continue
        r0, r1 = rows.min(), rows.max() + 1          # depth falls steadily with range, so also one block
        if reg.rectangle:
            mask[ch, i0:i1, r0:r1] |= 1
        else:
            # drawn polygon, test every pixel in its bounding box against the outline
            tt, dd = np.meshgrid(ping_s[i0:i1], depth[ch, r0:r1], indexing="ij")
            inside = Polygon(reg.poly).contains_points(np.column_stack([tt.ravel(), dd.ravel()]))
            mask[ch, i0:i1, r0:r1] |= inside.reshape(tt.shape).astype(np.uint8)
    return int(i1 - i0)


# dfo surface and near transducer lines (codes 4, 8 and 16)

def load_dfo_lines(export_dir, cache, refresh=False):
    # one row per 10 min interval from every WeekN_KrillSv.csv, cached in dfo_lines.csv
    if cache.exists() and not refresh:
        return pd.read_csv(cache, parse_dates=["start", "end"])
    if not Path(export_dir).exists():
        raise FileNotFoundError(f"No DFO cell exports at {export_dir} (dfo_cell_exports in config.yaml)")
    t = lambda d, h: pd.to_datetime(d.astype(str) + " " + h.str.strip().str[:8])
    weeks = []
    for week in range(1, 55):
        f = Path(export_dir) / f"Week{week}_KrillSv.csv"
        if not f.exists():
            continue
        cells = pd.read_csv(f, encoding="utf-8-sig",
                            usecols=["Interval", "Date_S", "Time_S", "Date_E", "Time_E",
                                     "Exclude_above_line_depth_mean", "Exclude_below_line_depth_mean"])
        cells["start"], cells["end"] = t(cells.Date_S, cells.Time_S), t(cells.Date_E, cells.Time_E)
        weeks.append(cells.groupby("Interval").agg(
            start=("start", "min"), end=("end", "max"),
            surface_depth=("Exclude_above_line_depth_mean", "first"),
            near_transducer_depth=("Exclude_below_line_depth_mean", "first"),
        ).assign(week=week))
    lines = pd.concat(weeks).sort_values("start").reset_index(drop=True)
    lines.to_csv(cache, index=False)
    print(f"Saved: {cache} ({len(lines)} intervals from {lines.week.nunique()} weeks)")
    return lines


def surface_per_ping(ping_s, lines):
    # dfo surface line depth for each ping, nan where no interval is within MAX_GAP_S
    s, e = epoch_seconds(lines["start"]), epoch_seconds(lines["end"])
    mid, value = (s + e) / 2, lines["surface_depth"].to_numpy()

    # interval before (or containing) each ping and the one after it
    prev = np.searchsorted(s, ping_s, side="right") - 1
    nxt = prev + 1
    has_prev, has_next = prev >= 0, nxt < len(s)
    gap_prev = np.where(has_prev, ping_s - e[np.clip(prev, 0, None)], np.inf)     # 0 or less means inside
    gap_next = np.where(has_next, s[np.clip(nxt, None, len(s) - 1)] - ping_s, np.inf)
    gap = np.minimum(np.clip(gap_prev, 0, None), gap_next)

    # inside an interval or in a short gap between two, interpolate between interval centres
    surface = np.interp(ping_s, mid, value)
    # in a long gap (night, edge of dfo's window), hold the nearest interval's value instead
    long_gap = has_prev & has_next & ((s[np.clip(nxt, None, len(s) - 1)] - e[np.clip(prev, 0, None)]) > MAX_GAP_S)
    long_gap |= ~has_prev | ~has_next
    nearest = np.where(np.clip(gap_prev, 0, None) <= gap_next, np.clip(prev, 0, None), np.clip(nxt, None, len(s) - 1))
    surface = np.where(long_gap & (gap > 0), value[nearest], surface)
    surface[gap > MAX_GAP_S] = np.nan
    return surface


def raise_line_over_regions(surface, ping_s, week_labelled):
    # during each labelled region, move the surface line up to the region's top if the region reaches above it
    # returns how far the line was raised for each region (m, 0 when it did not need to move)
    raised = {}
    for r in week_labelled.itertuples():
        i0 = np.searchsorted(ping_s, epoch_seconds([r.start_time])[0], side="left")
        i1 = np.searchsorted(ping_s, epoch_seconds([r.end_time])[0], side="right")
        before = surface[i0:i1].copy()
        surface[i0:i1] = np.fmin(surface[i0:i1], r.depth_top)
        raised[r.key] = round(float(np.nanmax(before - surface[i0:i1], initial=0)), 2)
    return raised


def mask_lines(mask, ping_s, ping_times, depth, lines, week_labelled):
    # codes 4 (above the surface line), 8 (below the near transducer line) and 16 (daytime, no surface line)
    surface = surface_per_ping(ping_s, lines)
    raised = raise_line_over_regions(surface, ping_s, week_labelled)
    near_transducer = float(lines["near_transducer_depth"].median())
    for ch in range(depth.shape[0]):
        above = depth[ch][None, :] < surface[:, None]         # nan surface compares false, so nothing masked
        mask[ch][above] |= 4
        mask[ch][:, depth[ch] > near_transducer] |= 8
    no_line = np.isnan(surface) & is_daytime(ping_times)
    mask[:, no_line, :] |= 16
    return surface, near_transducer, no_line, raised


# one week

def make_week_mask(week, sv_path, regions, bad_after, lines, labelled):
    ds = xr.open_zarr(sv_path)
    ping_times = pd.DatetimeIndex(ds.ping_time.values)
    ping_s = epoch_seconds(ping_times)
    # depth below the surface for each range sample, per channel (the same for every ping in a week)
    depth = ds["depth"].isel(ping_time=0).values
    mask = np.zeros(ds["Sv"].shape, dtype=np.uint8)      # (channel, ping_time, range_sample)

    listed = []
    week_regions = regions[(regions.t1 >= ping_s.min()) & (regions.t0 <= ping_s.max())]
    for reg in week_regions.itertuples():
        whole = covers_whole_ping(reg, depth)
        n = mask_region(mask, reg, ping_s, depth, whole)
        listed.append({
            "week": week, "code": 1, "reason": CODES[1],
            "source": f"{reg.file} region {reg.region_id}", "classification": reg.classification,
            "start": pd.to_datetime(reg.t0, unit="s").round("s"), "end": pd.to_datetime(reg.t1, unit="s").round("s"),
            "depth_top": np.nan if reg.whole_ping else round(reg.depth_top, 2),
            "depth_bottom": np.nan if reg.whole_ping else round(reg.depth_bottom, 2),
            "whole_ping": whole, "shape": "rectangle" if reg.rectangle else f"polygon ({reg.n_points} points)",
            "pings": n,
        })

    after = ping_s >= bad_after.value / 1e9
    if after.any():
        mask[:, after, :] |= 2
        listed.append({
            "week": week, "code": 2, "reason": CODES[2], "source": "config.yaml bad_data_after",
            "classification": "", "start": ping_times[after][0], "end": ping_times[after][-1],
            "depth_top": np.nan, "depth_bottom": np.nan, "whole_ping": True, "shape": "", "pings": int(after.sum()),
        })

    week_labelled = labelled[labelled.week == week]
    surface, near_transducer, no_line, raised = mask_lines(mask, ping_s, ping_times, depth, lines, week_labelled)
    daytime = is_daytime(ping_times)

    out = xr.Dataset(
        {"mask": (("channel", "ping_time", "range_sample"), mask),
         "surface_line_depth": (("ping_time",), surface.astype(np.float32))},
        coords={"channel": ds.channel, "ping_time": ds.ping_time, "range_sample": ds.range_sample},
        attrs={"week": week, "codes": "; ".join(f"{k} = {v}" for k, v in CODES.items()),
               "near_transducer_line_depth": near_transducer,
               "use": "Sv.where(mask == 0); a pixel's value is the sum of its reason codes"},
    )

    summary = {
        "week": week, "pings": mask.shape[1], "daytime_pings": int(daytime.sum()),
        "pings_fully_masked": int((mask != 0).all(axis=(0, 2)).sum()),
        "pct_pings_fully_masked": round(100 * (mask != 0).all(axis=(0, 2)).mean(), 2),
        "pct_pixels_masked": round(100 * (mask != 0).mean(), 2),
        "pct_daytime_pixels_masked": round(100 * (mask[:, daytime, :] != 0).mean(), 2) if daytime.any() else np.nan,
        "daytime_pings_no_surface_line": int(no_line.sum()),
    }
    for code in CODES:
        summary[f"pct_pixels_code{code}"] = round(100 * ((mask & code) != 0).mean(), 2)

    # how much of each labelled region is masked, per reason
    region_rows = []
    for r in week_labelled.itertuples():
        i0 = np.searchsorted(ping_s, epoch_seconds([r.start_time])[0], side="left")
        i1 = np.searchsorted(ping_s, epoch_seconds([r.end_time])[0], side="right")
        cells = np.concatenate([mask[ch, i0:i1][:, (depth[ch] >= r.depth_top) & (depth[ch] <= r.depth_bottom)].ravel()
                                for ch in range(depth.shape[0])])
        row = {"key": r.key, "label": r.label, "week": week, "split": r.split, "pixels": cells.size,
               "surface_line_raised_m": raised.get(r.key, 0.0),
               "pct_masked": round(100 * (cells != 0).mean(), 2) if cells.size else np.nan}
        for code in CODES:
            row[f"pct_code{code}"] = round(100 * ((cells & code) != 0).mean(), 2) if cells.size else np.nan
        region_rows.append(row)
    return out, listed, summary, region_rows


def main():
    parser = argparse.ArgumentParser(description="Data cleaning: bad data masks for each week")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--weeks", type=int, nargs="+")
    which.add_argument("--all", action="store_true")
    parser.add_argument("--refresh-lines", action="store_true", help="reread dfo's cell exports instead of dfo_lines.csv")
    args = parser.parse_args()

    sv_dir, out_dir = path("sv_output"), path("bad_data")
    os.makedirs(out_dir, exist_ok=True)
    bad_after = pd.Timestamp(setting("bad_data_after"))

    regions = load_bad_regions(path("regions_evr"))
    print(f"DFO bad data regions: {len(regions)} "
          f"({regions.whole_ping.sum()} whole ping, {regions.rectangle.sum()} rectangles, "
          f"{(~regions.rectangle).sum()} polygons)")
    print(f"End of deployment: everything from {bad_after} is bad")
    lines = load_dfo_lines(path("dfo_cell_exports"), out_dir / "dfo_lines.csv", args.refresh_lines)
    print(f"DFO lines: {len(lines)} intervals, surface line {lines.surface_depth.min():.2f} to "
          f"{lines.surface_depth.max():.2f} m, near transducer line {lines.near_transducer_depth.median():.1f} m")
    labelled = pd.read_csv(path("processed") / setting("split") / "regions_manifest.csv",
                           parse_dates=["start_time", "end_time"])

    weeks = range(1, 55) if args.all else args.weeks
    listed, summaries, region_rows = [], [], []
    for week in weeks:
        sv_path = sv_dir / f"week{week}_Sv.zarr"
        if not sv_path.exists():
            print(f"Week {week}: no Sv file, skipping")
            continue
        out, rows, summary, reg = make_week_mask(week, sv_path, regions, bad_after, lines, labelled)
        out_path = sv_dir / f"week{week}_mask.zarr"
        out.to_zarr(out_path, mode="w")
        listed += rows
        summaries.append(summary)
        region_rows += reg
        print(f"Week {week}: {summary['pct_daytime_pixels_masked']}% of daytime pixels masked, "
              f"{summary['daytime_pings_no_surface_line']} daytime pings without a surface line -> {out_path.name}")

    # keep earlier weeks' rows when only some weeks were rerun
    def save(rows, name, key):
        new = pd.DataFrame(rows)
        f = out_dir / name
        if f.exists() and not args.all:
            old = pd.read_csv(f)
            new = pd.concat([old[~old.week.isin(new.week.unique())], new])
        new.sort_values(key).to_csv(f, index=False)
        print(f"Saved: {f}")
        return new

    save(listed, "bad_data_list.csv", ["week", "code", "start"])
    save(summaries, "mask_summary.csv", ["week"])
    reg = save(region_rows, "regions_masked.csv", ["week", "key"])
    hit = reg[reg.pct_masked > 0]
    moved = reg[reg.surface_line_raised_m > 0]
    print(f"\nSurface line raised over {len(moved)} labelled regions, by up to {reg.surface_line_raised_m.max()} m")
    print(f"Labelled regions with masked pixels: {len(hit)} of {len(reg)}")
    for code in CODES:
        n = (reg[f"pct_code{code}"] > 0).sum()
        if n:
            print(f"  code {code:2d} ({CODES[code]}): {n} regions, largest {reg[f'pct_code{code}'].max()}%")


if __name__ == "__main__":
    main()
