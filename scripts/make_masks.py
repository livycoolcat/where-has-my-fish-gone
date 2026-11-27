"""
Data cleaning, step 1: mark bad data. Builds a mask for each week's Sv, without changing the Sv.

    python pipeline.py clean --weeks 3 53     # some weeks
    python pipeline.py clean --all            # every week with a saved Sv file

What counts as bad (step 1):
  code 1  DFO bad-data region: every region of type 0 "bad (no data)" in the .EVR files.
          Regions with depths of -9999.99 / 9999.99 cover the whole ping (Echoview's "full depth"), and so
          do rectangles drawn from above the surface to the transducer (within EDGE_TOLERANCE).
          Rectangles are masked as boxes; drawn polygons are masked by their outline.
          The region's classification (e.g. "Herring" on some single-ping bad regions) is ignored:
          the region type decides (still to confirm with DFO).
  code 2  End of deployment: everything from bad_data_after in config.yaml (signal lost on
          1 Mar 2026 ~17:03; DFO's own bad region starts 16:59:29 but their Week53.EVR ends
          6 Mar 17:07 and there is no Week54.EVR).

Later steps will add more codes (powers of 2, so a pixel can carry several reasons):
  4 no surface echo, 8 near the transducer, 16 at/above the sea surface, 32 noise spike ...

Outputs
  data/processed/Sv/weekN_mask.zarr        uint8 "mask" (channel, ping_time, range_sample), same grid as
                                           the Sv. 0 = good; otherwise the sum of the reason codes.
                                           Use it as:  Sv.where(mask == 0)
  data/processed/bad_data/bad_data_list.csv  one row per bad region / rule, with how many pings it hit
  data/processed/bad_data/mask_summary.csv   one row per week: % of pings and pixels masked per reason

DFO drew the regions on one echogram, so each region is applied to both channels (67 and 125 kHz).
Region depths are depth below the surface, compared with the Sv "depth" (INSTRUMENT_DEPTH - echo_range),
the same comparison the check plots use.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.path import Path as Polygon

from paths import path, setting
from parse_cropped_regions import parse_evr_file, get_week_number

CODES = {
    1: "DFO bad-data region (.EVR type 0)",
    2: "end of deployment (after bad_data_after in config.yaml)",
}
FULL_DEPTH = 9000          # |depth| above this in an .EVR means "whole ping" (Echoview writes 9999.99)
# A rectangle drawn from above the surface down to the transducer is also a whole ping. Its edges sit
# within a sample or so of the Sv's first/last depth (Echoview and this code place depth slightly
# differently), so allow this much (m) at each end instead of leaving a onesample sliver unmasked

# DFO's whole-echogram rectangles end at 115.24 m
# the Sv's first sample is at 115.25 m, slightly deeper, 
# so strip just above the transducer was left unmasked I added a 0.5 m allowance
EDGE_TOLERANCE = 0.5
BAD_TYPES = {"bad (no data)"}


def epoch_seconds(times):
    # ping_time has no time zone (instrument clock); parse_evr_file uses the same convention (UTC arithmetic)
    return pd.DatetimeIndex(times).asi8 / 1e9


def load_bad_regions(evr_dir):
    # Every bad-data region in every WeekN.EVR, with its outline in (epoch seconds, depth)
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
    # True for -9999.99/9999.99 regions and for rectangles drawn over the whole echogram
    if reg.whole_ping:
        return True
    return bool(reg.rectangle and reg.depth_top <= np.nanmin(depth) + EDGE_TOLERANCE
                and reg.depth_bottom >= np.nanmax(depth) - EDGE_TOLERANCE)


def mask_region(mask, reg, ping_s, depth, whole_ping):
    # Set code 1 inside one region. Returns the number of pings it touches.
    # ping times are sorted, so the region's pings are one block: i0 .. i1-1
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
            # drawn polygon: test every pixel in its bounding box against the outline
            tt, dd = np.meshgrid(ping_s[i0:i1], depth[ch, r0:r1], indexing="ij")
            inside = Polygon(reg.poly).contains_points(np.column_stack([tt.ravel(), dd.ravel()]))
            mask[ch, i0:i1, r0:r1] |= inside.reshape(tt.shape).astype(np.uint8)
    return int(i1 - i0)


def make_week_mask(week, sv_path, regions, bad_after):
    ds = xr.open_zarr(sv_path)
    ping_s = epoch_seconds(ds.ping_time.values)
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
            "classification": "", "start": pd.Timestamp(ds.ping_time.values[after][0]),
            "end": pd.Timestamp(ds.ping_time.values[after][-1]),
            "depth_top": np.nan, "depth_bottom": np.nan, "whole_ping": True, "shape": "",
            "pings": int(after.sum()),
        })

    out = xr.Dataset(
        {"mask": (("channel", "ping_time", "range_sample"), mask)},
        coords={"channel": ds.channel, "ping_time": ds.ping_time, "range_sample": ds.range_sample},
        attrs={"week": week, "codes": "; ".join(f"{k} = {v}" for k, v in CODES.items()),
               "use": "Sv.where(mask == 0); a pixel's value is the sum of its reason codes"},
    )

    n_pings = mask.shape[1]
    summary = {
        "week": week, "pings": n_pings,
        "pings_fully_masked": int((mask != 0).all(axis=(0, 2)).sum()),
        "pct_pings_fully_masked": round(100 * (mask != 0).all(axis=(0, 2)).mean(), 2),
        "pct_pixels_masked": round(100 * (mask != 0).mean(), 2),
    }
    for code in CODES:
        summary[f"pct_pixels_code{code}"] = round(100 * ((mask & code) != 0).mean(), 2)
    return out, listed, summary


def main():
    parser = argparse.ArgumentParser(description="Step 1 of data cleaning: bad-data masks for each week.")
    which = parser.add_mutually_exclusive_group(required=True)
    which.add_argument("--weeks", type=int, nargs="+")
    which.add_argument("--all", action="store_true")
    args = parser.parse_args()

    sv_dir, out_dir = path("sv_output"), path("bad_data")
    os.makedirs(out_dir, exist_ok=True)
    bad_after = pd.Timestamp(setting("bad_data_after"))

    regions = load_bad_regions(path("regions_evr"))
    print(f"DFO bad-data regions: {len(regions)} "
          f"({regions.whole_ping.sum()} whole-ping, {regions.rectangle.sum()} rectangles, "
          f"{(~regions.rectangle).sum()} polygons)")
    print(f"End of deployment: everything from {bad_after} is bad")

    weeks = range(1, 55) if args.all else args.weeks
    listed, summaries = [], []
    for week in weeks:
        sv_path = sv_dir / f"week{week}_Sv.zarr"
        if not sv_path.exists():
            print(f"Week {week}: no Sv file, skipping")
            continue
        out, rows, summary = make_week_mask(week, sv_path, regions, bad_after)
        out_path = sv_dir / f"week{week}_mask.zarr"
        out.to_zarr(out_path, mode="w")
        listed += rows
        summaries.append(summary)
        print(f"Week {week}: {summary['pct_pixels_masked']}% of pixels masked, "
              f"{summary['pings_fully_masked']} whole pings -> {out_path.name}")

    # Keep earlier weeks' rows when only some weeks were rerun
    def save(rows, name, key):
        new = pd.DataFrame(rows)
        f = out_dir / name
        if f.exists() and not args.all:
            old = pd.read_csv(f)
            new = pd.concat([old[~old.week.isin(new.week.unique())], new])
        new.sort_values(key).to_csv(f, index=False)
        print(f"Saved: {f}")

    save(listed, "bad_data_list.csv", ["week", "code", "start"])
    save(summaries, "mask_summary.csv", ["week"])


if __name__ == "__main__":
    main()
