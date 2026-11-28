#!/usr/bin/env python3
# cuts one fixed size window of Sv around every labelled region, for the cnn
# the window is the same size for every region (64 pings x 512 samples, about 13 min x 57 m),
# centred on the middle of the region's box, so school size and shape stay comparable between regions
# saved in raw dB, nothing clipped, scaled or resized, so the dB range, floor and input size can be
# chosen later (e.g. in a colab notebook) without cropping again
# masked pixels (make_masks.py), missing pings and anything outside the data are NaN
#
# to call: python pipeline.py crops   (or python scripts/make_crops.py)
#
# outputs in the crops folder of config.yaml:
#   crops.npz        crops (n, 2, 512, 64) float32 dB, channel 0 = 67 kHz, channel 1 = 125 kHz,
#                    row 0 = shallowest depth, column 0 = earliest ping
#                    region_mask (n, 512, 64) uint8, 1 inside DFO's region polygon
#                    keys (n,) region keys in the same order as crops_index.csv
#   crops_index.csv  one row per crop: key, label, week, window start time, depth of row 0, ...

import argparse

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib.path import Path as PolyPath

from paths import path, setting

INSTRUMENT_DEPTH = 116.5          # same as azfp_processing.py: depth = INSTRUMENT_DEPTH - echo_range
CHANNELS = ["55158-67-1", "55158-125-2"]
PING_SECONDS = 12                 # one ping every 12 s (azfp_sampling.csv, burst_interval_s)


def window_times(region):
    # ping times of the window: n_pings on a regular 12 s grid centred on the middle of the region
    n = setting("crop_window")["pings"]
    middle = region.start_time + (region.end_time - region.start_time) / 2
    offsets = (np.arange(n) - n // 2) * PING_SECONDS
    return middle + pd.to_timedelta(offsets, unit="s")


def window_rows(depth, middle_depth, n_rows):
    # range_sample index for each row of the window, row 0 shallowest, or -1 outside the data
    # depth falls as range_sample rises (upward looking), so shallower rows have higher indices
    centre = int(np.argmin(np.abs(depth - middle_depth)))
    idx = centre + n_rows // 2 - np.arange(n_rows)
    return np.where((idx >= 0) & (idx < len(depth)), idx, -1)


def crop_week(week, regions, polygons):
    # all crops for one week, returns crops, region masks and index rows
    n_pings, n_rows = setting("crop_window")["pings"], setting("crop_window")["samples"]
    sv_file = path("sv_output") / f"week{week}_Sv.zarr"
    mask_file = path("sv_output") / f"week{week}_mask.zarr"
    ds = xr.open_zarr(sv_file)
    mask = xr.open_zarr(mask_file)["mask"] if mask_file.exists() else None
    if mask is None:
        print(f"Week {week}: no mask file, crops are not masked")

    # nearest ping within half a ping interval, so a missing ping becomes a NaN column
    tolerance = np.timedelta64(PING_SECONDS // 2, "s")
    depth = {ch: INSTRUMENT_DEPTH - ds["echo_range"].sel(channel=ch).isel(ping_time=0).values for ch in CHANNELS}
    bin_m = float(np.abs(np.diff(depth[CHANNELS[0]][:2]))[0])

    crops, masks, rows = [], [], []
    for r in regions.itertuples():
        times = window_times(r)
        sv = ds["Sv"].reindex(ping_time=times.values, method="nearest", tolerance=tolerance)
        bad = None
        if mask is not None:
            bad = mask.reindex(ping_time=times.values, method="nearest", tolerance=tolerance)

        middle_depth = (r.depth_top + r.depth_bottom) / 2
        crop = np.full((2, n_rows, n_pings), np.nan, dtype=np.float32)
        for c, ch in enumerate(CHANNELS):
            idx = window_rows(depth[ch], middle_depth, n_rows)
            ok = idx >= 0
            values = sv.sel(channel=ch).isel(range_sample=idx[ok]).values.T     # rows x pings
            if bad is not None:
                m = bad.sel(channel=ch).isel(range_sample=idx[ok]).values.T
                values = np.where(m == 0, values, np.nan)                      # NaN mask value (missing ping) is dropped too
            crop[c, ok, :] = values

        # region polygon on the window grid (67 kHz depths), pixel centres inside the polygon are 1
        # depth of row 0 counted from the centre row, so it is right even when row 0 is above the data
        centre = int(np.argmin(np.abs(depth[CHANNELS[0]] - middle_depth)))
        row0_depth = float(depth[CHANNELS[0]][centre]) - (n_rows // 2) * bin_m
        poly = polygons[r.key]
        t0 = times[0].value / 1e9
        poly_cols = (poly[:, 0] - t0) / PING_SECONDS
        poly_rows = (poly[:, 1] - row0_depth) / bin_m
        cc, rr = np.meshgrid(np.arange(n_pings), np.arange(n_rows))
        inside = PolyPath(np.column_stack([poly_cols, poly_rows])).contains_points(
            np.column_stack([cc.ravel(), rr.ravel()]))
        region_mask = inside.reshape(n_rows, n_pings).astype(np.uint8)

        # the region is cut off if its box reaches past the window
        clipped = bool(poly_cols.min() < -0.5 or poly_cols.max() > n_pings - 0.5
                       or poly_rows.min() < -0.5 or poly_rows.max() > n_rows - 0.5)
        missing_pings = int(np.isnan(sv.isel(channel=0, range_sample=0).values).sum())

        crops.append(crop)
        masks.append(region_mask)
        rows.append({"key": r.key, "label": r.label, "week": week, "day": r.day, "season": r.season,
                     "window_start": times[0], "window_end": times[-1], "row0_depth_m": round(row0_depth, 3),
                     "bin_m": round(bin_m, 4), "region_pixels": int(region_mask.sum()), "clipped": clipped,
                     "missing_pings": missing_pings,
                     "nan_share_67": round(float(np.isnan(crop[0]).mean()), 3),
                     "nan_share_125": round(float(np.isnan(crop[1]).mean()), 3)})
    return crops, masks, rows


def main():
    parser = argparse.ArgumentParser(description="Fixed size Sv windows around every labelled region, for the CNN.")
    parser.add_argument("--split", default=setting("split"),
                        help="split whose regions_manifest.csv lists the regions (every split has the same regions)")
    parser.add_argument("--weeks", nargs="+", type=int, help="only these weeks (default: every week with regions)")
    args = parser.parse_args()

    folder = path("processed") / args.split
    regions = pd.read_csv(folder / "regions_manifest.csv", parse_dates=["start_time", "end_time"])
    polygons = np.load(folder / "regions_polygons.npz")
    weeks = sorted(regions["week"].unique()) if not args.weeks else args.weeks

    all_crops, all_masks, all_rows = [], [], []
    for week in weeks:
        week_regions = regions[regions["week"] == week]
        if not (path("sv_output") / f"week{week}_Sv.zarr").exists():
            print(f"Week {week}: no Sv file, skipped {len(week_regions)} regions (run python pipeline.py sv)")
            continue
        crops, masks, rows = crop_week(week, week_regions, polygons)
        all_crops += crops
        all_masks += masks
        all_rows += rows
        print(f"Week {week}: {len(crops)} crops")

    out = path("crops")
    out.mkdir(parents=True, exist_ok=True)
    index = pd.DataFrame(all_rows)
    np.savez_compressed(out / "crops.npz", crops=np.stack(all_crops), region_mask=np.stack(all_masks),
                        keys=index["key"].to_numpy())
    index.to_csv(out / "crops_index.csv", index=False)

    window = setting("crop_window")
    print(f"\nSaved {len(index)} crops of {window['pings']} pings x {window['samples']} samples to {out}")
    print(f"Cut off at the window edge: {int(index['clipped'].sum())}; "
          f"with missing pings: {int((index['missing_pings'] > 0).sum())}")
    print(index.groupby("label").size().to_string())


if __name__ == "__main__":
    main()
