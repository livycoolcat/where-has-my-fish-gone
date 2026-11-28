#!/usr/bin/env python3
# Parses cropped weekly Echoview .EVR files into CNN-ready label files.
# Checks against the start/end table and flags anything out of bounds.

# To call: python parse_cropped_regions.py <path_to_regions_folder> --split-frac <fraction>
# <fraction> is the decimal share assigned to BOTH the validation and test sets each (e.g., 0.10 creates an 80/10/10 split).

import argparse
import sys
import calendar
import re
from pathlib import Path
from paths import path, setting
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# Paths and the default split come from config.yaml
DEFAULT_REGIONS_DIR = path("regions_evr")
DEFAULT_WINDOWS_CSV = path("evr_windows")
DEFAULT_SPLIT = setting("split")                      # e.g. week_species_701515
DEFAULT_SPLIT_FRAC = setting("splits")[DEFAULT_SPLIT] # e.g. 0.15
# a split in config.yaml is either a number (week groups) or settings with frac, group and balance_season
if isinstance(DEFAULT_SPLIT_FRAC, dict):
    DEFAULT_SPLIT_FRAC = DEFAULT_SPLIT_FRAC.get("frac", 0.15)   # k fold splits have no frac

# labelled regions left out of every split, key (e.g. Week44_r14) and the reason, from config.yaml
EXCLUDED_REGIONS = setting("exclude_regions") or {}

# seasons used to balance splits, by month of the region start
SEASONS = {3: "mar_apr", 4: "mar_apr", 5: "may", 6: "jun_jul", 7: "jun_jul", 8: "aug_sep", 9: "aug_sep",
           10: "oct_nov", 11: "oct_nov", 12: "dec_feb", 1: "dec_feb", 2: "dec_feb"}

# types from echoview
REG_TYPES = {"0": "bad (no data)", "1": "analysis", "2": "marker", "3": "fishtracks", "4": "bad (empty water)"}
UNCLASSIFIED_LABELS = {"", "unclassified regions", "unclassified"}

# weeks run Sat-Fri
WEEK1_START = datetime(2025, 3, 1, tzinfo=timezone.utc)
week_regex = re.compile(r"week\s*0*(\d+)", re.IGNORECASE)

day_cache = {}

def get_epoch(d, t):
    # parses evr date '20250306' and time '1100100500' to epoch
    if d not in day_cache:
        day_cache[d] = calendar.timegm((int(d[:4]), int(d[4:6]), int(d[6:8]), 0, 0, 0))
    
    t = t.zfill(10)
    hours = int(t[:2]) * 3600
    mins = int(t[2:4]) * 60
    secs = int(t[4:6])
    frac = int(t[6:]) / 10000.0
    
    return day_cache[d] + hours + mins + secs + frac

def to_iso(seconds):
    # NOTE ON TIME: .EVR times are the AZFP instrument clock, the same clock as the Sv ping_time
    # We don't know what the time zone is for sure, most likely local standard time (UTC-8), not UTC
    # UTC is only used here as a neutral reference for the arithmetic; no time zone is written out
    return datetime.fromtimestamp(seconds, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")

def read_text_file(filepath):
    # Try a few encodings because echoview files can be weird
    raw_bytes = filepath.read_bytes()
    for encoding in ["utf-8-sig", "cp1252", "latin-1"]:
        try:
            return raw_bytes.decode(encoding).splitlines()
        except UnicodeDecodeError:
            pass
    raise Exception(f"Could not decode file: {filepath}")

def find_case_insensitive_path(p):
    if p.exists(): return p
    
    curr = Path(p.anchor) if p.is_absolute() else Path(".")
    parts = p.parts[1:] if p.is_absolute() else p.parts
    
    for part in parts:
        next_path = curr / part
        if not next_path.exists() and curr.is_dir():
            matches = [c for c in curr.iterdir() if c.name.lower() == part.lower()]
            if matches:
                next_path = matches[0]
        curr = next_path
    return curr

def get_week_number(filename, start_seconds):
    match = week_regex.search(filename)
    if match:
        return int(match.group(1))
    
    # fallback to calculating it from the start date
    diff = start_seconds - WEEK1_START.timestamp()
    return int(diff // (7 * 86400)) + 1

def parse_evr_file(filepath):
    lines = read_text_file(filepath)
    if not lines or not lines[0].startswith("EVRG"):
        print(f"Skipping {filepath}, missing EVRG header")
        return []
        
    parsed_regions = []
    idx = 2
    total_lines = len(lines)
    
    while idx < total_lines:
        tokens = lines[idx].split()
        if len(tokens) < 7 or not tokens[0].isdigit() or not tokens[1].isdigit():
            idx += 1
            continue
            
        num_points = int(tokens[1])
        reg_id = tokens[2]
        
        idx += 1
        idx += 1 + int(lines[idx]) # skip notes
        idx += 1 + int(lines[idx]) # skip detection settings
        
        class_name = lines[idx].strip()
        point_data = lines[idx + 1].split()
        
        reg_name = ""
        if idx + 2 < total_lines:
            reg_name = lines[idx + 2].strip()
            
        idx += 3
        
        if len(point_data) != 3 * num_points + 1 or point_data[-1] not in REG_TYPES:
            raise ValueError(f"Malformed point list in region {reg_id}")
            
        # extract time and depth arrays
        time_arr = []
        depth_arr = []
        for k in range(0, 3 * num_points, 3):
            time_arr.append(get_epoch(point_data[k], point_data[k + 1]))
            depth_arr.append(float(point_data[k + 2]))
            
        t_arr = np.array(time_arr)
        z_arr = np.array(depth_arr)
        
        parsed_regions.append({
            "region_id": reg_id,
            "name": reg_name,
            "classification": class_name,
            "region_type": REG_TYPES[point_data[-1]],
            "n_points": num_points,
            "poly": np.column_stack([t_arr, z_arr])
        })
        
    return parsed_regions

def group_counts(df, class_list, group, balance_season):
    # regions per group (rows) and per species, or per species and season (columns), for the regions kept
    valid_data = df[df["use_for_cnn"]]
    columns = ["label", "season"] if balance_season else ["label"]
    class_counts = valid_data.groupby([group] + columns).size().unstack(columns, fill_value=0)
    if not balance_season:
        class_counts = class_counts.reindex(columns=class_list, fill_value=0)
    return class_counts.index.to_numpy(), class_counts.to_numpy(float)


def balance_matrix(df, class_list, group):
    # regions per group for each species, then for each species in each season, side by side
    group_ids, species = group_counts(df, class_list, group, False)
    _, by_season = group_counts(df, class_list, group, True)
    return group_ids, np.hstack([species, by_season])


def local_search(count_matrix, shares, seed_val, restarts=20):
    # put every group (row) in one set so each set holds its share of every column, counted in regions
    # so big cells (may herring) weigh more than small ones (dec_feb salmon)
    # starts from a random assignment and keeps moving one group, or swapping two, while that gets closer
    # restarts a few times from new random starts and keeps the best, returns the set index of each group
    shares = np.asarray(shares, float)
    target = shares[:, None] * count_matrix.sum(axis=0)
    rand_gen = np.random.default_rng(seed_val)
    n, n_sets = len(count_matrix), len(shares)
    best_assignment = None

    def error(set_counts):
        return np.abs(set_counts - target).sum()

    for _ in range(restarts):
        set_of = rand_gen.choice(n_sets, n, p=shares)
        set_counts = np.array([count_matrix[set_of == s].sum(axis=0) for s in range(n_sets)])
        improved = True
        while improved:
            improved = False
            for i in rand_gen.permutation(n):
                # move group i to another set
                for s in range(n_sets):
                    si = set_of[i]
                    if s == si:
                        continue
                    before = error(set_counts)
                    set_counts[si] -= count_matrix[i]
                    set_counts[s] += count_matrix[i]
                    if error(set_counts) < before - 1e-9:
                        set_of[i] = s
                        improved = True
                        break
                    set_counts[si] += count_matrix[i]
                    set_counts[s] -= count_matrix[i]
                # swap group i with a few groups in other sets
                for j in rand_gen.permutation(n)[:30]:
                    si, sj = set_of[i], set_of[j]
                    if si == sj:
                        continue
                    before = error(set_counts)
                    change = count_matrix[j] - count_matrix[i]
                    set_counts[si] += change
                    set_counts[sj] -= change
                    if error(set_counts) < before - 1e-9:
                        set_of[i], set_of[j] = sj, si
                        improved = True
                    else:
                        set_counts[si] -= change
                        set_counts[sj] += change
        if best_assignment is None or error(set_counts) < best_assignment[0]:
            best_assignment = (error(set_counts), set_of.copy())

    return best_assignment[1]


def create_splits(df, class_list, seed_val, split_fraction, num_tries=5000, group="week", balance_season=False):
    # group is the column that holds the block (week, day, or key for one region per block, a random split)
    # all regions in one group go to the same set so close regions never leak across sets
    # balance_season: local search so each set holds its share of every species, overall and in each season
    # (random tries left val with 10 salmon and no may in week_season_701515)
    # otherwise: try random group assignments and keep the one whose val and test shares are closest to
    # split_fraction, kept as it was so week_species_701515, week_species_602020 and the random splits do not change
    if balance_season:
        group_ids, count_matrix = balance_matrix(df, class_list, group)
        set_of = local_search(count_matrix, [1 - 2 * split_fraction, split_fraction, split_fraction], seed_val)
        split_mapping = dict(zip(group_ids, np.array(["train", "val", "test"])[set_of]))
        return df[group].map(split_mapping).where(df["use_for_cnn"], "").fillna("")

    group_ids, count_matrix = group_counts(df, class_list, group, False)

    split_mapping = {g: "train" for g in group_ids}
    k = max(1, round(split_fraction * len(group_ids)))

    if len(group_ids) >= 3:
        totals = np.maximum(count_matrix.sum(axis=0), 1)
        rand_gen = np.random.default_rng(seed_val)
        best_assignment = None

        for _ in range(num_tries):
            p = rand_gen.permutation(len(group_ids))
            test_idx = p[:k]
            val_idx = p[k:2 * k]

            # how far the val and test shares are from the target share
            test_error = np.abs(count_matrix[test_idx].sum(axis=0) / totals - split_fraction).sum()
            val_error = np.abs(count_matrix[val_idx].sum(axis=0) / totals - split_fraction).sum()
            total_error = test_error + val_error

            if best_assignment is None or total_error < best_assignment[0]:
                best_assignment = (total_error, test_idx, val_idx)

        for j in best_assignment[1]:
            split_mapping[group_ids[j]] = "test"
        for j in best_assignment[2]:
            split_mapping[group_ids[j]] = "val"

    return df[group].map(split_mapping).where(df["use_for_cnn"], "").fillna("")


def create_folds(df, class_list, seed_val, n_folds, group="week", balance_season=False):
    # blocked k fold: every group goes to one of n_folds folds, so close regions never leak across folds
    # local search so every fold holds 1/n_folds of each species (and of each species in each season)
    if balance_season:
        group_ids, count_matrix = balance_matrix(df, class_list, group)
    else:
        group_ids, count_matrix = group_counts(df, class_list, group, False)
    fold_of = local_search(count_matrix, [1 / n_folds] * n_folds, seed_val)
    fold_mapping = dict(zip(group_ids, fold_of))
    return df[group].map(fold_mapping).where(df["use_for_cnn"]).astype("Int64")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parse weekly .EVR files into CNN-ready label files.")
    parser.add_argument("path", type=Path, nargs="?", default=Path(DEFAULT_REGIONS_DIR), help="Path to EVR folders")
    parser.add_argument("--windows", type=Path, default=Path(DEFAULT_WINDOWS_CSV), help="CSV with file,start,end")
    parser.add_argument("--out", type=Path, default=path("processed") / DEFAULT_SPLIT)
    parser.add_argument("--classes", nargs="+", default=["Herring", "Salmon"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-frac", type=float, default=DEFAULT_SPLIT_FRAC)
    parser.add_argument("--keep-outside-window", action="store_true")
    parser.add_argument("--group", choices=["week", "day", "region"], default="week",
                        help="keep each week or each day together in one set (default week), "
                             "region puts each region in a set on its own (random split, ignores time)")
    parser.add_argument("--balance-season", action="store_true",
                        help="also balance each species per season across train, val and test")
    parser.add_argument("--folds", type=int, default=0,
                        help="blocked k fold instead of train/val/test: writes a fold column 0 to folds-1")
    
    args = parser.parse_args()

    data_dir = find_case_insensitive_path(args.path)
    if not data_dir.is_dir():
        print(f"Error: Folder not found -> {args.path}")
        sys.exit(1)
        
    evr_files = []
    for p in data_dir.rglob("*"):
        if p.is_file() and p.suffix.lower() == ".evr" and not any(s.startswith(".") for s in p.relative_to(data_dir).parts):
            evr_files.append(p)
            
    evr_files = sorted(evr_files, key=lambda x: (get_week_number(x.name, 0), x.name))
    
    if not evr_files:
        print(f"No .evr files found in {data_dir}")
        sys.exit(1)

    win_path = args.windows
    possible_paths = [win_path, win_path.with_suffix(".xlsx")]
    possible_paths += [Path(__file__).resolve().parents[1] / p for p in possible_paths]
    
    actual_win_path = None
    for p in possible_paths:
        if p.exists():
            actual_win_path = p
            break
            
    if not actual_win_path:
        print(f"Window table missing: {args.windows}")
        sys.exit(1)
        
    print(f"Loading window table from: {actual_win_path}")
    if actual_win_path.suffix.lower() in [".xlsx", ".xls"]:
        win_df = pd.read_excel(actual_win_path)
    else:
        win_df = pd.read_csv(actual_win_path)
        
    win_df.columns = [str(c).strip().lower() for c in win_df.columns]
    
    if not {"file", "start", "end"}.issubset(win_df.columns):
        print(f"Window table missing required columns. Found: {list(win_df.columns)}")
        sys.exit(1)
        
    win_df["file"] = win_df["file"].astype(str).str.strip()
    win_df = win_df.dropna(subset=["start", "end"])
    
    win_df["start"] = [pd.Timestamp(x).strftime("%Y-%m-%d %H:%M:%S") for x in win_df["start"]]
    win_df["end"] = [pd.Timestamp(x).strftime("%Y-%m-%d %H:%M:%S") for x in win_df["end"]]
    win_df["key"] = win_df["file"].str.lower()
    
    win_df["w_start"] = [pd.Timestamp(s, tz="UTC").timestamp() for s in win_df["start"]]
    win_df["w_end"] = [pd.Timestamp(s, tz="UTC").timestamp() for s in win_df["end"]]
    win_df = win_df.set_index("key")

    all_rows = []
    poly_dict = {}
    check_results = []
    
    for f in evr_files:
        try:
            regions = parse_evr_file(f)
        except Exception as e:
            print(f"Error parsing {f.name}: {e}")
            continue
            
        if not regions:
            print(f"Warning: {f.name} has no valid regions")
            continue
            
        r_starts = [r["poly"][:, 0].min() for r in regions]
        r_ends = [r["poly"][:, 0].max() for r in regions]
        
        file_start = min(r_starts)
        file_end = max(r_ends)
        week_num = get_week_number(f.name, file_start)
        
        w_data = win_df.loc[f.name.lower()] if f.name.lower() in win_df.index else None
        
        check_info = {
            "file": f.name,
            "week": week_num,
            "n_regions": len(regions),
            "evr_start": to_iso(np.floor(file_start))[:19],
            "evr_end": to_iso(np.floor(file_end))[:19]
        }
        
        if w_data is None:
            check_info.update({"table_start": "", "table_end": "", "start_diff_s": np.nan, "end_diff_s": np.nan, "status": "NOT_IN_TABLE"})
        else:
            diff_start = np.floor(file_start) - w_data["w_start"]
            diff_end = np.floor(file_end) - w_data["w_end"]
            is_ok = abs(diff_start) <= 1.0 and abs(diff_end) <= 1.0
            check_info.update({
                "table_start": w_data["start"], 
                "table_end": w_data["end"], 
                "start_diff_s": diff_start, 
                "end_diff_s": diff_end,
                "status": "OK" if is_ok else "MISMATCH"
            })
            
        check_results.append(check_info)

        for reg, s, e in zip(regions, r_starts, r_ends):
            lbl = "Unclassified" if reg["classification"].lower() in UNCLASSIFIED_LABELS else reg["classification"]
            
            is_in_window = True
            if w_data is not None:
                is_in_window = (s >= w_data["w_start"] - 1.0) and (e <= w_data["w_end"] + 1.0)
                
            reg_key = f"{f.stem}_r{reg['region_id']}"
            all_rows.append({
                "key": reg_key,
                "file": f.name,
                "week": week_num,
                "region_id": reg["region_id"],
                "classification": reg["classification"],
                "label": lbl,
                "region_type": reg["region_type"],
                "n_points": reg["n_points"],
                "start_time": to_iso(s), # edited
                "end_time": to_iso(e), # edited
                "depth_top": float(reg["poly"][:, 1].min()),
                "depth_bottom": float(reg["poly"][:, 1].max()),
                "in_window": is_in_window
            })
            poly_dict[reg_key] = reg["poly"]

    final_df = pd.DataFrame(all_rows)
    
    if final_df["key"].duplicated().any():
        dupes = final_df.loc[final_df["key"].duplicated(), "key"].tolist()[:5]
        print(f"Warning: Found duplicate region keys: {dupes}")
        
    class_map = {c: i for i, c in enumerate(args.classes)}
    final_df["label_id"] = final_df["label"].map(class_map).fillna(-1).astype(int)
    
    is_valid_window = final_df["in_window"] | args.keep_outside_window
    final_df["use_for_cnn"] = (final_df["region_type"] == "analysis") & (final_df["label_id"] >= 0) & is_valid_window
    # regions listed in exclude_regions in config.yaml are dropped here, before the splits are made,
    # so they never count towards the balancing
    final_df["excluded"] = final_df["key"].isin(EXCLUDED_REGIONS)
    final_df["use_for_cnn"] &= ~final_df["excluded"]
    # day groups use the instrument clock date of the region start, season is by month of the start
    # both are kept for every split so the counts by season can be checked for week splits too
    starts = pd.to_datetime(final_df["start_time"], format="ISO8601")
    final_df["day"] = starts.dt.strftime("%Y-%m-%d")
    final_df["season"] = starts.dt.month.map(SEASONS)
    group_col = {"week": "week", "day": "day", "region": "key"}[args.group]
    if args.folds:
        # k fold regions have no train/val/test, the training code picks one fold as test on each run
        final_df["split"] = ""
        final_df["fold"] = create_folds(final_df, args.classes, args.seed, args.folds,
                                        group=group_col, balance_season=args.balance_season)
    else:
        final_df["split"] = create_splits(final_df, args.classes, args.seed, args.split_frac,
                                          group=group_col, balance_season=args.balance_season)

    # Export outputs
    args.out.mkdir(parents=True, exist_ok=True)
    
    chk_df = pd.DataFrame(check_results)
    chk_df.to_csv(args.out / "evr_window_check.csv", index=False)
    
    dropped_counts = final_df[~final_df["use_for_cnn"] & ~final_df["excluded"]].groupby(["label", "region_type"]).size()
    kept_df = final_df[final_df["use_for_cnn"]].drop(columns=["classification", "region_type", "in_window", "use_for_cnn", "excluded"]).reset_index(drop=True)
    kept_df.to_csv(args.out / "regions_manifest.csv", index=False)
    
    np.savez_compressed(args.out / "regions_polygons.npz", **{k: poly_dict[k] for k in kept_df["key"]})

    # regions per set (or fold) and species, by season, plus how many blocks each set holds
    set_col = "fold" if args.folds else "split"
    seasons = [s for s in dict.fromkeys(SEASONS.values()) if s in set(kept_df["season"])]
    counts = pd.crosstab([kept_df[set_col], kept_df["label"]], kept_df["season"]).reindex(columns=seasons, fill_value=0)
    counts["total"] = counts.sum(axis=1)
    counts["blocks"] = kept_df.groupby([set_col, "label"])[group_col].nunique()
    counts.to_csv(args.out / "split_counts.csv")

    # Print summary stats
    print(f"Parsed {len(chk_df)} files and {len(final_df)} regions.")
    print(f"Kept {len(kept_df)} regions (classes: {', '.join(args.classes)}).")
    
    dropped_str = "; ".join(f"{a} / {b}: {n}" for (a, b), n in dropped_counts.items())
    print(f"Dropped: {dropped_str if dropped_str else 'None'}")
    for key in EXCLUDED_REGIONS:
        found = final_df.loc[final_df["key"] == key, "label"]
        if found.empty:
            print(f"Warning: {key} is in exclude_regions in config.yaml but no region has that key")
        else:
            print(f"Excluded {key} ({found.iloc[0]}): {EXCLUDED_REGIONS[key]}")
    
    print("\n--- Window Checks ---")
    print(f"OK: {(chk_df['status'] == 'OK').sum()}")
    print(f"MISMATCH: {(chk_df['status'] == 'MISMATCH').sum()}")
    print(f"NOT_IN_TABLE: {(chk_df['status'] == 'NOT_IN_TABLE').sum()}")
    
    missing_files = sorted(set(win_df.index) - {c.lower() for c in chk_df["file"]})
    print(f"In table but missing EVR file: {len(missing_files)}")
    
    bad_checks = chk_df[chk_df["status"] != "OK"]
    if len(bad_checks) > 0:
        print("\nDetails on mismatched files:")
        print(bad_checks[["file", "table_start", "evr_start", "table_end", "evr_end", "status"]].to_string(index=False))
        
    out_of_window = final_df[(~final_df["in_window"]) & (final_df["region_type"] == "analysis") & (final_df["label_id"] >= 0)]
    if len(out_of_window) > 0:
        status_word = 'kept' if args.keep_outside_window else 'dropped'
        print(f"\n{len(out_of_window)} labelled regions were outside their window ({status_word}). Sample: {out_of_window['key'].tolist()[:8]}")
        
    kind = f"{args.folds} folds" if args.folds else "Train/Val/Test Split"
    print(f"\n--- {kind} (grouped by {args.group}"
          f"{', balanced by season' if args.balance_season else ''}) ---")
    print(pd.crosstab(kept_df[set_col], kept_df["label"]).to_string())
    print(counts.to_string())
    print(f"\nOutputs written to {args.out}")