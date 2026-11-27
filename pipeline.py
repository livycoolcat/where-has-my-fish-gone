"""
One command to run each step of the project, the same way on everyone's computer.
Paths come from config.yaml. See README "How to run".

    python pipeline.py check                  # are the paths set up?
    python pipeline.py sort                   # raw .01A files into week folders
    python pipeline.py count                  # count DFO regions per species
    python pipeline.py regions                # region CSVs for every split in config.yaml
    python pipeline.py regions --split split_8020   # only one split
    python pipeline.py sv --weeks 1 5         # Sv for some weeks (with check plots)
    python pipeline.py sv --all --no-plots    # Sv for every week
    python pipeline.py clean --all            # cleaning step 1: bad-data masks for every week
    python pipeline.py solar                  # sunrise and sunset, clock check, daylight check of the regions
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "scripts"))

from paths import path, setting  # noqa: E402


def run_script(script, *args):
    # Run one of the scripts in scripts/ with command-line arguments
    cmd = [sys.executable, str(REPO / "scripts" / script), *map(str, args)]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, cwd=REPO)


def split_manifest(split):
    # regions_manifest.csv for a split, e.g. data/processed/split_7030/regions_manifest.csv
    split = split or setting("split")
    if split not in setting("splits"):
        sys.exit(f"Unknown split '{split}'. Choose from: {', '.join(setting('splits'))}")
    return path("processed") / split / "regions_manifest.csv"


def check(args):
    # Show where each path points and whether it exists
    paths = {name: path(name) for name in ["raw_data", "weeks", "regions_evr", "evr_windows", "calibration"]}
    paths["split"] = split_manifest(None)
    missing = 0
    for name, p in paths.items():
        ok = p.exists()
        missing += not ok
        print(f"{'OK     ' if ok else 'MISSING'}  {name:13} {p}")
    print(f"\nSplit used by later steps: {setting('split')} (set in config.yaml)")
    if missing:
        print("Some paths are missing. See README 'Setup'.")


def sort(args):
    # Step 0: sort the raw .01A files into week folders
    import sort_raw_by_week as s
    s.DATA = path("raw_data")
    s.OUT = path("weeks")
    s.MODE = "copy" if args.copy else "move"
    s.DRY_RUN = args.dry_run
    s.main()


def count(args):
    # Count DFO's labelled regions per species and week
    # count_species_evr.py counts only "analysis" regions by default (add --all-types for every region)
    run_script("count_species_evr.py", path("regions_evr"), "--out", path("region_labels"))


def regions(args):
    # Region CSVs (times, depths, labels) and the train/val/test split, for each split in config.yaml
    if args.split and args.split not in setting("splits"):
        sys.exit(f"Unknown split '{args.split}'. Choose from: {', '.join(setting('splits'))}")
    for name, spec in setting("splits").items():
        if args.split and name != args.split:
            continue
        # a number means week groups, settings can ask for day groups and season balance
        spec = spec if isinstance(spec, dict) else {"frac": spec}
        extra = ["--group", spec.get("group", "week")]
        if spec.get("balance_season"):
            extra.append("--balance-season")
        run_script("parse_cropped_regions.py", path("regions_evr"),
                   "--windows", path("evr_windows"),
                   "--split-frac", spec["frac"],
                   "--out", path("processed") / name, *extra)


def sv(args):
    # Step 1: weekly Sv with calibration and depth, plus check plots
    import azfp_processing as a
    a.WEEKS_DIR = str(path("weeks"))
    a.OUTPUT_DIR = str(path("sv_output"))
    a.PLOTS_DIR = str(path("sv_plots"))
    a.CALIBRATION_PATH = str(path("calibration"))
    a.REGIONS_PATH = str(split_manifest(args.split))   # regions drawn on the check plots
    a.WEEKS_TO_RUN = None if args.all else args.weeks
    a.MAKE_PLOTS = not args.no_plots
    a.SKIP_EXISTING = not args.redo
    a.main()


def clean(args):
    # Data cleaning step 1: a bad-data mask for each week (DFO bad regions + end of deployment)
    run_script("make_masks.py", *(["--all"] if args.all else ["--weeks", *args.weeks]))


def solar(args):
    # sunrise and sunset in the instrument clock, checked against dfo's daytime intervals
    run_script("solar.py", *(["--split", args.split] if args.split else []))


def main():
    parser = argparse.ArgumentParser(description="Run a step of the fish classification pipeline.")
    steps = parser.add_subparsers(dest="step", required=True)

    steps.add_parser("check", help="show the paths from config.yaml and whether they exist")

    p = steps.add_parser("sort", help="sort raw .01A files into week folders")
    p.add_argument("--copy", action="store_true", help="copy files instead of moving them")
    p.add_argument("--dry-run", action="store_true", help="only print what would happen")

    steps.add_parser("count", help="count DFO regions per species and week")

    p = steps.add_parser("regions", help="make region CSVs and train/val/test splits")
    p.add_argument("--split", help="only this split, e.g. split_7030 (default: all in config.yaml)")

    p = steps.add_parser("sv", help="make weekly Sv files")
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--weeks", type=int, nargs="+", help="week numbers, e.g. --weeks 1 5")
    which.add_argument("--all", action="store_true", help="every week")
    p.add_argument("--no-plots", action="store_true", help="skip the check plots (faster)")
    p.add_argument("--redo", action="store_true", help="remake weeks that are already saved")
    p.add_argument("--split", help="split whose regions are drawn on the plots (default: split in config.yaml)")

    p = steps.add_parser("clean", help="bad-data masks for each week (needs the Sv files)")
    which = p.add_mutually_exclusive_group(required=True)
    which.add_argument("--weeks", type=int, nargs="+", help="week numbers, e.g. --weeks 3 53")
    which.add_argument("--all", action="store_true", help="every week with a saved Sv file")

    p = steps.add_parser("solar", help="sunrise and sunset, clock check, daylight check of the labelled regions")
    p.add_argument("--split", help="split whose regions are checked (default: split in config.yaml)")

    args = parser.parse_args()
    {"check": check, "sort": sort, "count": count, "regions": regions, "sv": sv,
     "clean": clean, "solar": solar}[args.step](args)


if __name__ == "__main__":
    main()
