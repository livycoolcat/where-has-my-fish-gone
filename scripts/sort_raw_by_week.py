"""
Sort AZFP .01A raw files into weekly folders (Saturday 00:00 -> Friday 23:59:59).

Week 1 starts on Sat 2025-03-01 (deployment start), matching the Echoview
weekly files / .EVR regions:  week = (file_date - 2025-03-01).days // 7 + 1

Each raw file is named YYMMDDHH.01A, so its week comes from the filename.
The month's .XML (needed by echopype to read the raw files) is copied into
every week folder that uses files from that month.

Run once with DRY_RUN = True to check the output, then set DRY_RUN = False.
"""

from datetime import datetime
from pathlib import Path
import shutil

# ----------------------------- EDIT ---------------------------------
DATA = Path("/Users/oliviawhitelaw/Documents/CSC 475/where-has-my-fish-gone/2025-03-01_MalaspinaStrait/"
            "2025-03-01_MalaspinaStrait/AZFP/55158/DATA")
OUT = DATA.parent / "Weeks"          # creates .../55158/Weeks/week1, week2, ...
WEEK1_START = datetime(2025, 3, 1)   # Saturday the deployment started
FOLDER_NAME = "week{}"               # use "week{:02d}" for week01, week02 (sorts nicer)
MODE = "move"                        # "move" or "copy"
DRY_RUN = False                       # True = only print what would happen
# --------------------------------------------------------------------


def week_of(raw: Path) -> int:
    t = datetime.strptime(raw.stem, "%y%m%d%H")
    return (t - WEEK1_START).days // 7 + 1


def main():
    raws = sorted(p for p in DATA.glob("20*/*")
                  if p.is_file() and p.suffix.upper() == ".01A")
    if not raws:
        print("No .01A files found under", DATA)
        return

    counts = {}
    xml_done = set()   # (week, xml name) already copied

    for raw in raws:
        try:
            wk = week_of(raw)
        except ValueError:
            print("Skipping (unexpected name):", raw.name)
            continue

        dest_dir = OUT / FOLDER_NAME.format(wk)
        dest = dest_dir / raw.name
        counts[wk] = counts.get(wk, 0) + 1

        if dest.exists():
            print("Already exists, skipping:", dest)
            continue

        if not DRY_RUN:
            dest_dir.mkdir(parents=True, exist_ok=True)
            if MODE == "move":
                shutil.move(str(raw), str(dest))
            else:
                shutil.copy2(raw, dest)

        # copy the month's XML alongside, once per week folder
        for xml in raw.parent.glob("*.XML"):
            key = (wk, xml.name)
            if key in xml_done:
                continue
            xml_done.add(key)
            if not DRY_RUN and not (dest_dir / xml.name).exists():
                shutil.copy2(xml, dest_dir / xml.name)

    print(f"\n{'DRY RUN - nothing moved' if DRY_RUN else MODE.upper() + ' done'}")
    print(f"{len(raws)} raw files -> {len(counts)} week folders in {OUT}\n")
    for wk in sorted(counts):
        n = counts[wk]
        flag = "" if n == 168 else "   <- not a full week"
        print(f"  {FOLDER_NAME.format(wk):>8}: {n:4d} files{flag}")


if __name__ == "__main__":
    main()