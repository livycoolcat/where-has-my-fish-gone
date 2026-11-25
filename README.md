# Where Oh Where Has My Fish Gone!
Classifying herring and salmon in DFO moored echosounder (AZFP) data from the Malaspina Strait.

---

## Setup (once)

1. **Clone the repo**
   ```bash
   git clone https://github.com/livycoolcat/where-has-my-fish-gone.git
   cd where-has-my-fish-gone
   ```

2. **Install the packages**
   ```bash
   pip install echopype pvlib pyyaml matplotlib
   ```

3. **Put the DFO data in `data/raw/55158`.** The data is confidential, so it is never committed (`data/raw/` is in `.gitignore`).
   If your copy is somewhere else, link to it instead of moving it:
   ```bash
   mkdir -p data/raw
   ln -s "/full/path/to/your/AZFP/55158" data/raw/55158
   ```
   The `55158` folder should contain `DATA/` (raw `.01A` files by month) and `Echoview/Exports/Regions/` (`WeekN.EVR` files).

   **Data laid out differently (e.g. OneDrive, or Windows)?** Copy `config.local.example.yaml` to `config.local.yaml` and put your own full paths in it. Only that file changes, and it is never committed:
   ```yaml
   raw_data:    "C:/Users/yourname/OneDrive - University of Victoria/Malaspina/DATA/RAW"
   weeks:       "C:/Users/yourname/OneDrive - University of Victoria/Malaspina/DATA/RAW_WEEKLY"
   regions_evr: "C:/Users/yourname/OneDrive - University of Victoria/Malaspina/DATA/Regions"
   ```

4. **Check the paths**
   ```bash
   python pipeline.py check
   ```
   Every line should say `OK` (except `weeks` until you run `sort`).

---

## How to run

Run everything from the repo folder with `pipeline.py`. Paths come from `config.yaml`, so nobody edits paths in the code.

| Step | Command | What it does |
|---|---|---|
| Check | `python pipeline.py check` | Shows each path and whether it exists |
| 0 | `python pipeline.py sort` | Sorts raw `.01A` files into `Weeks/week1 ... week54` (Sat-Fri). Add `--dry-run` to preview, `--copy` to copy instead of move |
| Labels | `python pipeline.py count` | Counts DFO regions per species and week |
| Labels | `python pipeline.py regions` | Region CSVs (times, depths, labels) and train/val/test splits, for every split in `config.yaml`. Add `--split split_8020` for just one |
| 1 | `python pipeline.py sv --weeks 1 5` | Weekly Sv with calibration and depth, plus check plots, for the weeks you list |
| 1 | `python pipeline.py sv --all --no-plots` | Sv for every week. Add `--redo` to remake weeks already saved |

**Choosing a split:** `split_7030` is 70% train / 15% val / 15% test, `split_8020` is 60% / 20% / 20% (split by week). Set the one everyone uses in `config.yaml` (`split: split_7030`), or add `--split split_8020` to a command.

Outputs: region CSVs in `data/processed/split_*/`, Sv in `data/processed/Sv/weekN_Sv.zarr`, plots in `data/processed/Sv/plots/`.

---

## Repo layout

```
config.yaml          all paths, relative to the repo
config.local.yaml    your own paths, if different (not committed)
pipeline.py          one command for each step
scripts/             the code for each step (paths.py reads config.yaml)
data/raw/            DFO data (not committed)
data/calibration/    calibration table and sources
data/region_labels/  region counts and EVR time windows
data/processed/      region CSVs per split (Sv outputs not committed)
notes/               background and research notes
```

---

## Adding a new step

1. Write the script in `scripts/`. Get paths with `from paths import path`, e.g. `path("weeks")`, never a typed-in path.
2. Add any new paths to `config.yaml`.
3. Add a command for it in `pipeline.py`.

---

## Notes

- **Time:** all times are the AZFP instrument clock (no time zone attached). Sv and DFO region times match. The time zone is unconfirmed, most likely local standard time (UTC-8).
- **Depth:** DFO region depths are depth below the surface. Sv has `echo_range` (distance from the transducer) and `depth` (`INSTRUMENT_DEPTH`, about 116.5 m, minus `echo_range`).
