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

   **One Drive set up** Copy `config.local.example.yaml` to `config.local.yaml` and put your own full paths in it. Only that file changes, and it is never committed:
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
| 0 | `python pipeline.py sort` | Sorts raw `.01A` files into `Weeks/week1 ... week54` (Sat-Fri). Add `--dry-run` to preview, `--copy` to copy instead of move. Once you run this `.01A` files will be moved out of DATA |
| Labels | `python pipeline.py count` | Counts DFO regions per species and week |
| Labels | `python pipeline.py regions` | Region CSVs (times, depths, labels) and train/val/test splits, for every split in `config.yaml`. Add `--split week_species_602020` for just one |
| 1 | `python pipeline.py sv --weeks 1 5` | Weekly Sv with calibration and depth, plus check plots, for the weeks you list |
| 1 | `python pipeline.py sv --all --no-plots` | Sv for every week. Add `--redo` to remake weeks already saved |
| 2 | `python pipeline.py clean --all` | Cleaning step 1: a bad-data mask per week (`data/processed/Sv/weekN_mask.zarr`, same grid as the Sv) from DFO's bad-data regions (.EVR type 0) and the end of deployment (`bad_data_after` in `config.yaml`). Lists what was masked in `data/processed/bad_data/`. Use as `Sv.where(mask == 0)` |
| 3 | `python pipeline.py solar` | Sunrise and sunset for every day in the instrument clock (`data/processed/daytime/sun_times.csv`), a check of `clock_utc_offset` against DFO's daytime cell exports, and a check that every labelled region is in daylight. Only daytime data are used for both models |

**Choosing a split:** every region in a block (a week or a day) goes to the same set, so regions close in time never end up on both sides. Names read as blocks, what is balanced, then the shares: `week_season_701515` is week blocks, each species balanced per season, 70% train / 15% val / 15% test. `602020` is 60% / 20% / 20%. These were called `split_7030`, `split_8020`, `split_day_7030` and so on before, so if your `config.local.yaml` has `split: split_7030`, change it to `split: week_species_701515`.

| Split | Blocks | Balanced on | Sets |
|---|---|---|---|
| `week_species_701515`, `week_species_602020` | week | species | train / val / test |
| `week_season_701515`, `week_season_602020` | week | species in each season | train / val / test |
| `day_season_701515`, `day_season_602020` | day | species in each season | train / val / test |
| `random_species_701515`, `random_species_602020` | none (each region alone) | species | train / val / test, baseline that ignores time |
| `kfold5_week_season`, `kfold5_day_season` | week or day | species in each season | `fold` column 0 to 4 for blocked cross validation, `split` is empty |

Splits balanced on species only keep the best of 5000 random tries (seed 42). Season balanced splits and the k folds use a search that moves and swaps blocks between sets until each set holds its share of every species, overall and in each season. Each split folder has `split_counts.csv` with regions per set (or fold), species and season. The shared default is `split: week_species_701515` in `config.yaml`. To use another one on your computer only, put `split: day_season_701515` (for example) in your `config.local.yaml`, or add `--split day_season_701515` to a single command.

Outputs: region CSVs in `data/processed/<split>/`, Sv in `data/processed/Sv/weekN_Sv.zarr`, plots in `data/processed/Sv/plots/`.

---

## Repo layout

```
config.yaml          all paths, relative to the repo
config.local.yaml    your own paths or split, if different (not committed)
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
