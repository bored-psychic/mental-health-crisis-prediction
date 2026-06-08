# Data folder

The project runs on **synthetic GLOBEM-structured data by default** — you do not
need to download anything to run the notebooks.

## To use the real datasets

### GLOBEM (primary, training)
1. Register (free) at <https://physionet.org> and sign the GLOBEM Data Use Agreement.
2. Download the pre-extracted daily feature CSVs only (~80 MB), **not** the raw
   ~120 GB sensor archive.
3. Place these files directly in this folder:
   ```
   data/INS-W_1.csv
   data/INS-W_2.csv
   data/INS-W_3.csv
   data/INS-W_4.csv
   ```
4. Re-run `python run.py` (or the notebook) — the loader auto-detects them.

> ⚠️ The real GLOBEM feature columns have long RAPIDS-style names like
> `f_slp:fitbit_sleep_intraday_rapids_sumdurationasleepunifiedmain:allday`.
> The loader resolves them automatically by keyword (see `FEATURE_PATTERNS` and
> `LABEL_KEYWORDS` in `src/pipeline.py`) and prints exactly which column it
> matched for each feature. If your release uses different conventions, add
> keyword sets to `FEATURE_PATTERNS`; the loader falls back to synthetic data if
> a feature or the label cannot be found. Weekly survey labels are automatically
> carried forward across each week's daily sensor rows.
>
> Run `python tests/test_real_loader.py` to validate the real-data path against a
> mock GLOBEM-style CSV (no download needed).

### StudentLife (secondary, cross-dataset validation)
Used only for generalisation testing. Download the pre-extracted daily feature
CSVs (~4 MB) from <https://studentlife.cs.dartmouth.edu>. The current pipeline
simulates a StudentLife-like cohort; wire in the real CSVs the same way if needed.

## Why these files are git-ignored
Real datasets are covered by data-use agreements and must not be committed.
`.gitignore` excludes `data/INS-W_*.csv` and StudentLife archives.
