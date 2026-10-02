# Chasing the Surge: Predicting NYC Taxi Demand for Fleet Positioning

MAST30034 Project 1 -- predicting NYC taxi demand at (taxi zone x hour) grain from
historical, weather, event, and holiday data, to support fleet positioning and shift
scheduling decisions.

The report is submitted separately via Turnitin (per the assignment spec) and written in
Overleaf -- it is not part of this repository. What *is* here is the full reproducible
pipeline and nine notebooks that compute and display every single number the report
cites, so an examiner can trace any figure or table back to exactly where it comes from
and re-run it themselves.

---

## Quick start (run this to reproduce every number in the report)

Four steps: get into the project folder, set up Python, place the raw taxi data, run the
notebooks. Each is expanded below.

### 1. Open a terminal in this folder

Every command below assumes your terminal's current directory is the root of this
project (the folder this `README.md` is in -- the one containing `src/`, `notebooks/`,
`requirements.txt`, etc.). After cloning the repository, `cd` into it (use whatever you
actually named the folder when you cloned it):

```bash
cd nyc_taxi_demand_prediction
```

> **Copy-pasting commands from this README**: every code block below is written to be
> pasted exactly as shown, with no trailing text after the command. Don't add anything
> after a command on the same line -- some shells (zsh, the macOS default, in particular)
> don't treat a trailing `# comment` as a comment when you paste it directly into an
> interactive prompt, and will error out trying to parse it as an argument.

If you already have a Python virtual environment active (your terminal prompt shows
something like `(.venv)`), deactivate it first so it doesn't interfere with the conda
environment created in the next step:

```bash
deactivate
```

### 2. Set up the environment

```bash
conda create --name nyc_taxi_demand_prediction python=3.10 -y
conda activate nyc_taxi_demand_prediction
pip install -r requirements.txt
```

Your terminal prompt should now start with `(nyc_taxi_demand_prediction)`, confirming
the right environment is active, and you should still be in the project folder from
Step 1.

This installs everything needed, including PySpark, and installs this project itself in
editable mode (`-e .` in `requirements.txt`), so `import src...` works from anywhere in
the project.

### 3. Place the raw taxi data

The only data this repository does *not* generate for itself is the raw monthly taxi
trip files -- everything else (cleaning, joining external sources, aggregating,
modelling) is code that runs against them. Download the Yellow and Green trip files for
**January 2024 through May 2026** from the [TLC Trip Record
Data](https://www.nyc.gov/site/tlc/about/tlc-trip-record-data.page) page and place them
here, unmodified, using the exact filenames TLC provides:

```
data/landing/yellow/Yellow Taxi Data <Month> <Year>.parquet
data/landing/green/Green Trip Data <Month> <Year>.parquet
```

e.g. `Yellow Taxi Data Apr 2025.parquet`, `Green Trip Data Apr 2025.parquet`. `<Month>`
is one of `Jan, Feb, Mar, Apr, May, June, July, Aug, Sept, Oct, Nov, Dec` (note `June`,
`July`, `Sept` -- not the usual 3-letter abbreviation for those three). 29 months x 2
taxi types = 58 files. The two folders already exist (`data/landing/yellow/`,
`data/landing/green/`) -- just drop the files in.

Weather, events, and holidays do **not** need to be manually sourced -- notebook `03`
fetches weather and events automatically, and the holiday schedule is already committed
(`data/external/holiday_schedule.csv`, a small manually-curated 34-row list with no
fetch script -- just don't delete it).

### 4. Run the notebooks, in order

Each notebook is a thin orchestration layer over `src/` -- it calls the same functions
`python -m src.<module>` would, then explicitly prints every report-cited number and
(re)generates any figures it's responsible for. Every notebook depends on the previous
one's output already existing on disk, so **run them in numeric order, 00 through 08**:

| # | Notebook | What it does | Roughly how long |
|---|---|---|---|
| 00 | `00_schema_and_raw_layer.ipynb` | Single-month (Jan 2024, Yellow) schema/outlier sanity check that justifies the ingestion design. No pipeline output. | < 1 min |
| 01 | `01_landing_to_raw.ipynb` | Landing -> Raw: ingests + standardises all 58 taxi files (`src.dataset`) | ~3 min |
| 02 | `02_raw_to_clean.ipynb` | Raw -> Clean: applies the documented row-removal rules (`src.cleaning`) | ~5 min |
| 03 | `03_external_data.ipynb` | Downloads + cleans weather (LGA) and NYC permitted events (`src.external_weather`, `src.external_events`) | ~3 min |
| 04 | `04_processed_and_features.ipynb` | Clean -> Processed: zero-demand grid, rolling baseline, external-source joins (`src.processed`), plus a feature-matrix demonstration (`src.features`) | ~2 min |
| 05 | `05_pre_model_figures_and_statistics.ipynb` | Report-cited-only EDA: distribution, temporal, spatial, weather/event figures and statistics | ~1 min |
| 06 | `06_modelling.ipynb` | Fits and saves Linear Regression, Ridge, and Gradient Boosting to `models/*.joblib` (`src.modeling.train`) | ~1 min |
| 07 | `07_post_model_figures_and_statistics.ipynb` | All four models' Table 3/4/5 numbers together (LASSO refit inline, tuned fresh), the reliability table, DTDI error | ~15-30 min (LASSO alpha search) |
| 08 | `08_recommendations.ipynb` | Recommendation 1 & 2's numbers | ~1 min |

Open each in Jupyter and use *Run All*, or from the command line:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/00_schema_and_raw_layer.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/01_landing_to_raw.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/02_raw_to_clean.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/03_external_data.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/04_processed_and_features.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/05_pre_model_figures_and_statistics.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/06_modelling.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/07_post_model_figures_and_statistics.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/08_recommendations.ipynb
```

**Event geocoding note**: notebook `03` geocodes every unique event location to a taxi
zone via a public geocoding service (Nominatim), rate-limited to 1 request/1.1s. Doing
this from scratch for all ~4,325 unique locations in this study period takes roughly
**80 minutes** -- so a complete lookup cache is committed
(`data/external/events/geocode_cache.json`), the one exception to this repo's rule of
not committing anything under `data/`. With it, this step is normally near-instant; only
genuinely new locations (unlikely, since the study period is fixed) would trigger a live
lookup. If you'd rather verify the geocoding from scratch yourself, delete that file
before running notebook `03` -- the pipeline regenerates it automatically, resuming
cleanly if interrupted partway through.

Not every event geocodes successfully (a free-text location string can fail to resolve
to a real place) -- currently **~94.9%** of events resolve to a taxi zone; the rest are
excluded from event counts (they can't contribute to any zone's `event_count`). Notebook
`03` prints the exact current rate directly, and it's also written to
`reports/data_quality/external/events_report.json` under `"geocoding"`.

---

## Where each report number comes from

Every number, table, and figure the report cites, with the exact notebook that computes
and prints it. Cell outputs match these values when the notebook is re-run against a
fresh `data/landing/` download.

### Introduction / Preprocessing

| Report item | Notebook |
|---|---|
| Table 1 -- TLC Yellow+Green row/feature counts | `01` |
| Table 1 -- weather/events/holidays row/feature counts | `03` |
| Section 2.1 -- TLC row-removal counts (`trip_distance`, `fare_amount`, `trip_duration_minutes` rules) | `02` |
| Section 2.2 -- external-join coverage (weather/events/holidays %) | `04` |
| Section 2.3.1 -- geographic zone count (263) | `04` |
| Section 2.3.2 -- baseline pivot (JFK/LaGuardia static vs. rolling deviation) | `04` |
| Figure 1 -- baseline deviation, static vs. rolling | `04` |
| Table 2 -- pipeline shape (Raw / Clean / Processed row counts, zero-demand rates) | `01`, `02`, `04` (spans all three stages) |

### Analysis and Geospatial Visualisation

| Report item | Notebook |
|---|---|
| Section 3.1 -- distribution statistics (median, mean, P99, max, zero-rate) | `05` |
| Section 3.1 -- spatial concentration (Manhattan/JFK/LaGuardia vs. outer boroughs) | `05` |
| Figure 2 -- distribution + spatial map | `05` |
| Section 3.2 -- temporal patterns (day-of-week swing, evening commute, midnight) | `05` |
| Figure 3 -- hourly demand by day of week | `05` |
| Section 3.3 -- weather/event relationships | `05` |
| Figure 4 -- demand vs. weather and events | `05` |

### Modelling and Discussion

| Report item | Notebook |
|---|---|
| Table 3 -- regression metrics, all four models | `07` |
| Table 4 -- top-k zone-hour overlap, all four models + random baseline | `07` |
| Table 5 -- GB permutation importance + LASSO retained/eliminated status | `07` |
| Section 5.3 -- DTDI prediction error statistics | `07` |
| Table (reliability under severe weather) -- DTDI compression, MAE ratio, and DTDI-drop-captured per model | `07` (also independently in `08`, since Recommendation 2 cites the same finding) |

Notebook `07` is the direct fix for the report's original problem: Table 3/4's LASSO
numbers used to live in a different notebook than the other three models. All four
models' evaluation now happens in one place, from one shared evaluation frame -- LASSO's
alpha is tuned fresh here (not hardcoded from an earlier run) via a held-out validation
slice from the training period.

### Recommendations

| Report item | Notebook |
|---|---|
| Section 6.1 -- Recommendation 1 (top-k overlap, 25 April 2026 case study) | `08` |
| Section 6.2 -- Recommendation 2 (severe-weather MAE/DTDI reliability evidence) | `08` |

---

## Verifying it worked

```bash
python -m pytest tests
```

Should print `111 passed`.

Then spot-check a few notebook outputs against the report:

- **Table 3** (notebook `07`): Linear Regression MAE=2.30, RMSE=10.16, R2=0.933.
- **Section 6.1** (notebook `08`): busiest test-period day is 2026-04-25; a top-20 list
  using Linear Regression correctly identifies 13/20 (65%) of the actual busiest
  zone-hours that day.
- **Section 6.2** (notebook `08`): severe-weather MAE is ~1.6x normal-weather MAE across
  all four models; mean actual DTDI is 0.898 in severe weather vs. 1.000 normal (demand
  suppressed, not elevated), and no model's predictions meaningfully track this shift.

**A note on Gradient Boosting and LASSO's numbers, if your re-run doesn't match exactly**:

`HistGradientBoostingRegressor` is not perfectly reproducible under multi-threading,
even with a fixed `random_state` -- floating-point summation order varies run to run.
This affects every GB-derived number in the report: Table 3's GB row, the GB column in
Table 4, GB's permutation importances in Table 5, and GB's figures in the reliability
table (Section 5.3/6.2). The swing can be larger than "small deltas" suggests -- across
separate re-runs during final testing, GB's RMSE/R2 flipped from *leading* Linear
Regression/Ridge to *trailing* them, and GB's DTDI "drop captured" figure flipped sign
more than once (report: +14%; independent re-runs: -17%, confirming this isn't a
one-off). If your run shows something similar, that's expected, not a bug in your setup.

LASSO's overall accuracy (MAE/RMSE/R2 in Tables 3/4) has been stable across every re-run
observed so far, but Table 5's retained/eliminated **status** for coefficients sitting
very close to the zero threshold is not -- most notably `taxi_type`, which has been
observed retaining both Yellow and Green, retaining neither, or retaining Yellow only,
across different runs. The train/test split itself is not the cause: it's a fixed
chronological date boundary applied to data already confirmed identical run-to-run (every
notebook `00`-`06` number has reproduced bit-for-bit across every fresh-clone test this
project has run), so it cannot vary. The more likely explanation is the same underlying
phenomenon as Gradient Boosting's non-determinism -- tiny floating-point differences from
Spark's distributed computation of the underlying features, invisible at the precision
anything gets printed to, but enough to nudge a coefficient that's already sitting right
at LASSO's regularisation boundary across zero.

None of this touches the report's actual conclusions: Linear Regression is fully
deterministic and reproduces exactly (including Recommendation 1's case study and
evidence), and Recommendation 2's core finding -- none of the four models reliably
capture the severe-weather demand shift -- holds regardless of which direction GB's
specific number lands on, or which side of zero a borderline LASSO coefficient lands on,
for a given run.

## Repository structure

```
data/                   Raw taxi files go in data/landing/ (Step 3); everything else is generated by the notebooks
models/                 Fitted models: linear_regression, ridge, gradient_boosting (from notebook 06)
notebooks/              Numbered pipeline notebooks 00-08, see the table above
src/                    Source package (ingestion, cleaning, external data, feature engineering, modelling, evaluation)
tests/                  pytest suite for src/
reports/
  figures/              Every figure the report embeds (Figures 1-5)
  data_quality/         Row-removal ledgers, ingestion logs, join-coverage summaries (regenerated by the notebooks)
resources/Geospatial/   TLC taxi zone shapefile + lookup, used for every choropleth map
```

## Modelling notes

Four models were compared: Linear Regression, Ridge, LASSO, and Gradient Boosting
(histogram-based). A Random Forest benchmark was tried early on and dropped after
evaluation -- it underperformed every other model on every metric and contributes
nothing to the final report, so it has been removed from this codebase entirely.
