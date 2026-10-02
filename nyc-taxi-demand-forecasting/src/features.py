"""Phase 9: feature engineering and temporal train/test split.

Turns `data/processed/pickups_zone_hour.parquet` (Phase 6 output) into a
model-ready feature matrix for each of the two Phase 10 models, and performs
the one-time chronological split. Fits nothing itself -- that's Phase 10.

Decisions locked 2026-08-17:
- Cyclical time features (hour/day-of-week/month) use one-hot encoding, not
  sin/cos -- deferred as a potential future improvement.
- Zone identity is asymmetric: Gradient Boosting gets the full `pu_location_id`
  (263 zones, handles high cardinality natively via categorical dtype support);
  Linear Regression gets `borough` (~6 categories) + `baseline_pickup_count`
  instead, to stay the interpretable benchmark and avoid 263 extra coefficients.
- Weather features: `weather_severity` + the raw numeric fields. `skyc1`/
  `wxcodes` dropped (weather_severity is already derived from wxcodes).
- Target: `log1p(pickup_count)` for both models -- back-transform with
  `expm1` before reporting metrics or feeding predictions into DTDI.

Usage:
    python -m src.features            # build both feature matrices, print shapes
"""

from pathlib import Path

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import typer

from src.geospatial import load_processed_zone_hour
from src.processed import (
    TAXI_ZONE_LOOKUP_PATH,
    TRAIN_PERIOD_END_EXCLUSIVE,
    load_geographic_zone_ids,
)

# Leakage: computed from the same hour's own actual trips, can't be known
# before the hour happens -- see Phase 6's aggregate_to_zone_hour().
LEAKAGE_COLUMNS = [
    "avg_trip_distance",
    "avg_trip_duration_minutes",
    "avg_passenger_count",
    "extreme_trip_distance_count",
    "extreme_trip_duration_count",
    "is_zero_demand_hour",
]

# Metadata/provenance, not model inputs.
NON_FEATURE_COLUMNS = ["holiday_name", "baseline_source", "skyc1", "wxcodes"]

WEATHER_NUMERIC_COLUMNS = ["tmpf", "dwpf", "relh", "sknt", "gust", "vsby", "p01i"]

# Shared across both models' feature sets.
BASE_ONEHOT_COLUMNS = ["taxi_type", "hour_of_day", "day_of_week", "month", "weather_severity"]
# `log1p_baseline_pickup_count`, not raw `baseline_pickup_count` -- found
# 2026-08-17 (Phase 10/11): a raw-scale baseline feeding a log1p-space
# target let a tiny, well-behaved coefficient (0.02) explode into a
# 2.1M-pickup prediction for the busiest zones (baseline ~640). log1p
# brings the feature onto the same scale as the target it's predicting.
# The *raw* `baseline_pickup_count` column is untouched on the source
# table -- DTDI computation (src/evaluate.py) always reads it directly
# from there, never from this log-transformed model input.
BASE_NUMERIC_COLUMNS = ["log1p_baseline_pickup_count", "event_count"] + WEATHER_NUMERIC_COLUMNS
BASE_BOOLEAN_COLUMNS = ["event_present", "is_holiday"]

TARGET_COLUMN = "pickup_count"

# Validation slice for LASSO alpha tuning (locked 2026-08-17): the last 2
# months of the training period, so tuning decisions never touch the 2026
# test set even indirectly. Must be strictly before
# TRAIN_PERIOD_END_EXCLUSIVE (2026-01-01).
VALIDATION_PERIOD_START = "2025-11-01 00:00:00"

# Fixed category domains for every one-hot column -- NOT inferred from
# whatever happens to appear in a given df slice. pandas.get_dummies() only
# creates a column for a category that's actually present, so encoding
# train and test separately from their own observed values silently
# produces different column sets between the two (found 2026-08-17: 65 vs
# 58 columns for the linear feature set) -- exactly the kind of mismatch
# that breaks a fitted sklearn model's .predict() on new data. Declaring
# every category up front, independent of any split, prevents that class
# of bug entirely, for splits by time as well as any other subset.
TAXI_TYPE_CATEGORIES = ["yellow", "green"]
HOUR_OF_DAY_CATEGORIES = list(range(24))
DAY_OF_WEEK_CATEGORIES = list(range(1, 8))
MONTH_CATEGORIES = list(range(1, 13))
WEATHER_SEVERITY_CATEGORIES = ["normal", "mild", "severe"]


def add_month_feature(df: pd.DataFrame) -> pd.DataFrame:
    """`month` (1-12), derived from `pickup_hour` -- not present on the Phase 6
    table. Motivated directly by the JFK/LaGuardia seasonality finding
    (Section 2.3.2 of the report): the Phase 6 baseline has no month
    component, so the model needs this to have any chance of learning
    seasonal effects the baseline alone can't capture."""
    return df.assign(month=df["pickup_hour"].dt.month)


def add_log_baseline_feature(df: pd.DataFrame) -> pd.DataFrame:
    """`log1p(baseline_pickup_count)` as a separate engineered column -- the
    Phase 10/11 fix for the raw-scale-baseline-vs-log-space-target mismatch
    (see the BASE_NUMERIC_COLUMNS comment above). Leaves the original
    `baseline_pickup_count` column untouched, since DTDI computation
    (src/evaluate.py) needs the real, non-transformed value."""
    return df.assign(log1p_baseline_pickup_count=np.log1p(df["baseline_pickup_count"]))


def load_zone_boroughs(lookup_path: Path = TAXI_ZONE_LOOKUP_PATH) -> pd.DataFrame:
    """`pu_location_id` -> `borough`, from the TLC taxi zone lookup CSV."""
    lookup = pd.read_csv(lookup_path)[["LocationID", "Borough"]].rename(
        columns={"LocationID": "pu_location_id", "Borough": "borough"}
    )
    return lookup


def borough_categories(lookup_path: Path = TAXI_ZONE_LOOKUP_PATH) -> list[str]:
    """The fixed set of boroughs across the 263 real geographic zones (not
    the "Unknown"/blank values that only apply to the 2 excluded placeholder
    zones) -- computed from the lookup file rather than hardcoded, so it
    stays correct if the file is ever updated."""
    boroughs = load_zone_boroughs(lookup_path)
    geographic_ids = set(load_geographic_zone_ids())
    return sorted(
        boroughs.loc[boroughs["pu_location_id"].isin(geographic_ids), "borough"].unique()
    )


def drop_missing_weather_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with no weather match at all -- the documented Phase 6 gap
    (weather data ends 2026-05-30 23:00, one day short of the study period's
    2026-05-31) -- and rows with a genuine station-outage gap in `sknt`/`vsby`
    (Phase 5: "26 hours are genuinely missing ... left as gaps, not imputed").

    NOTE: contrary to an earlier assumption here, not every weather column is
    populated together -- confirmed 2026-08-17 by inspection of the real
    feature matrix: `tmpf` alone was insufficient, `sknt`/`vsby` have their
    own small independent gaps. `gust`
    is NOT included here even though it's frequently NULL -- see
    `fill_gust_not_reported`, which fills it rather than dropping, since a
    NULL gust means "no gust to report" (calm/steady wind), not missing data.
    """
    return df.dropna(subset=["tmpf", "sknt", "vsby"])


def fill_gust_not_reported(df: pd.DataFrame) -> pd.DataFrame:
    """`gust` is NULL when there's nothing to report (calm/steady wind, no
    gust above the reporting threshold) -- a legitimate value, not missing
    data (documented in `src/external_weather.py`). Filled with 0 rather
    than dropped -- dropping would discard ~74% of rows (6.8M/9.2M train
    rows have NULL gust), which would gut the training set for no reason.
    """
    return df.assign(gust=df["gust"].fillna(0))


def split_train_test(
    df: pd.DataFrame, train_period_end_exclusive: str = TRAIN_PERIOD_END_EXCLUSIVE
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split on `pickup_hour` -- 2024-2025 train, 2026 test.
    Reuses the exact boundary already used to build the Phase 6 baseline
    (`src/processed.py::TRAIN_PERIOD_END_EXCLUSIVE`), not a new constant --
    the baseline and the split must agree on what "train" means.
    """
    boundary = pd.Timestamp(train_period_end_exclusive)
    train_df = df[df["pickup_hour"] < boundary]
    test_df = df[df["pickup_hour"] >= boundary]
    return train_df, test_df


def split_train_validation(
    train_df: pd.DataFrame, validation_period_start: str = VALIDATION_PERIOD_START
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carve a validation slice out of the *training* period only -- must be
    called on the output of `split_train_test`'s `train_df`, never on the
    full table, so the 2026 test set is never touched by tuning decisions
    even indirectly. Default: the last 2 months of 2025.
    """
    boundary = pd.Timestamp(validation_period_start)
    if boundary >= pd.Timestamp(TRAIN_PERIOD_END_EXCLUSIVE):
        raise ValueError(
            f"validation_period_start ({validation_period_start}) must be strictly "
            f"before TRAIN_PERIOD_END_EXCLUSIVE ({TRAIN_PERIOD_END_EXCLUSIVE}) -- "
            "a validation slice that overlaps the test period defeats its purpose."
        )
    train_sub_df = train_df[train_df["pickup_hour"] < boundary]
    validation_df = train_df[train_df["pickup_hour"] >= boundary]
    return train_sub_df, validation_df


def _onehot(df: pd.DataFrame, categories_by_column: dict[str, list]) -> pd.DataFrame:
    """One-hot encode each column against its full, fixed category list (not
    just the categories present in `df`) -- see the module-level comment on
    the *_CATEGORIES constants for why this matters."""
    df = df.copy()
    for column, categories in categories_by_column.items():
        df[column] = pd.Categorical(df[column], categories=categories)
    return pd.get_dummies(
        df, columns=list(categories_by_column), prefix=list(categories_by_column)
    )


def _categorical_native(df: pd.DataFrame, categories_by_column: dict[str, list]) -> pd.DataFrame:
    """Encode each column as a pandas 'category' dtype (NOT one-hot) against
    its full, fixed category list -- for models with native categorical
    support (`HistGradientBoostingRegressor(categorical_features="from_dtype")`).

    Added 2026-08-17 after one-hot encoding `pu_location_id` (263 columns)
    made a HistGradientBoostingRegressor fit take over 2 hours for just 30
    boosting iterations (extrapolated: many hours for a real fit) --
    sklearn's histogram-based splitter has to evaluate every one-hot column
    as its own binary feature, which is a poor fit for a 263-category
    variable. Native categorical support lets it use one column with an
    efficient partition-search algorithm instead, at a fraction of the cost.
    """
    df = df.copy()
    for column, categories in categories_by_column.items():
        df[column] = pd.Categorical(df[column], categories=categories)
    return df


BUILD_FEATURE_MATRIX_MODELS = {"linear", "gradient_boosting"}


def build_feature_matrix(processed_df: pd.DataFrame, model: str) -> tuple[pd.DataFrame, pd.Series]:
    """Build (X, y) for one of the Phase 10 models.

    `model`: "linear" (borough + baseline, no full zone identity, one-hot
    categoricals) -- Linear Regression, Ridge, LASSO. "gradient_boosting"
    (full `pu_location_id`, *native* pandas-categorical encoding, not
    one-hot -- see `_categorical_native`) -- HistGradientBoostingRegressor.
    """
    if model not in BUILD_FEATURE_MATRIX_MODELS:
        raise ValueError(f"model must be one of {BUILD_FEATURE_MATRIX_MODELS}, got {model!r}")

    df = add_month_feature(processed_df)
    df = drop_missing_weather_rows(df)
    df = fill_gust_not_reported(df)
    df = add_log_baseline_feature(df)

    categories_by_column = {
        "taxi_type": TAXI_TYPE_CATEGORIES,
        "hour_of_day": HOUR_OF_DAY_CATEGORIES,
        "day_of_week": DAY_OF_WEEK_CATEGORIES,
        "month": MONTH_CATEGORIES,
        "weather_severity": WEATHER_SEVERITY_CATEGORIES,
    }
    keep_columns = list(BASE_ONEHOT_COLUMNS) + BASE_NUMERIC_COLUMNS + BASE_BOOLEAN_COLUMNS

    if model == "linear":
        # .join(), not .merge() -- merge() resets the index to a fresh
        # RangeIndex even for a 1:1 left join, which silently breaks any
        # caller that aligns the returned X back to the original df by
        # index (e.g. src/evaluate.py::build_evaluation_frame, which raised
        # a "different row sets" error the first time this was exercised --
        # found and fixed 2026-08-17). join(on=...) preserves the caller's
        # index; the join key here (pu_location_id) has no duplicates in
        # the lookup, so this is a safe 1:1 join, not a fan-out risk.
        df = df.join(load_zone_boroughs().set_index("pu_location_id"), on="pu_location_id")
        categories_by_column["borough"] = borough_categories()
        keep_columns = keep_columns + ["borough"]
    else:
        # pu_location_id (263 zones) exceeds HistGradientBoostingRegressor's
        # native-categorical cardinality cap (255, found 2026-08-17 as a
        # real ValueError on this exact column) -- kept as a plain numeric
        # column instead (NOT added to categories_by_column), so sklearn
        # auto-bins it as a continuous feature (up to max_bins=255) rather
        # than raising. The other categorical columns above are all well
        # under the cap and still get native categorical treatment.
        keep_columns = keep_columns + ["pu_location_id"]

    y = pd.Series(
        np.log1p(df[TARGET_COLUMN].to_numpy()),
        index=df.index,
        name="log1p_pickup_count",
    )
    if model == "gradient_boosting":
        X = _categorical_native(df[keep_columns], categories_by_column)
    else:
        X = _onehot(df[keep_columns], categories_by_column)

    return X, y


# ---------------------------------------------------------------------------
# Standardization (2026-08-22): the "linear" feature set's continuous
# columns are on very different natural scales (log1p-baseline is a small
# handful of units; temperature is 0-100; wind speed is 0-50 knots; the
# one-hot/boolean columns are already 0/1). Ridge's L2 penalty shrinks
# coefficients by their raw magnitude, so unstandardized inputs let the
# regularization strength act very differently on different features for
# reasons that have nothing to do with genuine importance -- the
# preprocessing audit flagged this as the most concrete, checkable
# candidate for "would a change here actually improve something." This is
# a standalone experiment, not a change to the existing "linear" feature
# set: `fit_standard_scaler` must be called on *training* data only, then
# `apply_standard_scaler` reused (transform, not re-fit) on any other
# split -- fitting separately on train and test would let each split's own
# distribution leak into its own scaling, which is a (subtle) form of
# leakage the same way computing DTDI thresholds from test data would be.
CONTINUOUS_FEATURE_COLUMNS = [
    "log1p_baseline_pickup_count",
    "event_count",
] + WEATHER_NUMERIC_COLUMNS


def fit_standard_scaler(
    X_train: pd.DataFrame, columns: list[str] = CONTINUOUS_FEATURE_COLUMNS
) -> StandardScaler:
    """Fit a `StandardScaler` on the continuous columns of `X_train` only.
    Never call this on test/validation data -- see the module note above."""
    scaler = StandardScaler()
    scaler.fit(X_train[columns])
    return scaler


def apply_standard_scaler(
    X: pd.DataFrame, scaler: StandardScaler, columns: list[str] = CONTINUOUS_FEATURE_COLUMNS
) -> pd.DataFrame:
    """Transform (never re-fit) `X`'s continuous columns with an already-fit
    `scaler`. One-hot/boolean columns are left untouched -- they're already
    0/1, standardizing them would not be meaningful."""
    X = X.copy()
    X[columns] = scaler.transform(X[columns])
    return X


app = typer.Typer()


@app.command()
def main():
    """Build both feature matrices from the full Processed table and print shapes."""
    processed_df = load_processed_zone_hour()
    train_df, test_df = split_train_test(processed_df)
    train_sub_df, validation_df = split_train_validation(train_df)
    logger.info(
        f"Train rows: {len(train_df):,} (of which fit-sub: {len(train_sub_df):,}, "
        f"validation: {len(validation_df):,}) | Test rows: {len(test_df):,}"
    )

    for model in ["linear", "gradient_boosting"]:
        X_train, y_train = build_feature_matrix(train_df, model)
        X_test, y_test = build_feature_matrix(test_df, model)
        logger.info(
            f"[{model}] X_train {X_train.shape}, y_train {y_train.shape}, "
            f"X_test {X_test.shape}, y_test {y_test.shape}"
        )


if __name__ == "__main__":
    app()
