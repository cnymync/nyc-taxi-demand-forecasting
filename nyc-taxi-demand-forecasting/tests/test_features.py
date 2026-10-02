import numpy as np
import pandas as pd
import pytest

from src.features import (
    CONTINUOUS_FEATURE_COLUMNS,
    LEAKAGE_COLUMNS,
    NON_FEATURE_COLUMNS,
    add_month_feature,
    apply_standard_scaler,
    build_feature_matrix,
    drop_missing_weather_rows,
    fill_gust_not_reported,
    fit_standard_scaler,
    split_train_test,
    split_train_validation,
)


def _minimal_processed_df():
    # Zone 100 = Manhattan ("Garment District"), zone 200 = Bronx, in the
    # real TLC lookup -- used so the linear model's borough merge is exercised
    # against real data, mirroring the pattern in tests/test_processed.py.
    n = 4
    return pd.DataFrame(
        {
            "taxi_type": ["yellow", "yellow", "green", "yellow"],
            "pu_location_id": [100, 100, 200, 100],
            "pickup_hour": pd.to_datetime(
                [
                    "2025-12-31 23:00:00",
                    "2026-01-01 00:00:00",
                    "2026-01-01 01:00:00",
                    "2024-06-15 12:00:00",
                ]
            ),
            "day_of_week": [4, 5, 5, 1],
            "hour_of_day": [23, 0, 1, 12],
            "pickup_count": [10, 5, 2, 20],
            "avg_trip_distance": [3.0, 2.0, 1.0, 4.0],
            "avg_trip_duration_minutes": [15.0, 12.0, 10.0, 18.0],
            "avg_passenger_count": [1.2, 1.1, 1.0, 1.3],
            "extreme_trip_distance_count": [0, 0, 0, 1],
            "extreme_trip_duration_count": [0, 0, 0, 0],
            "is_zero_demand_hour": [False, False, False, False],
            "baseline_pickup_count": [9.5, 4.5, 2.5, 19.0],
            "baseline_source": ["zone_dow_hour"] * n,
            "tmpf": [40.0, 41.0, np.nan, 70.0],
            "dwpf": [30.0, 31.0, np.nan, 60.0],
            "relh": [50.0, 51.0, np.nan, 55.0],
            "sknt": [5.0, 6.0, np.nan, 3.0],
            "gust": [np.nan, np.nan, np.nan, np.nan],
            "vsby": [10.0, 10.0, np.nan, 10.0],
            "p01i": [0.0, 0.0, np.nan, 0.0],
            "skyc1": ["FEW", "CLR", None, "OVC"],
            "wxcodes": [None, None, None, "-RA"],
            "weather_severity": ["normal", "normal", None, "mild"],
            "event_count": [0, 1, 0, 2],
            "event_present": [False, True, False, True],
            "holiday_name": [None, "New Year's Day", "New Year's Day", None],
            "is_holiday": [False, True, True, False],
        }
    )


def test_add_month_feature_derives_from_pickup_hour():
    df = add_month_feature(_minimal_processed_df())
    assert list(df["month"]) == [12, 1, 1, 6]


def test_split_train_test_boundary_is_2026():
    df = _minimal_processed_df()
    train_df, test_df = split_train_test(df)
    assert set(train_df["pickup_hour"].dt.year) == {2025, 2024}
    assert set(test_df["pickup_hour"].dt.year) == {2026}
    assert len(train_df) + len(test_df) == len(df)


def test_drop_missing_weather_rows_removes_unmatched_hours():
    df = _minimal_processed_df()
    result = drop_missing_weather_rows(df)
    assert len(result) == 3
    assert result["tmpf"].isna().sum() == 0


def test_drop_missing_weather_rows_also_drops_sknt_vsby_gaps_not_just_tmpf():
    """Regression test: sknt/vsby can be NaN (genuine station-outage gaps)
    even when tmpf is present -- checking tmpf alone is not sufficient (a
    real bug found and fixed 2026-08-17: LinearRegression failed on the real
    training data with a NaN error even though drop_missing_weather_rows had
    already run, because sknt/vsby weren't part of the dropna subset)."""
    df = _minimal_processed_df()
    # Give one otherwise-valid row (has tmpf) a station-outage gap in sknt only.
    df.loc[df["pu_location_id"] == 100, "sknt"] = [np.nan, 6.0, 6.0]
    result = drop_missing_weather_rows(df)
    assert result["sknt"].isna().sum() == 0
    assert result["vsby"].isna().sum() == 0


def test_fill_gust_not_reported_fills_with_zero_not_dropped():
    df = _minimal_processed_df()
    assert df["gust"].isna().all()  # all NaN in the synthetic fixture, like most real rows
    result = fill_gust_not_reported(df)
    assert len(result) == len(df)  # no rows dropped
    assert (result["gust"] == 0).all()


def test_build_feature_matrix_has_no_remaining_nulls():
    """Regression test for the real bug: LinearRegression.fit() raised
    'Input X contains NaN' on the full real training set because gust was
    left as NaN. Every column in the built feature matrix must be NaN-free
    before it's handed to a model."""
    df = add_month_feature(_minimal_processed_df())
    for model in ["linear", "gradient_boosting"]:
        X, y = build_feature_matrix(df, model)
        assert X.isna().sum().sum() == 0, (
            f"{model}: NaNs remain in {X.columns[X.isna().any()].tolist()}"
        )
        assert not y.isna().any()


def test_build_feature_matrix_gradient_boosting_uses_native_categorical_not_onehot():
    """Regression test: one-hot-encoding pu_location_id (263 columns) made a
    HistGradientBoostingRegressor fit take over 2 hours for 30 iterations in
    a real timing test (2026-08-17) -- the 'gradient_boosting' feature set
    must use native pandas-categorical columns instead, verified here by
    column count and dtype rather than just trusting the code path.

    `pu_location_id` itself stays plain numeric (not categorical) -- a
    second real constraint found the same day: HistGradientBoostingRegressor
    caps native categorical cardinality at 255, and there are 263 zones.
    """
    df = add_month_feature(_minimal_processed_df())
    X, _ = build_feature_matrix(df, "gradient_boosting")

    assert not any(c.startswith("pu_location_id_") for c in X.columns)
    assert "pu_location_id" in X.columns
    assert not isinstance(X["pu_location_id"].dtype, pd.CategoricalDtype)
    assert isinstance(X["hour_of_day"].dtype, pd.CategoricalDtype)
    # Far fewer columns than a one-hot-encoded zone-identity matrix would be (~322 on real data).
    assert len(X.columns) < 20


def test_build_feature_matrix_linear_uses_borough_not_full_zone():
    df = add_month_feature(_minimal_processed_df())
    X, _ = build_feature_matrix(df, "linear")

    assert any(c.startswith("borough_") for c in X.columns)
    assert not any(c.startswith("pu_location_id_") for c in X.columns)
    # Manhattan rows should produce a borough_Manhattan column
    assert "borough_Manhattan" in X.columns


def test_build_feature_matrix_excludes_leakage_and_non_feature_columns():
    df = add_month_feature(_minimal_processed_df())
    for model in ["linear", "gradient_boosting"]:
        X, _ = build_feature_matrix(df, model)
        for col in LEAKAGE_COLUMNS + NON_FEATURE_COLUMNS:
            assert col not in X.columns, f"{col} leaked into the {model} feature matrix"
        assert "pickup_count" not in X.columns  # target, not a feature


def test_build_feature_matrix_uses_log1p_baseline_not_raw():
    """Regression test for the Phase 10/11 fix: the model must see
    log1p(baseline_pickup_count), not the raw value -- the raw scale is what
    caused a tiny coefficient (0.02) to explode into a 2.1M-pickup Linear
    Regression prediction for the busiest zones."""
    df = add_month_feature(_minimal_processed_df())
    for model in ["linear", "gradient_boosting"]:
        X, _ = build_feature_matrix(df, model)
        assert "log1p_baseline_pickup_count" in X.columns
        assert "baseline_pickup_count" not in X.columns
        row = X[X["log1p_baseline_pickup_count"] > 0].iloc[0]
        # sanity: the transform was actually applied, not just renamed
        assert row["log1p_baseline_pickup_count"] < 10  # log1p of any real baseline is small


def test_build_feature_matrix_rejects_unknown_model():
    df = add_month_feature(_minimal_processed_df())
    with pytest.raises(ValueError):
        build_feature_matrix(df, "xgboost")


def test_split_train_validation_carves_last_two_months_of_2025():
    train_df = pd.DataFrame(
        {
            "pickup_hour": pd.to_datetime(
                [
                    "2024-06-15 12:00:00",
                    "2025-06-01 00:00:00",
                    "2025-10-31 23:00:00",
                    "2025-11-01 00:00:00",
                    "2025-12-15 00:00:00",
                    "2025-12-31 23:00:00",
                ]
            )
        }
    )
    train_sub_df, validation_df = split_train_validation(train_df)

    assert list(train_sub_df["pickup_hour"].dt.strftime("%Y-%m-%d")) == [
        "2024-06-15",
        "2025-06-01",
        "2025-10-31",
    ]
    assert list(validation_df["pickup_hour"].dt.strftime("%Y-%m-%d")) == [
        "2025-11-01",
        "2025-12-15",
        "2025-12-31",
    ]
    assert len(train_sub_df) + len(validation_df) == len(train_df)


def test_split_train_validation_rejects_a_boundary_overlapping_the_test_period():
    train_df = pd.DataFrame({"pickup_hour": pd.to_datetime(["2025-06-01 00:00:00"])})
    with pytest.raises(ValueError):
        split_train_validation(train_df, validation_period_start="2026-01-01 00:00:00")


def test_build_feature_matrix_train_and_test_produce_identical_columns():
    """Regression test: pd.get_dummies() only creates a column for a category
    actually present in the data, so one-hot encoding a train split and a
    test split independently can silently produce different column sets if
    a category (e.g. a specific hour, borough, or zone) happens to be
    missing from one split -- this breaks a fitted sklearn model's
    .predict() on the other split. Caught in practice on the real Processed
    table (65 vs 58 columns for the linear feature set) before
    `_onehot`/`build_feature_matrix` were fixed to use fixed category lists.
    """
    df = add_month_feature(_minimal_processed_df())
    train_df, test_df = split_train_test(df)
    assert len(train_df) > 0 and len(test_df) > 0  # sanity: split actually split something

    for model in ["linear", "gradient_boosting"]:
        X_train, _ = build_feature_matrix(train_df, model)
        X_test, _ = build_feature_matrix(test_df, model)
        assert list(X_train.columns) == list(X_test.columns), (
            f"{model}: train/test column mismatch -- "
            f"only in train: {set(X_train.columns) - set(X_test.columns)}, "
            f"only in test: {set(X_test.columns) - set(X_train.columns)}"
        )


def test_build_feature_matrix_preserves_the_input_index_for_both_models():
    """Regression test: the 'linear' branch joins on `borough`, which (if
    done via `.merge()` instead of `.join()`) silently resets the index to
    a fresh RangeIndex, while the 'gradient_boosting' branch (no join) keeps
    the original index -- the two feature matrices would then have
    different index *labels* even with identical row *counts* and
    *columns*, breaking any caller that aligns predictions back to the
    source data by index (found 2026-08-17 via src/evaluate.py, which
    raised a "different row sets" error the first time it was exercised
    against real data, despite src/features.py's own tests all passing --
    those tests never checked index equality, only column equality).
    """
    df = add_month_feature(_minimal_processed_df())
    non_default_index = df.index + 100  # deliberately not 0..n-1, to catch a silent reset
    df.index = non_default_index

    X_linear, _ = build_feature_matrix(df, "linear")
    X_gb, _ = build_feature_matrix(df, "gradient_boosting")

    assert X_linear.index.equals(X_gb.index)
    assert list(X_linear.index) == [
        100,
        101,
        103,
    ]  # row 102 (zone 200) dropped for missing weather


# ---------------------------------------------------------------------------
# Standardization (2026-08-22)
# ---------------------------------------------------------------------------


def test_fit_standard_scaler_only_fits_continuous_columns():
    df = add_month_feature(_minimal_processed_df())
    X, _ = build_feature_matrix(df, "linear")
    scaler = fit_standard_scaler(X)
    assert list(scaler.feature_names_in_) == CONTINUOUS_FEATURE_COLUMNS
    assert len(scaler.mean_) == len(CONTINUOUS_FEATURE_COLUMNS)


def test_apply_standard_scaler_transforms_training_data_to_zero_mean_unit_std():
    df = add_month_feature(_minimal_processed_df())
    X, _ = build_feature_matrix(df, "linear")
    scaler = fit_standard_scaler(X)
    X_scaled = apply_standard_scaler(X, scaler)
    # Fit and applied on the *same* data -> exactly zero mean on every scaled
    # column, and unit (population) std on every column that actually varies
    # in this tiny fixture -- gust/vsby/p01i are constant across all 3
    # remaining rows here, and sklearn correctly leaves a zero-variance
    # column's scale at 1 (mean-centered, not divided by zero) rather than
    # forcing a unit std that isn't mathematically meaningful for it.
    assert np.allclose(X_scaled[CONTINUOUS_FEATURE_COLUMNS].mean(), 0.0, atol=1e-8)
    varying_columns = [c for c in CONTINUOUS_FEATURE_COLUMNS if X[c].std(ddof=0) > 0]
    assert varying_columns  # sanity: the fixture must exercise at least one
    assert np.allclose(X_scaled[varying_columns].std(ddof=0), 1.0, atol=1e-8)


def test_apply_standard_scaler_uses_train_statistics_not_a_fresh_fit_on_test():
    """The leakage-safety property that actually matters: a scaler fit on
    train and applied to a *different*-distribution test set must reuse
    train's mean/std, not silently re-center test data around its own mean
    -- the same class of leakage `compute_dtdi_percentile_thresholds` guards
    against for DTDI's own thresholds."""
    train_df = add_month_feature(_minimal_processed_df())
    X_train, _ = build_feature_matrix(train_df, "linear")
    scaler = fit_standard_scaler(X_train)

    # A test-like frame whose baseline is far outside the training range.
    test_df = _minimal_processed_df()
    test_df["pickup_hour"] = pd.to_datetime(["2026-02-01"] * 4)
    test_df["baseline_pickup_count"] = [500.0, 500.0, 500.0, 500.0]
    test_df = add_month_feature(test_df)
    X_test, _ = build_feature_matrix(test_df, "linear")

    X_test_scaled = apply_standard_scaler(X_test, scaler)
    # If this had been (incorrectly) re-fit on test's own data, a column
    # that is now constant across all test rows would scale to exactly 0
    # everywhere. Using train's statistics instead, a training-range-busting
    # baseline value scales to a large, non-zero number.
    assert (X_test_scaled["log1p_baseline_pickup_count"].abs() > 1.0).all()


def test_apply_standard_scaler_leaves_onehot_and_boolean_columns_untouched():
    df = add_month_feature(_minimal_processed_df())
    X, _ = build_feature_matrix(df, "linear")
    scaler = fit_standard_scaler(X)
    X_scaled = apply_standard_scaler(X, scaler)

    onehot_and_boolean_columns = [c for c in X.columns if c not in CONTINUOUS_FEATURE_COLUMNS]
    pd.testing.assert_frame_equal(
        X_scaled[onehot_and_boolean_columns], X[onehot_and_boolean_columns]
    )
