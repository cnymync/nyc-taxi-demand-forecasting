import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    DTDI_CATEGORY_LABELS,
    build_evaluation_frame,
    classify_dtdi,
    classify_dtdi_by_group,
    compute_dtdi,
    compute_dtdi_group_percentile_levels,
    compute_dtdi_percentile_thresholds,
    dtdi_classification_report,
    dtdi_confusion_matrix,
    regression_metrics,
    regression_metrics_by_group,
    top_k_zone_hour_overlap,
)


def test_regression_metrics_computes_mae_rmse_r2_on_real_values():
    actual = np.array([0.0, 10.0, 20.0, 30.0])
    predicted = np.array([0.0, 10.0, 20.0, 30.0])  # perfect predictions
    metrics = regression_metrics(actual, predicted)
    assert metrics["MAE"] == pytest.approx(0.0)
    assert metrics["RMSE"] == pytest.approx(0.0)
    assert metrics["R2"] == pytest.approx(1.0)
    assert metrics["n"] == 4


def test_regression_metrics_by_group_computes_one_row_per_group():
    df = pd.DataFrame(
        {
            "zone": ["A", "A", "B", "B"],
            "actual": [10.0, 20.0, 0.0, 0.0],
            "pred": [10.0, 10.0, 0.0, 5.0],
        }
    )
    result = regression_metrics_by_group(df, "zone", "actual", "pred")
    assert list(result.index) == ["A", "B"]
    assert result.loc["A", "n"] == 2
    assert result.loc["B", "MAE"] == pytest.approx(2.5)  # mean(|0-0|, |0-5|)


def test_compute_dtdi_is_ratio_and_nan_when_baseline_zero():
    demand = pd.Series([10.0, 5.0, 3.0])
    baseline = pd.Series([5.0, 0.0, 1.0])
    dtdi = compute_dtdi(demand, baseline)
    assert dtdi.iloc[0] == pytest.approx(2.0)
    assert pd.isna(dtdi.iloc[1])  # baseline == 0 -> undefined, not 0 or inf
    assert dtdi.iloc[2] == pytest.approx(3.0)


def test_classify_dtdi_uses_provisional_thresholds():
    dtdi = pd.Series([0.5, 1.2, 1.7, 2.5, np.nan])
    categories = classify_dtdi(dtdi)
    assert list(categories.astype(str)) == [
        "below_normal",
        "normal",
        "elevated",
        "very_elevated",
        "nan",
    ]


def test_dtdi_confusion_matrix_excludes_undefined_rows():
    actual_cat = pd.Series(["normal", "elevated", np.nan, "very_elevated"], dtype="category")
    predicted_cat = pd.Series(["normal", "normal", "elevated", "very_elevated"], dtype="category")
    cm = dtdi_confusion_matrix(actual_cat, predicted_cat)
    assert cm.shape == (len(DTDI_CATEGORY_LABELS), len(DTDI_CATEGORY_LABELS))
    assert cm.values.sum() == 3  # the NaN row excluded


def test_dtdi_classification_report_includes_macro_and_micro_averages():
    actual_cat = pd.Series(["normal", "normal", "elevated", "elevated"], dtype="category")
    predicted_cat = pd.Series(["normal", "elevated", "elevated", "elevated"], dtype="category")
    report = dtdi_classification_report(actual_cat, predicted_cat)
    assert "macro_avg" in report.index
    assert "micro_avg" in report.index
    # elevated: true=2 (both predicted elevated -> recall 1.0), predicted=3 (1 false positive) -> precision 2/3
    assert report.loc["elevated", "recall"] == pytest.approx(1.0)
    assert report.loc["elevated", "precision"] == pytest.approx(2 / 3)


def test_top_k_zone_hour_overlap_full_when_predictions_match_actual():
    df = pd.DataFrame({"actual": [5, 4, 3, 2, 1], "pred": [5, 4, 3, 2, 1]})
    assert top_k_zone_hour_overlap(df, k=2, actual_col="actual", predicted_col="pred") == 1.0


def test_top_k_zone_hour_overlap_partial_when_rankings_differ():
    df = pd.DataFrame({"actual": [10, 9, 1, 0], "pred": [1, 0, 10, 9]})
    # top-2 actual = rows 0,1; top-2 pred = rows 2,3 -- no overlap
    assert top_k_zone_hour_overlap(df, k=2, actual_col="actual", predicted_col="pred") == 0.0


def test_classify_dtdi_accepts_custom_percentile_bins():
    dtdi = pd.Series([0.5, 1.2, 2.0, 3.0])
    custom_bins = [-np.inf, 1.0, 1.8, 2.5, np.inf]  # narrower "elevated" band than the default
    categories = classify_dtdi(dtdi, bins=custom_bins)
    assert list(categories.astype(str)) == ["below_normal", "normal", "elevated", "very_elevated"]
    # 2.0 falls in "elevated" under the narrower custom bins but would be
    # "elevated" under the default 1.5/2.0 bins too -- use a value that
    # differs between the two schemes to prove `bins` is actually used.
    assert classify_dtdi(pd.Series([1.6]), bins=custom_bins).iloc[0] == "normal"
    assert classify_dtdi(pd.Series([1.6])).iloc[0] == "elevated"  # default bins: >1.5


def test_compute_dtdi_percentile_thresholds_is_leakage_safe_and_baseline_filtered():
    # 10 "reliable" rows (baseline >= 1.0) with a known DTDI spread, plus a
    # handful of tiny-baseline rows with extreme DTDI that must NOT affect
    # the computed percentiles.
    reliable_actual = list(range(1, 11))  # DTDI = 1..10 when baseline = 1.0
    tiny_baseline_actual = [50, 50]  # DTDI = 500 if baseline were 0.1 -- must be excluded
    train_df = pd.DataFrame(
        {
            "pickup_count": reliable_actual + tiny_baseline_actual,
            "baseline_pickup_count": [1.0] * 10 + [0.1, 0.1],
        }
    )
    bins = compute_dtdi_percentile_thresholds(train_df, min_baseline=1.0, percentiles=(75, 95))
    assert bins[0] == -np.inf
    assert bins[1] == 1.0  # fixed "below normal" anchor, unaffected by distribution
    assert bins[3] < 100  # would be ~500 if the tiny-baseline rows leaked in
    assert bins[4] == np.inf


# ---------------------------------------------------------------------------
# Per-(zone, hour) DTDI percentile thresholds (Round 4)
# ---------------------------------------------------------------------------


def _group_threshold_train_df():
    """Four (zone, hour) situations, all baseline == 1.0 so pickup_count IS
    the DTDI value directly:
    - zone=100, hour=8: 6 rows, tightly clustered DTDI (~1.0-1.2) -- a
      "busy zone, low natural variance" slot, plenty of its own history.
    - zone=200, hour=8: 6 rows, widely spread DTDI (1.0-5.0) -- a "quiet
      zone, high natural variance" slot, also plenty of its own history.
      Same row count as zone 100's slot, deliberately, so any threshold
      difference between them is due to the *data*, not sample size.
    - zone=300: only 2 rows at hour=8 (too thin for its own zone_hour
      threshold at min_group_obs=5) but 6 more rows at hour=9 -- the
      *zone*-level pool (8 rows total) is reliable, so hour=8 rows here
      should fall back to the zone level, not global.
    - zone=400: a single row total, at any hour -- both zone_hour and zone
      levels are too thin, so this must fall back all the way to global.
    """
    rows = []
    for dtdi in [1.0, 1.05, 1.1, 1.1, 1.15, 1.2]:
        rows.append((100, 8, dtdi, 1.0))
    for dtdi in [1.0, 2.0, 3.0, 4.0, 4.5, 5.0]:
        rows.append((200, 8, dtdi, 1.0))
    for dtdi in [1.0, 1.1]:
        rows.append((300, 8, dtdi, 1.0))
    for dtdi in [1.0, 1.2, 1.4, 1.6, 1.8, 2.0]:
        rows.append((300, 9, dtdi, 1.0))
    rows.append((400, 5, 1.0, 1.0))

    df = pd.DataFrame(
        rows, columns=["pu_location_id", "hour_of_day", "pickup_count", "baseline_pickup_count"]
    )
    return df


def test_compute_dtdi_group_percentile_levels_differ_by_group():
    train_df = _group_threshold_train_df()
    levels = compute_dtdi_group_percentile_levels(train_df, percentiles=(75, 95))

    zone_hour = levels["zone_hour"].set_index(["pu_location_id", "hour_of_day"])
    tight = zone_hour.loc[(100, 8)]
    wide = zone_hour.loc[(200, 8)]
    # Same sample size (6 rows each), but zone 200's spread-out DTDI must
    # produce a much higher P95 than zone 100's tightly clustered DTDI --
    # proving thresholds are genuinely computed per group, not one pooled
    # citywide number.
    assert wide["p_high"] > tight["p_high"] * 2


def test_classify_dtdi_by_group_uses_zone_hour_when_reliable():
    train_df = _group_threshold_train_df()
    levels = compute_dtdi_group_percentile_levels(train_df, percentiles=(75, 95))

    # A DTDI of 2.0 is "very_elevated" for the tight zone (its own P95 ~1.2)
    # but well within normal range for the wide zone (its own P95 ~4.9) --
    # same raw DTDI value, opposite classification, because each row uses
    # its own group's thresholds.
    to_classify = pd.DataFrame(
        {"pu_location_id": [100, 200], "hour_of_day": [8, 8], "dtdi": [2.0, 2.0]}
    )
    result = classify_dtdi_by_group(to_classify, levels, dtdi_col="dtdi", min_group_obs=5)
    assert list(result) == ["very_elevated", "normal"]


def test_classify_dtdi_by_group_falls_back_to_zone_when_zone_hour_is_thin():
    train_df = _group_threshold_train_df()
    levels = compute_dtdi_group_percentile_levels(train_df, percentiles=(75, 95))

    zone_hour_row = levels["zone_hour"].set_index(["pu_location_id", "hour_of_day"]).loc[(300, 8)]
    zone_row = levels["zone"].set_index("pu_location_id").loc[300]
    assert zone_hour_row["n_obs"] < 5  # too thin to trust on its own
    assert zone_row["n_obs"] >= 5  # the zone-wide pool (hour 8 + hour 9) is not

    to_classify = pd.DataFrame({"pu_location_id": [300], "hour_of_day": [8], "dtdi": [1.5]})
    result = classify_dtdi_by_group(to_classify, levels, dtdi_col="dtdi", min_group_obs=5)
    # 1.5 must be classified against the *zone* thresholds (computed from
    # all 8 zone-300 rows), not zone 300 hour 8's own too-thin 2 rows.
    zone_classified = classify_dtdi(
        pd.Series([1.5]), bins=[-np.inf, 1.0, zone_row["p_low"], zone_row["p_high"], np.inf]
    )
    assert list(result) == list(zone_classified.astype(str))


def test_classify_dtdi_by_group_falls_back_to_global_when_both_levels_are_thin():
    train_df = _group_threshold_train_df()
    levels = compute_dtdi_group_percentile_levels(train_df, percentiles=(75, 95))
    global_p_low, global_p_high = levels["global"]

    to_classify = pd.DataFrame({"pu_location_id": [400], "hour_of_day": [5], "dtdi": [1.5]})
    result = classify_dtdi_by_group(to_classify, levels, dtdi_col="dtdi", min_group_obs=5)
    expected = classify_dtdi(
        pd.Series([1.5]), bins=[-np.inf, 1.0, global_p_low, global_p_high, np.inf]
    )
    assert list(result) == list(expected.astype(str))


def test_classify_dtdi_by_group_leaves_undefined_dtdi_unclassified():
    train_df = _group_threshold_train_df()
    levels = compute_dtdi_group_percentile_levels(train_df, percentiles=(75, 95))

    to_classify = pd.DataFrame({"pu_location_id": [100], "hour_of_day": [8], "dtdi": [np.nan]})
    result = classify_dtdi_by_group(to_classify, levels, dtdi_col="dtdi", min_group_obs=5)
    assert pd.isna(result[0])


def test_classify_dtdi_by_group_preserves_row_order_regardless_of_input_index():
    # Deliberately shuffled, non-default index -- this project has already
    # been burned once by an operation that silently reset row order/index
    # (`.merge()` in src/features.py, found 2026-08-17); this locks in that
    # classify_dtdi_by_group does not repeat it.
    train_df = _group_threshold_train_df()
    levels = compute_dtdi_group_percentile_levels(train_df, percentiles=(75, 95))

    to_classify = pd.DataFrame(
        {"pu_location_id": [200, 100, 200], "hour_of_day": [8, 8, 8], "dtdi": [2.0, 2.0, 0.5]},
        index=[42, 7, 99],
    )
    result = classify_dtdi_by_group(to_classify, levels, dtdi_col="dtdi", min_group_obs=5)
    # Row 0 (zone 200) -> normal, row 1 (zone 100) -> very_elevated (same
    # 2.0 DTDI as the group-differentiation test above), row 2 -> below_normal.
    assert list(result) == ["normal", "very_elevated", "below_normal"]


def test_build_evaluation_frame_produces_one_prediction_column_per_model():
    from src.modeling.train import fit_gradient_boosting, fit_linear_regression

    df = pd.DataFrame(
        {
            "taxi_type": ["yellow"] * 6,
            "pu_location_id": [100] * 6,
            "pickup_hour": pd.date_range("2026-01-01", periods=6, freq="h"),
            "day_of_week": [5] * 6,
            "hour_of_day": list(range(6)),
            "month": [1] * 6,
            "pickup_count": [1, 2, 3, 4, 5, 6],
            "baseline_pickup_count": [1.0] * 6,
            "event_count": [0] * 6,
            "event_present": [False] * 6,
            "is_holiday": [False] * 6,
            "weather_severity": ["normal"] * 6,
            "tmpf": [50.0] * 6,
            "dwpf": [40.0] * 6,
            "relh": [50.0] * 6,
            "sknt": [5.0] * 6,
            "gust": [np.nan] * 6,
            "vsby": [10.0] * 6,
            "p01i": [0.0] * 6,
        }
    )
    from src.features import build_feature_matrix

    X, y = build_feature_matrix(df, "linear")
    linear_model = fit_linear_regression(X, y)
    X_gb, y_gb = build_feature_matrix(df, "gradient_boosting")
    gb_model = fit_gradient_boosting(X_gb, y_gb)

    eval_df = build_evaluation_frame(
        df, {"linear_regression": linear_model, "gradient_boosting": gb_model}
    )
    assert "pred_linear_regression" in eval_df.columns
    assert "pred_gradient_boosting" in eval_df.columns
    assert "actual" in eval_df.columns
    assert len(eval_df) == 6
