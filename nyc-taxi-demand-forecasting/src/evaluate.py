"""Phase 11: model evaluation, error analysis, and DTDI computation.

Two-layer evaluation framework:

- **Layer 1 -- regression metrics** (MAE/RMSE/R2, real-count scale, `expm1`-
  back-transformed): the primary evaluation of the demand-prediction models
  themselves. Answers "how accurate is the model."
- **Layer 2 -- DTDI and DTDI-derived classification-style metrics**
  (precision/recall/F1/confusion matrix, macro vs micro): a *derived*
  evaluation, not applicable to the raw regression models directly. Answers
  "did the model correctly flag unusually high demand," which is what the
  research question actually asks for.

**Round 2 (2026-08-17)**: extended to evaluate two more models (Ridge,
HistGradientBoosting -- see `src/modeling/train.py::MODEL_REGISTRY`), and to
address a DTDI small-baseline instability finding using the now-available
empirical distribution. `build_evaluation_frame` is model-agnostic (works
over `MODEL_REGISTRY`, not hardcoded to two named models). DTDI
classification supports **two** threshold schemes, computed with the same
`classify_dtdi`:

- `DTDI_CATEGORY_BINS` (original): provisional fixed 1.0/1.5/2.0 thresholds
  -- kept for comparison, still descriptive/operational, not statistically
  validated.
- `compute_dtdi_percentile_thresholds(train_df)`: data-driven upper
  thresholds (P75/P95 of training-period DTDI, computed only from zone-hours
  with `baseline_pickup_count >= MIN_BASELINE_FOR_DTDI`) with `<1.0` kept as
  the "below normal" anchor regardless of distribution. The minimum-baseline
  filter addresses the small-baseline instability found in the first
  evaluation pass. Percentiles are computed from **training-period data
  only** -- using 2026 test-period actual DTDI to define the categories a
  test-period evaluation is then judged against would leak test information
  into the evaluation itself.

**Round 4 (2026-08-17, later)**: the citywide percentile scheme above still
pools every reliable zone-hour into one P75/P95, and every model's recall
for `elevated`/`very_elevated` under it was found to be near 0% -- diagnosed
as the citywide pooling itself, not a modelling failure. Added
`compute_dtdi_group_percentile_levels`/`classify_dtdi_by_group`: per-
(`pu_location_id`, `hour_of_day`) reference-population thresholds, with a
`zone_hour -> zone -> global` fallback for thin groups.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    mean_absolute_error,
    precision_recall_fscore_support,
    r2_score,
    root_mean_squared_error,
)

from src.features import build_feature_matrix
from src.modeling.predict import predict_demand
from src.modeling.train import MODEL_REGISTRY

# ---------------------------------------------------------------------------
# Layer 1: regression metrics
# ---------------------------------------------------------------------------


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """MAE/RMSE/R2 in real-count units. Caller is responsible for having
    already back-transformed both arrays out of log-space (see
    `src/modeling/predict.py::predict_demand`, which does this for
    predictions -- `actual` here should be `np.expm1(y)`, never raw `y`)."""
    return {
        "MAE": mean_absolute_error(actual, predicted),
        "RMSE": root_mean_squared_error(actual, predicted),
        "R2": r2_score(actual, predicted),
        "n": len(actual),
    }


def regression_metrics_by_group(
    df: pd.DataFrame, group_col: str, actual_col: str, pred_col: str
) -> pd.DataFrame:
    """One row of regression_metrics per distinct value of `group_col` --
    the Phase 11 "error analysis by zone/hour/day-of-week/..." task."""
    rows = []
    for group_value, g in df.groupby(group_col, observed=True):
        metrics = regression_metrics(g[actual_col].to_numpy(), g[pred_col].to_numpy())
        rows.append({group_col: group_value, **metrics})
    return pd.DataFrame(rows).set_index(group_col).sort_index()


# ---------------------------------------------------------------------------
# Building the evaluation frame
# ---------------------------------------------------------------------------

EVAL_CONTEXT_COLUMNS = [
    "taxi_type",
    "pu_location_id",
    "pickup_hour",
    "day_of_week",
    "hour_of_day",
    "baseline_pickup_count",
    "event_present",
    "is_holiday",
    "weather_severity",
]


def build_evaluation_frame(test_df: pd.DataFrame, models: dict) -> pd.DataFrame:
    """One row per test zone-hour: actual demand + every model's prediction
    (all in real-count units, column `pred_<model_name>`) + the context
    columns needed for error breakdowns and DTDI.

    `models`: {model_name: fitted_model}, where each `model_name` must be a
    key in `src.modeling.train.MODEL_REGISTRY` (used to look up which
    feature set -- "linear" or "gradient_boosting" -- that model expects).
    Each feature set is only built once even if multiple models share it.
    Row-set consistency across feature sets is asserted, not assumed --
    found a real bug here on 2026-08-17 (see `src/features.py`'s
    `.join()` vs `.merge()` fix).
    """
    feature_matrices: dict[str, tuple] = {}
    eval_df = None
    reference_index = None

    for model_name, model in models.items():
        feature_set = MODEL_REGISTRY[model_name]["feature_set"]
        if feature_set not in feature_matrices:
            feature_matrices[feature_set] = build_feature_matrix(test_df, feature_set)
        X, y = feature_matrices[feature_set]

        if reference_index is None:
            reference_index = X.index
            eval_df = test_df.loc[reference_index, EVAL_CONTEXT_COLUMNS].copy()
            eval_df["actual"] = np.expm1(y.to_numpy())
        elif not X.index.equals(reference_index):
            raise ValueError(
                f"{model_name} ({feature_set} feature set) has a different row set than "
                "the first model evaluated -- expected identical preprocessing to "
                "produce identical rows across all feature sets."
            )

        eval_df[f"pred_{model_name}"] = predict_demand(model, X)

    return eval_df


# ---------------------------------------------------------------------------
# Layer 2: DTDI
# ---------------------------------------------------------------------------

# Provisional, descriptive/operational thresholds from resources/DTDI
# Definition and Reporting Plan.md -- explicitly NOT statistically
# validated significance thresholds. Kept as the default / comparison
# baseline against the percentile-based scheme below.
DTDI_CATEGORY_BINS = [-np.inf, 1.0, 1.5, 2.0, np.inf]
DTDI_CATEGORY_LABELS = ["below_normal", "normal", "elevated", "very_elevated"]

# A minimum baseline/observation threshold before using DTDI, found
# empirically necessary 2026-08-17: zone-hours with a tiny baseline (median
# 0.012 in the worst confusion-matrix cell) produce wildly unstable DTDI
# ratios even from a small, practically-meaningless predicted demand.
MIN_BASELINE_FOR_DTDI = 1.0


def compute_dtdi(demand: pd.Series, baseline: pd.Series) -> pd.Series:
    """DTDI = demand / baseline. Rows with baseline == 0 get NaN (the ratio
    is mathematically undefined, not silently 0 or infinity). On the real
    2026 test set, ~11% of rows have baseline == 0 (mostly Green taxi
    zone-hours) and are excluded from any DTDI-based analysis as a result.
    """
    return demand / baseline.replace(0, np.nan)


def compute_dtdi_percentile_thresholds(
    train_df: pd.DataFrame,
    min_baseline: float = MIN_BASELINE_FOR_DTDI,
    percentiles: tuple[float, float] = (75, 95),
) -> list[float]:
    """Data-driven `elevated`/`very_elevated` bin edges, computed from
    **training-period actual DTDI only** (2024-2025 -- never test-period
    outcomes, which would leak test information into the categories a test
    evaluation is then judged against). Restricted to
    `baseline_pickup_count >= min_baseline` so the small-baseline
    instability doesn't contaminate the thresholds themselves. `<1.0` is
    kept fixed as "below normal" regardless of distribution -- that
    boundary is definitionally meaningful (below baseline is below
    baseline) independent of how the rest of the distribution looks.
    """
    dtdi_train = compute_dtdi(train_df["pickup_count"], train_df["baseline_pickup_count"])
    reliable = dtdi_train[train_df["baseline_pickup_count"] >= min_baseline].dropna()
    p_low, p_high = np.percentile(reliable, percentiles)
    return [-np.inf, 1.0, p_low, p_high, np.inf]


def classify_dtdi(dtdi: pd.Series, bins: list[float] = DTDI_CATEGORY_BINS) -> pd.Series:
    """Categorise DTDI using either the fixed provisional thresholds
    (default) or a set of bin edges from `compute_dtdi_percentile_thresholds`.
    NaN DTDI (baseline == 0) stays NaN/unclassified, not forced into a
    category."""
    return pd.cut(dtdi, bins=bins, labels=DTDI_CATEGORY_LABELS)


# ---------------------------------------------------------------------------
# Layer 2b: per-(zone, hour) DTDI percentile thresholds (Round 4, 2026-08-17)
# ---------------------------------------------------------------------------

# `compute_dtdi_percentile_thresholds` above pools every reliable zone-hour
# citywide into one P75/P95 cutoff -- diagnosed as the root cause of a
# separate finding: every model's recall for `elevated`/`very_elevated` was
# near 0% under that scheme. Busy zones'
# demand naturally swings very little around their own baseline (large
# counts -> small relative variance), so their genuinely elevated moments
# rarely cross a cutoff calibrated mostly by the far noisier, far more
# numerous small-baseline zone-hours -- and the models' own smoother
# predictions rarely cross that same cutoff even where quiet-zone actual
# demand does. Classifying each (zone, hour) slot's DTDI against its own
# historical distribution, per the DTDI document's Section 6 ("the same
# zone", "the same hour of day" as candidate reference populations),
# targets that mechanism directly.

DTDI_THRESHOLD_GROUP_COLUMNS = ["pu_location_id", "hour_of_day"]

# A (zone, hour) slot needs at least this many reliable training
# observations before its own P75/P95 is trusted over the coarser zone-only
# level. Checked empirically against the real training data before picking
# this bar (2026-08-17): at 50, 76.1% of (zone, hour) slots are
# self-sufficient; the rest fall back to the zone level, which is almost
# always well-populated (median 7,005 training observations per zone).
MIN_GROUP_OBSERVATIONS_FOR_DTDI_THRESHOLD = 50


def compute_dtdi_group_percentile_levels(
    train_df: pd.DataFrame,
    group_cols: list[str] = DTDI_THRESHOLD_GROUP_COLUMNS,
    min_baseline: float = MIN_BASELINE_FOR_DTDI,
    percentiles: tuple[float, float] = (75, 95),
) -> dict:
    """Leakage-safe (training-period-only) DTDI percentile thresholds at
    each grain in a fallback chain -- `zone_hour` (most specific) -> `zone`
    -> `global` (a single scalar pair) -- mirroring the same fallback-chain
    shape `src/processed.py::compute_static_baseline_levels` uses for the
    baseline itself. This function only computes each grain's own raw
    P75/P95 independently; resolving which grain a given row should
    actually use is `classify_dtdi_by_group`'s job, not this one's.

    Restricted to `baseline_pickup_count >= min_baseline`, same as
    `compute_dtdi_percentile_thresholds`, so the small-baseline instability
    doesn't contaminate any grain's thresholds.
    """
    p_low_pct, p_high_pct = percentiles
    reliable = train_df.loc[train_df["baseline_pickup_count"] >= min_baseline].copy()
    reliable["dtdi"] = compute_dtdi(reliable["pickup_count"], reliable["baseline_pickup_count"])
    reliable = reliable.dropna(subset=["dtdi"])

    def _level(cols: list[str]) -> pd.DataFrame:
        grouped = reliable.groupby(cols)["dtdi"]
        out = grouped.quantile([p_low_pct / 100, p_high_pct / 100]).unstack()
        out.columns = ["p_low", "p_high"]
        out["n_obs"] = grouped.size()
        return out.reset_index()

    global_p_low, global_p_high = reliable["dtdi"].quantile([p_low_pct / 100, p_high_pct / 100])

    return {
        "zone_hour": _level(group_cols),
        "zone": _level([group_cols[0]]),
        "global": (float(global_p_low), float(global_p_high)),
    }


def classify_dtdi_by_group(
    df: pd.DataFrame,
    group_levels: dict,
    dtdi_col: str,
    group_cols: list[str] = DTDI_THRESHOLD_GROUP_COLUMNS,
    min_group_obs: int = MIN_GROUP_OBSERVATIONS_FOR_DTDI_THRESHOLD,
) -> pd.Series:
    """Classify `df[dtdi_col]` against the per-`group_cols` thresholds from
    `compute_dtdi_group_percentile_levels`, falling back to the zone level
    then the global level when a (zone, hour) slot has fewer than
    `min_group_obs` reliable training observations. `<1.0` ("below normal")
    stays a fixed anchor at every grain, matching `classify_dtdi`'s
    convention. `df` must contain `group_cols` alongside `dtdi_col`; NaN
    `dtdi_col` values (undefined baseline) stay unclassified.

    Joins are done via an explicit row-id column, not relying on merge/join
    preserving row order -- this codebase has already been burned once by
    an implicit-index-preservation assumption (`.merge()` resetting the
    index in `src/features.py`, found 2026-08-17 building
    `build_evaluation_frame`); this function does not repeat that mistake.
    """
    zone_col = group_cols[0]
    working = df[group_cols + [dtdi_col]].copy()
    working["_row_id"] = np.arange(len(working))

    merged = working.merge(group_levels["zone_hour"], on=group_cols, how="left")
    merged = merged.merge(group_levels["zone"], on=zone_col, how="left", suffixes=("", "_zone"))
    merged = merged.sort_values("_row_id")

    global_p_low, global_p_high = group_levels["global"]
    zone_hour_reliable = merged["n_obs"].fillna(0) >= min_group_obs
    zone_reliable = merged["n_obs_zone"].fillna(0) >= min_group_obs

    p_low = merged["p_low"].where(
        zone_hour_reliable, merged["p_low_zone"].where(zone_reliable, global_p_low)
    )
    p_high = merged["p_high"].where(
        zone_hour_reliable, merged["p_high_zone"].where(zone_reliable, global_p_high)
    )

    dtdi = merged[dtdi_col]
    labels = pd.Series(pd.NA, index=merged.index, dtype="object")
    defined = dtdi.notna()
    labels[defined & (dtdi < 1.0)] = DTDI_CATEGORY_LABELS[0]
    labels[defined & (dtdi >= 1.0) & (dtdi < p_low)] = DTDI_CATEGORY_LABELS[1]
    labels[defined & (dtdi >= p_low) & (dtdi < p_high)] = DTDI_CATEGORY_LABELS[2]
    labels[defined & (dtdi >= p_high)] = DTDI_CATEGORY_LABELS[3]

    # Positionally correct (via the _row_id sort above) but re-attached to
    # the *caller's* original index/row labels, not `merged`'s -- returns
    # a pd.Series like `classify_dtdi` does, not a bare Categorical, so
    # downstream code (`dtdi_confusion_matrix`, `.notna()` masking, etc.)
    # keeps working identically regardless of which classify function
    # produced the categories.
    return pd.Series(
        pd.Categorical(labels.to_numpy(), categories=DTDI_CATEGORY_LABELS), index=df.index
    )


def dtdi_confusion_matrix(
    actual_category: pd.Series, predicted_category: pd.Series
) -> pd.DataFrame:
    """Confusion matrix over the 4 DTDI categories, rows with either side
    NaN (undefined baseline) excluded first."""
    mask = actual_category.notna() & predicted_category.notna()
    labels = DTDI_CATEGORY_LABELS
    cm = confusion_matrix(actual_category[mask], predicted_category[mask], labels=labels)
    return pd.DataFrame(
        cm, index=[f"actual_{lab}" for lab in labels], columns=[f"pred_{lab}" for lab in labels]
    )


def dtdi_classification_report(
    actual_category: pd.Series, predicted_category: pd.Series
) -> pd.DataFrame:
    """Per-class precision/recall/F1, plus macro and micro averages, over
    the 4 DTDI categories. Macro treats every category equally; micro pools
    all predictions together (dominated by whichever category has the most
    rows -- typically 'normal'). Rows with either side NaN excluded first.
    """
    mask = actual_category.notna() & predicted_category.notna()
    y_true, y_pred = actual_category[mask], predicted_category[mask]
    labels = DTDI_CATEGORY_LABELS

    per_class = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    rows = [
        {"category": lab, "precision": p, "recall": r, "f1": f, "support": s}
        for lab, p, r, f, s in zip(labels, *per_class)
    ]
    for avg in ["macro", "micro"]:
        p, r, f, _ = precision_recall_fscore_support(
            y_true, y_pred, labels=labels, average=avg, zero_division=0
        )
        rows.append(
            {"category": f"{avg}_avg", "precision": p, "recall": r, "f1": f, "support": mask.sum()}
        )
    return pd.DataFrame(rows).set_index("category")


def top_k_zone_hour_overlap(
    df: pd.DataFrame, k: int, actual_col: str, predicted_col: str
) -> float:
    """Fraction overlap between the top-k rows by actual value and the
    top-k rows by predicted value -- more operationally relevant than
    per-row accuracy for fleet positioning, where only a limited number of
    zones can realistically be prioritised."""
    top_actual = set(df.nlargest(k, actual_col).index)
    top_predicted = set(df.nlargest(k, predicted_col).index)
    return len(top_actual & top_predicted) / k
