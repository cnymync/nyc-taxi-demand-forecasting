import pandas as pd
from pyspark.sql import functions as F
from pyspark.sql import types as T
import pytest

from src.processed import (
    PROCESSED_TAXI_COLUMNS,
    PROVENANCE_COLUMNS,
    add_historical_baseline,
    add_static_historical_baseline,
    add_zero_demand_rows,
    aggregate_to_zone_hour,
    build_zone_hour_grid,
    expand_events_to_zone_hour,
    join_events,
    join_holidays,
    join_weather,
    load_geographic_zone_ids,
    load_zone_boroughs,
    narrow_to_processed_columns,
    standardise_holiday_schedule,
)


def _minimal_clean_df(spark_session):
    schema = T.StructType(
        [
            T.StructField("taxi_type", T.StringType(), True),
            T.StructField("vendor_id", T.IntegerType(), True),
            T.StructField("pickup_datetime", T.StringType(), True),
            T.StructField("dropoff_datetime", T.StringType(), True),
            T.StructField("trip_duration_minutes", T.DoubleType(), True),
            T.StructField("passenger_count", T.IntegerType(), True),
            T.StructField("trip_distance", T.DoubleType(), True),
            T.StructField("ratecode_id", T.IntegerType(), True),
            T.StructField("pu_location_id", T.IntegerType(), True),
            T.StructField("do_location_id", T.IntegerType(), True),
            T.StructField("fare_amount", T.DoubleType(), True),
            T.StructField("is_extreme_trip_distance", T.BooleanType(), True),
            T.StructField("is_extreme_trip_duration", T.BooleanType(), True),
            T.StructField("source_dataset", T.StringType(), True),
            T.StructField("source_file", T.StringType(), True),
            T.StructField("source_year", T.IntegerType(), True),
            T.StructField("source_month", T.IntegerType(), True),
            T.StructField("ingestion_timestamp", T.StringType(), True),
        ]
    )
    rows = [
        (
            "yellow",
            1,
            "2024-01-01 10:05:00",
            "2024-01-01 10:20:00",
            15.0,
            1,
            3.5,
            1,
            100,
            200,
            15.0,
            False,
            False,
            "yellow_taxi",
            "f1",
            2024,
            1,
            "2024-01-01",
        ),
        (
            "yellow",
            1,
            "2024-01-01 10:45:00",
            "2024-01-01 11:10:00",
            25.0,
            1,
            40.0,
            1,
            100,
            200,
            15.0,
            True,
            False,
            "yellow_taxi",
            "f1",
            2024,
            1,
            "2024-01-01",
        ),
        (
            "yellow",
            1,
            "2024-01-01 11:05:00",
            "2024-01-01 11:20:00",
            15.0,
            2,
            2.0,
            1,
            100,
            200,
            15.0,
            False,
            False,
            "yellow_taxi",
            "f1",
            2024,
            1,
            "2024-01-01",
        ),
    ]
    df = spark_session.createDataFrame(rows, schema)
    return df.withColumn("pickup_datetime", F.to_timestamp("pickup_datetime")).withColumn(
        "dropoff_datetime", F.to_timestamp("dropoff_datetime")
    )


def test_narrow_to_processed_columns_selects_expected_set(spark_session):
    df = _minimal_clean_df(spark_session)
    narrowed = narrow_to_processed_columns(df)
    assert narrowed.columns == PROCESSED_TAXI_COLUMNS + PROVENANCE_COLUMNS
    assert "fare_amount" not in narrowed.columns
    assert "vendor_id" not in narrowed.columns


def test_aggregate_to_zone_hour_groups_by_hour_and_counts_correctly(spark_session):
    df = narrow_to_processed_columns(_minimal_clean_df(spark_session))
    agg = aggregate_to_zone_hour(df)
    rows = {(r["pu_location_id"], str(r["pickup_hour"])): r for r in agg.collect()}
    assert len(rows) == 2  # two distinct hours: 10:00 and 11:00

    hour_10 = rows[(100, "2024-01-01 10:00:00")]
    assert hour_10["pickup_count"] == 2
    assert hour_10["extreme_trip_distance_count"] == 1
    assert hour_10["extreme_trip_duration_count"] == 0

    hour_11 = rows[(100, "2024-01-01 11:00:00")]
    assert hour_11["pickup_count"] == 1


def test_load_geographic_zone_ids_excludes_placeholders():
    zone_ids = load_geographic_zone_ids()
    assert len(zone_ids) == 263
    assert 264 not in zone_ids
    assert 265 not in zone_ids
    assert 1 in zone_ids


def test_add_zero_demand_rows_fills_missing_combinations_as_zero(spark_session):
    df = narrow_to_processed_columns(_minimal_clean_df(spark_session))
    agg = aggregate_to_zone_hour(df)
    grid = build_zone_hour_grid(
        spark_session,
        "yellow",
        zone_ids=[100, 101],
        start="2024-01-01 10:00:00",
        end_exclusive="2024-01-01 12:00:00",
    )
    result = add_zero_demand_rows(grid, agg)

    # 2 zones x 2 hours = 4 rows total
    assert result.count() == 4

    zero_rows = result.where(F.col("pu_location_id") == 101).collect()
    assert len(zero_rows) == 2
    for row in zero_rows:
        assert row["pickup_count"] == 0
        assert row["is_zero_demand_hour"] is True
        assert row["avg_trip_distance"] is None  # no average of zero trips

    non_zero_row = result.where(
        (F.col("pu_location_id") == 100) & (F.col("pickup_hour") == "2024-01-01 10:00:00")
    ).collect()[0]
    assert non_zero_row["pickup_count"] == 2
    assert non_zero_row["is_zero_demand_hour"] is False


# ---------------------------------------------------------------------------
# Step 3: historical baseline
#
# Round 3 (2026-08-17): the production baseline (`add_historical_baseline`)
# is now a rolling 8-week window, not the original static 2024-2025 average
# -- see src/processed.py's module docstring for why. The original static
# implementation is preserved as `add_static_historical_baseline`, and its
# original test coverage below is kept, renamed to match, as a regression
# suite for that preserved function. New tests further down cover the
# rolling behaviour specifically (causal correctness, trend-tracking,
# fallback for insufficient history).
# ---------------------------------------------------------------------------

# Zone 100 ("Garment District") and zone 4 ("Alphabet City") are both
# Manhattan in the real TLC lookup; zone 200 ("Riverdale...") is the Bronx --
# used to exercise the borough/global fallback levels against real data.
# 2024-01-01 is a Monday (day_of_week=2 in Spark's dayofweek); 2026-01-05 is
# also a Monday, used to check the 2026 row never leaks into the baseline.


def _minimal_zone_hour_df(spark_session):
    schema = T.StructType(
        [
            T.StructField("taxi_type", T.StringType(), True),
            T.StructField("pu_location_id", T.IntegerType(), True),
            T.StructField("pickup_hour", T.StringType(), True),
            T.StructField("pickup_count", T.IntegerType(), True),
        ]
    )
    rows = []
    # Zone 100, Monday 08:00 -- 10 reliable non-zero training weeks (>= MIN_BASELINE_OBSERVATIONS).
    mondays_2024 = [
        "2024-01-01",
        "2024-01-08",
        "2024-01-15",
        "2024-01-22",
        "2024-01-29",
        "2024-02-05",
        "2024-02-12",
        "2024-02-19",
        "2024-02-26",
        "2024-03-04",
    ]
    for d in mondays_2024:
        rows.append(("yellow", 100, f"{d} 08:00:00", 10))
    # Zone 100, Tuesday 08:00 -- only 2 non-zero training weeks (sparse; below threshold).
    for d in ["2024-01-02", "2024-01-09"]:
        rows.append(("yellow", 100, f"{d} 08:00:00", 6))
    # Zone 4 (also Manhattan), no training data at all -- forces the borough fallback.
    # Zone 200 (Bronx), no training data at all, and no other Bronx zone supplied -- forces global fallback.
    # Both get a single 2026 test-period row so they appear in the full df.
    rows.append(("yellow", 4, "2026-01-05 08:00:00", 3))
    rows.append(("yellow", 200, "2026-01-06 08:00:00", 1))
    # Zone 100, Monday 08:00, 2026 -- the leakage check: an extreme value here
    # must NOT move the baseline computed from the 2024 Mondays above.
    rows.append(("yellow", 100, "2026-01-05 08:00:00", 999))

    df = spark_session.createDataFrame(rows, schema)
    return df.withColumn("pickup_hour", F.to_timestamp("pickup_hour"))


def test_add_static_historical_baseline_uses_specific_grain_when_reliable(spark_session):
    df = _minimal_zone_hour_df(spark_session)
    zone_boroughs = load_zone_boroughs(spark_session)
    result = add_static_historical_baseline(df, zone_boroughs)

    row = result.where(
        (F.col("pu_location_id") == 100) & (F.col("pickup_hour") == "2024-01-01 08:00:00")
    ).collect()[0]
    assert row["baseline_source"] == "zone_dow_hour"
    assert row["baseline_pickup_count"] == pytest.approx(10.0)


def test_add_static_historical_baseline_falls_back_when_specific_grain_is_sparse(spark_session):
    df = _minimal_zone_hour_df(spark_session)
    zone_boroughs = load_zone_boroughs(spark_session)
    result = add_static_historical_baseline(df, zone_boroughs)

    # Zone 100 Tuesday 08:00 has only 2 non-zero training weeks (< MIN_BASELINE_OBSERVATIONS),
    # so it must fall back to the zone+hour grain (which blends in the reliable Monday data).
    row = result.where(
        (F.col("pu_location_id") == 100) & (F.col("pickup_hour") == "2024-01-02 08:00:00")
    ).collect()[0]
    assert row["baseline_source"] == "zone_hour"


def test_add_static_historical_baseline_falls_back_to_borough_then_global(spark_session):
    df = _minimal_zone_hour_df(spark_session)
    zone_boroughs = load_zone_boroughs(spark_session)
    result = add_static_historical_baseline(df, zone_boroughs)

    # Zone 4 has zero training rows of its own -- must fall back to Manhattan (borough) level.
    zone_4_row = result.where(F.col("pu_location_id") == 4).collect()[0]
    assert zone_4_row["baseline_source"] == "borough_dow_hour"

    # Zone 200 (Bronx) has zero training rows and no other Bronx zone supplied -- global fallback.
    zone_200_row = result.where(F.col("pu_location_id") == 200).collect()[0]
    assert zone_200_row["baseline_source"] == "global"


def test_add_static_historical_baseline_never_leaks_2026_data(spark_session):
    df = _minimal_zone_hour_df(spark_session)
    zone_boroughs = load_zone_boroughs(spark_session)
    result = add_static_historical_baseline(df, zone_boroughs)

    # The 2026 pickup_count=999 row must receive a baseline computed purely
    # from the 2024 training Mondays (~10), not be pulled toward 999.
    row_2026 = result.where(
        (F.col("pu_location_id") == 100) & (F.col("pickup_hour") == "2026-01-05 08:00:00")
    ).collect()[0]
    assert row_2026["baseline_pickup_count"] == pytest.approx(10.0)
    assert (
        row_2026["pickup_count"] == 999
    )  # the actual value is untouched, only the baseline is safe


# ---------------------------------------------------------------------------
# Step 3b: rolling baseline (Round 3, 2026-08-17) -- add_historical_baseline
# ---------------------------------------------------------------------------

# Zone 100, Monday 08:00, yellow. A single (taxi_type, zone, dow, hour) slot
# with a deliberate step-change partway through: pickup_count=5 for the
# first 9 Mondays (2024-11-03 .. 2025-12-29, spanning the last stretch of
# the training period), then pickup_count=30 for 9 more Mondays crossing
# into the test period (2026-01-05 .. 2026-03-02). Used to prove the rolling
# baseline (a) never uses a row's own or a future week, and (b) actually
# tracks the step-change using real prior actuals -- including test-period
# actuals for later test-period rows -- rather than staying frozen at the
# stale pre-change static average, which is the whole point of Round 3.


def _rolling_baseline_step_change_df(spark_session):
    schema = T.StructType(
        [
            T.StructField("taxi_type", T.StringType(), True),
            T.StructField("pu_location_id", T.IntegerType(), True),
            T.StructField("pickup_hour", T.StringType(), True),
            T.StructField("pickup_count", T.IntegerType(), True),
        ]
    )
    pre_change_mondays = [
        "2025-11-03",
        "2025-11-10",
        "2025-11-17",
        "2025-11-24",
        "2025-12-01",
        "2025-12-08",
        "2025-12-15",
        "2025-12-22",
        "2025-12-29",
    ]
    post_change_mondays = [
        "2026-01-05",
        "2026-01-12",
        "2026-01-19",
        "2026-01-26",
        "2026-02-02",
        "2026-02-09",
        "2026-02-16",
        "2026-02-23",
        "2026-03-02",
    ]
    rows = [("yellow", 100, f"{d} 08:00:00", 5) for d in pre_change_mondays]
    rows += [("yellow", 100, f"{d} 08:00:00", 30) for d in post_change_mondays]
    df = spark_session.createDataFrame(rows, schema)
    return df.withColumn("pickup_hour", F.to_timestamp("pickup_hour"))


def test_add_historical_baseline_is_causal_never_uses_its_own_or_future_week(spark_session):
    df = _rolling_baseline_step_change_df(spark_session)
    zone_boroughs = load_zone_boroughs(spark_session)
    result = add_historical_baseline(df, zone_boroughs)

    # The FIRST post-change row (2026-01-05, actual=30) has exactly 8 prior
    # weeks available (the 8 pre-change Mondays, all =5) -- its baseline must
    # reflect only those, not its own new value or any later 30s.
    row = result.where(F.col("pickup_hour") == "2026-01-05 08:00:00").collect()[0]
    assert row["baseline_source"] == "zone_dow_hour"
    assert row["baseline_pickup_count"] == pytest.approx(5.0)
    assert row["pickup_count"] == 30  # actual is untouched; only the baseline stays causal


def test_add_historical_baseline_tracks_a_trend_once_the_window_catches_up(spark_session):
    df = _rolling_baseline_step_change_df(spark_session)
    zone_boroughs = load_zone_boroughs(spark_session)
    result = add_historical_baseline(df, zone_boroughs)

    # By the LAST row (2026-03-02), the trailing 8 weeks are entirely within
    # the post-change regime (2026-01-05 .. 2026-02-23, all =30) -- the
    # rolling baseline should track the new level exactly, unlike a static
    # 2024-2025 average which would still be dragged toward the old ~5.
    row = result.where(F.col("pickup_hour") == "2026-03-02 08:00:00").collect()[0]
    assert row["baseline_source"] == "zone_dow_hour"
    assert row["baseline_pickup_count"] == pytest.approx(30.0)


def test_add_historical_baseline_falls_back_when_window_is_incomplete(spark_session):
    df = _rolling_baseline_step_change_df(spark_session)
    zone_boroughs = load_zone_boroughs(spark_session)
    result = add_historical_baseline(df, zone_boroughs)

    # The 3rd row overall (2025-11-17) has only 2 prior weeks -- fewer than
    # ROLLING_BASELINE_WINDOW_WEEKS -- so zone_dow_hour isn't "reliable" yet.
    # With only this one slot's data in the fixture, zone_hour shares the
    # same 2 prior rows and so still resolves (isNotNull), giving that grain.
    row = result.where(F.col("pickup_hour") == "2025-11-17 08:00:00").collect()[0]
    assert row["baseline_source"] == "zone_hour"
    assert row["baseline_pickup_count"] == pytest.approx(5.0)


def test_add_historical_baseline_first_row_falls_back_to_static_coarse_level(spark_session):
    df = _rolling_baseline_step_change_df(spark_session)
    zone_boroughs = load_zone_boroughs(spark_session)
    result = add_historical_baseline(df, zone_boroughs)

    # The very first row in the whole partition has zero prior history at
    # every rolling grain -- must fall all the way back to the static
    # (train-only) borough_dow_hour level, the one genuinely-unavoidable
    # edge case documented in add_historical_baseline's docstring.
    row = result.where(F.col("pickup_hour") == "2025-11-03 08:00:00").collect()[0]
    assert row["baseline_source"] == "borough_dow_hour"


def test_add_historical_baseline_matches_static_baseline_column_schema(spark_session):
    # Both baseline implementations must be drop-in compatible for
    # src/features.py and everything downstream -- same output schema.
    df = _minimal_zone_hour_df(spark_session)
    zone_boroughs = load_zone_boroughs(spark_session)
    rolling_result = add_historical_baseline(df, zone_boroughs)
    static_result = add_static_historical_baseline(df, zone_boroughs)
    assert set(rolling_result.columns) == set(static_result.columns)


# ---------------------------------------------------------------------------
# Step 4: external-source join (weather, events, holidays)
# ---------------------------------------------------------------------------


def _minimal_zone_hour_grid(spark_session):
    schema = T.StructType(
        [
            T.StructField("pu_location_id", T.IntegerType(), True),
            T.StructField("pickup_hour", T.StringType(), True),
        ]
    )
    rows = [
        (100, "2024-01-01 00:00:00"),
        (100, "2024-01-01 01:00:00"),
        (100, "2024-01-02 00:00:00"),  # different day -- not a holiday in this test
    ]
    df = spark_session.createDataFrame(rows, schema)
    return df.withColumn("pickup_hour", F.to_timestamp("pickup_hour"))


def test_join_weather_matches_on_hour_and_leaves_unmatched_hours_null(spark_session):
    grid = _minimal_zone_hour_grid(spark_session)
    weather_schema = T.StructType(
        [
            T.StructField("pickup_hour", T.StringType(), True),
            T.StructField("tmpf", T.DoubleType(), True),
        ]
    )
    weather_df = spark_session.createDataFrame(
        [("2024-01-01 00:00:00", 40.0)], weather_schema
    ).withColumn("pickup_hour", F.to_timestamp("pickup_hour"))

    result = join_weather(grid, weather_df)
    rows = {str(r["pickup_hour"]): r["tmpf"] for r in result.collect()}
    assert rows["2024-01-01 00:00:00"] == 40.0
    assert (
        rows["2024-01-01 01:00:00"] is None
    )  # no matching weather hour -- real gap, not fabricated


def test_expand_events_to_zone_hour_covers_full_span_and_counts_overlaps(spark_session):
    schema = T.StructType(
        [
            T.StructField("pu_location_id", T.IntegerType(), True),
            T.StructField("start_date_time", T.StringType(), True),
            T.StructField("end_date_time", T.StringType(), True),
        ]
    )
    rows = [
        (100, "2024-01-01 00:00:00", "2024-01-01 02:00:00"),  # covers hours 00, 01, 02
        (100, "2024-01-01 01:00:00", "2024-01-01 01:00:00"),  # overlaps hour 01 only
    ]
    df = spark_session.createDataFrame(rows, schema)
    df = df.withColumn("start_date_time", F.to_timestamp("start_date_time")).withColumn(
        "end_date_time", F.to_timestamp("end_date_time")
    )
    expanded = expand_events_to_zone_hour(df)
    counts = {str(r["pickup_hour"]): r["event_count"] for r in expanded.collect()}
    assert counts["2024-01-01 00:00:00"] == 1
    assert counts["2024-01-01 01:00:00"] == 2  # both events cover this hour
    assert counts["2024-01-01 02:00:00"] == 1


def test_join_events_fills_no_event_as_zero_not_missing(spark_session):
    grid = _minimal_zone_hour_grid(spark_session)
    events_schema = T.StructType(
        [
            T.StructField("pu_location_id", T.IntegerType(), True),
            T.StructField("pickup_hour", T.StringType(), True),
            T.StructField("event_count", T.LongType(), True),
        ]
    )
    events_zone_hour_df = spark_session.createDataFrame(
        [(100, "2024-01-01 00:00:00", 2)], events_schema
    ).withColumn("pickup_hour", F.to_timestamp("pickup_hour"))

    result = join_events(grid, events_zone_hour_df)
    rows = {str(r["pickup_hour"]): r for r in result.collect()}
    assert rows["2024-01-01 00:00:00"]["event_count"] == 2
    assert rows["2024-01-01 00:00:00"]["event_present"] is True
    assert rows["2024-01-01 01:00:00"]["event_count"] == 0
    assert rows["2024-01-01 01:00:00"]["event_present"] is False


def test_standardise_holiday_schedule_reformats_date_to_iso_style(tmp_path):
    raw_path = tmp_path / "holiday_schedule.csv"
    clean_path = tmp_path / "holiday_schedule_clean.csv"
    pd.DataFrame(
        {"Date": ["1/1/24", "12/25/25"], "Holiday": ["New Year's Day", "Christmas Day"]}
    ).to_csv(raw_path, index=False)

    result = standardise_holiday_schedule(raw_path, clean_path)

    assert list(result["date"]) == ["2024-01-01 00:00:00", "2025-12-25 00:00:00"]
    assert list(result["holiday_name"]) == ["New Year's Day", "Christmas Day"]
    assert clean_path.exists()
    on_disk = pd.read_csv(clean_path)
    assert list(on_disk["date"]) == ["2024-01-01 00:00:00", "2025-12-25 00:00:00"]


def test_join_holidays_flags_every_hour_of_a_holiday_date(spark_session):
    grid = _minimal_zone_hour_grid(spark_session)
    holidays_schema = T.StructType(
        [
            T.StructField("date", T.StringType(), True),
            T.StructField("holiday_name", T.StringType(), True),
        ]
    )
    holidays_df = spark_session.createDataFrame(
        [("2024-01-01 00:00:00", "New Year's Day")], holidays_schema
    ).withColumn("date", F.to_timestamp("date"))

    result = join_holidays(grid, holidays_df)
    rows = {str(r["pickup_hour"]): r for r in result.collect()}
    assert rows["2024-01-01 00:00:00"]["is_holiday"] is True
    assert rows["2024-01-01 00:00:00"]["holiday_name"] == "New Year's Day"
    assert rows["2024-01-01 01:00:00"]["is_holiday"] is True  # same date, different hour
    assert rows["2024-01-02 00:00:00"]["is_holiday"] is False
