from pathlib import Path
import shutil

from pyspark.sql import functions as F
from pyspark.sql import types as T
import pytest

from src.clean_rules import (
    CLEAN_SCHEMA_COLUMNS,
    ESSENTIAL_FIELDS,
    STUDY_PERIOD_START,
    VALID_LOCATION_ID_RANGE,
)
from src.cleaning import (
    apply_duplicate_removal,
    apply_removal_rule,
    clean_filename,
    clean_taxi_type,
    standardise_to_clean_schema,
    write_clean_dataset_per_month,
)
from src.config import RAW_DATA_DIR

# ---------------------------------------------------------------------------
# Unit tests: schema standardisation
# ---------------------------------------------------------------------------


def _minimal_yellow_raw_df(spark_session):
    # pickup/dropoff built as strings and cast to timestamp by the caller --
    # createDataFrame's type inference doesn't like mixed None/timestamp tuples.
    schema = T.StructType(
        [
            T.StructField("vendor_id", T.IntegerType(), True),
            T.StructField("tpep_pickup_datetime", T.StringType(), True),
            T.StructField("tpep_dropoff_datetime", T.StringType(), True),
            T.StructField("passenger_count", T.IntegerType(), True),
            T.StructField("trip_distance", T.DoubleType(), True),
            T.StructField("ratecode_id", T.IntegerType(), True),
            T.StructField("store_and_fwd_flag", T.StringType(), True),
            T.StructField("pu_location_id", T.IntegerType(), True),
            T.StructField("do_location_id", T.IntegerType(), True),
            T.StructField("payment_type", T.IntegerType(), True),
            T.StructField("fare_amount", T.DoubleType(), True),
            T.StructField("extra", T.DoubleType(), True),
            T.StructField("mta_tax", T.DoubleType(), True),
            T.StructField("tip_amount", T.DoubleType(), True),
            T.StructField("tolls_amount", T.DoubleType(), True),
            T.StructField("improvement_surcharge", T.DoubleType(), True),
            T.StructField("total_amount", T.DoubleType(), True),
            T.StructField("congestion_surcharge", T.DoubleType(), True),
            T.StructField("airport_fee", T.DoubleType(), True),
            T.StructField("cbd_congestion_fee", T.DoubleType(), True),
            T.StructField("source_dataset", T.StringType(), True),
            T.StructField("source_file", T.StringType(), True),
            T.StructField("source_year", T.IntegerType(), True),
            T.StructField("source_month", T.IntegerType(), True),
            T.StructField("ingestion_timestamp", T.StringType(), True),
        ]
    )
    rows = [
        (
            1,
            "2024-01-01 10:00:00",
            "2024-01-01 10:20:00",
            1,
            3.5,
            1,
            "N",
            100,
            200,
            1,
            15.0,
            1.0,
            0.5,
            2.0,
            0.0,
            1.0,
            19.5,
            2.5,
            0.0,
            None,
            "yellow_taxi",
            "Yellow Taxi Data Jan 2024.parquet",
            2024,
            1,
            "2026-01-01T00:00:00+00:00",
        )
    ]
    return spark_session.createDataFrame(rows, schema=schema)


def test_standardise_to_clean_schema_yellow(spark_session):
    df = _minimal_yellow_raw_df(spark_session)
    df = df.withColumn("tpep_pickup_datetime", F.col("tpep_pickup_datetime").cast("timestamp"))
    df = df.withColumn("tpep_dropoff_datetime", F.col("tpep_dropoff_datetime").cast("timestamp"))

    out = standardise_to_clean_schema(df, "yellow")

    assert set(CLEAN_SCHEMA_COLUMNS).issubset(set(out.columns))
    assert "pickup_datetime" in out.columns
    assert "dropoff_datetime" in out.columns
    assert "tpep_pickup_datetime" not in out.columns
    row = out.first()
    assert row["taxi_type"] == "yellow"
    # Yellow-only Clean row has NULL for Green-only concepts (structural, not missing).
    assert row["ehail_fee"] is None
    assert row["trip_type"] is None
    # trip_duration_minutes computed correctly (20 minutes in the fixture).
    assert row["trip_duration_minutes"] == pytest.approx(20.0)


def test_pickup_dropoff_are_timestamps(spark_session):
    df = _minimal_yellow_raw_df(spark_session)
    df = df.withColumn("tpep_pickup_datetime", F.col("tpep_pickup_datetime").cast("timestamp"))
    df = df.withColumn("tpep_dropoff_datetime", F.col("tpep_dropoff_datetime").cast("timestamp"))
    out = standardise_to_clean_schema(df, "yellow")
    dtypes = dict(out.dtypes)
    assert dtypes["pickup_datetime"] == "timestamp"
    assert dtypes["dropoff_datetime"] == "timestamp"


# ---------------------------------------------------------------------------
# Unit tests: removal rules
# ---------------------------------------------------------------------------


def test_apply_removal_rule_removes_only_matching_rows(spark_session):
    df = spark_session.createDataFrame([(1, -1), (2, 5), (3, None)], ["id", "value"])
    ledger = []
    kept = apply_removal_rule(
        df,
        F.col("value") < 0,
        rule="value < 0",
        reason="test",
        dataset="test",
        original_count=3,
        ledger=ledger,
    )
    kept_ids = {row["id"] for row in kept.collect()}
    assert kept_ids == {
        2,
        3,
    }  # id=1 removed (value<0); id=3 kept (NULL is not evidence of invalidity)
    assert ledger[0]["records_removed"] == 1
    assert ledger[0]["pct_of_original"] == pytest.approx(100 / 3, rel=1e-3)


def test_apply_removal_rule_null_condition_is_not_removed(spark_session):
    """A NULL bad_condition (e.g. from comparing a NULL column) must not remove the row."""
    schema = T.StructType(
        [
            T.StructField("id", T.IntegerType(), True),
            T.StructField("dropoff_before_pickup", T.BooleanType(), True),
        ]
    )
    df = spark_session.createDataFrame([(1, None)], schema=schema)
    ledger = []
    kept = apply_removal_rule(
        df,
        F.col("dropoff_before_pickup"),  # NULL boolean column
        rule="dummy",
        reason="test",
        dataset="test",
        original_count=1,
        ledger=ledger,
    )
    assert kept.count() == 1
    assert ledger[0]["records_removed"] == 0


def test_apply_duplicate_removal(spark_session):
    df = spark_session.createDataFrame([(1, "a"), (1, "a"), (2, "b")], ["id", "value"])
    ledger = []
    deduped, profile = apply_duplicate_removal(
        df,
        subset_cols=["id", "value"],
        dataset="test",
        original_count=3,
        before_count=3,
        ledger=ledger,
    )
    assert deduped.count() == 2
    assert profile["duplicate_extra_rows"] == 1
    assert ledger[0]["records_removed"] == 1


# ---------------------------------------------------------------------------
# Integration test: full clean_taxi_type on a real (small) Raw file
# ---------------------------------------------------------------------------


def _small_raw_fixture(taxi_type: str, filename: str, tmp_path: Path) -> Path:
    src = RAW_DATA_DIR / taxi_type / filename
    if not src.exists():
        pytest.skip(f"{src} not present in this environment")
    raw_dir = tmp_path / "raw"
    (raw_dir / taxi_type).mkdir(parents=True)
    shutil.copy(src, raw_dir / taxi_type / filename)
    return raw_dir


@pytest.mark.parametrize(
    "taxi_type,filename",
    [("green", "green-02-26.parquet"), ("yellow", "yellow-02-24.parquet")],
)
def test_clean_taxi_type_end_to_end(spark_session, tmp_path, taxi_type, filename):
    raw_dir = _small_raw_fixture(taxi_type, filename, tmp_path)
    result = clean_taxi_type(spark_session, taxi_type, raw_dir=raw_dir)

    clean_df = result["clean_df"]

    # Common schema present.
    assert set(CLEAN_SCHEMA_COLUMNS).issubset(set(clean_df.columns))

    # taxi_type populated correctly.
    taxi_types = {row["taxi_type"] for row in clean_df.select("taxi_type").distinct().collect()}
    assert taxi_types == {taxi_type}

    # Datetimes are proper timestamps.
    dtypes = dict(clean_df.dtypes)
    assert dtypes["pickup_datetime"] == "timestamp"
    assert dtypes["dropoff_datetime"] == "timestamp"

    # Essential fields are never NULL in Clean output (removal rule enforced).
    for field in ESSENTIAL_FIELDS:
        assert clean_df.where(F.col(field).isNull()).count() == 0

    # pu_location_id is always within the documented valid range.
    lo, hi = VALID_LOCATION_ID_RANGE
    assert (
        clean_df.where((F.col("pu_location_id") < lo) | (F.col("pu_location_id") > hi)).count()
        == 0
    )

    # No logically-impossible or explicitly-dropped-by-policy trips survive.
    assert (
        clean_df.where(
            F.col("dropoff_datetime").isNotNull()
            & F.col("pickup_datetime").isNotNull()
            & (F.col("dropoff_datetime") < F.col("pickup_datetime"))
        ).count()
        == 0
    )
    assert clean_df.where(F.col("trip_distance") <= 0).count() == 0  # user policy: drop zero too
    assert clean_df.where(F.col("passenger_count") < 0).count() == 0
    assert (
        clean_df.where(F.col("fare_amount") < 0).count() == 0
    )  # user policy: drop negative fares
    assert (
        clean_df.where(
            (F.col("trip_duration_minutes") <= 0) | (F.col("trip_duration_minutes") > 1440)
        ).count()
        == 0
    )  # user policy: drop 0 or >24h trips
    assert (
        clean_df.where((F.col("do_location_id") < lo) | (F.col("do_location_id") > hi)).count()
        == 0
    )  # user policy: drop out-of-range do_location_id too

    # Study period enforced.
    assert clean_df.where(F.col("pickup_datetime") < F.lit(STUDY_PERIOD_START)).count() == 0

    # Cleaning must not unexpectedly remove a large proportion of a normal month's data
    # (the new user-specified rules remove more than before, but should still be a small minority).
    retained_pct = 100 * result["clean_row_count"] / result["raw_row_count"]
    assert retained_pct > 90.0


def test_clean_filename_matches_taxi_type_cleaned_mm_yy_convention():
    assert clean_filename("yellow", 2024, 1) == "yellow-cleaned-01-24.parquet"
    assert clean_filename("green", 2026, 2) == "green-cleaned-02-26.parquet"


def test_yellow_and_green_union_produces_single_logical_dataset(spark_session, tmp_path):
    green_raw_dir = _small_raw_fixture("green", "green-02-26.parquet", tmp_path / "g")
    yellow_raw_dir = _small_raw_fixture("yellow", "yellow-02-24.parquet", tmp_path / "y")

    green_result = clean_taxi_type(spark_session, "green", raw_dir=green_raw_dir)
    yellow_result = clean_taxi_type(spark_session, "yellow", raw_dir=yellow_raw_dir)

    combined_taxi_types = {
        row["taxi_type"]
        for df in (yellow_result["clean_df"], green_result["clean_df"])
        for row in df.select("taxi_type").distinct().collect()
    }
    assert combined_taxi_types == {"yellow", "green"}

    # Write each taxi type via the real writer -- one flat, named file per
    # month -- into the same flat directory, mirroring how `main()` builds
    # the unified Clean dataset.
    out_dir = tmp_path / "clean_taxi"
    write_clean_dataset_per_month(yellow_result["clean_df"], "yellow", out_dir)
    write_clean_dataset_per_month(green_result["clean_df"], "green", out_dir)

    written_files = sorted(p.name for p in out_dir.glob("*.parquet"))
    assert written_files == ["green-cleaned-02-26.parquet", "yellow-cleaned-02-24.parquet"]

    # Readable as a single logical dataset (mirrors `spark.read.parquet("data/clean/taxi/")`).
    reread = spark_session.read.parquet(str(out_dir))
    assert reread.count() == yellow_result["clean_row_count"] + green_result["clean_row_count"]
    assert {row["taxi_type"] for row in reread.select("taxi_type").distinct().collect()} == {
        "yellow",
        "green",
    }
