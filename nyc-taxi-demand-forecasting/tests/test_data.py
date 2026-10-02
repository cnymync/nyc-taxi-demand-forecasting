from pathlib import Path

from pyspark.sql import functions as F
import pytest

from src.config import LANDING_DATA_DIR
from src.dataset import ingest_file, parse_filename, raw_filename, standardise_schema
from src.schemas import RAW_SCHEMAS

# --- filename parsing ----------------------------------------------------


@pytest.mark.parametrize(
    "taxi_type,filename,expected",
    [
        ("yellow", "Yellow Taxi Data Jan 2024.parquet", (2024, 1)),
        ("yellow", "Yellow Taxi Data June 2025.parquet", (2025, 6)),
        ("yellow", "Yellow Taxi Data Sept 2025.parquet", (2025, 9)),
        ("green", "Green Trip Data Dec 2025.parquet", (2025, 12)),
        ("green", "Green Trip Data July 2024.parquet", (2024, 7)),
    ],
)
def test_parse_filename(taxi_type, filename, expected):
    assert parse_filename(taxi_type, filename) == expected


def test_parse_filename_rejects_unexpected_pattern():
    with pytest.raises(ValueError):
        parse_filename("yellow", "not_a_taxi_file.parquet")


def test_month_name_lookup_covers_all_landing_filenames():
    """Every month token actually used in data/landing filenames must resolve."""
    for taxi_type, subdir in [("yellow", "yellow"), ("green", "green")]:
        landing_dir = LANDING_DATA_DIR / subdir
        if not landing_dir.exists():
            pytest.skip(f"{landing_dir} not present in this environment")
        for path in landing_dir.glob("*.parquet"):
            parse_filename(taxi_type, path.name)  # raises ValueError if unrecognised


def test_raw_filename_matches_taxi_type_mm_yy_convention():
    assert raw_filename("yellow", 2024, 1) == "yellow-01-24.parquet"
    assert raw_filename("green", 2025, 12) == "green-12-25.parquet"
    assert raw_filename("yellow", 2026, 5) == "yellow-05-26.parquet"


# --- schema standardisation -----------------------------------------------


def test_standardise_schema_adds_missing_column_as_null(spark_session):
    # Simulates a pre-2025 Yellow file: cbd_congestion_fee doesn't exist yet.
    pre_2025_df = spark_session.createDataFrame(
        [(1, 2, "N", 100, 200, 1)],
        [
            "VendorID",
            "RatecodeID",
            "store_and_fwd_flag",
            "PULocationID",
            "DOLocationID",
            "payment_type",
        ],
    )
    out = standardise_schema(pre_2025_df, "yellow")
    assert {f.name for f in RAW_SCHEMAS["yellow"].fields} == set(out.columns)
    assert out.select("cbd_congestion_fee").first()[0] is None
    assert out.count() == pre_2025_df.count()


def test_standardise_schema_renames_columns(spark_session):
    df = spark_session.createDataFrame(
        [(1, 2, 100, 200)], ["VendorID", "RatecodeID", "PULocationID", "DOLocationID"]
    )
    out = standardise_schema(df, "yellow")
    assert "vendor_id" in out.columns
    assert "VendorID" not in out.columns


def test_standardise_schema_preserves_nulls(spark_session):
    from pyspark.sql.types import IntegerType, StructField, StructType

    schema = StructType(
        [
            StructField("VendorID", IntegerType(), True),
            StructField("RatecodeID", IntegerType(), True),
        ]
    )
    df = spark_session.createDataFrame([(None, None)], schema)
    out = standardise_schema(df, "yellow")
    row = out.select("vendor_id", "ratecode_id").first()
    assert row["vendor_id"] is None
    assert row["ratecode_id"] is None


# --- end-to-end ingestion on a real (small) landing file ------------------

GREEN_SAMPLE_FILE = "Green Trip Data Feb 2026.parquet"


def _green_sample_path() -> Path:
    return LANDING_DATA_DIR / "green" / GREEN_SAMPLE_FILE


requires_sample_file = pytest.mark.skipif(
    not _green_sample_path().exists(), reason="sample landing file not present in this environment"
)


@requires_sample_file
def test_ingest_file_preserves_row_count_and_does_not_touch_landing(spark_session, tmp_path):
    landing_path = _green_sample_path()
    original_bytes = landing_path.read_bytes()
    original_mtime = landing_path.stat().st_mtime

    raw_dir = tmp_path / "raw"
    stats = ingest_file(spark_session, "green", landing_path, raw_dir)

    assert stats["status"] == "ok"
    assert stats["landing_row_count"] == stats["raw_row_count"]
    assert stats["raw_column_count"] == len(RAW_SCHEMAS["green"].fields)

    # Landing file must be untouched by ingestion.
    assert landing_path.read_bytes() == original_bytes
    assert landing_path.stat().st_mtime == original_mtime

    # Raw output actually exists as a single flat .parquet file mirroring
    # the Landing layout, and is readable with matching row count.
    output_path = raw_dir / "green" / raw_filename("green", stats["year"], stats["month"])
    assert output_path.is_file()
    assert output_path.suffix == ".parquet"
    raw_df = spark_session.read.parquet(str(output_path))
    assert raw_df.count() == stats["landing_row_count"]


@requires_sample_file
def test_ingest_file_preserves_missing_values(spark_session, tmp_path):
    landing_path = _green_sample_path()
    landing_df = spark_session.read.parquet(str(landing_path))
    landing_null_total = sum(
        landing_df.select(
            [F.sum(F.col(c).isNull().cast("int")).alias(c) for c in landing_df.columns]
        )
        .first()
        .asDict()
        .values()
    )

    raw_dir = tmp_path / "raw"
    stats = ingest_file(spark_session, "green", landing_path, raw_dir)
    raw_null_total = sum(stats["raw_null_counts"].values())

    # Ingestion must not silently drop or impute missing values.
    assert raw_null_total == landing_null_total


@requires_sample_file
def test_ingest_file_is_idempotent(spark_session, tmp_path):
    landing_path = _green_sample_path()
    raw_dir = tmp_path / "raw"

    first = ingest_file(spark_session, "green", landing_path, raw_dir)
    second = ingest_file(spark_session, "green", landing_path, raw_dir)

    assert first["raw_row_count"] == second["raw_row_count"]
    assert first["raw_column_count"] == second["raw_column_count"]
