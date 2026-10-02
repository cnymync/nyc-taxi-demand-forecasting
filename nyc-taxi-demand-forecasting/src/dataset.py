"""Landing -> Raw ingestion pipeline for the NYC taxi datasets.

Reads monthly TLC parquet files from `data/landing/{yellow,green}/`,
standardises column names and types against the canonical schemas in
`src/schemas.py`, attaches provenance metadata, and writes the result to
`data/raw/{yellow,green}/`, mirroring the Landing layout: one flat
`.parquet` file per month, named `{taxi_type}-{MM}-{YY}.parquet`
(e.g. `yellow-01-24.parquet` for January 2024).

This stage does NOT clean, filter, deduplicate, or impute the data.
Invalid-looking values (negative fares, zero passengers, etc.) are
preserved as-is and reported in the ingestion log, not removed.

Usage:
    python -m src.dataset            # ingest yellow + green
    python -m src.dataset yellow     # ingest yellow only
    python -m src.dataset green      # ingest green only
"""

from datetime import datetime, timezone
import os
from pathlib import Path
import re
import shutil
import time

from loguru import logger
import pandas as pd
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
import typer

# Must happen before the JVM starts -- spark.sql.session.timeZone alone is
# not enough, since PySpark's collect()-to-Python datetime conversion
# follows the process/JVM's default timezone, not the Spark session conf.
# Without this, a driver machine set to a DST-observing timezone (e.g.
# Australia/Melbourne) can silently corrupt date/time SQL operations -- see
# the same fix in src/processed.py for the bug this caused there.
os.environ["TZ"] = "UTC"
time.tzset()

from src.config import LANDING_DATA_DIR, PROJ_ROOT, RAW_DATA_DIR, REPORTS_DIR
from src.schemas import COLUMN_RENAMES, MONTH_NAME_TO_NUMBER, RAW_SCHEMAS

FILENAME_PATTERNS = {
    "yellow": re.compile(r"Yellow Taxi Data (?P<month>[A-Za-z]+) (?P<year>\d{4})\.parquet$"),
    "green": re.compile(r"Green Trip Data (?P<month>[A-Za-z]+) (?P<year>\d{4})\.parquet$"),
}

DATA_QUALITY_DIR = REPORTS_DIR / "data_quality"

app = typer.Typer()


def parse_filename(taxi_type: str, filename: str) -> tuple[int, int]:
    """Extract (year, month) from a Landing filename.

    Raises ValueError rather than guessing if the filename doesn't match
    the expected pattern -- an ingestion failure here should be loud, not
    silently skipped.
    """
    pattern = FILENAME_PATTERNS[taxi_type]
    match = pattern.match(filename)
    if not match:
        raise ValueError(f"Filename does not match expected {taxi_type} pattern: {filename!r}")
    month_name = match.group("month").lower()
    if month_name not in MONTH_NAME_TO_NUMBER:
        raise ValueError(f"Unrecognised month name {match.group('month')!r} in {filename!r}")
    return int(match.group("year")), MONTH_NAME_TO_NUMBER[month_name]


def standardise_schema(df: DataFrame, taxi_type: str) -> DataFrame:
    """Rename columns and cast to the canonical Raw schema for `taxi_type`.

    A canonical column missing from this month's Landing file (e.g.
    `cbd_congestion_fee` before Jan 2025) is added as a typed NULL column,
    never dropped or backfilled with a guessed value. No rows are filtered
    here.
    """
    renames = COLUMN_RENAMES[taxi_type]
    for old_name, new_name in renames.items():
        if old_name in df.columns:
            df = df.withColumnRenamed(old_name, new_name)

    canonical_schema = RAW_SCHEMAS[taxi_type]
    select_exprs = []
    for field in canonical_schema.fields:
        if field.name in df.columns:
            select_exprs.append(F.col(field.name).cast(field.dataType).alias(field.name))
        else:
            select_exprs.append(F.lit(None).cast(field.dataType).alias(field.name))
    return df.select(*select_exprs)


def add_provenance_columns(
    df: DataFrame,
    taxi_type: str,
    source_file: str,
    year: int,
    month: int,
    ingestion_ts: str,
) -> DataFrame:
    """Attach columns that trace a Raw record back to its Landing source."""
    return (
        df.withColumn("source_dataset", F.lit(f"{taxi_type}_taxi"))
        .withColumn("source_file", F.lit(source_file))
        .withColumn("source_year", F.lit(year))
        .withColumn("source_month", F.lit(month))
        .withColumn("ingestion_timestamp", F.lit(ingestion_ts))
    )


def profile_dataframe(df: DataFrame, columns: list[str]) -> dict:
    """Row/column/null counts for before-vs-after data-quality logging.

    Nulls are counted, never removed -- this is reporting only.
    """
    row_count = df.count()
    null_counts = (
        df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in columns])
        .collect()[0]
        .asDict()
    )
    return {"row_count": row_count, "column_count": len(columns), "null_counts": null_counts}


def raw_filename(taxi_type: str, year: int, month: int) -> str:
    """`{taxi_type}-{MM}-{YY}.parquet`, e.g. `yellow-01-24.parquet` for Jan 2024."""
    return f"{taxi_type}-{month:02d}-{year % 100:02d}.parquet"


def write_single_parquet_file(df: DataFrame, output_path: Path) -> None:
    """Write `df` as one flat `.parquet` file at `output_path`.

    Spark only writes directories of part-files, so this writes to a
    temporary directory with a single partition, then moves the one part
    file to the desired flat filename and removes the temporary directory
    -- mirroring the one-file-per-month layout used in `data/landing/`.
    """
    tmp_dir = output_path.parent / f"_tmp_{output_path.stem}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    df.coalesce(1).write.mode("overwrite").parquet(str(tmp_dir))

    part_files = list(tmp_dir.glob("part-*.parquet"))
    if len(part_files) != 1:
        raise RuntimeError(f"Expected exactly one part file in {tmp_dir}, found {len(part_files)}")

    if output_path.exists():
        output_path.unlink()
    shutil.move(str(part_files[0]), str(output_path))
    shutil.rmtree(tmp_dir)


def ingest_file(spark: SparkSession, taxi_type: str, landing_path: Path, raw_dir: Path) -> dict:
    """Ingest a single Landing month file into the Raw layer.

    Returns a stats dict used for the data-quality/ingestion log. Landing
    files are only ever read, never written to.
    """
    year, month = parse_filename(taxi_type, landing_path.name)
    ingestion_ts = datetime.now(timezone.utc).isoformat()

    logger.info(f"Reading {landing_path.name}")
    landing_df = spark.read.parquet(str(landing_path))
    before = profile_dataframe(landing_df, landing_df.columns)

    raw_df = standardise_schema(landing_df, taxi_type)
    canonical_columns = [f.name for f in RAW_SCHEMAS[taxi_type].fields]
    raw_df = add_provenance_columns(
        raw_df, taxi_type, landing_path.name, year, month, ingestion_ts
    )

    output_dir = raw_dir / taxi_type
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / raw_filename(taxi_type, year, month)

    after = profile_dataframe(raw_df, canonical_columns)

    logger.info(f"Writing {before['row_count']:,} rows -> {output_path}")
    write_single_parquet_file(raw_df, output_path)

    status = "ok" if before["row_count"] == after["row_count"] else "ROW_COUNT_MISMATCH"
    if status != "ok":
        logger.error(
            f"{landing_path.name}: row count changed during ingestion "
            f"({before['row_count']} -> {after['row_count']})"
        )

    try:
        output_path_str = str(output_path.relative_to(PROJ_ROOT))
    except ValueError:
        output_path_str = str(output_path)

    return {
        "taxi_type": taxi_type,
        "source_file": landing_path.name,
        "output_path": output_path_str,
        "year": year,
        "month": month,
        "landing_row_count": before["row_count"],
        "raw_row_count": after["row_count"],
        "landing_column_count": before["column_count"],
        "raw_column_count": after["column_count"],
        "landing_null_counts": before["null_counts"],
        "raw_null_counts": after["null_counts"],
        "ingestion_timestamp": ingestion_ts,
        "status": status,
    }


def ingest_taxi_type(
    spark: SparkSession, taxi_type: str, raw_dir: Path = RAW_DATA_DIR
) -> list[dict]:
    """Ingest every Landing month file for one taxi type."""
    landing_dir = LANDING_DATA_DIR / taxi_type
    landing_files = sorted(landing_dir.glob("*.parquet"))
    if not landing_files:
        logger.warning(f"No landing files found for {taxi_type} in {landing_dir}")
        return []

    return [ingest_file(spark, taxi_type, path, raw_dir) for path in landing_files]


def write_ingestion_log(logs: list[dict], log_dir: Path = DATA_QUALITY_DIR) -> None:
    """Write both a detailed JSON log and a flat CSV summary for review."""
    log_dir.mkdir(parents=True, exist_ok=True)

    detail_df = pd.DataFrame(logs)
    detail_df.to_json(log_dir / "landing_to_raw_ingestion_log.json", orient="records", indent=2)

    summary_cols = [
        "taxi_type",
        "source_file",
        "output_path",
        "year",
        "month",
        "landing_row_count",
        "raw_row_count",
        "landing_column_count",
        "raw_column_count",
        "ingestion_timestamp",
        "status",
    ]
    summary_df = (
        detail_df[summary_cols] if not detail_df.empty else pd.DataFrame(columns=summary_cols)
    )
    summary_df.to_csv(log_dir / "landing_to_raw_ingestion_summary.csv", index=False)

    logger.info(f"Ingestion log written to {log_dir}")


@app.command()
def main(
    taxi_type: str = typer.Argument(
        "all", help="Which dataset to ingest: 'yellow', 'green', or 'all'."
    ),
):
    """Run the Landing -> Raw ingestion pipeline."""
    if taxi_type not in {"all", "yellow", "green"}:
        raise typer.BadParameter("taxi_type must be 'all', 'yellow', or 'green'")

    spark = (
        SparkSession.builder.appName("nyc-taxi-landing-to-raw")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    taxi_types = ["yellow", "green"] if taxi_type == "all" else [taxi_type]
    all_logs: list[dict] = []
    for tt in taxi_types:
        logger.info(f"Ingesting {tt} taxi: Landing -> Raw")
        all_logs.extend(ingest_taxi_type(spark, tt))

    write_ingestion_log(all_logs)

    n_mismatch = sum(1 for entry in all_logs if entry["status"] != "ok")
    if n_mismatch:
        logger.error(f"Ingested {len(all_logs)} file(s); {n_mismatch} row-count mismatch(es).")
    else:
        logger.success(f"Ingested {len(all_logs)} file(s); all row counts preserved.")

    spark.stop()


if __name__ == "__main__":
    app()
