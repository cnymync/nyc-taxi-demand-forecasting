"""Raw -> Clean pipeline for the NYC taxi datasets.

Cleans Yellow and Green Raw data *separately* (they have different schemas
and field definitions -- see the Yellow/Green TLC data dictionaries in
`resources/`), standardises both to a common schema, then writes both to
`data/clean/taxi/` as one flat `.parquet` file per taxi type per month
(`{taxi_type}-cleaned-{MM}-{YY}.parquet`, mirroring the Raw layer's
naming) -- together forming one logical unified Clean taxi dataset,
readable as `spark.read.parquet("data/clean/taxi/")`.

This stage removes only observations with a specific, documented reason
(see `src/clean_rules.py` for the exact removal rules) -- either
demonstrably invalid (e.g. an impossible timestamp order) or explicitly
out-of-policy per user instruction (e.g. negative fares). Values that are
merely unusual or statistically extreme are profiled and, where useful,
flagged, never silently dropped. Every removal rule is logged to a ledger
with the exact record count and percentage affected, computed from the
data (never invented).

Usage:
    python -m src.cleaning            # clean yellow + green, write Clean
    python -m src.cleaning yellow     # profile/clean yellow only (no union/write)
    python -m src.cleaning green      # profile/clean green only (no union/write)
"""

from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import time

from loguru import logger
import pandas as pd
from pyspark.sql import Column, DataFrame, SparkSession
from pyspark.sql import functions as F
import typer

# Must happen before the JVM starts -- spark.sql.session.timeZone alone is
# not enough, since PySpark's collect()-to-Python datetime conversion
# follows the process/JVM's default timezone, not the Spark session conf.
# Without this, a driver machine set to a DST-observing timezone (e.g.
# Australia/Melbourne) can silently corrupt date/time SQL operations --
# this is what caused a real duplicate-row bug in src/processed.py's
# zone-hour grid, fixed there and applied here for the same reason: this
# module's trip_duration_minutes (unix_timestamp subtraction) is timezone-
# sensitive for any trip spanning exactly a DST transition instant.
os.environ["TZ"] = "UTC"
time.tzset()

from src.clean_rules import (
    CLEAN_SCHEMA_COLUMNS,
    FIELD_RULES,
    PICKUP_DROPOFF_RENAMES,
    STRUCTURALLY_ONE_SIDED_FIELDS,
    STUDY_PERIOD_END_EXCLUSIVE,
    STUDY_PERIOD_START,
    UNKNOWN_NA_ZONE_IDS,
    VALID_LOCATION_ID_RANGE,
    iqr_outlier_multiplier,
)
from src.config import CLEAN_DATA_DIR, RAW_DATA_DIR, REPORTS_DIR

PROVENANCE_COLUMNS = [
    "source_dataset",
    "source_file",
    "source_year",
    "source_month",
    "ingestion_timestamp",
]

CLEAN_DATA_QUALITY_DIR = REPORTS_DIR / "data_quality" / "clean"

app = typer.Typer()


# ---------------------------------------------------------------------------
# Loading and schema standardisation
# ---------------------------------------------------------------------------


def load_raw(spark: SparkSession, taxi_type: str, raw_dir: Path = RAW_DATA_DIR) -> DataFrame:
    """Read every Raw month file for one taxi type as a single DataFrame."""
    input_path = raw_dir / taxi_type / "*.parquet"
    return spark.read.parquet(str(input_path))


def standardise_to_clean_schema(df: DataFrame, taxi_type: str) -> DataFrame:
    """Rename Yellow/Green-specific columns to the common Clean schema.

    Adds `taxi_type`, computes `trip_duration_minutes`, and fills in the
    columns that only exist conceptually for the *other* taxi type
    (`airport_fee` for Green, `ehail_fee`/`trip_type` for Yellow) as typed
    NULL -- structurally not-applicable, not missing data.
    """
    for old_name, new_name in PICKUP_DROPOFF_RENAMES[taxi_type].items():
        if old_name in df.columns:
            df = df.withColumnRenamed(old_name, new_name)

    df = df.withColumn("taxi_type", F.lit(taxi_type))

    for column, owning_taxi_type in STRUCTURALLY_ONE_SIDED_FIELDS.items():
        if column not in df.columns:
            dtype = FIELD_RULES[owning_taxi_type][column].dtype
            df = df.withColumn(column, F.lit(None).cast(dtype))

    df = df.withColumn(
        "trip_duration_minutes",
        (F.unix_timestamp("dropoff_datetime") - F.unix_timestamp("pickup_datetime")) / 60.0,
    )

    return df.select(*CLEAN_SCHEMA_COLUMNS, *PROVENANCE_COLUMNS)


# ---------------------------------------------------------------------------
# Profiling (report-only -- never removes rows)
# ---------------------------------------------------------------------------


def profile_missingness(df: DataFrame, columns: list[str], total: int) -> dict:
    """1 action: null count per column."""
    null_counts = (
        df.select([F.sum(F.col(c).isNull().cast("int")).alias(c) for c in columns])
        .collect()[0]
        .asDict()
    )
    return {
        c: {"null_count": n, "pct_null": round(100 * n / total, 4) if total else 0.0}
        for c, n in null_counts.items()
    }


def profile_numeric(df: DataFrame, columns: list[str]) -> dict:
    """count/min/percentiles/max for each numeric column, nulls excluded.

    3 actions total regardless of how many columns are passed (one combined
    non-null-count aggregate, one combined min/max aggregate, one
    multi-column approxQuantile call) -- avoids one scan per column on a
    dataset with 100M+ rows.
    """
    probs = [0.01, 0.05, 0.5, 0.75, 0.95, 0.99, 0.999]
    counts = df.select([F.count(F.col(c)).alias(c) for c in columns]).collect()[0].asDict()
    minmax = df.select(
        *[F.min(c).alias(f"{c}__min") for c in columns],
        *[F.max(c).alias(f"{c}__max") for c in columns],
    ).collect()[0]
    quantiles_by_col = dict(zip(columns, df.approxQuantile(columns, probs, 0.001)))

    result = {}
    for c in columns:
        if counts[c] == 0:
            result[c] = {"count": 0}
            continue
        q = quantiles_by_col[c]
        result[c] = {
            "count": counts[c],
            "min": minmax[f"{c}__min"],
            "p1": q[0],
            "p5": q[1],
            "median": q[2],
            "p75": q[3],
            "p95": q[4],
            "p99": q[5],
            "p99_9": q[6],
            "max": minmax[f"{c}__max"],
        }
    return result


def profile_categorical(df: DataFrame, column: str, valid_values: tuple | None) -> dict:
    """1 action per categorical field (a groupBy aggregate, not a full scan-per-value)."""
    counts = {str(row[column]): row["count"] for row in df.groupBy(column).count().collect()}
    if valid_values is None:
        return {"value_counts": counts, "unexpected_values": {}}
    valid_strs = {str(v) for v in valid_values}
    unexpected = {v: n for v, n in counts.items() if v != "None" and v not in valid_strs}
    return {
        "value_counts": counts,
        "valid_values": list(valid_values),
        "unexpected_values": unexpected,
    }


def profile_location_validity(df: DataFrame, total: int) -> dict:
    """1 action: null/out-of-range/unknown-zone counts for both location columns."""
    lo, hi = VALID_LOCATION_ID_RANGE

    def _exprs(col: str) -> list[Column]:
        c = F.col(col)
        return [
            F.sum(c.isNull().cast("int")).alias(f"{col}__null"),
            F.sum((c.isNotNull() & ((c < lo) | (c > hi))).cast("int")).alias(f"{col}__oor"),
            F.sum(c.isin(*UNKNOWN_NA_ZONE_IDS).cast("int")).alias(f"{col}__unk"),
        ]

    row = df.select(*_exprs("pu_location_id"), *_exprs("do_location_id")).collect()[0]

    def _stats(col: str) -> dict:
        null_count, oor, unk = row[f"{col}__null"], row[f"{col}__oor"], row[f"{col}__unk"]
        return {
            "null_count": null_count,
            "out_of_range_count": oor,
            "unknown_or_na_zone_count": unk,
            "pct_null": round(100 * null_count / total, 4) if total else 0.0,
            "pct_out_of_range": round(100 * oor / total, 4) if total else 0.0,
        }

    return {"pu_location_id": _stats("pu_location_id"), "do_location_id": _stats("do_location_id")}


def profile_study_period(df: DataFrame, total: int) -> dict:
    """1 action: earliest/latest pickup and out-of-study-period count."""
    outside_cond = (F.col("pickup_datetime") < F.lit(STUDY_PERIOD_START)) | (
        F.col("pickup_datetime") >= F.lit(STUDY_PERIOD_END_EXCLUSIVE)
    )
    row = df.agg(
        F.min("pickup_datetime").alias("min"),
        F.max("pickup_datetime").alias("max"),
        F.sum(outside_cond.cast("int")).alias("outside"),
    ).collect()[0]
    outside = row["outside"] or 0
    return {
        "earliest_pickup_datetime": str(row["min"]),
        "latest_pickup_datetime": str(row["max"]),
        "study_period_start": STUDY_PERIOD_START,
        "study_period_end_exclusive": STUDY_PERIOD_END_EXCLUSIVE,
        "records_outside_study_period": outside,
        "pct_outside_study_period": round(100 * outside / total, 4) if total else 0.0,
    }


def profile_temporal_consistency(df: DataFrame, total: int) -> dict:
    """1 action: duration/consistency counts, all as conditional sums over the full df."""
    both_present = F.col("pickup_datetime").isNotNull() & F.col("dropoff_datetime").isNotNull()
    duration = F.col("trip_duration_minutes")
    row = df.agg(
        F.sum(both_present.cast("int")).alias("both"),
        F.sum(
            (both_present & (F.col("dropoff_datetime") < F.col("pickup_datetime"))).cast("int")
        ).alias("neg"),
        F.sum((both_present & (duration == 0)).cast("int")).alias("zero"),
        F.sum((both_present & (duration > 180)).cast("int")).alias("over_3h"),
        F.sum((both_present & (duration > 1440)).cast("int")).alias("over_24h"),
    ).collect()[0]
    n_both = row["both"] or 0
    return {
        "total_rows": total,
        "rows_with_both_timestamps": n_both,
        "rows_missing_dropoff_datetime": total - n_both,
        "negative_duration": row["neg"] or 0,
        "zero_duration": row["zero"] or 0,
        "duration_over_3h": row["over_3h"] or 0,
        "duration_over_24h": row["over_24h"] or 0,
    }


# ---------------------------------------------------------------------------
# Removal rules -- each documented, each computed from the data
# ---------------------------------------------------------------------------


def apply_removal_rule(
    df: DataFrame,
    bad_condition: Column,
    rule: str,
    reason: str,
    dataset: str,
    original_count: int,
    ledger: list[dict],
    stage: str = "invalid",
) -> DataFrame:
    """Remove rows matching `bad_condition`, logging the exact count to `ledger`.

    `bad_condition` is coalesced to False before use: a row where the
    condition is indeterminate (e.g. comparing a NULL column) is kept, not
    removed -- absence of evidence of invalidity is not evidence of
    invalidity.
    """
    safe_bad = F.coalesce(bad_condition, F.lit(False))
    removed = df.where(safe_bad).count()
    kept = df.where(~safe_bad)
    ledger.append(
        {
            "stage": stage,
            "dataset": dataset,
            "rule": rule,
            "reason": reason,
            "records_removed": removed,
            "pct_of_original": round(100 * removed / original_count, 4) if original_count else 0.0,
        }
    )
    if removed:
        logger.info(f"[{dataset}] removed {removed:,} rows: {rule}")
    return kept


def apply_duplicate_removal(
    df: DataFrame,
    subset_cols: list[str],
    dataset: str,
    original_count: int,
    before_count: int,
    ledger: list[dict],
) -> tuple[DataFrame, dict]:
    """Drop exact duplicate records (on `subset_cols`), profiling and removing in one pass.

    `before_count` is the caller's already-known row count going into this
    step (tracked through the pipeline rather than recomputed here) so this
    function needs only one action (`dropDuplicates(...).count()`) instead
    of two full-table passes over a wide 100M+ row dataset.
    """
    deduped = df.dropDuplicates(subset_cols)
    after = deduped.count()
    removed = before_count - after
    ledger.append(
        {
            "stage": "invalid",
            "dataset": dataset,
            "rule": "exact duplicate business-column record",
            "reason": "identical trip-defining values (all Clean business fields) already represented by another row",
            "records_removed": removed,
            "pct_of_original": round(100 * removed / original_count, 4) if original_count else 0.0,
        }
    )
    if removed:
        logger.info(f"[{dataset}] removed {removed:,} duplicate rows")
    profile = {
        "total_rows": before_count,
        "distinct_rows": after,
        "duplicate_extra_rows": removed,
        "pct_duplicate": round(100 * removed / before_count, 4) if before_count else 0.0,
    }
    return deduped, profile


def add_extreme_value_flags(df: DataFrame, taxi_type: str, row_count: int) -> DataFrame:
    """Flag (never remove) statistically extreme distance/duration values.

    Threshold = Q3 + k * IQR, with k from the Lecture 3 IQR outlier rule
    (`iqr_outlier_multiplier`, `src/clean_rules.py`) computed from this
    taxi type's own row count -- see src/clean_rules.py for why a fixed
    high percentile (e.g. P99.9) was rejected in favour of this at this
    row count.
    """
    k = iqr_outlier_multiplier(row_count)
    q1_d, q3_d = df.approxQuantile("trip_distance", [0.25, 0.75], 0.001)
    q1_t, q3_t = df.approxQuantile("trip_duration_minutes", [0.25, 0.75], 0.001)
    distance_threshold = q3_d + k * (q3_d - q1_d)
    duration_threshold = q3_t + k * (q3_t - q1_t)
    logger.info(
        f"[{taxi_type}] extreme-value flag thresholds (Q3 + {k:.4f}*IQR, N={row_count:,}): "
        f"trip_distance > {distance_threshold}, trip_duration_minutes > {duration_threshold}"
    )
    return df.withColumn(
        "is_extreme_trip_distance",
        F.coalesce(F.col("trip_distance") > F.lit(distance_threshold), F.lit(False)),
    ).withColumn(
        "is_extreme_trip_duration",
        F.coalesce(F.col("trip_duration_minutes") > F.lit(duration_threshold), F.lit(False)),
    )


# ---------------------------------------------------------------------------
# Per-taxi-type cleaning
# ---------------------------------------------------------------------------


def clean_taxi_type(spark: SparkSession, taxi_type: str, raw_dir: Path = RAW_DATA_DIR) -> dict:
    """Run the full Raw -> Clean workflow for one taxi type.

    Returns a dict with the cleaned DataFrame, the removal-rule ledger, the
    data-quality profile report, and before/after row counts.
    """
    ledger: list[dict] = []

    raw_df = load_raw(spark, taxi_type, raw_dir)
    raw_row_count = raw_df.count()
    logger.info(f"[{taxi_type}] loaded {raw_row_count:,} Raw rows")

    df = standardise_to_clean_schema(raw_df, taxi_type)
    original_count = raw_row_count  # rename/select never changes row count

    rules = FIELD_RULES[taxi_type]
    categorical_fields = [name for name, rule in rules.items() if rule.valid_values is not None]
    numeric_profile_fields = [
        "passenger_count",
        "trip_distance",
        "trip_duration_minutes",
        "fare_amount",
        "total_amount",
    ]

    profile = {
        "raw_row_count": raw_row_count,
        "missingness": profile_missingness(df, CLEAN_SCHEMA_COLUMNS, raw_row_count),
        "numeric": profile_numeric(df, numeric_profile_fields),
        "categorical": {
            field: profile_categorical(df, field, rules[field].valid_values)
            for field in categorical_fields
        },
        "location_validity": profile_location_validity(df, raw_row_count),
        "study_period": profile_study_period(df, raw_row_count),
        "temporal_consistency": profile_temporal_consistency(df, raw_row_count),
    }

    # --- removal rules: only demonstrably invalid observations -----------
    df = apply_removal_rule(
        df,
        F.col("pickup_datetime").isNull(),
        rule="pickup_datetime IS NULL",
        reason="pickup_datetime is essential for zone-hour pickup-demand analysis",
        dataset=taxi_type,
        original_count=original_count,
        ledger=ledger,
        stage="essential_missing",
    )
    df = apply_removal_rule(
        df,
        F.col("pu_location_id").isNull(),
        rule="pu_location_id IS NULL",
        reason="pu_location_id is essential for zone-hour pickup-demand analysis",
        dataset=taxi_type,
        original_count=original_count,
        ledger=ledger,
        stage="essential_missing",
    )
    lo, hi = VALID_LOCATION_ID_RANGE
    df = apply_removal_rule(
        df,
        (F.col("pu_location_id") < lo) | (F.col("pu_location_id") > hi),
        rule=f"pu_location_id outside valid TLC zone range [{lo}, {hi}]",
        reason="pu_location_id is essential and must reference a real/documented TLC taxi zone",
        dataset=taxi_type,
        original_count=original_count,
        ledger=ledger,
        stage="invalid",
    )
    df = apply_removal_rule(
        df,
        F.col("do_location_id").isNotNull()
        & ((F.col("do_location_id") < lo) | (F.col("do_location_id") > hi)),
        rule=f"do_location_id outside valid TLC zone range [{lo}, {hi}]",
        reason="dropoff location must reference a real/documented TLC taxi zone (per user cleaning policy)",
        dataset=taxi_type,
        original_count=original_count,
        ledger=ledger,
        stage="invalid",
    )
    df = apply_removal_rule(
        df,
        F.col("trip_distance") <= 0,
        rule="trip_distance <= 0",
        reason="negative distance is physically impossible; zero-distance trips dropped per user cleaning policy",
        dataset=taxi_type,
        original_count=original_count,
        ledger=ledger,
        stage="invalid",
    )
    df = apply_removal_rule(
        df,
        F.col("passenger_count") < 0,
        rule="passenger_count < 0",
        reason="a physically impossible passenger count",
        dataset=taxi_type,
        original_count=original_count,
        ledger=ledger,
        stage="invalid",
    )
    df = apply_removal_rule(
        df,
        F.col("fare_amount") < 0,
        rule="fare_amount < 0",
        reason="negative fares dropped per user cleaning policy",
        dataset=taxi_type,
        original_count=original_count,
        ledger=ledger,
        stage="invalid",
    )
    df = apply_removal_rule(
        df,
        (F.col("pickup_datetime").isNotNull())
        & (F.col("dropoff_datetime").isNotNull())
        & ((F.col("trip_duration_minutes") <= 0) | (F.col("trip_duration_minutes") > 1440)),
        rule="trip_duration_minutes <= 0 or > 1440 (24h)",
        reason="non-positive or >24h trips dropped per user cleaning policy (supersedes the narrower "
        "dropoff < pickup check, which this subsumes)",
        dataset=taxi_type,
        original_count=original_count,
        ledger=ledger,
        stage="invalid",
    )
    # Running count tracked from the ledger (pure arithmetic, no Spark
    # action) so duplicate removal doesn't need a redundant count() of its
    # input on top of the expensive dropDuplicates() shuffle itself.
    running_count = original_count - sum(entry["records_removed"] for entry in ledger)
    df, duplicate_profile = apply_duplicate_removal(
        df,
        subset_cols=[c for c in CLEAN_SCHEMA_COLUMNS if c != "trip_duration_minutes"],
        dataset=taxi_type,
        original_count=original_count,
        before_count=running_count,
        ledger=ledger,
    )
    profile["duplicates"] = duplicate_profile
    df = apply_removal_rule(
        df,
        (F.col("pickup_datetime") < F.lit(STUDY_PERIOD_START))
        | (F.col("pickup_datetime") >= F.lit(STUDY_PERIOD_END_EXCLUSIVE)),
        rule=f"pickup_datetime outside study period [{STUDY_PERIOD_START}, {STUDY_PERIOD_END_EXCLUSIVE})",
        reason="out of the established study period; not a data-validity issue",
        dataset=taxi_type,
        original_count=original_count,
        ledger=ledger,
        stage="study_period",
    )

    clean_row_count = df.count()
    df = add_extreme_value_flags(df, taxi_type, clean_row_count)

    logger.success(
        f"[{taxi_type}] Clean: {clean_row_count:,} / {raw_row_count:,} rows retained "
        f"({100 * clean_row_count / raw_row_count:.2f}%)"
    )

    return {
        "taxi_type": taxi_type,
        "clean_df": df,
        "ledger": ledger,
        "profile": profile,
        "raw_row_count": raw_row_count,
        "clean_row_count": clean_row_count,
    }


# ---------------------------------------------------------------------------
# Union + write + reporting
# ---------------------------------------------------------------------------


def clean_filename(taxi_type: str, year: int, month: int) -> str:
    """`{taxi_type}-cleaned-{MM}-{YY}.parquet`, mirroring the Raw layer's naming."""
    return f"{taxi_type}-cleaned-{month:02d}-{year % 100:02d}.parquet"


def write_clean_dataset_per_month(
    df: DataFrame, taxi_type: str, output_dir: Path = CLEAN_DATA_DIR / "taxi"
) -> None:
    """Write one flat `.parquet` file per (source_year, source_month), mirroring
    the Raw layer's one-file-per-month layout (`write_single_parquet_file` in
    `src/dataset.py`), but for many months at once from an already-fully-computed
    DataFrame.

    A plain `df.write.partitionBy(...)` would still leave multiple part-files
    per month (however many tasks touch that partition), so this repartitions
    by the exact (year, month) key first -- one Spark partition per month,
    hence exactly one output file per month -- then renames each into the
    flat `{taxi_type}-cleaned-{MM}-{YY}.parquet` file and removes the
    intermediate Hive-style partition directories.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / f"_tmp_{taxi_type}"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)

    n_months = df.select("source_year", "source_month").distinct().count()
    (
        df.repartition(n_months, "source_year", "source_month")
        .write.mode("overwrite")
        .partitionBy("source_year", "source_month")
        .parquet(str(tmp_dir))
    )

    n_written = 0
    for year_dir in sorted(tmp_dir.glob("source_year=*")):
        year = int(year_dir.name.split("=")[1])
        for month_dir in sorted(year_dir.glob("source_month=*")):
            month = int(month_dir.name.split("=")[1])
            part_files = list(month_dir.glob("part-*.parquet"))
            if len(part_files) != 1:
                raise RuntimeError(
                    f"Expected exactly one part file in {month_dir}, found {len(part_files)}"
                )
            target = output_dir / clean_filename(taxi_type, year, month)
            if target.exists():
                target.unlink()
            shutil.move(str(part_files[0]), str(target))
            n_written += 1

    shutil.rmtree(tmp_dir)
    logger.info(f"[{taxi_type}] wrote {n_written} monthly Clean file(s) to {output_dir}")


def write_reports(
    results: list[dict], combined_row_count: int, report_dir: Path = CLEAN_DATA_QUALITY_DIR
) -> None:
    """Write the removal ledger (CSV + JSON), the per-taxi-type + combined
    row-count summary, and each taxi type's full data-quality profile."""
    report_dir.mkdir(parents=True, exist_ok=True)
    run_ts = datetime.now(timezone.utc).isoformat()

    all_ledger_rows = [row for r in results for row in r["ledger"]]
    ledger_df = pd.DataFrame(all_ledger_rows)
    ledger_df.to_csv(report_dir / "cleaning_ledger.csv", index=False)
    ledger_df.to_json(report_dir / "cleaning_ledger.json", orient="records", indent=2)

    summary_rows = [
        {
            "taxi_type": r["taxi_type"],
            "raw_row_count": r["raw_row_count"],
            "clean_row_count": r["clean_row_count"],
            "rows_removed": r["raw_row_count"] - r["clean_row_count"],
            "pct_removed": round(
                100 * (r["raw_row_count"] - r["clean_row_count"]) / r["raw_row_count"], 4
            ),
        }
        for r in results
    ]
    summary_rows.append(
        {
            "taxi_type": "combined",
            "raw_row_count": sum(r["raw_row_count"] for r in results),
            "clean_row_count": combined_row_count,
            "rows_removed": sum(r["raw_row_count"] for r in results) - combined_row_count,
            "pct_removed": round(
                100
                * (sum(r["raw_row_count"] for r in results) - combined_row_count)
                / sum(r["raw_row_count"] for r in results),
                4,
            ),
        }
    )
    pd.DataFrame(summary_rows).to_csv(report_dir / "cleaning_summary.csv", index=False)

    for r in results:
        profile_path = report_dir / f"{r['taxi_type']}_profile.json"
        with open(profile_path, "w") as f:
            pd.Series({"run_timestamp": run_ts, **r["profile"]}).to_json(f, indent=2)

    logger.info(f"Clean data-quality reports written to {report_dir}")


@app.command()
def main(
    taxi_type: str = typer.Argument(
        "all",
        help="Which dataset to clean: 'yellow', 'green', or 'all' (unions and writes Clean).",
    ),
):
    """Run the Raw -> Clean pipeline."""
    if taxi_type not in {"all", "yellow", "green"}:
        raise typer.BadParameter("taxi_type must be 'all', 'yellow', or 'green'")

    # Yellow alone is ~109M rows / ~2.4GB across 29 files read together (a
    # wildcard glob, unlike the one-file-at-a-time Landing->Raw pipeline),
    # plus wide aggregations and a dropDuplicates() shuffle over ~20
    # columns -- the default ~1GB driver heap OOMs on this. Bump it well
    # above what's needed on a 36GB-RAM dev machine; adjust down if run
    # somewhere smaller.
    spark = (
        SparkSession.builder.appName("nyc-taxi-raw-to-clean")
        .master("local[*]")
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.driver.memory", "16g")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")

    if taxi_type == "all":
        results = [clean_taxi_type(spark, "yellow"), clean_taxi_type(spark, "green")]
        for r in results:
            write_clean_dataset_per_month(r["clean_df"], r["taxi_type"])
        combined_row_count = sum(r["clean_row_count"] for r in results)
        write_reports(results, combined_row_count)
        logger.success(
            f"Combined Clean taxi dataset: {combined_row_count:,} rows "
            f"(Yellow {results[0]['clean_row_count']:,} + Green {results[1]['clean_row_count']:,})"
        )
    else:
        result = clean_taxi_type(spark, taxi_type)
        write_reports([result], result["clean_row_count"])
        logger.info(
            f"Ran in single-taxi-type mode ('{taxi_type}'); Clean dataset not written (use 'all')."
        )

    spark.stop()


if __name__ == "__main__":
    app()
