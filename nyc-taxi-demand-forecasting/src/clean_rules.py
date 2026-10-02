"""Raw -> Clean validation/cleaning parameters for the NYC taxi datasets.

Every threshold and valid-value set here is taken directly from the TLC
Trip Record data dictionaries (Yellow and Green/LPEP), not invented. Where
a parameter is *not* dictionary-derived (the taxi-zone ID range, the
study-period boundary), that is called out explicitly below with the
reasoning, flagging uncertain decisions rather than silently choosing them.

This module is deliberately declarative/data-only: `src/cleaning.py`
consumes it to build the actual PySpark logic.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class FieldRule:
    """Documents one field's expected shape and how Clean should treat it.

    Attributes:
        clean_name: column name in the unified Clean schema.
        dtype: expected Spark type (simpleString), inherited from Raw.
        valid_values: documented categorical codes, or None for non-categorical fields.
        null_valid: whether NULL is a legitimate value per the dictionary/TLC behaviour.
        zero_valid: whether 0 is a legitimate value.
        negative_valid: whether a negative value is a legitimate value.
        essential: whether the field is required for zone-hour pickup-demand analysis.
        retain_in_clean: whether the field is kept in the Clean output.
        invalid_definition: what specifically makes an observed value in this field
            "demonstrably invalid" (used to justify a removal rule) as opposed to
            merely unusual (which is reported, not removed).
    """

    clean_name: str
    dtype: str
    valid_values: tuple | None
    null_valid: bool
    zero_valid: bool
    negative_valid: bool
    essential: bool
    retain_in_clean: bool
    invalid_definition: str


# ---------------------------------------------------------------------------
# Categorical codes, taken verbatim from the two data dictionaries.
# ---------------------------------------------------------------------------

# Yellow dictionary lists VendorID 1, 2, 6, 7. Green (LPEP) dictionary lists
# only 1, 2, 6 -- Helix (7) is not documented as an LPEP provider.
VENDOR_ID_VALID = {"yellow": (1, 2, 6, 7), "green": (1, 2, 6)}

# Identical in both dictionaries.
RATECODE_ID_VALID = (1, 2, 3, 4, 5, 6, 99)
PAYMENT_TYPE_VALID = (0, 1, 2, 3, 4, 5, 6)
STORE_AND_FWD_FLAG_VALID = ("Y", "N")

# Green-only field (Yellow trips are definitionally street-hail).
TRIP_TYPE_VALID = (1, 2)

# ---------------------------------------------------------------------------
# Yellow field rules (source: z NYC Yellow Taxi Trip Records Data Dictionary.pdf)
# ---------------------------------------------------------------------------

YELLOW_FIELD_RULES: dict[str, FieldRule] = {
    "vendor_id": FieldRule(
        "vendor_id",
        "int",
        VENDOR_ID_VALID["yellow"],
        True,
        False,
        False,
        essential=False,
        retain_in_clean=True,
        invalid_definition="not applicable: no removal rule, values outside {1,2,6,7} are reported, not removed",
    ),
    "pickup_datetime": FieldRule(
        "pickup_datetime",
        "timestamp",
        None,
        False,
        False,
        False,
        essential=True,
        retain_in_clean=True,
        invalid_definition="NULL, or outside the established study period",
    ),
    "dropoff_datetime": FieldRule(
        "dropoff_datetime",
        "timestamp",
        None,
        True,
        False,
        False,
        essential=False,
        retain_in_clean=True,
        invalid_definition="earlier than pickup_datetime, or resulting trip_duration_minutes <= 0 or > 1440 (24h), when both timestamps are present (per user cleaning policy)",
    ),
    "passenger_count": FieldRule(
        "passenger_count",
        "int",
        None,
        True,
        True,
        False,
        essential=False,
        retain_in_clean=True,
        invalid_definition="negative (a physically impossible count); 0 is retained (documented as ambiguous)",
    ),
    "trip_distance": FieldRule(
        "trip_distance",
        "double",
        None,
        True,
        True,
        False,
        essential=False,
        retain_in_clean=True,
        invalid_definition="<= 0 (negative is physically impossible; zero dropped per user cleaning policy); large values retained and flagged only",
    ),
    "ratecode_id": FieldRule(
        "ratecode_id",
        "int",
        RATECODE_ID_VALID,
        True,
        False,
        False,
        essential=False,
        retain_in_clean=True,
        invalid_definition="not applicable: no removal rule, values outside the documented set are reported",
    ),
    "store_and_fwd_flag": FieldRule(
        "store_and_fwd_flag",
        "string",
        STORE_AND_FWD_FLAG_VALID,
        True,
        False,
        False,
        essential=False,
        retain_in_clean=True,
        invalid_definition="not applicable: no removal rule, values outside {Y,N} are reported",
    ),
    "pu_location_id": FieldRule(
        "pu_location_id",
        "int",
        None,
        False,
        False,
        False,
        essential=True,
        retain_in_clean=True,
        invalid_definition="NULL, or outside the valid TLC Taxi Zone ID range",
    ),
    "do_location_id": FieldRule(
        "do_location_id",
        "int",
        None,
        True,
        False,
        False,
        essential=False,
        retain_in_clean=True,
        invalid_definition="outside the valid TLC Taxi Zone ID range (removed per user cleaning policy; not an essential field, so NULL is still retained)",
    ),
    "payment_type": FieldRule(
        "payment_type",
        "int",
        PAYMENT_TYPE_VALID,
        True,
        True,
        False,
        essential=False,
        retain_in_clean=True,
        invalid_definition="not applicable: no removal rule, values outside the documented set are reported",
    ),
    "fare_amount": FieldRule(
        "fare_amount",
        "double",
        None,
        True,
        True,
        True,
        essential=False,
        retain_in_clean=True,
        invalid_definition="negative (removed per user cleaning policy; other monetary fields are still retained/reported only, as negative extra/tolls/tax etc. may be legitimate refunds or adjustments)",
    ),
    "extra": FieldRule(
        "extra",
        "double",
        None,
        True,
        True,
        True,
        essential=False,
        retain_in_clean=True,
        invalid_definition="not applicable: reported only",
    ),
    "mta_tax": FieldRule(
        "mta_tax",
        "double",
        None,
        True,
        True,
        True,
        essential=False,
        retain_in_clean=True,
        invalid_definition="not applicable: reported only",
    ),
    "tip_amount": FieldRule(
        "tip_amount",
        "double",
        None,
        True,
        True,
        True,
        essential=False,
        retain_in_clean=True,
        invalid_definition="not applicable: reported only (cash tips are legitimately unrecorded, i.e. 0)",
    ),
    "tolls_amount": FieldRule(
        "tolls_amount",
        "double",
        None,
        True,
        True,
        True,
        essential=False,
        retain_in_clean=True,
        invalid_definition="not applicable: reported only",
    ),
    "improvement_surcharge": FieldRule(
        "improvement_surcharge",
        "double",
        None,
        True,
        True,
        True,
        essential=False,
        retain_in_clean=True,
        invalid_definition="not applicable: reported only",
    ),
    "total_amount": FieldRule(
        "total_amount",
        "double",
        None,
        True,
        True,
        True,
        essential=False,
        retain_in_clean=True,
        invalid_definition="not applicable: reported only",
    ),
    "congestion_surcharge": FieldRule(
        "congestion_surcharge",
        "double",
        None,
        True,
        True,
        True,
        essential=False,
        retain_in_clean=True,
        invalid_definition="not applicable: reported only",
    ),
    "airport_fee": FieldRule(
        "airport_fee",
        "double",
        None,
        True,
        True,
        True,
        essential=False,
        retain_in_clean=True,
        invalid_definition="not applicable: reported only; 0 for non-airport trips is expected",
    ),
    "cbd_congestion_fee": FieldRule(
        "cbd_congestion_fee",
        "double",
        None,
        True,
        True,
        True,
        essential=False,
        retain_in_clean=True,
        invalid_definition="not applicable: NULL before Jan 2025 is structural, not missing; reported only",
    ),
}

# ---------------------------------------------------------------------------
# Green field rules (source: Trip Records Data Dictionary.pdf, LPEP)
# ---------------------------------------------------------------------------

GREEN_FIELD_RULES: dict[str, FieldRule] = {
    "vendor_id": FieldRule(
        "vendor_id",
        "int",
        VENDOR_ID_VALID["green"],
        True,
        False,
        False,
        essential=False,
        retain_in_clean=True,
        invalid_definition="not applicable: no removal rule, values outside {1,2,6} are reported, not removed",
    ),
    "pickup_datetime": FieldRule(
        "pickup_datetime",
        "timestamp",
        None,
        False,
        False,
        False,
        essential=True,
        retain_in_clean=True,
        invalid_definition="NULL, or outside the established study period",
    ),
    "dropoff_datetime": FieldRule(
        "dropoff_datetime",
        "timestamp",
        None,
        True,
        False,
        False,
        essential=False,
        retain_in_clean=True,
        invalid_definition="earlier than pickup_datetime, or resulting trip_duration_minutes <= 0 or > 1440 (24h), when both timestamps are present (per user cleaning policy)",
    ),
    "store_and_fwd_flag": FieldRule(
        "store_and_fwd_flag",
        "string",
        STORE_AND_FWD_FLAG_VALID,
        True,
        False,
        False,
        essential=False,
        retain_in_clean=True,
        invalid_definition="not applicable: no removal rule, values outside {Y,N} are reported",
    ),
    "ratecode_id": FieldRule(
        "ratecode_id",
        "int",
        RATECODE_ID_VALID,
        True,
        False,
        False,
        essential=False,
        retain_in_clean=True,
        invalid_definition="not applicable: no removal rule, values outside the documented set are reported",
    ),
    "pu_location_id": FieldRule(
        "pu_location_id",
        "int",
        None,
        False,
        False,
        False,
        essential=True,
        retain_in_clean=True,
        invalid_definition="NULL, or outside the valid TLC Taxi Zone ID range",
    ),
    "do_location_id": FieldRule(
        "do_location_id",
        "int",
        None,
        True,
        False,
        False,
        essential=False,
        retain_in_clean=True,
        invalid_definition="outside the valid TLC Taxi Zone ID range (removed per user cleaning policy; not an essential field, so NULL is still retained)",
    ),
    "passenger_count": FieldRule(
        "passenger_count",
        "int",
        None,
        True,
        True,
        False,
        essential=False,
        retain_in_clean=True,
        invalid_definition="negative (a physically impossible count); 0 is retained (documented as ambiguous)",
    ),
    "trip_distance": FieldRule(
        "trip_distance",
        "double",
        None,
        True,
        True,
        False,
        essential=False,
        retain_in_clean=True,
        invalid_definition="<= 0 (negative is physically impossible; zero dropped per user cleaning policy); large values retained and flagged only",
    ),
    "fare_amount": FieldRule(
        "fare_amount",
        "double",
        None,
        True,
        True,
        True,
        essential=False,
        retain_in_clean=True,
        invalid_definition="negative (removed per user cleaning policy; other monetary fields are still retained/reported only, as negative extra/tolls/tax etc. may be legitimate refunds or adjustments)",
    ),
    "extra": FieldRule(
        "extra",
        "double",
        None,
        True,
        True,
        True,
        essential=False,
        retain_in_clean=True,
        invalid_definition="not applicable: reported only",
    ),
    "mta_tax": FieldRule(
        "mta_tax",
        "double",
        None,
        True,
        True,
        True,
        essential=False,
        retain_in_clean=True,
        invalid_definition="not applicable: reported only",
    ),
    "tip_amount": FieldRule(
        "tip_amount",
        "double",
        None,
        True,
        True,
        True,
        essential=False,
        retain_in_clean=True,
        invalid_definition="not applicable: reported only (cash tips are legitimately unrecorded, i.e. 0)",
    ),
    "tolls_amount": FieldRule(
        "tolls_amount",
        "double",
        None,
        True,
        True,
        True,
        essential=False,
        retain_in_clean=True,
        invalid_definition="not applicable: reported only",
    ),
    "ehail_fee": FieldRule(
        "ehail_fee",
        "double",
        None,
        True,
        True,
        True,
        essential=False,
        retain_in_clean=True,
        invalid_definition="not applicable: this field is effectively deprecated in the source data; reported only",
    ),
    "improvement_surcharge": FieldRule(
        "improvement_surcharge",
        "double",
        None,
        True,
        True,
        True,
        essential=False,
        retain_in_clean=True,
        invalid_definition="not applicable: reported only",
    ),
    "total_amount": FieldRule(
        "total_amount",
        "double",
        None,
        True,
        True,
        True,
        essential=False,
        retain_in_clean=True,
        invalid_definition="not applicable: reported only",
    ),
    "payment_type": FieldRule(
        "payment_type",
        "int",
        PAYMENT_TYPE_VALID,
        True,
        True,
        False,
        essential=False,
        retain_in_clean=True,
        invalid_definition="not applicable: no removal rule, values outside the documented set are reported",
    ),
    "trip_type": FieldRule(
        "trip_type",
        "int",
        TRIP_TYPE_VALID,
        True,
        False,
        False,
        essential=False,
        retain_in_clean=True,
        invalid_definition="not applicable: no removal rule, values outside {1,2} are reported",
    ),
    "congestion_surcharge": FieldRule(
        "congestion_surcharge",
        "double",
        None,
        True,
        True,
        True,
        essential=False,
        retain_in_clean=True,
        invalid_definition="not applicable: reported only",
    ),
    "cbd_congestion_fee": FieldRule(
        "cbd_congestion_fee",
        "double",
        None,
        True,
        True,
        True,
        essential=False,
        retain_in_clean=True,
        invalid_definition="not applicable: NULL before Jan 2025 is structural, not missing; reported only",
    ),
}

FIELD_RULES: dict[str, dict[str, FieldRule]] = {
    "yellow": YELLOW_FIELD_RULES,
    "green": GREEN_FIELD_RULES,
}

# ---------------------------------------------------------------------------
# Fields that are ESSENTIAL for zone-hour pickup-demand analysis. Missing or
# invalid values here make an observation unusable for the research question
# and are the only grounds (together with the demonstrably-invalid rules
# below) for row removal.
# ---------------------------------------------------------------------------

ESSENTIAL_FIELDS = ("pickup_datetime", "pu_location_id")

# ---------------------------------------------------------------------------
# Pickup/dropoff column renames to the common Clean schema (per Step 5.1).
# ---------------------------------------------------------------------------

PICKUP_DROPOFF_RENAMES = {
    "yellow": {
        "tpep_pickup_datetime": "pickup_datetime",
        "tpep_dropoff_datetime": "dropoff_datetime",
    },
    "green": {
        "lpep_pickup_datetime": "pickup_datetime",
        "lpep_dropoff_datetime": "dropoff_datetime",
    },
}

# Fields that exist conceptually for only one taxi type. They are filled with
# NULL for the other type in the unified Clean schema -- this NULL is
# structural ("this concept doesn't apply to this service"), not missing
# data, and is documented as such rather than being reported as a data-
# quality problem.
STRUCTURALLY_ONE_SIDED_FIELDS = {
    "airport_fee": "yellow",  # Yellow only; airport pickup fee doesn't apply to Green
    "ehail_fee": "green",  # Green only; e-hail dispatch fee doesn't apply to Yellow
    "trip_type": "green",  # Green only; Yellow trips are definitionally street-hail
}

# Unified Clean schema column order (business columns; provenance columns
# from Raw -- source_dataset, source_file, source_year, source_month,
# ingestion_timestamp -- are carried through unchanged and appended after).
CLEAN_SCHEMA_COLUMNS = [
    "taxi_type",
    "vendor_id",
    "pickup_datetime",
    "dropoff_datetime",
    "trip_duration_minutes",
    "passenger_count",
    "trip_distance",
    "ratecode_id",
    "store_and_fwd_flag",
    "pu_location_id",
    "do_location_id",
    "payment_type",
    "trip_type",
    "fare_amount",
    "extra",
    "mta_tax",
    "tip_amount",
    "tolls_amount",
    "ehail_fee",
    "improvement_surcharge",
    "total_amount",
    "congestion_surcharge",
    "cbd_congestion_fee",
    "airport_fee",
]

# ---------------------------------------------------------------------------
# Parameters NOT derived from the data dictionaries -- flagged explicitly.
# ---------------------------------------------------------------------------

# Validity here is checked against the TLC's documented zone-ID range
# rather than an exact zone-by-zone match against the lookup file
# (`resources/Geospatial/Taxi Zone Lookup.csv`, `src/config.py::TAXI_ZONE_LOOKUP_CSV`,
# used elsewhere for zone names/boroughs/geometry): both data dictionaries
# describe PULocationID/DOLocationID as "TLC Taxi Zone" identifiers from the
# TLC's fixed, published zone scheme, publicly documented as running from 1
# to 263 real zones plus 264 ("Unknown") and 265 ("N/A / outside NYC").
VALID_LOCATION_ID_RANGE = (1, 265)
UNKNOWN_NA_ZONE_IDS = (264, 265)

# Training period: 2024-2025; test period: 2026 (see
# `src/processed.py::TRAIN_PERIOD_END_EXCLUSIVE`, which reuses this same
# boundary). The study period below matches the Landing/Raw data actually
# acquired (Jan 2024 - May 2026 inclusive).
STUDY_PERIOD_START = "2024-01-01 00:00:00"
STUDY_PERIOD_END_EXCLUSIVE = (
    "2026-06-01 00:00:00"  # exclusive upper bound (through May 2026 inclusive)
)

# Threshold used to *flag* (never remove) statistically extreme trip
# distance/duration values, computed per taxi_type from the data itself
# rather than an invented absolute cutoff (see Step 13/15).
#
# A fixed high percentile (e.g. P99.9) was tried first and rejected: at
# ~109M Yellow rows, trip_distance's extreme tail is dominated by a small
# number of severely corrupted values (max observed: 398,608 "miles"), and
# Spark's approxQuantile (bounded relative error, necessary at this scale)
# collapsed P99.9 to essentially the dataset max -- so the flag caught
# ~0 rows, defeating its purpose.
#
# Replaced with the IQR outlier rule from MAST30034 Applied Data Science,
# Lecture 3 "Data Cleaning" (Dr. Liam Hodgkinson, University of Melbourne,
# 13 Aug 2026), slide 13 "Outlier Detection":
#
#   flag if more than k * IQR away from the nearest quartile, where
#   k = 1.5                  if N <= 100
#   k = sqrt(log(N)) - 0.5   if N > 100
#
# (log = natural log, matching the same slide's Z-score rule
# |Z| > sqrt(2 log N), the standard extreme-value-theory bound on the
# expected maximum of N iid standard normals.) N here is the taxi type's
# own Clean row count (always > 100), so this always uses the second
# branch, computed fresh per taxi type -- e.g. for Yellow's ~109M rows this
# works out to k ~= 3.8, for Green's ~1.46M rows k ~= 3.3.


def iqr_outlier_multiplier(n: int) -> float:
    """k in `Q3 + k*IQR` / `Q1 - k*IQR`, per the Lecture 3 slide 13 IQR rule."""
    import math

    if n > 100:
        return math.sqrt(math.log(n)) - 0.5
    return 1.5
