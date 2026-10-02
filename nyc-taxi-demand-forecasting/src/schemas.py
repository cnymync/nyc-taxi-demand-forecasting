"""Canonical Raw-layer schemas for the NYC taxi datasets.

These define the standardised (but not analytically cleaned) column names
and types used in `data/raw/`. They were derived from the TLC Trip Record
data dictionaries (Yellow and Green/LPEP), cross-checked against Landing
files spanning Jan 2024 - May 2026 for both Yellow and Green taxi
(see `notebooks/00_schema_and_raw_layer.ipynb`).

Renaming is scoped to casing/naming consistency *within* each taxi type
(e.g. `VendorID` -> `vendor_id`). Harmonising Yellow (`tpep_*`) and Green
(`lpep_*`) column names against each other happens one stage later, in
`src/cleaning.py`, not a Raw-layer concern.
"""

from pyspark.sql import types as T

YELLOW_COLUMN_RENAMES = {
    "VendorID": "vendor_id",
    "RatecodeID": "ratecode_id",
    "PULocationID": "pu_location_id",
    "DOLocationID": "do_location_id",
    "Airport_fee": "airport_fee",
}

GREEN_COLUMN_RENAMES = {
    "VendorID": "vendor_id",
    "RatecodeID": "ratecode_id",
    "PULocationID": "pu_location_id",
    "DOLocationID": "do_location_id",
}

# `cbd_congestion_fee` was introduced by TLC from Jan 2025 onward (MTA
# Congestion Relief Zone fee). It is declared here as nullable so that
# pre-2025 and post-2025 months share one schema: absent in a given month's
# Landing file simply becomes NULL in Raw, never dropped or invented.
YELLOW_RAW_SCHEMA = T.StructType(
    [
        T.StructField("vendor_id", T.IntegerType(), True),
        T.StructField("tpep_pickup_datetime", T.TimestampType(), True),
        T.StructField("tpep_dropoff_datetime", T.TimestampType(), True),
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
    ]
)

GREEN_RAW_SCHEMA = T.StructType(
    [
        T.StructField("vendor_id", T.IntegerType(), True),
        T.StructField("lpep_pickup_datetime", T.TimestampType(), True),
        T.StructField("lpep_dropoff_datetime", T.TimestampType(), True),
        T.StructField("store_and_fwd_flag", T.StringType(), True),
        T.StructField("ratecode_id", T.IntegerType(), True),
        T.StructField("pu_location_id", T.IntegerType(), True),
        T.StructField("do_location_id", T.IntegerType(), True),
        T.StructField("passenger_count", T.IntegerType(), True),
        T.StructField("trip_distance", T.DoubleType(), True),
        T.StructField("fare_amount", T.DoubleType(), True),
        T.StructField("extra", T.DoubleType(), True),
        T.StructField("mta_tax", T.DoubleType(), True),
        T.StructField("tip_amount", T.DoubleType(), True),
        T.StructField("tolls_amount", T.DoubleType(), True),
        T.StructField("ehail_fee", T.DoubleType(), True),
        T.StructField("improvement_surcharge", T.DoubleType(), True),
        T.StructField("total_amount", T.DoubleType(), True),
        T.StructField("payment_type", T.IntegerType(), True),
        T.StructField("trip_type", T.IntegerType(), True),
        T.StructField("congestion_surcharge", T.DoubleType(), True),
        T.StructField("cbd_congestion_fee", T.DoubleType(), True),
    ]
)

RAW_SCHEMAS = {"yellow": YELLOW_RAW_SCHEMA, "green": GREEN_RAW_SCHEMA}
COLUMN_RENAMES = {"yellow": YELLOW_COLUMN_RENAMES, "green": GREEN_COLUMN_RENAMES}

# Landing filenames use full/mixed month names, e.g. "June", "July", "Sept"
# (not "Jun"/"Jul"/"Sep") -- confirmed against every file in data/landing/.
MONTH_NAME_TO_NUMBER = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "aug": 8,
    "sept": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
