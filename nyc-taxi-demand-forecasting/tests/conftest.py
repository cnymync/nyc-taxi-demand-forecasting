import os
import time

from pyspark.sql import SparkSession
import pytest

# Pinned to UTC (both the OS-level TZ the JVM reads at startup and the Spark
# SQL session timezone) so tests behave the same regardless of the machine's
# own timezone -- see src/processed.py's SparkSession config for the real
# bug (Melbourne DST corrupting an hourly grid) this avoids reproducing.
# Setting spark.sql.session.timeZone alone is not enough: PySpark's
# collect()-to-Python datetime conversion follows the process/JVM's default
# timezone, not the Spark session conf, so TZ must be set before the JVM
# starts.
os.environ["TZ"] = "UTC"
time.tzset()


@pytest.fixture(scope="session")
def spark_session():
    spark = (
        SparkSession.builder.appName("nyc-taxi-tests")
        .master("local[1]")
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("ERROR")
    yield spark
    spark.stop()
