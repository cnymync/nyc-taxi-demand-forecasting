import os
from pathlib import Path
import sys

from dotenv import load_dotenv
from loguru import logger

# Pin PySpark's worker subprocesses to the exact same Python interpreter
# running this driver process. Without this, PySpark falls back to
# searching PATH for a `python3` to launch workers with, which can silently
# resolve to a *different* Python installation (Homebrew, pyenv, system
# Python, ...) than the active conda/venv environment -- causing every
# Spark job to fail with PYSPARK_VERSION_MISMATCH the moment it needs a
# Python worker (found 2026-08-27: driver correctly used the conda env's
# Python 3.10, but workers picked up an unrelated Python 3.13 from PATH).
# setdefault(), not direct assignment, so an operator who has deliberately
# set these themselves is never silently overridden.
os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

DATA_DIR = PROJ_ROOT / "data"
LANDING_DATA_DIR = DATA_DIR / "landing"
RAW_DATA_DIR = DATA_DIR / "raw"
CLEAN_DATA_DIR = DATA_DIR / "clean"
INTERIM_DATA_DIR = DATA_DIR / "interim"
CURATED_DATA_DIR = DATA_DIR / "curated"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
EXTERNAL_WEATHER_DIR = EXTERNAL_DATA_DIR / "weather"
EXTERNAL_EVENTS_DIR = EXTERNAL_DATA_DIR / "events"

MODELS_DIR = PROJ_ROOT / "models"

REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RESOURCES_DIR = PROJ_ROOT / "resources"
TAXI_ZONES_SHAPEFILE = RESOURCES_DIR / "Geospatial" / "taxi_zones" / "taxi_zones.shp"
TAXI_ZONE_LOOKUP_CSV = RESOURCES_DIR / "Geospatial" / "Taxi Zone Lookup.csv"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
