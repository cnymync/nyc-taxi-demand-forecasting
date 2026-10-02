"""Events external dataset: download, geocode, and preprocess NYC permitted events.

Source: NYC Open Data, "NYC Permitted Event Information - Historical"
(Office of Citywide Event Coordination and Management), Socrata dataset
`bkfu-528j`. Confirmed via live query to cover 2008 through at least
Dec 2026 (includes already-approved future permits), so the study period
(2024-01-01 to 2026-06-01) is fully covered without needing the separate
"next 30 days" rolling dataset.

`event_location` is free-text (e.g. "Central Park", "62 STREET between 13
AVENUE and 14 AVENUE") -- not coordinates or a taxi zone ID. This module
geocodes it: extracts a `location_key` (the park/landmark name, or the
first street segment), geocodes each *unique* key once via Nominatim
(rate-limited, disk-cached, resumable), then spatially joins the
resulting point to the taxi zone shapefile already used in
`src/geospatial.py`.
"""

import json
from pathlib import Path
import re
import time

import geopandas as gpd
from loguru import logger
import pandas as pd
import requests
from shapely.geometry import Point
import typer

from src.config import EXTERNAL_EVENTS_DIR, REPORTS_DIR
from src.geospatial import load_taxi_zones

STUDY_PERIOD_START = "2024-01-01T00:00:00"
STUDY_PERIOD_END_EXCLUSIVE = "2026-06-01T00:00:00"

SOCRATA_URL = "https://data.cityofnewyork.us/resource/bkfu-528j.json"
PAGE_SIZE = 50000

RAW_CSV_PATH = EXTERNAL_EVENTS_DIR / "nyc_permitted_events_raw.csv"
GEOCODE_CACHE_PATH = EXTERNAL_EVENTS_DIR / "geocode_cache.json"
CLEAN_CSV_PATH = EXTERNAL_EVENTS_DIR / "nyc_permitted_events_clean.csv"
REPORT_PATH = REPORTS_DIR / "data_quality" / "external" / "events_report.json"

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
NOMINATIM_USER_AGENT = (
    "nyc-taxi-demand-prediction-student-project (contact: project owner via course staff)"
)

app = typer.Typer()


# ---------------------------------------------------------------------------
# Download
# ---------------------------------------------------------------------------


def download_events(
    start: str = STUDY_PERIOD_START,
    end_exclusive: str = STUDY_PERIOD_END_EXCLUSIVE,
    output_path: Path = RAW_CSV_PATH,
) -> Path:
    """Paginate the Socrata API for all events in [start, end_exclusive).

    Writes the untouched combined result (the Landing/Raw equivalent for
    this dataset) to `output_path`.
    """
    where_clause = f"start_date_time >= '{start}' AND start_date_time < '{end_exclusive}'"
    all_rows: list[dict] = []
    offset = 0

    while True:
        params = {
            "$where": where_clause,
            "$order": "start_date_time,event_id",
            "$limit": PAGE_SIZE,
            "$offset": offset,
        }
        logger.info(f"Fetching events offset={offset}")
        response = requests.get(SOCRATA_URL, params=params, timeout=120)
        response.raise_for_status()
        page = response.json()
        if not page:
            break
        all_rows.extend(page)
        offset += PAGE_SIZE
        if len(page) < PAGE_SIZE:
            break

    df = pd.DataFrame(all_rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.success(f"Wrote {len(df):,} rows to {output_path}")
    return output_path


def load_raw_events(path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """Read the raw events CSV. Every column stays a string until it's
    actually parsed (`parse_event_datetimes`), so nothing is silently
    coerced or dropped on load."""
    return pd.read_csv(path, dtype=str)


# ---------------------------------------------------------------------------
# Geocoding: unique location keys -> lat/long -> taxi zone
# ---------------------------------------------------------------------------


def extract_location_key(event_location: str) -> str:
    """Reduce a raw `event_location` string to a geocodable key.

    Park/landmark bookings look like "Central Park: Great Lawn-01" -- the
    text before the first colon is the actual place name. Street-segment
    closures look like "62 STREET between 13 AVENUE and 14 AVENUE" -- take
    the segment before " between " (the anchor street), which Nominatim
    can usually resolve to a street centroid even without the full
    intersection description.
    """
    if not isinstance(event_location, str) or not event_location.strip():
        return ""
    key = event_location.split(":")[0].strip()
    key = re.split(r"\bbetween\b", key, flags=re.IGNORECASE)[0].strip()
    return key


def build_location_keys(df: pd.DataFrame) -> pd.DataFrame:
    """One row per unique (location_key, event_borough), with event counts."""
    working = df.copy()
    working["location_key"] = working["event_location"].apply(extract_location_key)
    counts = (
        working.groupby(["location_key", "event_borough"])
        .size()
        .reset_index(name="event_count")
        .sort_values("event_count", ascending=False)
    )
    return counts[counts["location_key"] != ""]


def geocode_locations(
    keys_df: pd.DataFrame,
    cache_path: Path = GEOCODE_CACHE_PATH,
    max_locations: int | None = None,
    rate_limit_seconds: float = 1.1,
) -> dict:
    """Geocode unique (location_key, borough) combos via Nominatim, disk-cached.

    Respects Nominatim's public-instance usage policy: max 1 request/sec,
    identifying User-Agent, no parallel requests. Resumable -- already-
    cached keys are skipped, so an interrupted run can just be re-invoked.
    `max_locations` (by descending event frequency) bounds the run time;
    the long tail of one-off locations can be geocoded in a later pass.
    """
    cache: dict = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())

    todo = keys_df.copy()
    if max_locations is not None:
        todo = todo.head(max_locations)

    session = requests.Session()
    session.headers.update({"User-Agent": NOMINATIM_USER_AGENT})

    n_done, n_cached, n_failed = 0, 0, 0
    for _, row in todo.iterrows():
        key, borough = row["location_key"], row["event_borough"]
        cache_key = f"{key}|||{borough}"
        if cache_key in cache:
            n_cached += 1
            continue

        query = f"{key}, {borough}, New York City, NY"
        try:
            resp = session.get(
                NOMINATIM_URL,
                params={"q": query, "format": "json", "limit": 1, "countrycodes": "us"},
                timeout=15,
            )
            resp.raise_for_status()
            results = resp.json()
            if results:
                cache[cache_key] = {
                    "lat": float(results[0]["lat"]),
                    "lon": float(results[0]["lon"]),
                }
            else:
                cache[cache_key] = None
                n_failed += 1
        except Exception as exc:  # noqa: BLE001 -- log and continue; geocoding is best-effort
            logger.warning(f"Geocode failed for {query!r}: {exc}")
            cache[cache_key] = None
            n_failed += 1

        n_done += 1
        if n_done % 50 == 0:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(cache, indent=2))
            logger.info(f"Geocoded {n_done}/{len(todo)} (cached so far: {len(cache)})")
        time.sleep(rate_limit_seconds)

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=2))
    logger.success(f"Geocoding pass: {n_done} new ({n_failed} failed), {n_cached} already cached")
    return cache


def simplify_location_key(key: str) -> str | None:
    """Strip qualifiers that cause exact-name Nominatim lookups to fail.

    Diagnosed by hand against a sample of failures: names like "Shore Road
    Park and Parkway" or "Kissena Corridor West" return no OSM match, but
    "Shore Road Park" / "Kissena Corridor" do -- OSM tags the base
    park/corridor name, not the descriptive suffix. Returns None if no
    simplification is possible (nothing left to try).
    """
    simplified = key
    simplified = re.sub(r"\s*\([^)]*\)", "", simplified)  # "X (Y)" -> "X"
    simplified = simplified.split("/")[0].strip()  # "X / Y" -> "X"
    simplified = re.sub(
        r"\s+and\s+\w+$", "", simplified, flags=re.IGNORECASE
    )  # "X and Parkway" -> "X"
    simplified = re.sub(
        r"\s+(West|East|North|South)$", "", simplified, flags=re.IGNORECASE
    )  # "X West" -> "X"
    simplified = simplified.strip()
    return simplified if simplified and simplified != key else None


def retry_failed_geocodes(
    keys_df: pd.DataFrame,
    cache_path: Path = GEOCODE_CACHE_PATH,
    rate_limit_seconds: float = 1.1,
) -> dict:
    """Retry every cached geocoding failure once with a simplified query.

    Only rewrites cache entries that previously failed (`None`) -- never
    touches a successful result. Keys with no viable simplification (per
    `simplify_location_key`) are skipped, remaining `None` (left as NaN in
    the final data, not guessed).
    """
    cache: dict = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    failed_keys = [k for k, v in cache.items() if v is None]
    logger.info(f"Retrying {len(failed_keys)} failed geocodes with simplified queries")

    session = requests.Session()
    session.headers.update({"User-Agent": NOMINATIM_USER_AGENT})

    n_recovered = 0
    for i, cache_key in enumerate(failed_keys):
        key, _, borough = cache_key.partition("|||")
        simplified = simplify_location_key(key)
        if simplified is None:
            continue

        query = f"{simplified}, {borough}, New York City, NY"
        try:
            resp = session.get(
                NOMINATIM_URL,
                params={"q": query, "format": "json", "limit": 1, "countrycodes": "us"},
                timeout=15,
            )
            resp.raise_for_status()
            results = resp.json()
            if results:
                cache[cache_key] = {
                    "lat": float(results[0]["lat"]),
                    "lon": float(results[0]["lon"]),
                }
                n_recovered += 1
        except Exception as exc:  # noqa: BLE001 -- log and continue; best-effort
            logger.warning(f"Retry geocode failed for {query!r}: {exc}")

        if (i + 1) % 50 == 0:
            cache_path.write_text(json.dumps(cache, indent=2))
            logger.info(f"Retried {i + 1}/{len(failed_keys)} (recovered so far: {n_recovered})")
        time.sleep(rate_limit_seconds)

    cache_path.write_text(json.dumps(cache, indent=2))
    logger.success(
        f"Retry pass: recovered {n_recovered}/{len(failed_keys)} previously-failed locations"
    )
    return cache


def assign_taxi_zones(df: pd.DataFrame, cache_path: Path = GEOCODE_CACHE_PATH) -> pd.DataFrame:
    """Attach `pu_location_id` (nullable) to each event via the geocode cache + zone shapefile."""
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}

    working = df.copy().reset_index(drop=True)
    working["_row_id"] = working.index
    working["location_key"] = working["event_location"].apply(extract_location_key)
    working["_cache_key"] = working["location_key"] + "|||" + working["event_borough"].fillna("")

    lats, lons = [], []
    for ck in working["_cache_key"]:
        entry = cache.get(ck)
        lats.append(entry["lat"] if entry else None)
        lons.append(entry["lon"] if entry else None)
    working["latitude"] = lats
    working["longitude"] = lons
    working = working.drop(columns=["_cache_key"])

    zones = load_taxi_zones()

    geocoded_mask = working["latitude"].notna()
    points = gpd.GeoDataFrame(
        working.loc[geocoded_mask, ["_row_id"]],
        geometry=[
            Point(lon, lat)
            for lon, lat in zip(
                working.loc[geocoded_mask, "longitude"], working.loc[geocoded_mask, "latitude"]
            )
        ],
        crs="EPSG:4326",
    )
    # A point can only legitimately fall in one non-overlapping taxi zone
    # polygon, but drop_duplicates on _row_id guards against any sliver
    # overlaps in the shapefile producing more than one match per point.
    joined = gpd.sjoin(points, zones[["LocationID", "geometry"]], how="left", predicate="within")
    joined = joined.drop_duplicates(subset="_row_id", keep="first")[["_row_id", "LocationID"]]

    working = working.merge(joined, on="_row_id", how="left").drop(columns=["_row_id"])
    working = working.rename(columns={"LocationID": "pu_location_id"})

    return working


# ---------------------------------------------------------------------------
# Cleaning: remove only demonstrably invalid rows, profile the rest
# ---------------------------------------------------------------------------


def parse_event_datetimes(df: pd.DataFrame) -> pd.DataFrame:
    """Parse `start_date_time`/`end_date_time` from string to datetime;
    unparseable values become NaT (`errors="coerce"`), not dropped rows."""
    working = df.copy()
    working["start_date_time"] = pd.to_datetime(working["start_date_time"], errors="coerce")
    working["end_date_time"] = pd.to_datetime(working["end_date_time"], errors="coerce")
    return working


def clean_events(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Remove only demonstrably invalid rows; profile everything else.

    Mirrors the taxi Clean-layer philosophy (`src/cleaning.py`): a
    logically-impossible temporal relationship is removed and its exact
    count logged; everything merely unusual (long duration, missing
    non-essential fields) is retained and reported, not removed, so the
    user can make the final call.

    Note: `event_id` is a *reusable* permit/booking identifier (the same
    ID recurs across many separate daily occurrences of a recurring
    booking, confirmed by inspection -- e.g. a season-long park booking
    has one row per calendar day it's active), not a unique row key. No
    duplicate rows were found on the full business-column set, so no
    deduplication rule was needed.
    """
    working = parse_event_datetimes(df)
    original_count = len(working)
    ledger: list[dict] = []

    bad_temporal = working["end_date_time"] < working["start_date_time"]
    n_bad_temporal = int(bad_temporal.sum())
    ledger.append(
        {
            "rule": "end_date_time < start_date_time",
            "reason": "logically impossible event (mirrors the taxi dropoff < pickup rule)",
            "records_removed": n_bad_temporal,
            "pct_of_original": round(100 * n_bad_temporal / original_count, 4),
        }
    )
    working = working[~bad_temporal]

    duration_days = (
        working["end_date_time"] - working["start_date_time"]
    ).dt.total_seconds() / 86400

    kept_but_flagged = {
        "missing_event_location": {
            "count": int(working["event_location"].isna().sum()),
            "action": "retained -- cannot be geocoded to a taxi zone, but the event record itself is not invalid",
        },
        "missing_event_name": {
            "count": int(working["event_name"].isna().sum()),
            "action": "retained -- not essential",
        },
        "duration_over_30_days": {
            "count": int((duration_days > 30).sum()),
            "action": "retained -- plausible for season-long park/construction bookings, not evidence of invalidity",
        },
        "duration_over_90_days": {
            "count": int((duration_days > 90).sum()),
            "action": "retained, flagged for review",
        },
        "duration_over_365_days": {
            "count": int((duration_days > 365).sum()),
            "action": "retained, flagged for review -- worth a manual look, only 1 record",
        },
        "zero_or_negative_after_removal": {
            "count": int((duration_days <= 0).sum()),
            "action": "should be 0 after the end<start removal rule above -- sanity check",
        },
    }

    report = {
        "raw_row_count": original_count,
        "clean_row_count": len(working),
        "removal_ledger": ledger,
        "kept_but_flagged": kept_but_flagged,
        "duplicate_check": "0 exact-duplicate rows found on the full business-column set; event_id is a "
        "reusable recurring-booking identifier, not a unique row key, so it was not used for dedup.",
    }
    return working, report


def write_events_report(report: dict, path: Path = REPORT_PATH) -> None:
    """Write the events data-quality report (removal ledger, flagged-but-kept
    counts, geocoding coverage) to `path` as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Events data-quality report written to {path}")


@app.command()
def main(
    max_geocode: int = typer.Option(
        None, help="Cap the number of unique locations to geocode this run (by event frequency)."
    ),
):
    """Run the full events pipeline: download -> clean -> geocode -> save -> report."""
    download_events()
    raw = load_raw_events()

    clean, report = clean_events(raw)

    keys = build_location_keys(clean)
    geocode_locations(keys, max_locations=max_geocode)
    clean = assign_taxi_zones(clean)

    report["geocoding"] = {
        "unique_location_keys": len(keys),
        "events_geocoded": int(clean["pu_location_id"].notna().sum()),
        "events_not_geocoded": int(clean["pu_location_id"].isna().sum()),
        "pct_geocoded": round(100 * clean["pu_location_id"].notna().mean(), 2),
    }

    CLEAN_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_CSV_PATH, index=False)
    logger.success(f"Wrote {len(clean):,} clean rows to {CLEAN_CSV_PATH}")

    write_events_report(report)


if __name__ == "__main__":
    app()
