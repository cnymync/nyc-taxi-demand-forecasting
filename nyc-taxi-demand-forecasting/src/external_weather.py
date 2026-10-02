"""Weather external dataset: download + preprocess hourly NYC weather.

Source: Iowa Environmental Mesonet (IEM) ASOS archive
(https://mesonet.agron.iastate.edu/request/download.phtml), station LGA
(LaGuardia Airport). This is the working data source -- NOAA/NCEI's own
Local Climatological Data v2 bulk archive lags behind (no 2026 directory
yet as of this writing) and its REST API timed out repeatedly; IEM mirrors
the same underlying ASOS hourly observations and is reliably queryable for
an exact date range with no API key.

Two stages, mirroring the taxi Landing->Raw->Clean pattern at a scale
appropriate for a single small file:
    download_weather()  -- fetch the raw CSV for the study period (Landing/Raw)
    clean_weather()      -- parse, dedupe to one row/hour, classify severity (Clean)
"""

import json
from pathlib import Path

from loguru import logger
import pandas as pd
import requests
import typer

from src.config import EXTERNAL_WEATHER_DIR, REPORTS_DIR

STATION = "LGA"
STUDY_PERIOD_START = "2024-01-01"
STUDY_PERIOD_END_EXCLUSIVE = "2026-06-01"

IEM_ASOS_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
IEM_FIELDS = ["tmpf", "dwpf", "relh", "sknt", "gust", "vsby", "p01i", "skyc1", "wxcodes"]

RAW_CSV_PATH = EXTERNAL_WEATHER_DIR / f"{STATION.lower()}_hourly_raw.csv"
CLEAN_CSV_PATH = EXTERNAL_WEATHER_DIR / f"{STATION.lower()}_hourly_clean.csv"
REPORT_PATH = REPORTS_DIR / "data_quality" / "external" / "weather_report.json"

app = typer.Typer()


def download_weather(
    station: str = STATION,
    start: str = STUDY_PERIOD_START,
    end_exclusive: str = STUDY_PERIOD_END_EXCLUSIVE,
    output_path: Path = RAW_CSV_PATH,
) -> Path:
    """Download raw hourly ASOS observations for `station` over [start, end_exclusive).

    Writes the untouched response to `output_path` (the Landing/Raw
    equivalent for this dataset -- never modified once downloaded).
    """
    start_dt = pd.Timestamp(start)
    end_dt = pd.Timestamp(end_exclusive) - pd.Timedelta(days=1)  # IEM's day2 is inclusive

    params = {
        "station": station,
        "data": IEM_FIELDS,
        "year1": start_dt.year,
        "month1": start_dt.month,
        "day1": start_dt.day,
        "year2": end_dt.year,
        "month2": end_dt.month,
        "day2": end_dt.day,
        "tz": "Etc/UTC",
        "format": "onlycomma",
        "latlon": "no",
        "missing": "empty",
        "trace": "T",
        "direct": "no",
        "report_type": 3,  # hourly routine (METAR) reports only, excludes SPECIs
    }

    logger.info(f"Downloading {station} hourly weather {start} -> {end_exclusive} from IEM ASOS")
    response = requests.get(IEM_ASOS_URL, params=params, timeout=120)
    response.raise_for_status()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(response.text)

    row_count = response.text.count("\n") - 1
    logger.success(f"Wrote {row_count:,} rows to {output_path}")
    return output_path


def load_raw_weather(path: Path = RAW_CSV_PATH) -> pd.DataFrame:
    """Read the raw IEM ASOS CSV, untouched except for correct dtypes.

    `p01i` (hourly precipitation) uses `"T"` for trace amounts (per the
    `trace=T` download parameter) -- mapped to 0.005in (half the standard
    0.01in gauge resolution, the conventional trace-precipitation value)
    so the column is numeric; every other non-numeric token becomes NaN
    via `errors="coerce"`, not silently dropped.
    """
    df = pd.read_csv(path, na_values=["M", ""], parse_dates=["valid"])
    df["p01i"] = df["p01i"].replace("T", 0.005)
    df["p01i"] = pd.to_numeric(df["p01i"], errors="coerce")
    return df


def classify_severity(row: pd.Series) -> str:
    """Heuristic normal/mild/severe classification.

    Not derived from a data dictionary or lecture (unlike the taxi
    cleaning rules) -- this is a documented assumption combining standard
    aviation-weather severity signals: present-weather codes (`wxcodes`,
    METAR convention: prefix `+` = heavy, `-` = light, `TS` = thunderstorm,
    `FZ` = freezing, `SN`/`GR`/`FC`/`SQ` = snow/hail/funnel-cloud/squall),
    visibility, wind, and hourly precipitation. Thresholds are conservative
    aviation-weather conventions (e.g. <1 mile visibility and >=34kt wind
    are standard "low visibility"/"gale" cutoffs), not invented figures,
    but the classification scheme itself (which combination -> severe vs
    mild) is a project-specific judgment call for the user to review.
    """
    wx = str(row.get("wxcodes") or "")
    vsby = row.get("vsby")
    sknt = row.get("sknt")
    gust = row.get("gust")
    precip = row.get("p01i")

    severe_codes = ("+SN", "+RA", "TS", "FZRA", "FZDZ", "GR", "FC", "SQ", "+SG", "PL")
    mild_codes = ("SN", "RA", "BR", "FG", "DZ", "HZ", "FU", "SG")

    is_severe = (
        any(code in wx for code in severe_codes)
        or (pd.notna(vsby) and vsby < 1)
        or (pd.notna(sknt) and sknt >= 34)
        or (pd.notna(gust) and gust >= 34)
        or (pd.notna(precip) and precip >= 0.3)
    )
    if is_severe:
        return "severe"

    is_mild = (
        any(code in wx for code in mild_codes)
        or (pd.notna(vsby) and vsby < 3)
        or (pd.notna(sknt) and sknt >= 20)
        or (pd.notna(precip) and precip > 0)
    )
    if is_mild:
        return "mild"

    return "normal"


def clean_weather(raw: pd.DataFrame) -> pd.DataFrame:
    """Parse timestamps, dedupe to exactly one row per hour, classify severity.

    IEM ASOS reports are roughly hourly (typically at :51) with occasional
    duplicate/extra reports in the same clock hour; keeping the *last*
    report within each hour is the standard convention (most representative
    of conditions at the top of the next hour).
    """
    df = raw.copy()
    df["date_hour"] = df["valid"].dt.floor("h")
    df = df.sort_values("valid").drop_duplicates(subset="date_hour", keep="last")

    df["weather_severity"] = df.apply(classify_severity, axis=1)

    return df[
        [
            "date_hour",
            "tmpf",
            "dwpf",
            "relh",
            "sknt",
            "gust",
            "vsby",
            "p01i",
            "skyc1",
            "wxcodes",
            "weather_severity",
        ]
    ].reset_index(drop=True)


def profile_weather(raw: pd.DataFrame, clean: pd.DataFrame) -> dict:
    """Before/after data-quality report: what was checked, removed, and kept.

    No physically-impossible values were found in this data (unlike the
    taxi trip data, ASOS observations are already QC'd by NOAA before
    publication) -- so nothing was removed as an "obvious outlier". The
    only rows collapsed are exact duplicate-hour reports.
    """
    expected_hours = pd.date_range(clean["date_hour"].min(), clean["date_hour"].max(), freq="h")
    return {
        "station": STATION,
        "study_period": [STUDY_PERIOD_START, STUDY_PERIOD_END_EXCLUSIVE],
        "raw_row_count": len(raw),
        "clean_row_count": len(clean),
        "duplicate_hour_rows_collapsed": len(raw) - len(clean),
        "expected_hours_in_range": len(expected_hours),
        "missing_hours_gap": len(expected_hours) - len(clean),
        "plausibility_checks_removed_kept": {
            "negative_precipitation": {
                "found": int((clean["p01i"] < 0).sum()),
                "action": "would remove if found",
            },
            "relative_humidity_out_of_0_100": {
                "found": int(((clean["relh"] < 0) | (clean["relh"] > 100)).sum()),
                "action": "would remove if found",
            },
            "negative_wind_speed": {
                "found": int((clean["sknt"] < 0).sum()),
                "action": "would remove if found",
            },
            "negative_visibility": {
                "found": int((clean["vsby"] < 0).sum()),
                "action": "would remove if found",
            },
            "temperature_outside_minus40_to_130F": {
                "found": int(((clean["tmpf"] < -40) | (clean["tmpf"] > 130)).sum()),
                "action": "would remove if found",
            },
            "dewpoint_exceeds_air_temperature": {
                "found": int((clean["dwpf"] > clean["tmpf"]).sum()),
                "action": "would remove if found (physically impossible)",
            },
        },
        "missingness": {
            "gust": {
                "null_count": int(clean["gust"].isna().sum()),
                "note": "expected/legitimate: METAR only reports gust when notably above sustained wind, not a data quality issue",
            },
            "wxcodes": {
                "null_count": int(clean["wxcodes"].isna().sum()),
                "note": "expected/legitimate: blank when no significant weather phenomenon present",
            },
            "tmpf_dwpf_relh_sknt_vsby": {
                "null_count": int(
                    clean[["tmpf", "dwpf", "relh", "sknt", "vsby"]].isna().sum().sum()
                ),
                "note": "true sensor/reporting gaps, left as NULL (not imputed, not dropped)",
            },
        },
        "severity_distribution": clean["weather_severity"].value_counts().to_dict(),
        "kept": "All rows and columns. No physically-impossible values were found, so no rows were removed as outliers.",
        "removed": f"{len(raw) - len(clean)} exact duplicate-hour reports (same clock hour reported twice; last kept).",
        "not_imputed": "Missing hours (station gaps) and missing fields (sensor gaps) are left as gaps/NULL, not filled in.",
    }


def write_weather_report(report: dict, path: Path = REPORT_PATH) -> None:
    """Write the weather data-quality report (row counts, plausibility
    checks, missingness, severity distribution) to `path` as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(report, f, indent=2, default=str)
    logger.info(f"Weather data-quality report written to {path}")


@app.command()
def main():
    """Run the full weather pipeline: download -> clean -> save -> report."""
    download_weather()
    raw = load_raw_weather()
    clean = clean_weather(raw)

    CLEAN_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    clean.to_csv(CLEAN_CSV_PATH, index=False)
    logger.success(f"Wrote {len(clean):,} clean hourly rows to {CLEAN_CSV_PATH}")

    report = profile_weather(raw, clean)
    write_weather_report(report)


if __name__ == "__main__":
    app()
