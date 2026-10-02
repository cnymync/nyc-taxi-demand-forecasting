import pandas as pd
import pytest

from src.external_weather import classify_severity, clean_weather


def _row(**overrides):
    base = {"wxcodes": None, "vsby": 10.0, "sknt": 5.0, "gust": None, "p01i": 0.0}
    base.update(overrides)
    return pd.Series(base)


def test_classify_severity_normal_by_default():
    assert classify_severity(_row()) == "normal"


@pytest.mark.parametrize(
    "overrides",
    [
        {"wxcodes": "+SN"},
        {"wxcodes": "TS"},
        {"vsby": 0.5},
        {"sknt": 40.0},
        {"gust": 40.0},
        {"p01i": 0.5},
    ],
)
def test_classify_severity_severe_cases(overrides):
    assert classify_severity(_row(**overrides)) == "severe"


@pytest.mark.parametrize(
    "overrides",
    [
        {"wxcodes": "RA"},
        {"wxcodes": "BR"},
        {"vsby": 2.0},
        {"sknt": 25.0},
        {"p01i": 0.05},
    ],
)
def test_classify_severity_mild_cases(overrides):
    assert classify_severity(_row(**overrides)) == "mild"


def test_classify_severity_missing_values_do_not_crash():
    row = pd.Series({"wxcodes": None, "vsby": None, "sknt": None, "gust": None, "p01i": None})
    assert classify_severity(row) == "normal"


def test_clean_weather_dedupes_to_one_row_per_hour():
    raw = pd.DataFrame(
        {
            "valid": pd.to_datetime(["2024-01-01 05:51", "2024-01-01 05:53", "2024-01-01 06:51"]),
            "tmpf": [40.0, 41.0, 42.0],
            "dwpf": [30.0, 30.0, 30.0],
            "relh": [50.0, 50.0, 50.0],
            "sknt": [5.0, 5.0, 5.0],
            "gust": [None, None, None],
            "vsby": [10.0, 10.0, 10.0],
            "p01i": [0.0, 0.0, 0.0],
            "skyc1": ["CLR", "CLR", "CLR"],
            "wxcodes": [None, None, None],
        }
    )
    clean = clean_weather(raw)
    assert len(clean) == 2  # the two 05:51/05:53 reports collapse to one hour
    # last report within the hour is kept (05:53 -> tmpf 41.0), per the documented convention
    assert (
        clean.loc[clean["date_hour"] == pd.Timestamp("2024-01-01 05:00:00"), "tmpf"].iloc[0]
        == 41.0
    )


def test_clean_weather_preserves_missing_values_as_nan():
    raw = pd.DataFrame(
        {
            "valid": pd.to_datetime(["2024-01-01 00:51"]),
            "tmpf": [40.0],
            "dwpf": [30.0],
            "relh": [50.0],
            "sknt": [5.0],
            "gust": [None],
            "vsby": [10.0],
            "p01i": [0.0],
            "skyc1": ["CLR"],
            "wxcodes": [None],
        }
    )
    clean = clean_weather(raw)
    assert pd.isna(clean.loc[0, "gust"])
    assert pd.isna(clean.loc[0, "wxcodes"])
