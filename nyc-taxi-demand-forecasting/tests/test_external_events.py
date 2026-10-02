import pandas as pd

from src.external_events import build_location_keys, clean_events, extract_location_key


def test_extract_location_key_park_style():
    assert extract_location_key("Central Park: Great Lawn-01") == "Central Park"


def test_extract_location_key_street_segment_style():
    assert extract_location_key("62 STREET between 13 AVENUE and 14 AVENUE") == "62 STREET"


def test_extract_location_key_no_colon_or_between():
    assert extract_location_key("Astoria Park") == "Astoria Park"


def test_extract_location_key_missing():
    assert extract_location_key(None) == ""
    assert extract_location_key("") == ""


def test_build_location_keys_deduplicates_and_counts():
    df = pd.DataFrame(
        {
            "event_location": [
                "Central Park: Soccer-01",
                "Central Park: Soccer-02",
                "Astoria Park: Track-01",
            ],
            "event_borough": ["Manhattan", "Manhattan", "Queens"],
        }
    )
    keys = build_location_keys(df)
    assert set(keys["location_key"]) == {"Central Park", "Astoria Park"}
    central = keys[keys["location_key"] == "Central Park"].iloc[0]
    assert central["event_count"] == 2


def _events_df():
    return pd.DataFrame(
        {
            "event_id": ["1", "2", "3"],
            "event_name": ["A", "B", None],
            "start_date_time": [
                "2024-01-01T08:00:00.000",
                "2024-01-02T08:00:00.000",
                "2024-01-03T08:00:00.000",
            ],
            "end_date_time": [
                "2024-01-01T20:00:00.000",
                "2024-01-02T06:00:00.000",  # ends before it starts -> invalid
                "2024-01-03T20:00:00.000",
            ],
            "event_borough": ["Manhattan", "Brooklyn", "Queens"],
            "event_location": ["Central Park", "Prospect Park", "Astoria Park"],
        }
    )


def test_clean_events_removes_only_end_before_start():
    clean, report = clean_events(_events_df())
    assert len(clean) == 2
    assert set(clean["event_id"]) == {"1", "3"}
    assert report["removal_ledger"][0]["records_removed"] == 1


def test_clean_events_retains_missing_name_not_removed():
    clean, _ = clean_events(_events_df())
    # event_id 3 has a missing event_name and must still be present
    assert "3" in set(clean["event_id"])
