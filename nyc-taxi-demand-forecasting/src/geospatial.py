"""Geospatial helpers: taxi zone shapefile loading and choropleth data prep.

Used by notebooks `04_processed_and_features.ipynb` (Figure 1, baseline
deviation static vs. rolling) and `05_pre_model_figures_and_statistics.ipynb`
(Figure 2b, mean demand per zone) to join a per-zone aggregate onto the TLC
taxi zone shapefile for plotting.
"""

from pathlib import Path

import geopandas as gpd
import pandas as pd

from src.config import PROCESSED_DATA_DIR, TAXI_ZONE_LOOKUP_CSV, TAXI_ZONES_SHAPEFILE


def load_taxi_zones() -> gpd.GeoDataFrame:
    """Load the TLC taxi zone shapefile, joined with the zone lookup CSV.

    Reprojects to WGS84 (lat/long) for standard plotting. The shapefile has
    263 zones (LocationIDs 1-265 minus the two non-geographic placeholders,
    264 "Unknown" and 265 "N/A", which have no geometry).
    """
    zones_geom = gpd.read_file(TAXI_ZONES_SHAPEFILE).to_crs("EPSG:4326")
    zones_lookup = pd.read_csv(TAXI_ZONE_LOOKUP_CSV)
    return gpd.GeoDataFrame(
        zones_geom.merge(zones_lookup, on="LocationID", how="left", suffixes=("", "_lookup"))
    )


def load_processed_zone_hour(
    processed_dir: Path = PROCESSED_DATA_DIR,
) -> pd.DataFrame:
    """Read the Processed zone-hour table (already at a grain and size --
    ~11M rows -- suited to pandas directly, no Spark session needed)."""
    return pd.read_parquet(processed_dir / "pickups_zone_hour.parquet")


def build_zone_choropleth_data(
    agg_df: pd.DataFrame, zones: gpd.GeoDataFrame, value_col: str
) -> gpd.GeoDataFrame:
    """Left-join zone geometry with a per-zone aggregate (mean demand or mean
    baseline deviation). Does NOT fill missing values with 0 -- a zone
    absent from `agg_df` here would mean it's missing from the Processed
    table's zero-demand grid entirely, which would itself be a bug worth
    surfacing, not silently zeroing out.
    """
    merged = zones[["LocationID", "zone", "borough", "geometry"]].merge(
        agg_df, left_on="LocationID", right_on="pu_location_id", how="left"
    )
    missing = merged[value_col].isna().sum()
    if missing:
        raise ValueError(
            f"{missing} zone(s) have no Processed-table row for '{value_col}' -- "
            "expected every real geographic zone to be present in the zero-demand grid."
        )
    return gpd.GeoDataFrame(merged, geometry="geometry", crs=zones.crs)
