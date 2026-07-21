from __future__ import annotations

import geopandas as gpd
from shapely.geometry import shape
from shapely.ops import unary_union

MERGE_BUFFER_DEG = 5e-7  # ≈0.05m: closes 1-2px gaps at tile seams


def run_merge(
    records: list[tuple[dict, str]],
    merge_buffer_deg: float = MERGE_BUFFER_DEG,
) -> gpd.GeoDataFrame:
    if not records:
        return gpd.GeoDataFrame(columns=["geometry", "source_tile"], crs="EPSG:4326")

    geometries = [shape(geom) for geom, _ in records]
    source_tiles = [name for _, name in records]

    gdf = gpd.GeoDataFrame(
        {"source_tile": source_tiles},
        geometry=geometries,
        crs="EPSG:4326",
    )

    print(f"Merging {len(gdf)} polygons across tiles...")
    buffered = gdf.geometry.buffer(merge_buffer_deg)
    merged = unary_union(buffered)
    shrunk = merged.buffer(-merge_buffer_deg)

    gdf_merged = gpd.GeoDataFrame(geometry=[shrunk], crs="EPSG:4326")
    gdf_merged = gdf_merged.explode(index_parts=False).reset_index(drop=True)

    print(f"Merge done: {len(gdf)} → {len(gdf_merged)} polygons")
    return gdf_merged
