from __future__ import annotations

from pathlib import Path

import geopandas as gpd
from pyproj import Geod

_GEOD = Geod(ellps="WGS84")


def _compute_area_m2(geometry) -> float:
    area, _ = _GEOD.geometry_area_perimeter(geometry)
    return abs(area)


def run_export(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    min_area_sqm: float = 11.5,
    simplify_tol_deg: float = 2e-6,
    output_format: str = "GPKG",
) -> dict[str, int]:
    total_input = len(gdf)

    gdf = gdf.copy()
    gdf["area_m2"] = gdf.geometry.apply(_compute_area_m2)
    gdf = gdf[gdf["area_m2"] >= min_area_sqm].reset_index(drop=True)
    after_filter = len(gdf)

    gdf.geometry = gdf.geometry.simplify(simplify_tol_deg, preserve_topology=True)

    if "source_tile" not in gdf.columns:
        gdf["source_tile"] = ""

    gdf = gdf[["geometry", "area_m2", "source_tile"]]

    output_path.parent.mkdir(parents=True, exist_ok=True)

    driver = "GPKG" if output_format == "GPKG" else "GeoJSON"
    layer_name = output_path.stem if driver == "GPKG" else None
    write_kwargs: dict = {"driver": driver}
    if layer_name:
        write_kwargs["layer"] = layer_name

    gdf.to_file(output_path, **write_kwargs)

    print(
        f"Export done: input={total_input}, after_filter={after_filter}, "
        f"written={after_filter} → {output_path}"
    )
    return {"total_input": total_input, "after_filter": after_filter, "written": after_filter}
