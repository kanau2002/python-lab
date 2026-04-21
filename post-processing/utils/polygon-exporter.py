from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import numpy as np
from pyproj import Geod
from shapely.affinity import rotate
from shapely.geometry import MultiPolygon, Polygon

_GEOD = Geod(ellps="WGS84")


def _regularize_polygon(polygon, tol_deg: float = 1e-5):
    try:
        mrr = polygon.minimum_rotated_rectangle
        if not hasattr(mrr, "exterior"):
            return polygon

        mrr_coords = np.array(mrr.exterior.coords[:-1])
        if len(mrr_coords) < 3:
            return polygon

        edges = np.diff(np.vstack([mrr_coords, mrr_coords[:1]]), axis=0)
        longest_edge = edges[np.argmax(np.linalg.norm(edges, axis=1))]
        theta = np.degrees(np.arctan2(longest_edge[1], longest_edge[0]))

        centroid = polygon.centroid
        p_derot = rotate(polygon, -theta, origin=centroid)
        pts = np.array(p_derot.exterior.coords[:-1])

        # 各軸ごとに近い座標値をクラスタリングして同一直線にスナップ
        for axis in (1, 0):
            vals = pts[:, axis].copy()
            visited = np.zeros(len(vals), dtype=bool)
            for i in range(len(vals)):
                if visited[i]:
                    continue
                mask = np.abs(vals - vals[i]) < tol_deg
                pts[mask, axis] = vals[mask].mean()
                visited[mask] = True

        result = Polygon(pts)
        if not result.is_valid:
            result = result.buffer(0)
        if isinstance(result, MultiPolygon):
            result = max(result.geoms, key=lambda g: g.area)
        if result.is_empty:
            return polygon

        p_final = rotate(result, theta, origin=centroid)
        if not p_final.is_valid:
            p_final = p_final.buffer(0)
        if p_final.is_empty:
            return polygon

        return p_final

    except Exception:
        return polygon


def _compute_area_m2(geometry) -> float:
    area, _ = _GEOD.geometry_area_perimeter(geometry)
    return abs(area)


def run_export(
    gdf: gpd.GeoDataFrame,
    output_path: Path,
    min_area_sqm: float = 11.5,
    simplify_tol_deg: float = 2e-6,
    output_format: str = "GPKG",
    regularize: bool = True,
) -> dict[str, int]:
    total_input = len(gdf)

    gdf = gdf.copy()
    gdf["area_m2"] = gdf.geometry.apply(_compute_area_m2)
    gdf = gdf[gdf["area_m2"] >= min_area_sqm].reset_index(drop=True)
    after_filter = len(gdf)

    gdf.geometry = gdf.geometry.simplify(simplify_tol_deg, preserve_topology=True)

    if regularize:
        gdf.geometry = gdf.geometry.apply(
            lambda geom: _regularize_polygon(geom, tol_deg=simplify_tol_deg * 2)
        )

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
