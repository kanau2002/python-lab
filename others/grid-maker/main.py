import math
from pathlib import Path

import geopandas as gpd
from shapely.geometry import box

CITY = "酒々井町"
ZOOM = 20
STEP_SIZE = 33

INPUT_BOUNDARY = Path(__file__).parent / "input_boundary" / "N03-20250101_12.shp"
OUTPUT_DIR = Path(__file__).parent / "output"


def lon_to_tile_x(lon: float, zoom: int) -> int:
    return int((lon + 180.0) / 360.0 * (2**zoom))


def lat_to_tile_y(lat: float, zoom: int) -> int:
    lat_rad = math.radians(lat)
    return int((1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * (2**zoom))


def tile_to_bounds(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
    n = 2.0**zoom
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return west, south, east, north


def main() -> None:
    gdf = gpd.read_file(INPUT_BOUNDARY)
    city_gdf = gdf[gdf["N03_004"] == CITY]
    if city_gdf.empty:
        raise ValueError(f"City not found in shapefile: {CITY}")

    city_union = city_gdf.dissolve().geometry.iloc[0]
    minx, miny, maxx, maxy = city_union.bounds

    min_tx = lon_to_tile_x(minx, ZOOM)
    max_tx = lon_to_tile_x(maxx, ZOOM)
    min_ty = lat_to_tile_y(maxy, ZOOM)
    max_ty = lat_to_tile_y(miny, ZOOM)

    base_x0 = min_tx - 1
    base_y0 = min_ty - 1

    x_groups = (max_tx - min_tx + STEP_SIZE) // STEP_SIZE
    y_groups = (max_ty - min_ty + STEP_SIZE) // STEP_SIZE

    records = []
    group_id = 0
    for gx in range(x_groups):
        for gy in range(y_groups):
            base_x = base_x0 + gx * STEP_SIZE
            base_y = base_y0 + gy * STEP_SIZE

            west, south, _, _ = tile_to_bounds(base_x, base_y + STEP_SIZE - 1, ZOOM)
            _, _, east, north = tile_to_bounds(base_x + STEP_SIZE - 1, base_y, ZOOM)

            cell = box(west, south, east, north)
            if city_union.intersects(cell):
                records.append({
                    "group_id": group_id,
                    "base_x": base_x,
                    "base_y": base_y,
                    "geometry": cell,
                })
            group_id += 1

    result_gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")

    OUTPUT_DIR.mkdir(exist_ok=True)
    output_path = OUTPUT_DIR / f"{CITY}_groups_step{STEP_SIZE}.geojson"
    result_gdf.to_file(output_path, driver="GeoJSON")
    print(f"Saved {len(result_gdf)} groups → {output_path}")


if __name__ == "__main__":
    main()
