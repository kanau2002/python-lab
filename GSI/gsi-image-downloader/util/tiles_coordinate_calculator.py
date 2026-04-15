from pathlib import Path
import math
import geopandas as gpd
from shapely.geometry import box

def lat_lon_to_tile(lat: float, lon: float, zoom: int):
    n = 2.0 ** zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return (x, y)

def tile_to_bounds(x: int, y: int, zoom: int):
    """タイルの境界ボックスを取得（west, south, east, north）"""
    n = 2.0 ** zoom
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return (west, south, east, north)

def get_tiles_in_city(city_name: str, zoom: int):
    """指定された市の領域内のタイル座標リストを取得"""
    boundary_dir = Path(__file__).parent.parent / "input_boundary"
    boundary_files = list(boundary_dir.glob("**/*.shp")) + list(boundary_dir.glob("**/*.geojson"))

    gdf = gpd.read_file(boundary_files[0])
    city_col = next((col for col in ['N03_004', '市区町村名', 'CITY_NAME', 'name'] if col in gdf.columns), None)
    city_data = gdf[gdf[city_col].str.contains(city_name, na=False)]

    if city_data.empty:
        return []

    polygon = city_data.geometry.unary_union
    min_lon, min_lat, max_lon, max_lat = polygon.bounds

    x_west, y_north = lat_lon_to_tile(max_lat, min_lon, zoom)
    x_east, y_south = lat_lon_to_tile(min_lat, max_lon, zoom)

    tiles = []
    for x in range(x_west, x_east + 1):
        for y in range(y_north, y_south + 1):
            west, south, east, north = tile_to_bounds(x, y, zoom)
            tile_box = box(west, south, east, north)
            if polygon.intersects(tile_box):
                tiles.append((x, y))

    return tiles
