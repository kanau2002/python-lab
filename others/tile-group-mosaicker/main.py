from pathlib import Path
import re
import numpy as np
import rasterio
from rasterio.transform import from_bounds
import math

def parse_tile_filename(filename: str):
    match = re.match(r'tile_z(\d+)_x(\d+)_y(\d+)\.tif', filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None

def tile_to_bounds(x: int, y: int, zoom: int):
    n = 2.0 ** zoom
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return (west, south, east, north)

def create_mosaic_group(tiles_dict: dict, base_x: int, base_y: int, zoom: int, output_path: Path, group_size: int = 32, tile_size: int = 154):
    output_size = group_size * tile_size
    mosaic = np.zeros((output_size, output_size, 4), dtype=np.uint8)
    tile_count = 0
    
    for i in range(group_size):
        for j in range(group_size):
            tile_x = base_x + i
            tile_y = base_y + j
            
            if (tile_x, tile_y) in tiles_dict:
                tile_count += 1
                with rasterio.open(tiles_dict[(tile_x, tile_y)]) as src:
                    tile_data = np.transpose(src.read(), (1, 2, 0))
                    tile_rgba = np.dstack([tile_data, np.full((tile_size, tile_size), 255, dtype=np.uint8)])
                    
                    y_start = j * tile_size
                    x_start = i * tile_size
                    mosaic[y_start:y_start+tile_size, x_start:x_start+tile_size] = tile_rgba
    
    if tile_count == 0:
        return False
    
    min_west, min_south, _, _ = tile_to_bounds(base_x, base_y + group_size - 1, zoom)
    _, _, max_east, max_north = tile_to_bounds(base_x + group_size - 1, base_y, zoom)
    transform = from_bounds(min_west, min_south, max_east, max_north, output_size, output_size)
    
    with rasterio.open(output_path, 'w', driver='GTiff', height=output_size, width=output_size, 
                       count=4, dtype=np.uint8, crs='EPSG:4326', transform=transform) as dst:
        for i in range(4):
            dst.write(mosaic[:, :, i], i + 1)
    
    return True

def main():
    input_dir = Path(__file__).parent / "input"
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tiles_dict = {}
    zoom_level = None
    
    for tile_file in input_dir.glob("tile_z*.tif"):
        parsed = parse_tile_filename(tile_file.name)
        if parsed:
            z, x, y = parsed
            tiles_dict[(x, y)] = tile_file
            if zoom_level is None:
                zoom_level = z
    
    x_coords = [x for x, y in tiles_dict.keys()]
    y_coords = [y for x, y in tiles_dict.keys()]
    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)
    
    tile_size = 155  # タイルのサイズ（ピクセル）
    group_size = 35  # 35x35 tiles = 5390x5390px
    step_size = 33   # Step by 33 tiles to create 1-tile overlap
    x_groups = (max_x - min_x + step_size) // step_size
    y_groups = (max_y - min_y + step_size) // step_size
    
    group_id = 0
    for gx in range(x_groups):
        for gy in range(y_groups):
            base_x = min_x - 1 + gx * step_size  # Start 1 tile before
            base_y = min_y - 1 + gy * step_size  # Start 1 tile before
            output_path = output_dir / f"mosaic_group_{group_id:03d}_x{base_x}_y{base_y}.tif"
            if create_mosaic_group(tiles_dict, base_x, base_y, zoom_level, output_path, group_size, tile_size):
                group_id += 1

if __name__ == "__main__":
    main()