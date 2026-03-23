from pathlib import Path
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from PIL import Image
import math
import re

def tile_to_bounds(x, y, zoom):
    n = 2.0 ** zoom
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return west, south, east, north

def parse_filename(filename):
    match = re.match(r'tile_z(\d+)_x(\d+)_y(\d+)\.tif', filename)
    return int(match.group(1)), int(match.group(2)), int(match.group(3))

def downsample_tile(input_path, output_path, new_size):
    zoom, x, y = parse_filename(input_path.name)
    
    with rasterio.open(input_path) as src:
        img_array = src.read()
        west, south, east, north = tile_to_bounds(x, y, zoom)
        
        img_rgb = np.transpose(img_array, (1, 2, 0))
        resized_array = np.array(Image.fromarray(img_rgb).resize((new_size, new_size), Image.Resampling.LANCZOS))
        
        transform = from_bounds(west, south, east, north, new_size, new_size)
        
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=new_size,
            width=new_size,
            count=3,
            dtype=resized_array.dtype,
            crs='EPSG:4326',
            transform=transform
        ) as dst:
            for i in range(3):
                dst.write(resized_array[:, :, i], i + 1)

def main():
    input_dir = Path(__file__).parent / "input"
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    tile_files = list(input_dir.glob("tile_z*.tif"))
    
    zoom, x, y = parse_filename(tile_files[0].name)
    west, south, east, north = tile_to_bounds(x, y, zoom)
    center_lat = (north + south) / 2
    lon_distance = (east - west) * 111320 * math.cos(math.radians(center_lat))
    lat_distance = (north - south) * 110540
    new_size = int((lon_distance + lat_distance) / 2 / 0.2)
    
    for tile_file in tile_files:
        downsample_tile(tile_file, output_dir / tile_file.name, new_size)

if __name__ == "__main__":
    main()