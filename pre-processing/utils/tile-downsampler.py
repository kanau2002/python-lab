from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_bounds

TILE_FILENAME_PATTERN = re.compile(r"tile_z(\d+)_x(\d+)_y(\d+)\.tif$")

TARGET_RESOLUTION_M = 0.2
WORKERS = 8
LOG_INTERVAL = 1000


def tile_to_bounds(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
	n = 2.0 ** zoom
	west = x / n * 360.0 - 180.0
	east = (x + 1) / n * 360.0 - 180.0
	north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
	south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
	return west, south, east, north


def parse_filename(filename: str) -> tuple[int, int, int]:
	match = TILE_FILENAME_PATTERN.match(filename)
	if not match:
		raise ValueError(f"Invalid tile filename: {filename}")
	return int(match.group(1)), int(match.group(2)), int(match.group(3))


def calculate_target_tile_size_px(tile_file: Path) -> int:
	zoom, x, y = parse_filename(tile_file.name)
	west, south, east, north = tile_to_bounds(x, y, zoom)
	center_lat = (north + south) / 2
	lon_distance = (east - west) * 111320 * math.cos(math.radians(center_lat))
	lat_distance = (north - south) * 110540
	average_ground_distance = (lon_distance + lat_distance) / 2
	return max(1, int(round(average_ground_distance / TARGET_RESOLUTION_M)))


def downsample_tile(input_path: Path, output_path: Path, new_size: int) -> None:
	zoom, x, y = parse_filename(input_path.name)
	west, south, east, north = tile_to_bounds(x, y, zoom)

	with rasterio.open(input_path) as src:
		resized_array = src.read(
			out_shape=(src.count, new_size, new_size),
			resampling=Resampling.lanczos,
		)
		transform = from_bounds(west, south, east, north, new_size, new_size)

		profile = src.profile.copy()
		profile.update(
			{
				"driver": "GTiff",
				"height": new_size,
				"width": new_size,
				"count": resized_array.shape[0],
				"dtype": resized_array.dtype,
				"crs": src.crs or "EPSG:4326",
				"transform": transform,
			}
		)

		with rasterio.open(output_path, "w", **profile) as dst:
			dst.write(resized_array)


def run_downsampling(input_dir: Path, output_dir: Path) -> dict[str, int]:
	if not input_dir.exists() or not input_dir.is_dir():
		raise FileNotFoundError(f"Input directory not found: {input_dir}")

	output_dir.mkdir(parents=True, exist_ok=True)
	tile_files = sorted(input_dir.glob("tile_z*_x*_y*.tif"))

	if not tile_files:
		raise FileNotFoundError(f"No tile files found in: {input_dir}")

	new_size = calculate_target_tile_size_px(tile_files[0])
	total_files = len(tile_files)
	processed = 0

	jobs = [(tf, output_dir / tf.name, new_size) for tf in tile_files]
	with ThreadPoolExecutor(max_workers=WORKERS) as executor:
		future_map = {
			executor.submit(downsample_tile, input_path, output_path, size): None
			for input_path, output_path, size in jobs
		}
		for future in as_completed(future_map):
			future.result()
			processed += 1
			if processed % LOG_INTERVAL == 0 or processed == total_files:
				print(f"Downsampling progress: {processed}/{total_files}")

	return {"total": total_files, "processed": processed, "tile_size": new_size}
