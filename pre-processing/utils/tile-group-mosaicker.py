# 衛星画像タイル群をグループ単位のモザイク画像に合成するモジュール。
# tile_z{zoom}_x{x}_y{y}.tif 形式のタイルを GROUP_SIZE×GROUP_SIZE のグリッドで
# 結合し、GeoTIFF として出力する。iter_mosaic_groups() でメモリ上に配列を
# yield することで、中間ファイルなしに後段の処理へ渡すことができる。

from __future__ import annotations

import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import rasterio
from rasterio.env import Env

TILE_FILENAME_PATTERN = re.compile(r"tile_z(\d+)_x(\d+)_y(\d+)\.tif$")

GROUP_SIZE = 35
STEP_SIZE = 33
LOG_INTERVAL = 10
TILE_READ_WORKERS = 32


def parse_tile_filename(filename: str) -> tuple[int, int, int] | None:
	match = TILE_FILENAME_PATTERN.match(filename)
	if not match:
		return None
	return int(match.group(1)), int(match.group(2)), int(match.group(3))


def tile_to_bounds(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
	n = 2.0 ** zoom
	west = x / n * 360.0 - 180.0
	east = (x + 1) / n * 360.0 - 180.0
	north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
	south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
	return west, south, east, north


def _determine_tile_size(tile_path: Path) -> int:
	with rasterio.open(tile_path) as src:
		return int(src.width)


def _read_tile(path: Path) -> np.ndarray:
	with rasterio.open(path) as src:
		return src.read()


def _to_rgba(tile_data: np.ndarray, tile_size: int) -> np.ndarray:
	if tile_data.shape[0] >= 4:
		return tile_data[:4]
	rgba = np.zeros((4, tile_size, tile_size), dtype=np.uint8)
	if tile_data.shape[0] == 3:
		rgba[:3] = tile_data
	else:
		rgba[0] = rgba[1] = rgba[2] = tile_data[0]
	rgba[3] = 255
	return rgba


def _build_mosaic_array(
	tiles_dict: dict[tuple[int, int], Path],
	base_x: int,
	base_y: int,
	zoom: int,
	group_size: int,
	tile_size: int,
) -> tuple[np.ndarray, tuple[float, float, float, float]] | None:
	jobs = [
		(i, j, path)
		for i in range(group_size)
		for j in range(group_size)
		if (path := tiles_dict.get((base_x + i, base_y + j))) is not None
	]
	if not jobs:
		return None

	output_size = group_size * tile_size
	mosaic = np.zeros((4, output_size, output_size), dtype=np.uint8)

	with ThreadPoolExecutor(max_workers=TILE_READ_WORKERS) as executor:
		future_to_pos = {executor.submit(_read_tile, path): (i, j) for i, j, path in jobs}
		for future in as_completed(future_to_pos):
			i, j = future_to_pos[future]
			tile_rgba = _to_rgba(future.result(), tile_size)
			y_start = j * tile_size
			x_start = i * tile_size
			mosaic[:, y_start : y_start + tile_size, x_start : x_start + tile_size] = tile_rgba

	min_west, min_south, _, _ = tile_to_bounds(base_x, base_y + group_size - 1, zoom)
	_, _, max_east, max_north = tile_to_bounds(base_x + group_size - 1, base_y, zoom)
	return mosaic, (min_west, min_south, max_east, max_north)


def iter_mosaic_groups(
	input_dir: Path,
	group_size: int = GROUP_SIZE,
	step_size: int = STEP_SIZE,
):
	if not input_dir.exists() or not input_dir.is_dir():
		raise FileNotFoundError(f"Input directory not found: {input_dir}")

	tiles_dict: dict[tuple[int, int], Path] = {}
	zoom_level: int | None = None

	tile_files = sorted(input_dir.glob("tile_z*_x*_y*.tif"))
	if not tile_files:
		raise FileNotFoundError(f"No tile files found in: {input_dir}")

	for tile_file in tile_files:
		parsed = parse_tile_filename(tile_file.name)
		if parsed is None:
			continue
		z, x, y = parsed
		tiles_dict[(x, y)] = tile_file
		if zoom_level is None:
			zoom_level = z

	if zoom_level is None or not tiles_dict:
		raise RuntimeError(f"No valid tile files found in: {input_dir}")

	tile_size = _determine_tile_size(next(iter(tiles_dict.values())))

	x_coords = [x for x, _ in tiles_dict.keys()]
	y_coords = [y for _, y in tiles_dict.keys()]
	min_x, max_x = min(x_coords), max(x_coords)
	min_y, max_y = min(y_coords), max(y_coords)

	x_groups = (max_x - min_x + step_size) // step_size
	y_groups = (max_y - min_y + step_size) // step_size
	total_groups = x_groups * y_groups

	group_id = 0
	processed = 0

	with Env(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR"):
		for gx in range(x_groups):
			for gy in range(y_groups):
				base_x = min_x - 1 + gx * step_size
				base_y = min_y - 1 + gy * step_size

				result = _build_mosaic_array(tiles_dict, base_x, base_y, zoom_level, group_size, tile_size)
				if result is not None:
					mosaic_array, bounds = result
					yield mosaic_array, bounds, group_id, base_x, base_y

				group_id += 1
				processed += 1
				if processed % LOG_INTERVAL == 0 or processed == total_groups:
					print(f"Mosaicking progress: {processed}/{total_groups}")
