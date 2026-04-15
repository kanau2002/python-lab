from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_bounds

TILE_FILENAME_PATTERN = re.compile(r"tile_z(\d+)_x(\d+)_y(\d+)\.tif$")

GROUP_SIZE = 10
STEP_SIZE = 8
LOG_INTERVAL = 1000


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


def _create_mosaic_group(
    tiles_dict: dict[tuple[int, int], Path],
    base_x: int,
    base_y: int,
    zoom: int,
    output_path: Path,
    group_size: int,
    tile_size: int,
) -> bool:
    output_size = group_size * tile_size
    mosaic = np.zeros((4, output_size, output_size), dtype=np.uint8)
    tile_count = 0

    for i in range(group_size):
        for j in range(group_size):
            tile_path = tiles_dict.get((base_x + i, base_y + j))
            if tile_path is None:
                continue

            tile_count += 1
            with rasterio.open(tile_path) as src:
                tile_data = src.read()

            if tile_data.shape[0] >= 4:
                tile_rgba = tile_data[:4]
            elif tile_data.shape[0] == 3:
                tile_rgba = np.zeros((4, tile_size, tile_size), dtype=np.uint8)
                tile_rgba[:3] = tile_data
                tile_rgba[3] = 255
            else:
                tile_rgba = np.zeros((4, tile_size, tile_size), dtype=np.uint8)
                tile_rgba[0] = tile_data[0]
                tile_rgba[1] = tile_data[0]
                tile_rgba[2] = tile_data[0]
                tile_rgba[3] = 255

            y_start = j * tile_size
            x_start = i * tile_size
            mosaic[:, y_start : y_start + tile_size, x_start : x_start + tile_size] = tile_rgba

    if tile_count == 0:
        return False

    min_west, min_south, _, _ = tile_to_bounds(base_x, base_y + group_size - 1, zoom)
    _, _, max_east, max_north = tile_to_bounds(base_x + group_size - 1, base_y, zoom)
    transform = from_bounds(min_west, min_south, max_east, max_north, output_size, output_size)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=output_size,
        width=output_size,
        count=4,
        dtype=np.uint8,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(mosaic)

    return True


def run_group_mosaicking(input_dir: Path, output_dir: Path) -> dict[str, int]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

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

    x_groups = (max_x - min_x + STEP_SIZE) // STEP_SIZE
    y_groups = (max_y - min_y + STEP_SIZE) // STEP_SIZE
    total_groups = x_groups * y_groups

    group_id = 0
    created = 0
    empty = 0
    processed_groups = 0

    for gx in range(x_groups):
        for gy in range(y_groups):
            base_x = min_x - 1 + gx * STEP_SIZE
            base_y = min_y - 1 + gy * STEP_SIZE

            output_path = output_dir / f"mosaic_group_{group_id:03d}_x{base_x}_y{base_y}.tif"

            if _create_mosaic_group(
                tiles_dict=tiles_dict,
                base_x=base_x,
                base_y=base_y,
                zoom=zoom_level,
                output_path=output_path,
                group_size=GROUP_SIZE,
                tile_size=tile_size,
            ):
                created += 1
            else:
                empty += 1

            group_id += 1
            processed_groups += 1
            if processed_groups % LOG_INTERVAL == 0 or processed_groups == total_groups:
                print(
                    f"Group mosaicking progress: {processed_groups}/{total_groups} "
                    f"(created={created}, empty={empty})"
                )

    return {"total_groups": total_groups, "created": created, "empty": empty}
