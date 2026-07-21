# モザイクグループ画像を目標解像度（0.2 m/px）にアップサンプリングするモジュール。
# tile-group-mosaicker.py が生成した numpy 配列（元解像度は目標解像度より粗い）を
# 旧方式（タイル単位アップサンプリング）と同じ Resampling.cubic でグループ全体を
# 一括リサンプルし、GeoTIFF として保存する。

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.io import MemoryFile
from rasterio.transform import from_bounds

TARGET_RESOLUTION_M = 0.2


def calculate_target_size(bounds: tuple[float, float, float, float]) -> int:
    west, south, east, north = bounds
    center_lat = (north + south) / 2
    lon_distance = (east - west) * 111320 * math.cos(math.radians(center_lat))
    lat_distance = (north - south) * 110540
    average_ground_distance = (lon_distance + lat_distance) / 2
    return max(1, int(round(average_ground_distance / TARGET_RESOLUTION_M)))


def upsample_group_array(
    mosaic_array: np.ndarray,
    bounds: tuple[float, float, float, float],
    output_path: Path,
) -> None:
    west, south, east, north = bounds
    target_size = calculate_target_size(bounds)
    src_count, src_height, src_width = mosaic_array.shape
    src_transform = from_bounds(west, south, east, north, src_width, src_height)

    with MemoryFile() as memfile:
        with memfile.open(
            driver="GTiff",
            height=src_height,
            width=src_width,
            count=src_count,
            dtype=mosaic_array.dtype,
            crs="EPSG:4326",
            transform=src_transform,
        ) as src:
            src.write(mosaic_array)
            resized = src.read(
                out_shape=(src_count, target_size, target_size),
                resampling=Resampling.cubic,
            )

    dst_transform = from_bounds(west, south, east, north, target_size, target_size)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=target_size,
        width=target_size,
        count=src_count,
        dtype=resized.dtype,
        crs="EPSG:4326",
        transform=dst_transform,
    ) as dst:
        dst.write(resized)
