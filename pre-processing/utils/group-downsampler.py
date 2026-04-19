# モザイクグループ画像を目標解像度（0.2 m/px）にダウンサンプルするモジュール。
# tile-group-mosaicker.py が生成した numpy 配列、または保存済みの
# mosaic_group_*.tif を受け取り、Lanczos リサンプルして GeoTIFF として保存する。

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_bounds

TARGET_RESOLUTION_M = 0.2


def calculate_target_size(bounds: tuple[float, float, float, float]) -> int:
	west, south, east, north = bounds
	center_lat = (north + south) / 2
	lon_distance = (east - west) * 111320 * math.cos(math.radians(center_lat))
	lat_distance = (north - south) * 110540
	average_ground_distance = (lon_distance + lat_distance) / 2
	return max(1, int(round(average_ground_distance / TARGET_RESOLUTION_M)))


def downsample_group_array(
	mosaic_array: np.ndarray,
	bounds: tuple[float, float, float, float],
	output_path: Path,
) -> None:
	west, south, east, north = bounds
	target_size = calculate_target_size(bounds)
	src_count = mosaic_array.shape[0]

	# CHW → HWC でPILに渡してLanczosリサンプル後、CHWに戻す
	hwc = mosaic_array.transpose(1, 2, 0)
	mode = {4: "RGBA", 3: "RGB", 1: "L"}.get(src_count, "RGBA")
	img_input = hwc[:, :, 0] if mode == "L" else hwc
	resized_hwc = np.array(
		Image.fromarray(img_input, mode=mode).resize((target_size, target_size), Image.LANCZOS)
	)
	resized = resized_hwc[np.newaxis] if mode == "L" else resized_hwc.transpose(2, 0, 1)

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


def run_group_downsampling(input_dir: Path, output_dir: Path) -> dict[str, int]:
	if not input_dir.exists() or not input_dir.is_dir():
		raise FileNotFoundError(f"Input directory not found: {input_dir}")

	output_dir.mkdir(parents=True, exist_ok=True)
	group_files = sorted(input_dir.glob("mosaic_group_*.tif"))

	if not group_files:
		raise FileNotFoundError(f"No mosaic group files found in: {input_dir}")

	total = len(group_files)
	processed = 0

	for group_file in group_files:
		with rasterio.open(group_file) as src:
			mosaic_array = src.read()
			b = src.bounds
			bounds = (b.left, b.bottom, b.right, b.top)

		output_path = output_dir / group_file.name
		downsample_group_array(mosaic_array, bounds, output_path)
		processed += 1
		print(f"Group downsampling progress: {processed}/{total}")

	return {"total": total, "processed": processed}
