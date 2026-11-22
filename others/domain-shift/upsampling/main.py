# google-satellite_50cm.tif を目標解像度（0.2 m/px）にアップサンプルするスクリプト。
# others/downsampling/main.py の逆処理。地上距離から目標ピクセル数を算出し、
# Lanczos リサンプルして同じ bounds の GeoTIFF として保存する。

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.transform import from_bounds

TARGET_RESOLUTION_M = 0.2

BASE_DIR = Path(__file__).parent
INPUT_PATH = BASE_DIR / "input" / "gsi-satellite_z18.tif"
OUTPUT_PATH = BASE_DIR / "output" / "gsi-satellite_z18_20cm.tif"


def calculate_target_size(bounds: tuple[float, float, float, float]) -> int:
	west, south, east, north = bounds
	center_lat = (north + south) / 2
	lon_distance = (east - west) * 111320 * math.cos(math.radians(center_lat))
	lat_distance = (north - south) * 110540
	average_ground_distance = (lon_distance + lat_distance) / 2
	return max(1, int(round(average_ground_distance / TARGET_RESOLUTION_M)))


def upsample_array(
	array: np.ndarray,
	bounds: tuple[float, float, float, float],
	crs,
	output_path: Path,
) -> int:
	west, south, east, north = bounds
	target_size = calculate_target_size(bounds)
	src_count = array.shape[0]

	# CHW → HWC でPILに渡してLanczosリサンプル後、CHWに戻す
	hwc = array.transpose(1, 2, 0)
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
		crs=crs,
		transform=dst_transform,
	) as dst:
		dst.write(resized)

	return target_size


def main() -> None:
	if not INPUT_PATH.exists():
		raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

	with rasterio.open(INPUT_PATH) as src:
		array = src.read()
		b = src.bounds
		bounds = (b.left, b.bottom, b.right, b.top)
		crs = src.crs
		print(f"Input : {INPUT_PATH.name} ({src.width}x{src.height}, {src.count} bands)")

	target_size = upsample_array(array, bounds, crs, OUTPUT_PATH)
	print(f"Output: {OUTPUT_PATH.name} ({target_size}x{target_size}, {TARGET_RESOLUTION_M} m/px)")


if __name__ == "__main__":
	main()
