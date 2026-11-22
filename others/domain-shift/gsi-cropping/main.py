# google-satellite_20cm.tif と同じ領域の国土地理院（GSI）衛星画像を1枚作成するスクリプト。
# Google 画像の bounds から必要な z18 タイルを特定し、モザイクして同じ bounds で切り出す。
# 解像度は GSI タイルのネイティブ解像度（z18, 約0.48 m/px）のまま。

from __future__ import annotations

import math
from pathlib import Path

import rasterio
from rasterio.merge import merge

GSI_TILE_DIR = Path("/Volumes/HDPH-UTV/GSI/gsi-satellite-image/酒々井町")
ZOOM = 18

BASE_DIR = Path(__file__).parent
REFERENCE_PATH = BASE_DIR.parent / "downsampling" / "input" / "google-satellite_20cm.tif"
OUTPUT_PATH = BASE_DIR / "output" / "gsi-satellite_z18.tif"


def lonlat_to_tile(lon: float, lat: float, zoom: int) -> tuple[float, float]:
	n = 2.0 ** zoom
	x = (lon + 180.0) / 360.0 * n
	y = (1.0 - math.asinh(math.tan(math.radians(lat))) / math.pi) / 2.0 * n
	return x, y


def find_covering_tiles(bounds: tuple[float, float, float, float]) -> list[Path]:
	west, south, east, north = bounds
	x_min, y_min = lonlat_to_tile(west, north, ZOOM)
	x_max, y_max = lonlat_to_tile(east, south, ZOOM)

	tiles: list[Path] = []
	missing: list[str] = []
	for x in range(math.floor(x_min), math.floor(x_max) + 1):
		for y in range(math.floor(y_min), math.floor(y_max) + 1):
			path = GSI_TILE_DIR / f"tile_z{ZOOM}_x{x}_y{y}.tif"
			if path.exists():
				tiles.append(path)
			else:
				missing.append(path.name)

	if missing:
		print(f"Warning: {len(missing)} tiles missing (skipped): {missing}")
	if not tiles:
		raise FileNotFoundError(f"No covering tiles found in: {GSI_TILE_DIR}")
	return tiles


def approx_resolution_m(transform, center_lat: float) -> tuple[float, float]:
	x_res = transform.a * 111320 * math.cos(math.radians(center_lat))
	y_res = -transform.e * 110540
	return x_res, y_res


def main() -> None:
	if not REFERENCE_PATH.exists():
		raise FileNotFoundError(f"Reference file not found: {REFERENCE_PATH}")
	if not GSI_TILE_DIR.exists():
		raise FileNotFoundError(f"GSI tile directory not found (is the disk mounted?): {GSI_TILE_DIR}")

	with rasterio.open(REFERENCE_PATH) as ref:
		b = ref.bounds
		bounds = (b.left, b.bottom, b.right, b.top)
		print(f"Reference: {REFERENCE_PATH.name} ({ref.width}x{ref.height})")
		print(f"Bounds   : {bounds}")

	tiles = find_covering_tiles(bounds)
	print(f"Tiles    : {len(tiles)}")

	srcs = [rasterio.open(t) for t in tiles]
	try:
		crs = srcs[0].crs
		mosaic, transform = merge(srcs, bounds=bounds)
	finally:
		for src in srcs:
			src.close()

	OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	with rasterio.open(
		OUTPUT_PATH,
		"w",
		driver="GTiff",
		height=mosaic.shape[1],
		width=mosaic.shape[2],
		count=mosaic.shape[0],
		dtype=mosaic.dtype,
		crs=crs,
		transform=transform,
		compress="lzw",
	) as dst:
		dst.write(mosaic)

	x_res, y_res = approx_resolution_m(transform, (bounds[1] + bounds[3]) / 2)
	print(
		f"Output   : {OUTPUT_PATH.name} ({mosaic.shape[2]}x{mosaic.shape[1]}, "
		f"{mosaic.shape[0]} bands, ~{x_res:.3f}x{y_res:.3f} m/px)"
	)


if __name__ == "__main__":
	main()
