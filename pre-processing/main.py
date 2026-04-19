# 衛星画像の前処理パイプラインのエントリポイント。
# 各都市のタイル群を tile-group-mosaicker.py でモザイク合成し、
# group-downsampler.py でダウンサンプルして mosaic-groups/ に保存する。

from __future__ import annotations

import importlib.util
import sys
import time
from pathlib import Path
from types import ModuleType


SSD_ROOT = Path("/Volumes/T7 Touch")
INPUT_ROOT = SSD_ROOT / "google-satellite-image"
OUTPUT_ROOT = SSD_ROOT / "pre-processing"


def _load_module(module_name: str, file_path: Path) -> ModuleType:
	spec = importlib.util.spec_from_file_location(module_name, file_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Failed to load module spec for: {file_path}")
	module = importlib.util.module_from_spec(spec)
	sys.modules[module_name] = module
	spec.loader.exec_module(module)
	return module


base_dir = Path(__file__).resolve().parent
group_mosaicker = _load_module("tile_group_mosaicker_module", base_dir / "utils" / "tile-group-mosaicker.py")
group_downsampler = _load_module("group_downsampler_module", base_dir / "utils" / "group-downsampler.py")


def run_city(city: str) -> None:
	input_dir = INPUT_ROOT / city
	if not input_dir.exists():
		print(f"skip: input not found: {input_dir}")
		return

	mosaic_group_dir = OUTPUT_ROOT / city / "mosaic-groups"

	if mosaic_group_dir.exists() and any(mosaic_group_dir.iterdir()):
		print(f"skip: already processed: {city}")
		return

	mosaic_group_dir.mkdir(parents=True, exist_ok=True)
	print(f"start city: {city}")

	created = 0
	t0 = time.perf_counter()

	for mosaic_array, bounds, group_id, base_x, base_y in group_mosaicker.iter_mosaic_groups(input_dir):
		output_path = mosaic_group_dir / f"mosaic_group_{group_id:03d}_x{base_x}_y{base_y}.tif"
		group_downsampler.downsample_group_array(mosaic_array, bounds, output_path)
		created += 1
		print(f"  saved group_{group_id:03d} (total saved: {created}, elapsed: {time.perf_counter() - t0:.1f}s)")

	print(f"finish city: {city}, created={created}, total: {time.perf_counter() - t0:.1f}s")


def run_all_cities() -> None:
	if not INPUT_ROOT.exists():
		raise FileNotFoundError(f"Input root not found: {INPUT_ROOT}")

	for city_dir in sorted(INPUT_ROOT.iterdir()):
		if city_dir.is_dir() and "_downloading" not in city_dir.name:
			run_city(city_dir.name)


if __name__ == "__main__":
	# CITY に以下のいずれかを設定（"all" or "浦安市".etc）
	CITY = "千葉市"
	if CITY == "all":
		run_all_cities()
	else:
		run_city(CITY)
