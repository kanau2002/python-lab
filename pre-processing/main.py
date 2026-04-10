from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

HDD_ROOT = Path("/Volumes/HDPH-UTV")
INPUT_ROOT = HDD_ROOT / "google-satellite-image"
OUTPUT_ROOT = HDD_ROOT / "pre-processing"


def _load_module(module_name: str, file_path: Path) -> ModuleType:
	spec = importlib.util.spec_from_file_location(module_name, file_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Failed to load module spec for: {file_path}")
	module = importlib.util.module_from_spec(spec)
	sys.modules[module_name] = module
	spec.loader.exec_module(module)
	return module


base_dir = Path(__file__).resolve().parent
downsampler = _load_module("tile_downsampler_module", base_dir / "utils" / "tile-downsampler.py")
group_mosaicker = _load_module("tile_group_mosaicker_module", base_dir / "utils" / "tile-group-mosaicker.py")


def run_city(city: str) -> None:
	input_dir = INPUT_ROOT / city
	if not input_dir.exists():
		print(f"skip: input not found: {input_dir}")
		return

	downsample_dir = OUTPUT_ROOT / city / "downsampled"
	mosaic_group_dir = OUTPUT_ROOT / city / "mosaic-groups"

	if mosaic_group_dir.exists() and any(mosaic_group_dir.iterdir()):
		print(f"skip: already processed: {city}")
		return

	print(f"start city: {city}")
	downsample_result = downsampler.run_downsampling(input_dir, downsample_dir)
	print(f"downsample done: total={downsample_result['total']}, processed={downsample_result['processed']}")

	mosaic_result = group_mosaicker.run_group_mosaicking(downsample_dir, mosaic_group_dir)
	print(f"mosaic done: total_groups={mosaic_result['total_groups']}, created={mosaic_result['created']}, empty={mosaic_result['empty']}")
	print(f"finish city: {city}")


def run_all_cities() -> None:
	if not INPUT_ROOT.exists():
		raise FileNotFoundError(f"Input root not found: {INPUT_ROOT}")

	for city_dir in sorted(INPUT_ROOT.iterdir()):
		if city_dir.is_dir() and "_downloading" not in city_dir.name:
			run_city(city_dir.name)


if __name__ == "__main__":
	# CITY に以下のいずれかを設定（"all" or "浦安市".etc）
	CITY = "鎌ケ谷市"
	if CITY == "all":
		run_all_cities()
	else:
		run_city(CITY)
