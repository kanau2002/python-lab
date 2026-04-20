from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

SSD_ROOT = Path("/Volumes/T7 Touch")
INPUT_ROOT = SSD_ROOT / "predicted-mask"
OUTPUT_ROOT = SSD_ROOT / "post-processing"

# overlay-cropper
GROUP_SIZE_FOR_CROP = 35

# mask-vectorizer
MORPH_OPEN_ITER = 1
MORPH_CLOSE_ITER = 2

# polygon-merger
MERGE_BUFFER_DEG = 1e-5  # ≈1.1m: compensates ~4px erosion at tile seams

# polygon-exporter
MIN_AREA_SQM = 11.5      # 国土交通省基準: 5.0m × 2.3m
SIMPLIFY_TOL_DEG = 2e-6  # ≈0.2m = 1px at 0.2m/px resolution
OUTPUT_FORMAT = "GPKG"

WORKERS = 4
LOG_INTERVAL = 1000


def _load_module(module_name: str, file_path: Path) -> ModuleType:
	spec = importlib.util.spec_from_file_location(module_name, file_path)
	if spec is None or spec.loader is None:
		raise ImportError(f"Failed to load module spec for: {file_path}")
	module = importlib.util.module_from_spec(spec)
	sys.modules[module_name] = module
	spec.loader.exec_module(module)
	return module


base_dir = Path(__file__).resolve().parent
overlay_cropper = _load_module(
	"overlay_cropper_module",
	base_dir / "utils" / "overlay-cropper.py",
)
mask_vectorizer = _load_module(
	"mask_vectorizer_module",
	base_dir / "utils" / "mask-vectorizer.py",
)
polygon_merger = _load_module(
	"polygon_merger_module",
	base_dir / "utils" / "polygon-merger.py",
)
polygon_exporter = _load_module(
	"polygon_exporter_module",
	base_dir / "utils" / "polygon-exporter.py",
)


def run_city(city: str) -> None:
	input_dir = INPUT_ROOT / city
	if not input_dir.exists():
		print(f"skip: input not found: {input_dir}")
		return

	output_gpkg = OUTPUT_ROOT / city / f"{city}_parking_lots.gpkg"

	print(f"[{city}] start")

	# Step 1: overlay cropping (in-memory)
	tiles, crop_stats = overlay_cropper.run_overlay_cropping_in_memory(
		input_dir,
		group_size_for_crop=GROUP_SIZE_FOR_CROP,
		workers=WORKERS,
		log_interval=LOG_INTERVAL,
	)
	print(
		f"[{city}] overlay crop done: total={crop_stats['total']}, "
		f"processed={crop_stats['processed']}, too_small={crop_stats['too_small']}, "
		f"crop_range=[{crop_stats['min_crop']}, {crop_stats['max_crop']}]"
	)

	# Step 2: per-tile vectorization (in-memory)
	records = mask_vectorizer.run_vectorization_from_memory(
		tiles,
		morph_open_iter=MORPH_OPEN_ITER,
		morph_close_iter=MORPH_CLOSE_ITER,
		workers=WORKERS,
		log_interval=LOG_INTERVAL,
	)

	# Step 3: cross-tile seam stitching
	gdf = polygon_merger.run_merge(records, merge_buffer_deg=MERGE_BUFFER_DEG)

	# Step 4: filter, simplify, export
	export_result = polygon_exporter.run_export(
		gdf, output_gpkg,
		min_area_sqm=MIN_AREA_SQM,
		simplify_tol_deg=SIMPLIFY_TOL_DEG,
		output_format=OUTPUT_FORMAT,
	)
	print(
		f"[{city}] export done: "
		f"input={export_result['total_input']}, "
		f"after_filter={export_result['after_filter']}, "
		f"written={export_result['written']}"
	)

	print(f"[{city}] finish → {output_gpkg}")


def run_all_cities() -> None:
	if not INPUT_ROOT.exists():
		raise FileNotFoundError(f"Input root not found: {INPUT_ROOT}")

	for city_dir in sorted(INPUT_ROOT.iterdir()):
		if city_dir.is_dir():
			run_city(city_dir.name)


if __name__ == "__main__":
	# CITY に以下のいずれかを設定（"all" or "浦安市".etc）
	CITY = "千葉市"
	if CITY == "all":
		run_all_cities()
	else:
		run_city(CITY)
