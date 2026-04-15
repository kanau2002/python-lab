from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

HDD_ROOT = Path("/Volumes/HDPH-UTV")
INPUT_ROOT = HDD_ROOT / "GSI" / "predicted-mask"
OUTPUT_ROOT = HDD_ROOT / "GSI" / "post-processing"


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


def run_city(city: str) -> None:
    input_dir = INPUT_ROOT / city
    if not input_dir.exists():
        print(f"skip: input not found: {input_dir}")
        return

    output_dir = OUTPUT_ROOT / city

    print(f"start city: {city}")
    result = overlay_cropper.run_overlay_cropping(input_dir, output_dir)
    print(
        f"overlay crop done: total={result['total']}, "
        f"processed={result['processed']}, skipped={result['skipped']}, "
        f"too_small={result['too_small']}, "
        f"crop_range=[{result['min_crop']}, {result['max_crop']}]"
    )
    print(f"finish city: {city}")


def run_all_cities() -> None:
    if not INPUT_ROOT.exists():
        raise FileNotFoundError(f"Input root not found: {INPUT_ROOT}")

    for city_dir in sorted(INPUT_ROOT.iterdir()):
        if city_dir.is_dir():
            run_city(city_dir.name)


if __name__ == "__main__":
    # CITY に以下のいずれかを設定（"all" or "浦安市" etc）
    CITY = "浦安市"
    if CITY == "all":
        run_all_cities()
    else:
        run_city(CITY)
