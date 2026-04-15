from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import rasterio
from rasterio.transform import from_bounds

GROUP_SIZE_FOR_CROP = 10  # GSI版: tile-group-mosaicker の GROUP_SIZE に合わせる
WORKERS = 4
LOG_INTERVAL = 1000
OVERWRITE = False


def _calculate_safe_crop_size(height: int, width: int) -> int:
    # For 10x10 mosaics, crop 1 tile per edge
    estimated_tile_size = int(round(min(height, width) / GROUP_SIZE_FOR_CROP))
    safe_limit = min((height - 1) // 2, (width - 1) // 2)
    return max(0, min(estimated_tile_size, safe_limit))


def _crop_tif(input_path: Path, output_path: Path) -> tuple[str, int]:
    if output_path.exists() and not OVERWRITE:
        return "skipped", 0

    with rasterio.open(input_path) as src:
        data = src.read()
        h, w = data.shape[1], data.shape[2]
        crop_size = _calculate_safe_crop_size(h, w)

        if crop_size <= 0 or h <= crop_size * 2 or w <= crop_size * 2:
            return "too_small", crop_size

        cropped = data[:, crop_size : h - crop_size, crop_size : w - crop_size]
        new_h, new_w = cropped.shape[1], cropped.shape[2]

        bounds = src.bounds
        pw = (bounds.right - bounds.left) / w
        ph = (bounds.top - bounds.bottom) / h

        new_bounds = (
            bounds.left + crop_size * pw,
            bounds.bottom + crop_size * ph,
            bounds.right - crop_size * pw,
            bounds.top - crop_size * ph,
        )
        transform = from_bounds(*new_bounds, new_w, new_h)

        profile = src.profile.copy()
        profile.update(
            {
                "driver": "GTiff",
                "height": new_h,
                "width": new_w,
                "count": src.count,
                "dtype": src.dtypes[0],
                "transform": transform,
                "nodata": 0,
            }
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(cropped)

    return "processed", crop_size


def run_overlay_cropping(input_dir: Path, output_dir: Path) -> dict[str, int]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    output_dir.mkdir(parents=True, exist_ok=True)

    tif_files = sorted(p for p in input_dir.glob("*.tif") if not p.name.startswith("._"))
    if not tif_files:
        raise FileNotFoundError(f"No tif files found in: {input_dir}")

    total = len(tif_files)
    processed = 0
    skipped = 0
    too_small = 0
    min_crop_used: int | None = None
    max_crop_used: int | None = None

    jobs = [(tif_file, output_dir / tif_file.name) for tif_file in tif_files]

    if WORKERS <= 1:
        for idx, (input_path, output_path) in enumerate(jobs, start=1):
            status, crop_used = _crop_tif(input_path, output_path)
            if status == "processed":
                processed += 1
                if min_crop_used is None or crop_used < min_crop_used:
                    min_crop_used = crop_used
                if max_crop_used is None or crop_used > max_crop_used:
                    max_crop_used = crop_used
            elif status == "skipped":
                skipped += 1
            else:
                too_small += 1

            if idx % LOG_INTERVAL == 0 or idx == total:
                print(
                    "Overlay crop progress: "
                    f"{idx}/{total} "
                    f"(processed={processed}, skipped={skipped}, too_small={too_small})"
                )
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {
                executor.submit(_crop_tif, input_path, output_path): (input_path, output_path)
                for input_path, output_path in jobs
            }
            for future in as_completed(futures):
                status, crop_used = future.result()
                if status == "processed":
                    processed += 1
                    if min_crop_used is None or crop_used < min_crop_used:
                        min_crop_used = crop_used
                    if max_crop_used is None or crop_used > max_crop_used:
                        max_crop_used = crop_used
                elif status == "skipped":
                    skipped += 1
                else:
                    too_small += 1

                completed += 1
                if completed % LOG_INTERVAL == 0 or completed == total:
                    print(
                        "Overlay crop progress: "
                        f"{completed}/{total} "
                        f"(processed={processed}, skipped={skipped}, too_small={too_small})"
                    )

    return {
        "total": total,
        "processed": processed,
        "skipped": skipped,
        "too_small": too_small,
        "min_crop": min_crop_used if min_crop_used is not None else 0,
        "max_crop": max_crop_used if max_crop_used is not None else 0,
    }
