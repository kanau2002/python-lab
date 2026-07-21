from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import rasterio.features
from scipy.ndimage import binary_closing, binary_opening

def _vectorize_tif(
    tif_path: Path, morph_open_iter: int, morph_close_iter: int
) -> tuple[list[dict], str]:
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        transform = src.transform

    mask = (data > 0).astype(np.uint8)
    mask = binary_opening(mask, iterations=morph_open_iter).astype(np.uint8)
    mask = binary_closing(mask, iterations=morph_close_iter).astype(np.uint8)

    geometries = [
        geom
        for geom, val in rasterio.features.shapes(mask, transform=transform)
        if val == 1
    ]
    return geometries, tif_path.name


def _vectorize_tile(
    data: np.ndarray, transform: Any, filename: str,
    morph_open_iter: int, morph_close_iter: int,
) -> tuple[list[dict], str]:
    mask = (data > 0).astype(np.uint8)
    mask = binary_opening(mask, iterations=morph_open_iter).astype(np.uint8)
    mask = binary_closing(mask, iterations=morph_close_iter).astype(np.uint8)
    geometries = [
        geom
        for geom, val in rasterio.features.shapes(mask, transform=transform)
        if val == 1
    ]
    return geometries, filename


def run_vectorization_from_memory(
    tiles: list[tuple[np.ndarray, Any, str]],
    morph_open_iter: int = 1,
    morph_close_iter: int = 2,
    workers: int = 4,
    log_interval: int = 1000,
) -> list[tuple[dict, str]]:
    total = len(tiles)
    results: list[tuple[dict, str]] = []

    if workers <= 1:
        for idx, (data, transform, filename) in enumerate(tiles, start=1):
            geoms, name = _vectorize_tile(data, transform, filename, morph_open_iter, morph_close_iter)
            results.extend((g, name) for g in geoms)
            if idx % log_interval == 0 or idx == total:
                print(f"Vectorize progress: {idx}/{total}, polygons so far={len(results)}")
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_vectorize_tile, d, t, f, morph_open_iter, morph_close_iter): f
                for d, t, f in tiles
            }
            for future in as_completed(futures):
                geoms, name = future.result()
                results.extend((g, name) for g in geoms)
                completed += 1
                if completed % log_interval == 0 or completed == total:
                    print(f"Vectorize progress: {completed}/{total}, polygons so far={len(results)}")

    print(f"Vectorization done: {total} tiles, {len(results)} raw polygons")
    return results


def run_vectorization(
    input_dir: Path,
    morph_open_iter: int = 1,
    morph_close_iter: int = 2,
    workers: int = 4,
    log_interval: int = 1000,
) -> list[tuple[dict, str]]:
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    tif_files = sorted(
        p for p in input_dir.glob("*.tif") if not p.name.startswith("._")
    )
    if not tif_files:
        raise FileNotFoundError(f"No tif files found in: {input_dir}")

    total = len(tif_files)
    results: list[tuple[dict, str]] = []

    if workers <= 1:
        for idx, tif_path in enumerate(tif_files, start=1):
            geoms, name = _vectorize_tif(tif_path, morph_open_iter, morph_close_iter)
            results.extend((g, name) for g in geoms)
            if idx % log_interval == 0 or idx == total:
                print(f"Vectorize progress: {idx}/{total}, polygons so far={len(results)}")
    else:
        completed = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_vectorize_tif, p, morph_open_iter, morph_close_iter): p
                for p in tif_files
            }
            for future in as_completed(futures):
                geoms, name = future.result()
                results.extend((g, name) for g in geoms)
                completed += 1
                if completed % log_interval == 0 or completed == total:
                    print(
                        f"Vectorize progress: {completed}/{total}, polygons so far={len(results)}"
                    )

    print(f"Vectorization done: {total} tiles, {len(results)} raw polygons")
    return results
