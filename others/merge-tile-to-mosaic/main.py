import shutil
import subprocess
import sys
from pathlib import Path

import rasterio
from rasterio.merge import merge

INPUT = '/Volumes/T7 Touch/google-satellite-image/酒々井町'
OUTPUT_DIR = Path(__file__).parent / 'output'


def find_tiles(input_dir: str) -> list[Path]:
    tiles = sorted(Path(input_dir).rglob('*.tif'))
    print(f"Found {len(tiles)} tiles")
    return tiles


def _write_filelist(tiles: list[Path], filelist_path: Path) -> None:
    with open(filelist_path, 'w', encoding='utf-8') as f:
        for t in tiles:
            f.write(str(t) + '\n')


def create_mosaic_gdal(tiles: list[Path], output_path: Path) -> None:
    vrt_path = output_path.with_suffix('.vrt')
    filelist_path = output_path.with_suffix('.txt')

    _write_filelist(tiles, filelist_path)

    print("Building VRT...")
    subprocess.run(
        ['gdalbuildvrt', '-input_file_list', str(filelist_path), str(vrt_path)],
        check=True,
    )

    print("Translating to GeoTIFF...")
    subprocess.run(
        [
            'gdal_translate',
            '-of', 'GTiff',
            '-co', 'COMPRESS=LZW',
            '-co', 'TILED=YES',
            '-co', 'BLOCKXSIZE=512',
            '-co', 'BLOCKYSIZE=512',
            '-co', 'BIGTIFF=IF_SAFER',
            str(vrt_path),
            str(output_path),
        ],
        check=True,
    )

    print("Building overviews...")
    subprocess.run(
        ['gdaladdo', '-r', 'average', str(output_path), '2', '4', '8', '16', '32'],
        check=True,
    )

    vrt_path.unlink(missing_ok=True)
    filelist_path.unlink(missing_ok=True)


def create_mosaic_rasterio(tiles: list[Path], output_path: Path) -> None:
    print("Building VRT... (rasterio)")
    src_files = [rasterio.open(t) for t in tiles]
    try:
        mosaic, transform = merge(src_files)
    finally:
        for src in src_files:
            src.close()

    print("Writing GeoTIFF...")
    meta = rasterio.open(tiles[0]).meta.copy()
    meta.update({
        'driver': 'GTiff',
        'height': mosaic.shape[1],
        'width': mosaic.shape[2],
        'transform': transform,
        'compress': 'lzw',
        'tiled': True,
        'blockxsize': 512,
        'blockysize': 512,
        'bigtiff': 'IF_SAFER',
    })
    with rasterio.open(output_path, 'w', **meta) as dst:
        dst.write(mosaic)

    print("Building overviews...")
    with rasterio.open(output_path, 'r+') as dst:
        dst.build_overviews([2, 4, 8, 16, 32], rasterio.enums.Resampling.average)
        dst.update_tags(ns='rio_overview', resampling='average')


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tiles = find_tiles(INPUT)
    if not tiles:
        print("No tiles found. Exiting.")
        sys.exit(1)

    output_path = OUTPUT_DIR / 'mosaic.tif'

    if shutil.which('gdalbuildvrt'):
        create_mosaic_gdal(tiles, output_path)
    else:
        print("GDAL CLI not found, falling back to rasterio...")
        create_mosaic_rasterio(tiles, output_path)

    print(f"Done: {output_path}")


if __name__ == '__main__':
    main()
