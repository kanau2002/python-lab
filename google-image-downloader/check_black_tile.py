from pathlib import Path
import rasterio
import numpy as np

tile_dir = Path("/Volumes/T7 Touch/google-satellite-image/八街市_downloading")
black_tiles = []
for f in sorted(tile_dir.glob("tile_z*_x*_y*.tif")):
    with rasterio.open(f) as src:
        data = src.read()
    if data.max() == 0:
        black_tiles.append(f.name)

print(f"黒いタイル: {len(black_tiles)}枚")
for name in black_tiles[:10]:
    print(name)
