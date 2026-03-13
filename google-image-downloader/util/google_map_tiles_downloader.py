import math
import os
import requests
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
import io

def tile_to_bounds(x: int, y: int, zoom: int):
    """タイルの境界ボックスを取得（west, south, east, north）"""
    n = 2.0 ** zoom
    west = x / n * 360.0 - 180.0
    east = (x + 1) / n * 360.0 - 180.0
    north = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    south = math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * (y + 1) / n))))
    return (west, south, east, north)

class GoogleMapTilesDownloader:
    
    def __init__(self, output_dir: str = "output"):
        load_dotenv()
        self.api_key = os.getenv("API_KEY")
        self.output_dir = Path(output_dir)
        self.base_url = "https://tile.googleapis.com/v1/2dtiles"
        self.session_url = "https://tile.googleapis.com/v1/createSession"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.session_token = self._create_session()
    
    def _create_session(self):
        payload = {"mapType": "satellite", "language": "ja", "region": "JP"}
        response = requests.post(f"{self.session_url}?key={self.api_key}", json=payload)
        return response.json().get("session")
    
    def download_tile(self, zoom: int, x: int, y: int) -> str:
        url = f"{self.base_url}/{zoom}/{x}/{y}?session={self.session_token}&key={self.api_key}"
        response = requests.get(url)
        
        image = Image.open(io.BytesIO(response.content))
        img_array = np.array(image)
        
        filename = f"tile_z{zoom}_x{x}_y{y}.tif"
        filepath = self.output_dir / filename
        
        west, south, east, north = tile_to_bounds(x, y, zoom)
        transform = from_bounds(west, south, east, north, img_array.shape[1], img_array.shape[0])
        
        with rasterio.open(
            filepath, 'w',
            driver='GTiff',
            height=img_array.shape[0],
            width=img_array.shape[1],
            count=3,
            dtype=img_array.dtype,
            crs='EPSG:4326',
            transform=transform
        ) as dst:
            for i in range(3):
                dst.write(img_array[:, :, i], i + 1)
        
        return str(filepath)
    
    def download_tiles(self, tiles: list, zoom: int):
        """タイル座標のリストから画像をダウンロード"""
        total = len(tiles)
        for i, (x, y) in enumerate(tiles, 1):
            self.download_tile(zoom, x, y)
            if i % 10 == 0 or i == total:
                print(f"進捗: {i}/{total}枚")