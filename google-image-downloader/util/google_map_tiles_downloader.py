import math
import os
import io
import time
import requests
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from pathlib import Path
from PIL import Image
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED


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
        filename = f"tile_z{zoom}_x{x}_y{y}.tif"
        filepath = self.output_dir / filename

        url = f"{self.base_url}/{zoom}/{x}/{y}?session={self.session_token}&key={self.api_key}"

        for attempt in range(5):
            try:
                response = requests.get(url)
                response.raise_for_status()

                image = Image.open(io.BytesIO(response.content))
                img_array = np.array(image)

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

            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)  # 1, 2, 4, 8秒と増加

    def download_tiles(self, tiles: list, zoom: int, check_start: str = "all"):
        start_index = 0 if check_start == "all" else int(check_start)
        existing_files = {
            path.name for path in self.output_dir.glob(f"tile_z{zoom}_x*_y*.tif")
        }
        pending = [
            (x, y) for x, y in tiles[start_index:]
            if f"tile_z{zoom}_x{x}_y{y}.tif" not in existing_files
        ]
        total = len(tiles)
        print(f"取得済: {total - len(pending)}/{total}枚 残り: {len(pending)}枚")

        last_saved = None
        count = 0

        with ThreadPoolExecutor(max_workers=8) as executor:
            in_flight = {}
            pending_iter = iter(pending)

            def submit_next():
                try:
                    x, y = next(pending_iter)
                except StopIteration:
                    return False
                in_flight[executor.submit(self.download_tile, zoom, x, y)] = (x, y)
                return True

            for _ in range(min(64, len(pending))):
                submit_next()

            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    in_flight.pop(future, None)
                    try:
                        last_saved = future.result()
                        count += 1
                        if count % 100 == 0:
                            print(f"進捗: {count}/{len(pending)}枚")
                        submit_next()
                    except Exception as e:
                        print(f"エラーが発生したため本日の取得を終了します: {e}")
                        for f in in_flight:
                            f.cancel()
                        in_flight.clear()
                        break

        print(f"本日の取得完了: {count}枚")
        if last_saved:
            print(f"最後に保存したファイル: {last_saved}")