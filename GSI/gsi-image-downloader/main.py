from pathlib import Path
from util.tiles_coordinate_calculator import get_tiles_in_city
from util.gsi_tiles_downloader import GsiTilesDownloader

OUTPUT_ROOT = Path("/Volumes/HDPH-UTV/GSI/gsi-satellite-image")
# OUTPUT_ROOT = Path(__file__).parent / "output"
CHECK_START = "all"

def main():
    city_name = "浦安市"
    zoom_level = 18

    tiles = get_tiles_in_city(city_name, zoom_level)

    output_dir = OUTPUT_ROOT / f"{city_name}_downloading"
    downloader = GsiTilesDownloader(str(output_dir))

    downloader.download_tiles(tiles, zoom_level, CHECK_START)

if __name__ == "__main__":
    main()
