from pathlib import Path
from util.tiles_coordinate_calculator import get_tiles_in_city
from util.google_map_tiles_downloader import GoogleMapTilesDownloader

OUTPUT_ROOT = Path("/Volumes/T7 Touch/google-satellite-image")
# OUTPUT_ROOT = Path(__file__).parent / "output"
CHECK_START = "all"

def main():
    city_name = "多古町"
    zoom_level = 20
    
    tiles = get_tiles_in_city(city_name, zoom_level)

    output_dir = OUTPUT_ROOT / f"{city_name}_downloading"
    downloader = GoogleMapTilesDownloader(str(output_dir))
    
    downloader.download_tiles(tiles, zoom_level, CHECK_START)

if __name__ == "__main__":
    main()