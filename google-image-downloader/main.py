from pathlib import Path
from util.tiles_coordinate_calculator import get_tiles_in_city
from util.google_map_tiles_downloader import GoogleMapTilesDownloader

def main():
    city_name = "長生村"
    zoom_level = 20
    
    tiles = get_tiles_in_city(city_name, zoom_level)

    output_dir = Path(__file__).parent / f"output_{city_name}"
    downloader = GoogleMapTilesDownloader(str(output_dir))
    
    downloader.download_tiles(tiles, zoom_level)

if __name__ == "__main__":
    main()