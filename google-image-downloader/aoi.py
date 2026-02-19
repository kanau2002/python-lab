from pathlib import Path
from lib.tiles_coordinate_calculator import lat_lon_to_tile
from lib.google_map_tiles_downloader import GoogleMapTilesDownloader

def main():
    north_lat = 35.369656
    west_lon = 139.903748
    south_lat = 35.362094
    east_lon = 139.910869
    zoom_level = 20
    
    x_west, y_north = lat_lon_to_tile(north_lat, west_lon, zoom_level)
    x_east, y_south = lat_lon_to_tile(south_lat, east_lon, zoom_level)
    
    tiles = [(x, y) for x in range(x_west, x_east + 1) for y in range(y_north, y_south + 1)]
    
    output_dir = Path(__file__).parent / "output_aoi"
    downloader = GoogleMapTilesDownloader(str(output_dir))
    downloader.download_tiles(tiles, zoom_level)

if __name__ == "__main__":
    main()