from pathlib import Path
from util.tiles_coordinate_calculator import get_tiles_in_city
from util.gsi_tiles_downloader import GsiTilesDownloader

OUTPUT_ROOT = Path("/Volumes/HDPH-UTV/GSI/gsi-satellite-image")
# OUTPUT_ROOT = Path(__file__).parent / "output"
CHECK_START = "all"

def main():
    selected_city = ["浦安市", "酒々井町", "神崎町", "習志野市", "鎌ケ谷市", "一宮町", "九十九里町", "御宿町",
  "白子町", "長生村", "栄町", "四街道市", "流山市", "睦沢町", "白井市", "我孫子市",
  "芝山町", "鋸南町", "東庄町", "長柄町", "八千代市", "富里市", "市川市", "大網白里市",
  "松戸市", "長南町", "横芝光町", "多古町", "八街市", "銚子市", "船橋市", "東金市",
  "勝浦市", "袖ケ浦市", "茂原市", "匝瑳市", "佐倉市", "野田市", "館山市", "柏市",
  "印西市", "大多喜町", "旭市", "木更津市", "山武市", "いすみ市", "鴨川市", "富津市",
  "成田市", "南房総市", "香取市", "千葉市", "君津市", "市原市"]
    zoom_level = 18

    for city_name in selected_city:
        tiles = get_tiles_in_city(city_name, zoom_level)

        output_dir = OUTPUT_ROOT / f"{city_name}_downloading"
        downloader = GsiTilesDownloader(str(output_dir))

        downloader.download_tiles(tiles, zoom_level, CHECK_START)

if __name__ == "__main__":
    main()
