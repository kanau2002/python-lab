from pathlib import Path

import geopandas as gpd

INPUT_BOUNDARY = Path(__file__).parent / "input_boudary" / "N03-20250101_12.shp"
OUTPUT_PATH = Path(__file__).parent / "output" / "chiba_dissolved.geojson"


def main() -> None:
    gdf = gpd.read_file(INPUT_BOUNDARY)
    gdf = gdf[gdf["N03_004"] != "所属未定地"]

    dissolved = gdf.dissolve(by="N03_004", aggfunc="first").reset_index()

    if len(dissolved) != 54:
        raise ValueError(f"Expected 54 municipalities, got {len(dissolved)}")

    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    dissolved.to_file(OUTPUT_PATH, driver="GeoJSON")
    print(f"Saved {len(dissolved)} municipalities → {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
