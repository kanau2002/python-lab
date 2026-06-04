import geopandas as gpd
import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).parent
INPUT_PARKING_DIR = BASE_DIR / "input_parking"
INPUT_BOUNDARY_DIR = BASE_DIR / "input_boundary"
OUTPUT_PATH = BASE_DIR / "output/area_by_region.csv"

# 境界データから市町村ごとの総面積を算出
boundary_files = list(INPUT_BOUNDARY_DIR.glob("*.geojson")) + list(INPUT_BOUNDARY_DIR.glob("*.shp"))
boundary_gdf = gpd.read_file(boundary_files[0]).to_crs(epsg=6677)
boundary_gdf["area_m2"] = boundary_gdf.geometry.area
municipality_area = (
    boundary_gdf.groupby("N03_004")["area_m2"].sum()
)

records = []

for gpkg_path in sorted(INPUT_PARKING_DIR.glob("*.gpkg")):
    region = gpkg_path.stem
    gdf = gpd.read_file(gpkg_path)

    if "area_m2" in gdf.columns:
        parking_m2 = gdf["area_m2"].sum()
    else:
        gdf_proj = gdf.to_crs(epsg=6677)
        parking_m2 = gdf_proj.geometry.area.sum()

    total_municipality_m2 = municipality_area.get(region, float("nan"))

    records.append({
        "地域名": region,
        "市町村総面積_km2": round(total_municipality_m2 / 1_000_000, 6),
        "駐車場数": len(gdf),
        "駐車場総面積_km2": round(parking_m2 / 1_000_000, 6),
    })

result = pd.DataFrame(records).sort_values("市町村総面積_km2", ascending=False).reset_index(drop=True)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
result.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")

print(result.to_string(index=False))
print(f"\n保存先: {OUTPUT_PATH}")
