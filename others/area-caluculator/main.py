import geopandas as gpd

shp_path = "input_boundary/N03-20250101_12.shp"
output_path = "output/municipality_area.csv"

gdf = gpd.read_file(shp_path)

# 面積計算のため平面直角座標系（JGD2011 第9系）に変換
gdf_proj = gdf.to_crs(epsg=6677)

gdf_proj["area_km2"] = gdf_proj.geometry.area / 1_000_000

result = (
    gdf_proj.groupby("N03_004", as_index=False)["area_km2"]
    .sum()
    .sort_values("area_km2", ascending=False)
    .reset_index(drop=True)
)

result.to_csv(output_path, index=False, encoding="utf-8-sig")

print(result.to_string(index=False))
print(f"\n保存先: {output_path}")
