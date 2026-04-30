from pathlib import Path

import geopandas as gpd
import pandas as pd

base = Path(__file__).parent
input_dir = base / "input"
output_path = base / "output" / "merged.geojson"
output_path.parent.mkdir(parents=True, exist_ok=True)

gpkg_files = sorted(input_dir.glob("*.gpkg"))
if not gpkg_files:
    raise FileNotFoundError(f"No .gpkg files found in {input_dir}")

gdfs = []
for gpkg_path in gpkg_files:
    municipality = gpkg_path.stem
    print(f"Reading: {municipality}")
    for layer in gpd.list_layers(gpkg_path)["name"]:
        gdf = gpd.read_file(gpkg_path, layer=layer)
        if gdf.empty:
            continue
        if gdf.crs is not None and gdf.crs.to_epsg() != 4326:
            gdf = gdf.to_crs(epsg=4326)
        gdf["municipality"] = municipality
        gdfs.append(gdf)

if not gdfs:
    raise ValueError("No features found in any .gpkg file")

merged = pd.concat(gdfs, ignore_index=True)
merged.to_file(output_path, driver="GeoJSON")
print(f"Saved: {output_path}  ({len(merged)} features, {merged['municipality'].nunique()} municipalities)")
