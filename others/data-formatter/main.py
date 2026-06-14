import os
import pandas as pd

BASE_DIR = os.path.dirname(__file__)

pop_df = pd.read_excel(
    os.path.join(BASE_DIR, "input_population/population.xlsx"),
    header=None,
    skiprows=5,
    usecols=[0, 1],
    names=["地域名", "人口"],
)
ward_names = ["県計", "中央区", "花見川区", "稲毛区", "若葉区", "緑区", "美浜区"]
pop_df = pop_df[~pop_df["地域名"].isin(ward_names)].reset_index(drop=True)

park_df = pd.read_csv(os.path.join(BASE_DIR, "input_parking/area_by_region.csv"))

merged = pd.merge(park_df, pop_df, on="地域名")

output_dir = os.path.join(BASE_DIR, "output")
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, "area_by_region.csv")
merged.to_csv(output_path, index=False)
print(f"保存: {output_path}")
