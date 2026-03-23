from pathlib import Path
import rasterio

CROP_SIZE = 155

def main():
    input_dir = Path(__file__).parent / "input"
    output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(exist_ok=True)
    
    for tif_file in input_dir.glob("*.tif"):
        with rasterio.open(tif_file) as src:
            data = src.read()
            h, w = data.shape[1], data.shape[2]
            
            cropped = data[:, CROP_SIZE:h-CROP_SIZE, CROP_SIZE:w-CROP_SIZE]
            new_h, new_w = cropped.shape[1], cropped.shape[2]
            
            bounds = src.bounds
            pw = (bounds.right - bounds.left) / w
            ph = (bounds.top - bounds.bottom) / h
            
            new_bounds = (
                bounds.left + CROP_SIZE * pw,
                bounds.bottom + CROP_SIZE * ph,
                bounds.right - CROP_SIZE * pw,
                bounds.top - CROP_SIZE * ph
            )
            
            transform = rasterio.transform.from_bounds(*new_bounds, new_w, new_h)
            
            with rasterio.open(
                output_dir / tif_file.name,
                'w',
                driver='GTiff',
                height=new_h,
                width=new_w,
                count=src.count,
                dtype=src.dtypes[0],
                crs=src.crs,
                transform=transform,
                nodata=0
            ) as dst:
                dst.write(cropped)

if __name__ == "__main__":
    main()