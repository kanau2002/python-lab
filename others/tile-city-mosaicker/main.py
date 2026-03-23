from pathlib import Path
import rasterio
from rasterio.merge import merge

def mosaic_tiles(input_dir: str, output_file: str):
    """タイル画像を結合して1枚のモザイク画像を作成"""
    
    input_path = Path(input_dir)
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # すべてのGeoTIFFファイルを取得
    tile_files = sorted(input_path.glob("tile_*.tif"))
    
    if not tile_files:
        print(f"エラー: {input_path} にタイルファイルが見つかりません")
        return
    
    print(f"タイル数: {len(tile_files)}枚")
    print("タイルを結合中...")
    
    # rasterioでファイルを開く
    src_files_to_mosaic = []
    
    try:
        for tile_file in tile_files:
            src = rasterio.open(tile_file)
            src_files_to_mosaic.append(src)
        
        # タイルを結合
        mosaic, out_trans = merge(src_files_to_mosaic)
        
        # メタデータをコピー
        out_meta = src_files_to_mosaic[0].meta.copy()
        
        # 結合後のメタデータを更新
        out_meta.update({
            "driver": "GTiff",
            "height": mosaic.shape[1],
            "width": mosaic.shape[2],
            "transform": out_trans,
        })
        
        # 結合画像を保存
        print(f"画像を保存中: {output_path}")
        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(mosaic)
        
        print(f"✓ 結合完了: {output_path}")
        print(f"  サイズ: {mosaic.shape[2]} x {mosaic.shape[1]} ピクセル")
        
    finally:
        # ファイルを閉じる
        for src in src_files_to_mosaic:
            src.close()

def main():
    input_dir = Path(__file__).parent / "input"
    output_file = Path(__file__).parent / "output" / "mosaic_urayasu.tif"
    
    mosaic_tiles(str(input_dir), str(output_file))

if __name__ == "__main__":
    main()