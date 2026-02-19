from PIL import Image
import os

# スクリプトのディレクトリに移動
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# TIFファイルを開いてPNGで保存
input_file = "mosaic_group_011_x931757_y413000.tif"
output_file = input_file.replace('.tif', '.png')

img = Image.open(input_file)
img.save(output_file, 'PNG')
img.close()

print(f"変換完了: {output_file}")