from PIL import Image
import os

INPUT = "input/20cm_5944px.png"
OUTPUT = "output/20cm_5944px_5944.png"
SIZE = 5944

script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

img = Image.open(INPUT)
width, height = img.size

left = width // 2 - SIZE // 2
top = height // 2 - SIZE // 2
right = width // 2 + SIZE // 2
bottom = height // 2 + SIZE // 2

cropped = img.crop((left, top, right, bottom))
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
cropped.save(OUTPUT)