import random
import shutil
from pathlib import Path

SEED = 42
TRAIN_COUNT = 544

input_dir = Path(__file__).parent / "input"
output_dir = Path(__file__).parent / "output"

mask_files = sorted(p.name for p in (input_dir / "mask").iterdir() if p.suffix == ".png")

random.seed(SEED)
random.shuffle(mask_files)

splits = {"train": mask_files[:TRAIN_COUNT], "validation": mask_files[TRAIN_COUNT:]}

for split, files in splits.items():
    for subdir in ("mask", "origin"):
        (output_dir / split / subdir).mkdir(parents=True, exist_ok=True)
    for name in files:
        for subdir in ("mask", "origin"):
            shutil.copy2(input_dir / subdir / name, output_dir / split / subdir / name)
    print(f"{split}: {len(files)}枚コピー完了")
