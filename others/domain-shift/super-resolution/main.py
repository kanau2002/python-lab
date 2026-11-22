# google-satellite_50cm.tif を学習済み超解像モデルで 20cm/px 相当に復元するスクリプト。
# モデル（Real-ESRGAN x4plus / SwinIR-M classical x4）で ×4 に拡大した後、
# Lanczos で目標解像度（0.2 m/px）のピクセル数に縮小し、同じ bounds の GeoTIFF として保存する。

from __future__ import annotations

import os

# MPS 未対応の演算があれば CPU に逃がす（torch の import より前に設定する必要がある）
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import argparse
import math
import shutil
import ssl
import time
import urllib.request
from pathlib import Path

import certifi
import numpy as np
import rasterio
import torch
from PIL import Image
from rasterio.transform import from_bounds
from spandrel import ImageModelDescriptor, ModelLoader

TARGET_RESOLUTION_M = 0.2
TILE_PAD = 16

BASE_DIR = Path(__file__).parent
INPUT_PATH = BASE_DIR / "input" / "gsi-satellite_z18.tif"
OUTPUT_DIR = BASE_DIR / "output"
WEIGHTS_DIR = BASE_DIR / "weights"

# tile: 1回の推論に渡す入力タイルの一辺（px）。SwinIR は attention がメモリを食うため小さめにする
MODELS = {
	"realesrgan": {
		"url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
		"tile": 512,
	},
	"swinir": {
		"url": "https://github.com/JingyunLiang/SwinIR/releases/download/v0.0/001_classicalSR_DF2K_s64w8_SwinIR-M_x4.pth",
		"tile": 256,
	},
}


def calculate_target_size(bounds: tuple[float, float, float, float]) -> int:
	west, south, east, north = bounds
	center_lat = (north + south) / 2
	lon_distance = (east - west) * 111320 * math.cos(math.radians(center_lat))
	lat_distance = (north - south) * 110540
	average_ground_distance = (lon_distance + lat_distance) / 2
	return max(1, int(round(average_ground_distance / TARGET_RESOLUTION_M)))


def get_device() -> torch.device:
	if torch.backends.mps.is_available():
		return torch.device("mps")
	if torch.cuda.is_available():
		return torch.device("cuda")
	return torch.device("cpu")


def download_weights(name: str) -> Path:
	url = MODELS[name]["url"]
	weights_path = WEIGHTS_DIR / Path(url).name
	if not weights_path.exists():
		WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
		print(f"Downloading weights: {weights_path.name}")
		tmp_path = weights_path.with_suffix(weights_path.suffix + ".part")
		# python.org 版 macOS Python は証明書が未設定のことがあるため certifi の CA を使う
		ssl_context = ssl.create_default_context(cafile=certifi.where())
		with urllib.request.urlopen(url, context=ssl_context) as response, open(tmp_path, "wb") as f:
			shutil.copyfileobj(response, f)
		tmp_path.rename(weights_path)
	return weights_path


def load_model(name: str, device: torch.device) -> ImageModelDescriptor:
	model = ModelLoader().load_from_file(download_weights(name))
	if not isinstance(model, ImageModelDescriptor):
		raise TypeError(f"Unsupported model type for {name}: {type(model)}")
	return model.to(device).eval()


def run_tiled(
	model: ImageModelDescriptor,
	rgb: np.ndarray,
	tile: int,
	device: torch.device,
	pad: int = TILE_PAD,
) -> np.ndarray:
	# タイルごとに周囲 pad px を足して推論し、出力から足した分を切り落として貼り合わせる（継ぎ目対策）
	scale = model.scale
	height, width, _ = rgb.shape
	output = np.zeros((height * scale, width * scale, 3), dtype=np.uint8)
	image = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0)

	ys = range(0, height, tile)
	xs = range(0, width, tile)
	total = len(ys) * len(xs)
	done = 0

	with torch.inference_mode():
		for y0 in ys:
			y1 = min(y0 + tile, height)
			py0, py1 = max(y0 - pad, 0), min(y1 + pad, height)
			for x0 in xs:
				x1 = min(x0 + tile, width)
				px0, px1 = max(x0 - pad, 0), min(x1 + pad, width)

				tile_in = image[:, py0:py1, px0:px1].unsqueeze(0).to(device)
				tile_out = model(tile_in).squeeze(0).clamp(0, 1).mul(255).round().byte().cpu()

				oy0 = (y0 - py0) * scale
				ox0 = (x0 - px0) * scale
				oy1 = oy0 + (y1 - y0) * scale
				ox1 = ox0 + (x1 - x0) * scale
				output[y0 * scale:y1 * scale, x0 * scale:x1 * scale] = (
					tile_out[:, oy0:oy1, ox0:ox1].permute(1, 2, 0).numpy()
				)

				done += 1
				print(f"\r  tiles: {done}/{total}", end="", flush=True)
	print()
	return output


def super_resolve(name: str, device: torch.device) -> Path:
	with rasterio.open(INPUT_PATH) as src:
		array = src.read()
		b = src.bounds
		bounds = (b.left, b.bottom, b.right, b.top)
		crs = src.crs

	west, south, east, north = bounds
	target_size = calculate_target_size(bounds)
	src_count = array.shape[0]
	if src_count < 3:
		raise ValueError(f"RGB input is required, got {src_count} band(s)")

	model = load_model(name, device)
	print(f"[{name}] {model.architecture.name} x{model.scale} on {device}")

	start = time.perf_counter()
	rgb = array[:3].transpose(1, 2, 0)
	sr_rgb = run_tiled(model, rgb, MODELS[name]["tile"], device)
	elapsed = time.perf_counter() - start

	# ×4 出力を目標解像度のピクセル数に縮小（50cm → 20cm は ×2.5 のため）
	resized_rgb = np.array(
		Image.fromarray(sr_rgb, mode="RGB").resize((target_size, target_size), Image.LANCZOS)
	)
	bands = [resized_rgb.transpose(2, 0, 1)]

	# alpha などの残りのバンドは Lanczos で拡大して戻す
	for extra in array[3:]:
		resized_extra = np.array(
			Image.fromarray(extra, mode="L").resize((target_size, target_size), Image.LANCZOS)
		)
		bands.append(resized_extra[np.newaxis])
	result = np.concatenate(bands, axis=0)

	output_path = OUTPUT_DIR / f"google-satellite_20cm_{name}.tif"
	output_path.parent.mkdir(parents=True, exist_ok=True)
	dst_transform = from_bounds(west, south, east, north, target_size, target_size)

	with rasterio.open(
		output_path,
		"w",
		driver="GTiff",
		height=target_size,
		width=target_size,
		count=src_count,
		dtype=result.dtype,
		crs=crs,
		transform=dst_transform,
	) as dst:
		dst.write(result)

	print(f"[{name}] {sr_rgb.shape[1]}x{sr_rgb.shape[0]} -> {target_size}x{target_size}, inference {elapsed:.1f}s")
	print(f"[{name}] saved: {output_path.name}")

	del model
	if device.type == "mps":
		torch.mps.empty_cache()
	return output_path


def main() -> None:
	parser = argparse.ArgumentParser(description="Super-resolve 50cm/px imagery to 20cm/px")
	parser.add_argument("--model", choices=[*MODELS, "all"], default="all")
	args = parser.parse_args()

	if not INPUT_PATH.exists():
		raise FileNotFoundError(f"Input file not found: {INPUT_PATH}")

	device = get_device()
	names = list(MODELS) if args.model == "all" else [args.model]
	for name in names:
		super_resolve(name, device)


if __name__ == "__main__":
	main()
