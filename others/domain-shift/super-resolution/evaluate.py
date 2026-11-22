# 超解像・アップサンプリング結果を元の 20cm 画像（正解）と比較し、PSNR / SSIM を表示するスクリプト。
# RGB 3バンドのみで評価する（alpha は除外）。

from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

BASE_DIR = Path(__file__).parent
DOMAIN_SHIFT_DIR = BASE_DIR.parent

GROUND_TRUTH_PATH = DOMAIN_SHIFT_DIR / "downsampling" / "input" / "google-satellite_20cm.tif"
CANDIDATES = {
	"Lanczos": DOMAIN_SHIFT_DIR / "upsampling" / "output" / "google-satellite_20cm.tif",
	"Real-ESRGAN": BASE_DIR / "output" / "google-satellite_20cm_realesrgan.tif",
	"SwinIR": BASE_DIR / "output" / "google-satellite_20cm_swinir.tif",
}


def read_rgb(path: Path) -> np.ndarray:
	with rasterio.open(path) as src:
		return src.read([1, 2, 3]).transpose(1, 2, 0)


def main() -> None:
	if not GROUND_TRUTH_PATH.exists():
		raise FileNotFoundError(f"Ground truth not found: {GROUND_TRUTH_PATH}")

	ground_truth = read_rgb(GROUND_TRUTH_PATH)
	print(f"Ground truth: {GROUND_TRUTH_PATH.name} ({ground_truth.shape[1]}x{ground_truth.shape[0]})")
	print()
	print(f"{'Method':<12} {'PSNR [dB]':>10} {'SSIM':>8}")
	print("-" * 32)

	for method, path in CANDIDATES.items():
		if not path.exists():
			print(f"{method:<12} {'(missing)':>10}")
			continue

		prediction = read_rgb(path)
		if prediction.shape != ground_truth.shape:
			raise ValueError(
				f"{method}: shape {prediction.shape} does not match ground truth {ground_truth.shape}"
			)

		psnr = peak_signal_noise_ratio(ground_truth, prediction, data_range=255)
		ssim = structural_similarity(ground_truth, prediction, channel_axis=-1, data_range=255)
		print(f"{method:<12} {psnr:>10.2f} {ssim:>8.4f}")


if __name__ == "__main__":
	main()
