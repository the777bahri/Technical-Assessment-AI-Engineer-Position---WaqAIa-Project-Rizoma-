"""Runs the trained model on one new RGB + NIR image pair and reports percent area affected.

Usage:
    python predict.py path/to/rgb.jpg path/to/nir.jpg

Needs both an RGB image and its matching NIR image, same size, since that is what the model
was trained on. A normal RGB-only photo will not work.
"""
import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from model_utils import load_model, predict_mask

CKPT_PATH = "best_deeplabv3plus.pth"
THRESHOLD = 0.45  # calibrated threshold for this checkpoint, see Methodology.md


def predict(rgb_path, nir_path):
    rgb = np.array(Image.open(rgb_path).convert("RGB"))
    nir = np.array(Image.open(nir_path).convert("L"))
    rgb_nir = np.concatenate([rgb, nir[:, :, None]], axis=2)

    model = load_model(CKPT_PATH, encoder_name="resnet50")
    mask = predict_mask(model, rgb_nir, threshold=THRESHOLD)

    pct_affected = mask.mean() * 100
    print(f"percent of image affected: {pct_affected:.1f}%")

    overlay = rgb.copy()
    overlay[mask > 0] = [255, 0, 0]
    out_path = "prediction_overlay.png"
    plt.imsave(out_path, overlay)
    print(f"saved overlay to {out_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python predict.py path/to/rgb.jpg path/to/nir.jpg")
        sys.exit(1)
    predict(sys.argv[1], sys.argv[2])
