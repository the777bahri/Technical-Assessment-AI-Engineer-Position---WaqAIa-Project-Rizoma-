"""Shared model loading and prediction logic, used by predict.py and by
Models Comparison/Models_Comparison.ipynb, so both point at the same code instead of each
keeping their own copy.
"""
import numpy as np
import torch
import segmentation_models_pytorch as smp


def load_model(ckpt_path, encoder_name="resnet50", device="cpu"):
    model = smp.DeepLabV3Plus(
        encoder_name=encoder_name, encoder_weights=None, in_channels=4, classes=1, activation=None
    )
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def predict_mask(model, rgb_nir, device="cpu", threshold=0.45):
    """rgb_nir: (H, W, 4) uint8 array, RGB channels then NIR. Returns a (H, W) uint8 mask of 0/1."""
    x = torch.from_numpy(rgb_nir.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return (probs > threshold).astype(np.uint8)
