from __future__ import annotations

import cv2
import numpy as np


def dice_coefficient(pred_mask: np.ndarray, gt_mask: np.ndarray, smooth: float = 1e-6) -> float:
    """Compute Dice Coefficient for two binary masks."""
    pred = (pred_mask > 0).astype(np.uint8)
    gt = (gt_mask > 0).astype(np.uint8)

    if pred.shape != gt.shape:
        gt = cv2.resize(gt, (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
        gt = (gt > 0).astype(np.uint8)

    intersection = float(np.sum(pred * gt))
    denominator = float(np.sum(pred) + np.sum(gt))
    return (2.0 * intersection + smooth) / (denominator + smooth)


def load_mask_grayscale(path: str) -> np.ndarray:
    mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise FileNotFoundError(f"Mask not found or unreadable: {path}")
    return mask
