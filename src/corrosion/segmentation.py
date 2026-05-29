from __future__ import annotations

import cv2
import numpy as np

from src.corrosion.config import SegmentationConfig


def _ensure_odd_kernel(size: int) -> int:
    size = max(3, int(size))
    return size if size % 2 == 1 else size + 1


def segment_corrosion_roi(roi_bgr: np.ndarray, config: SegmentationConfig) -> np.ndarray:
    """Segment corrosion pixels inside one YOLO bounding box ROI.

    The default rule targets reddish/brown/yellow corrosion tones in HSV, then
    cleans noise with morphology. If the HSV mask is nearly empty, an adaptive
    threshold fallback is used to capture darker, rough corrosion texture.
    """
    if roi_bgr.size == 0:
        return np.zeros((0, 0), dtype=np.uint8)

    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)
    lower = np.array(config.lower_hsv, dtype=np.uint8)
    upper = np.array(config.upper_hsv, dtype=np.uint8)
    hsv_mask = cv2.inRange(hsv, lower, upper)

    kernel_size = _ensure_odd_kernel(config.kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    if config.use_adaptive_fallback and cv2.countNonZero(mask) < config.min_area_px:
        gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)
        adaptive = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            4,
        )
        mask = cv2.bitwise_or(mask, adaptive)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    return remove_small_components(mask, config.min_area_px)


def remove_small_components(mask: np.ndarray, min_area_px: int) -> np.ndarray:
    if mask.size == 0:
        return mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    cleaned = np.zeros(mask.shape, dtype=np.uint8)

    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area >= min_area_px:
            cleaned[labels == label] = 255

    return cleaned


def paste_roi_mask(full_shape: tuple[int, int], roi_mask: np.ndarray, bbox_xyxy: list[int]) -> np.ndarray:
    full_mask = np.zeros(full_shape, dtype=np.uint8)
    x1, y1, x2, y2 = bbox_xyxy
    full_mask[y1:y2, x1:x2] = roi_mask
    return full_mask


def mask_area_px(mask: np.ndarray) -> int:
    return int(cv2.countNonZero(mask))


def area_cm2(area_px: int, mm_per_pixel: float | None) -> float | None:
    if mm_per_pixel is None:
        return None
    area_mm2 = float(area_px) * (mm_per_pixel**2)
    return area_mm2 / 100.0
