from __future__ import annotations

import cv2
import numpy as np

from src.corrosion.schema import DetectionResult


SEVERITY_COLORS = {
    "none": (160, 160, 160),
    "low": (61, 184, 112),
    "medium": (34, 159, 235),
    "high": (44, 85, 220),
}


def draw_annotations(image_bgr: np.ndarray, detections: list[DetectionResult], full_mask: np.ndarray) -> np.ndarray:
    annotated = image_bgr.copy()

    if full_mask.size > 0 and cv2.countNonZero(full_mask) > 0:
        overlay = annotated.copy()
        overlay[full_mask > 0] = (0, 165, 255)
        annotated = cv2.addWeighted(overlay, 0.22, annotated, 0.78, 0)

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox_xyxy
        color = SEVERITY_COLORS.get(detection.severity, (255, 255, 255))
        thickness = max(2, round(min(annotated.shape[:2]) / 220))
        font_scale = max(0.55, min(0.9, annotated.shape[1] / 1150))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, thickness)

        label = f"#{detection.id} {detection.confidence:.2f} {detection.severity}"
        draw_label(annotated, label, x1, y1, color, font_scale)

    return annotated


def draw_label(image: np.ndarray, text: str, x: int, y: int, color: tuple[int, int, int], font_scale: float) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    padding = 6
    text_size, baseline = cv2.getTextSize(text, font, font_scale, thickness)
    text_w, text_h = text_size

    label_x1 = max(0, x)
    label_y2 = y - 4
    label_y1 = label_y2 - text_h - baseline - padding * 2
    if label_y1 < 0:
        label_y1 = y + 4
        label_y2 = label_y1 + text_h + baseline + padding * 2

    label_x2 = min(image.shape[1] - 1, label_x1 + text_w + padding * 2)
    cv2.rectangle(image, (label_x1, label_y1), (label_x2, label_y2), color, -1)
    cv2.rectangle(image, (label_x1, label_y1), (label_x2, label_y2), (255, 255, 255), 1)
    cv2.putText(
        image,
        text,
        (label_x1 + padding, label_y2 - baseline - padding),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
