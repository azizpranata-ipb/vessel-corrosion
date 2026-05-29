from __future__ import annotations

import cv2
import numpy as np

from src.corrosion.schema import DetectionResult


SEVERITY_COLORS = {
    "none": (160, 160, 160),
    "mild": (0, 180, 0),
    "moderate": (0, 180, 255),
    "severe": (0, 80, 255),
    "critical": (0, 0, 255),
}


def draw_annotations(image_bgr: np.ndarray, detections: list[DetectionResult], full_mask: np.ndarray) -> np.ndarray:
    annotated = image_bgr.copy()

    if full_mask.size > 0 and cv2.countNonZero(full_mask) > 0:
        overlay = annotated.copy()
        overlay[full_mask > 0] = (0, 0, 255)
        annotated = cv2.addWeighted(overlay, 0.35, annotated, 0.65, 0)

    for detection in detections:
        x1, y1, x2, y2 = detection.bbox_xyxy
        color = SEVERITY_COLORS.get(detection.severity, (255, 255, 255))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        label = f"{detection.class_name} {detection.confidence:.2f} {detection.severity}"
        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
            cv2.LINE_AA,
        )

    return annotated
