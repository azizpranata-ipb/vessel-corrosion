from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from ultralytics import YOLO

from src.corrosion.config import AppConfig
from src.corrosion.dice import dice_coefficient, load_mask_grayscale
from src.corrosion.schema import ArtifactResult, DetectionResult, ImageInfo, PredictionResponse, SummaryResult
from src.corrosion.segmentation import area_cm2, mask_area_px, paste_roi_mask, segment_corrosion_roi
from src.corrosion.severity import aggregate_severity, classify_severity, severity_score
from src.corrosion.visualization import draw_annotations


class CorrosionAnalyzer:
    def __init__(self, config: AppConfig, model_path: str | Path | None = None) -> None:
        self.config = config
        self.model_path = str(model_path or config.model.path)
        self.model = YOLO(self.model_path)

    def analyze_image(
        self,
        image_path: str | Path,
        output_dir: str | Path | None = None,
        mm_per_pixel: float | None = None,
        gt_mask_path: str | Path | None = None,
        artifact_url_prefix: str = "/outputs/predictions",
    ) -> PredictionResponse:
        image_path = Path(image_path)
        image_bgr = cv2.imread(str(image_path))
        if image_bgr is None:
            raise FileNotFoundError(f"Image not found or unreadable: {image_path}")

        height, width = image_bgr.shape[:2]
        image_area = width * height
        result = self.model.predict(
            source=str(image_path),
            conf=self.config.model.confidence_threshold,
            iou=self.config.model.iou_threshold,
            imgsz=self.config.model.image_size,
            verbose=False,
        )[0]

        full_mask = np.zeros((height, width), dtype=np.uint8)
        detections: list[DetectionResult] = []
        gt_mask = load_mask_grayscale(str(gt_mask_path)) if gt_mask_path else None

        boxes = result.boxes
        if boxes is not None:
            for index, box in enumerate(boxes, start=1):
                x1, y1, x2, y2 = [int(round(v)) for v in box.xyxy[0].tolist()]
                x1 = max(0, min(x1, width - 1))
                y1 = max(0, min(y1, height - 1))
                x2 = max(x1 + 1, min(x2, width))
                y2 = max(y1 + 1, min(y2, height))

                roi = image_bgr[y1:y2, x1:x2]
                roi_mask = segment_corrosion_roi(roi, self.config.segmentation)
                bbox_area = (x2 - x1) * (y2 - y1)
                corrosion_area = mask_area_px(roi_mask)
                ratio_in_bbox = corrosion_area / bbox_area if bbox_area else 0.0
                severity = classify_severity(ratio_in_bbox, self.config.severity)

                roi_full_mask = paste_roi_mask((height, width), roi_mask, [x1, y1, x2, y2])
                full_mask = cv2.bitwise_or(full_mask, roi_full_mask)

                dice_value = None
                if gt_mask is not None:
                    gt_roi = gt_mask[y1:y2, x1:x2]
                    dice_value = round(float(dice_coefficient(roi_mask, gt_roi)), 6)

                cls_id = int(box.cls[0].item()) if box.cls is not None else 0
                class_name = self.model.names.get(cls_id, "corrosion")
                confidence = float(box.conf[0].item()) if box.conf is not None else 0.0

                detections.append(
                    DetectionResult(
                        id=index,
                        class_name=str(class_name),
                        confidence=round(confidence, 6),
                        bbox_xyxy=[x1, y1, x2, y2],
                        bbox_area_px=bbox_area,
                        corrosion_area_px=corrosion_area,
                        corrosion_area_cm2=area_cm2(corrosion_area, mm_per_pixel),
                        corrosion_ratio_in_bbox=round(ratio_in_bbox, 6),
                        severity=severity,
                        severity_score=severity_score(severity),
                        dice_coefficient=dice_value,
                    )
                )

        total_area_px = mask_area_px(full_mask)
        total_ratio = total_area_px / image_area if image_area else 0.0
        summary_severity, summary_score, severity_counts = aggregate_severity(
            [detection.severity for detection in detections]
        )

        artifacts = ArtifactResult()
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            stem = f"{image_path.stem}_{uuid4().hex[:10]}"
            annotated_name = f"annotated_{stem}.jpg"
            mask_name = f"mask_{stem}.png"
            annotated = draw_annotations(image_bgr, detections, full_mask)
            cv2.imwrite(str(output_path / annotated_name), annotated)
            cv2.imwrite(str(output_path / mask_name), full_mask)
            artifacts = ArtifactResult(
                annotated_image_url=f"{artifact_url_prefix}/{annotated_name}",
                mask_image_url=f"{artifact_url_prefix}/{mask_name}",
            )

        return PredictionResponse(
            image=ImageInfo(filename=image_path.name, width=width, height=height),
            summary=SummaryResult(
                detections=len(detections),
                total_corrosion_area_px=total_area_px,
                total_corrosion_area_cm2=area_cm2(total_area_px, mm_per_pixel),
                corrosion_ratio=round(total_ratio, 6),
                severity=summary_severity,
                severity_score=summary_score,
                severity_counts=severity_counts,
            ),
            detections=detections,
            artifacts=artifacts,
        )
