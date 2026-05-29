from __future__ import annotations

from pydantic import BaseModel, Field


class ImageInfo(BaseModel):
    filename: str
    width: int
    height: int


class DetectionResult(BaseModel):
    id: int
    class_name: str
    confidence: float
    bbox_xyxy: list[int] = Field(..., min_length=4, max_length=4)
    bbox_area_px: int
    corrosion_area_px: int
    corrosion_area_cm2: float | None
    corrosion_ratio_in_bbox: float
    severity: str
    dice_coefficient: float | None = None


class SummaryResult(BaseModel):
    detections: int
    total_corrosion_area_px: int
    total_corrosion_area_cm2: float | None
    corrosion_ratio: float
    severity: str


class ArtifactResult(BaseModel):
    annotated_image_url: str | None = None
    mask_image_url: str | None = None


class PredictionResponse(BaseModel):
    image: ImageInfo
    summary: SummaryResult
    detections: list[DetectionResult]
    artifacts: ArtifactResult
