from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class ModelConfig:
    path: str
    confidence_threshold: float
    iou_threshold: float
    image_size: int


@dataclass(frozen=True)
class SegmentationConfig:
    min_area_px: int
    kernel_size: int
    lower_hsv: tuple[int, int, int]
    upper_hsv: tuple[int, int, int]
    use_adaptive_fallback: bool


@dataclass(frozen=True)
class SeverityConfig:
    mild_max_ratio: float
    moderate_max_ratio: float
    severe_max_ratio: float


@dataclass(frozen=True)
class AppConfig:
    model: ModelConfig
    segmentation: SegmentationConfig
    severity: SeverityConfig


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw: dict[str, Any] = yaml.safe_load(file)

    model = raw["model"]
    segmentation = raw["segmentation"]
    severity = raw["severity"]

    return AppConfig(
        model=ModelConfig(
            path=str(model["path"]),
            confidence_threshold=float(model["confidence_threshold"]),
            iou_threshold=float(model["iou_threshold"]),
            image_size=int(model["image_size"]),
        ),
        segmentation=SegmentationConfig(
            min_area_px=int(segmentation["min_area_px"]),
            kernel_size=int(segmentation["kernel_size"]),
            lower_hsv=tuple(int(v) for v in segmentation["lower_hsv"]),
            upper_hsv=tuple(int(v) for v in segmentation["upper_hsv"]),
            use_adaptive_fallback=bool(segmentation["use_adaptive_fallback"]),
        ),
        severity=SeverityConfig(
            mild_max_ratio=float(severity["mild_max_ratio"]),
            moderate_max_ratio=float(severity["moderate_max_ratio"]),
            severe_max_ratio=float(severity["severe_max_ratio"]),
        ),
    )
