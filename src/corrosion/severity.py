from __future__ import annotations

from src.corrosion.config import SeverityConfig


def classify_severity(corrosion_ratio: float, config: SeverityConfig) -> str:
    """Classify severity from corrosion ratio.

    Ratio can be total corrosion area / image area or corrosion area / bbox area.
    Tune thresholds according to your research rule.
    """
    if corrosion_ratio <= 0:
        return "none"
    if corrosion_ratio < config.mild_max_ratio:
        return "mild"
    if corrosion_ratio < config.moderate_max_ratio:
        return "moderate"
    if corrosion_ratio < config.severe_max_ratio:
        return "severe"
    return "critical"
