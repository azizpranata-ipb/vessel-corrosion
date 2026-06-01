from __future__ import annotations

from src.corrosion.config import SeverityConfig


def classify_severity(corrosion_ratio: float, config: SeverityConfig) -> str:
    """Classify per-detection severity from corrosion ratio.

    Low/medium/high labels are easier to aggregate than many severity levels.
    Tune thresholds according to your research rule in configs/app.yaml.
    """
    if corrosion_ratio <= 0:
        return "none"
    if corrosion_ratio < config.mild_max_ratio:
        return "low"
    if corrosion_ratio < config.moderate_max_ratio:
        return "medium"
    return "high"


def severity_score(severity: str) -> int:
    return {
        "none": 0,
        "low": 1,
        "medium": 2,
        "high": 3,
    }.get(severity, 0)


def aggregate_severity(severities: list[str]) -> tuple[str, float, dict[str, int]]:
    if not severities:
        return "none", 0.0, {"low": 0, "medium": 0, "high": 0}

    counts = {
        "low": severities.count("low"),
        "medium": severities.count("medium"),
        "high": severities.count("high"),
    }
    average_score = sum(severity_score(severity) for severity in severities) / len(severities)

    if average_score < 1.5:
        return "low", round(average_score, 3), counts
    if average_score < 2.5:
        return "medium", round(average_score, 3), counts
    return "high", round(average_score, 3), counts
