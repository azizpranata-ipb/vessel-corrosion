from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def resolve_data_yaml(data_yaml: str | Path, project_root: str | Path | None = None) -> Path:
    """Create a YOLO data.yaml copy with an absolute dataset path.

    Ultralytics may resolve relative dataset paths against its global
    datasets_dir setting. This helper makes training independent from that
    setting by writing a temporary config with an absolute `path`.
    """
    data_yaml = Path(data_yaml)
    root = Path(project_root) if project_root else Path.cwd()

    with data_yaml.open("r", encoding="utf-8") as file:
        payload: dict[str, Any] = yaml.safe_load(file)

    dataset_path = Path(payload.get("path", ""))
    if not dataset_path.is_absolute():
        payload["path"] = str((root / dataset_path).resolve())

    output_dir = root / "outputs" / "reports"
    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_path = output_dir / f"resolved_{data_yaml.name}"
    with resolved_path.open("w", encoding="utf-8") as file:
        yaml.safe_dump(payload, file, sort_keys=False)

    return resolved_path
