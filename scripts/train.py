from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

from src.corrosion.yolo_data import resolve_data_yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 for ship hull corrosion detection.")
    parser.add_argument("--data", default="configs/data.yaml", help="Path to YOLO data.yaml.")
    parser.add_argument("--model", default="yolov8n.pt", help="YOLOv8 base model or checkpoint.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="ship_corrosion")
    parser.add_argument("--device", default=None, help="Example: 0 for GPU, cpu for CPU.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = resolve_data_yaml(args.data)
    model = YOLO(args.model)
    model.train(
        data=str(data_yaml),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        patience=30,
        pretrained=True,
        plots=True,
    )


if __name__ == "__main__":
    main()
