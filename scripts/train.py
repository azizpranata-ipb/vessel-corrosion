from __future__ import annotations

import argparse

from ultralytics import YOLO


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
    model = YOLO(args.model)
    model.train(
        data=args.data,
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
