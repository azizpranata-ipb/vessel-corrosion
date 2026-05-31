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
    parser = argparse.ArgumentParser(description="Evaluate a trained YOLOv8 corrosion detector.")
    parser.add_argument("--weights", required=True, help="Path to trained model, e.g. runs/detect/.../best.pt")
    parser.add_argument("--data", default="configs/data.yaml")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_yaml = resolve_data_yaml(args.data)
    model = YOLO(args.weights)
    metrics = model.val(
        data=str(data_yaml),
        imgsz=args.imgsz,
        batch=args.batch,
        split=args.split,
        device=args.device,
        plots=True,
    )

    print("YOLOv8 Detection Metrics")
    print(f"mAP50-95: {metrics.box.map:.6f}")
    print(f"mAP50   : {metrics.box.map50:.6f}")
    print(f"mAP75   : {metrics.box.map75:.6f}")
    print(f"Precision mean: {metrics.box.mp:.6f}")
    print(f"Recall mean   : {metrics.box.mr:.6f}")


if __name__ == "__main__":
    main()
