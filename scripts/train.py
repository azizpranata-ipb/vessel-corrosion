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
    parser.add_argument("--lr0", type=float, default=0.01, help="Initial learning rate.")
    parser.add_argument("--lrf", type=float, default=0.01, help="Final learning rate fraction.")
    parser.add_argument("--optimizer", default="auto", help="Optimizer, e.g. auto, SGD, Adam, AdamW.")
    parser.add_argument("--patience", type=int, default=30, help="Early stopping patience in epochs.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--no-auto-augment",
        action="store_true",
        help="Disable YOLO runtime augmentations for experiments with manually augmented data.",
    )
    parser.add_argument(
        "--light-augment",
        action="store_true",
        help="Use a conservative runtime augmentation preset for corrosion detection.",
    )
    parser.add_argument("--project", default="runs/detect")
    parser.add_argument("--name", default="ship_corrosion")
    parser.add_argument("--device", default=None, help="Example: 0 for GPU, cpu for CPU.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.no_auto_augment and args.light_augment:
        raise ValueError("--no-auto-augment and --light-augment cannot be used together.")
    if args.no_auto_augment:
        disable_albumentations()

    data_yaml = resolve_data_yaml(args.data)
    model = YOLO(args.model)
    train_kwargs = {
        "data": str(data_yaml),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "project": args.project,
        "name": args.name,
        "device": args.device,
        "lr0": args.lr0,
        "lrf": args.lrf,
        "optimizer": args.optimizer,
        "patience": args.patience,
        "seed": args.seed,
        "workers": args.workers,
        "pretrained": True,
        "plots": True,
    }

    if args.no_auto_augment:
        train_kwargs.update(
            {
                "hsv_h": 0.0,
                "hsv_s": 0.0,
                "hsv_v": 0.0,
                "degrees": 0.0,
                "translate": 0.0,
                "scale": 0.0,
                "shear": 0.0,
                "perspective": 0.0,
                "flipud": 0.0,
                "fliplr": 0.0,
                "mosaic": 0.0,
                "mixup": 0.0,
                "copy_paste": 0.0,
                "auto_augment": None,
                "erasing": 0.0,
            }
        )
    elif args.light_augment:
        train_kwargs.update(
            {
                "hsv_h": 0.01,
                "hsv_s": 0.20,
                "hsv_v": 0.20,
                "degrees": 7.0,
                "translate": 0.05,
                "scale": 0.10,
                "shear": 0.0,
                "perspective": 0.0,
                "flipud": 0.0,
                "fliplr": 0.5,
                "mosaic": 0.20,
                "mixup": 0.0,
                "copy_paste": 0.0,
            }
        )

    model.train(**train_kwargs)


def disable_albumentations() -> None:
    """Disable Ultralytics' built-in Albumentations transforms for a clean manual-augmentation run."""
    import ultralytics.data.augment as yolo_augment

    class NoOpAlbumentations:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, labels):
            return labels

    yolo_augment.Albumentations = NoOpAlbumentations


if __name__ == "__main__":
    main()
