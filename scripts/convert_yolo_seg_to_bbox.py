from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
SPLIT_MAP = {
    "train": "train",
    "valid": "val",
    "val": "val",
    "test": "test",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert YOLOv8 segmentation polygon labels to YOLO detection bounding boxes."
    )
    parser.add_argument(
        "--input-root",
        default="data/corrosion Instance Segmentation",
        help="Input dataset root containing train/valid/test images and labels.",
    )
    parser.add_argument(
        "--out-root",
        default="data/roboflow_yolo_bbox",
        help="Output YOLO detection dataset root.",
    )
    parser.add_argument("--clear", action="store_true", help="Clear output folder before conversion.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    out_root = Path(args.out_root)

    if args.clear and out_root.exists():
        shutil.rmtree(out_root)

    totals = {}
    for input_split, output_split in SPLIT_MAP.items():
        image_dir = input_root / input_split / "images"
        label_dir = input_root / input_split / "labels"
        if not image_dir.exists() or not label_dir.exists():
            continue

        out_image_dir = out_root / "images" / output_split
        out_label_dir = out_root / "labels" / output_split
        out_image_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)

        image_count = 0
        label_count = 0
        box_count = 0
        empty_label_count = 0
        for image_path in sorted(image_dir.iterdir()):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            label_path = label_dir / f"{image_path.stem}.txt"
            if not label_path.exists():
                print(f"Skip missing label: {image_path}")
                continue

            shutil.copy2(image_path, out_image_dir / image_path.name)
            boxes = convert_label_file(label_path)
            write_detection_labels(out_label_dir / label_path.name, boxes)

            image_count += 1
            label_count += 1
            box_count += len(boxes)
            if not boxes:
                empty_label_count += 1

        totals[output_split] = (image_count, label_count, box_count, empty_label_count)

    write_data_yaml(out_root)
    for split, (images, labels, boxes, empty) in totals.items():
        print(f"{split}: {images} images, {labels} labels, {boxes} boxes, {empty} empty labels")


def convert_label_file(label_path: Path) -> list[str]:
    boxes = []
    for line_number, line in enumerate(label_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue

        parts = stripped.split()
        if len(parts) < 5:
            raise ValueError(f"Invalid label with fewer than 5 values: {label_path}:{line_number}")

        class_id = int(float(parts[0]))
        values = [float(value) for value in parts[1:]]
        if len(values) == 4:
            x_center, y_center, width, height = values
        else:
            if len(values) % 2 != 0:
                raise ValueError(f"Invalid polygon coordinate count: {label_path}:{line_number}")
            xs = values[0::2]
            ys = values[1::2]
            if not xs or not ys:
                continue
            x_min = clamp(min(xs))
            x_max = clamp(max(xs))
            y_min = clamp(min(ys))
            y_max = clamp(max(ys))
            width = x_max - x_min
            height = y_max - y_min
            if width <= 0 or height <= 0:
                continue
            x_center = x_min + width / 2.0
            y_center = y_min + height / 2.0

        if width <= 0 or height <= 0:
            continue

        boxes.append(
            f"{class_id} {clamp(x_center):.6f} {clamp(y_center):.6f} "
            f"{clamp(width):.6f} {clamp(height):.6f}"
        )
    return boxes


def write_detection_labels(label_path: Path, boxes: list[str]) -> None:
    label_path.write_text("\n".join(boxes) + ("\n" if boxes else ""), encoding="utf-8")


def write_data_yaml(out_root: Path) -> None:
    payload = """path: .
train: images/train
val: images/val
test: images/test

names:
  0: corrosion
"""
    (out_root / "data.yaml").write_text(payload, encoding="utf-8")


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


if __name__ == "__main__":
    main()
