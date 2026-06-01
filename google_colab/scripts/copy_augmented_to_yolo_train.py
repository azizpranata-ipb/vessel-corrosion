from __future__ import annotations

import argparse
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy manually augmented YOLO train data into data/yolo train split.")
    parser.add_argument("--aug-images", default=None)
    parser.add_argument("--aug-labels", default=None)
    parser.add_argument("--yolo-images", default="data/yolo/images/train")
    parser.add_argument("--yolo-labels", default="data/yolo/labels/train")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    aug_image_dir = Path(args.aug_images) if args.aug_images else first_existing_path(
        [
            "data/processed/augmented/images/train-augment",
            "data/processed/augmented/images/train",
        ]
    )
    aug_label_dir = Path(args.aug_labels) if args.aug_labels else first_existing_path(
        [
            "data/processed/augmented/labels/train-augment",
            "data/processed/augmented/labels/train",
        ]
    )
    yolo_image_dir = Path(args.yolo_images)
    yolo_label_dir = Path(args.yolo_labels)

    pairs = collect_pairs(aug_image_dir, aug_label_dir)
    if not pairs:
        raise ValueError("No augmented image/label pairs found.")

    collisions = []
    for image_path, label_path in pairs:
        if (yolo_image_dir / image_path.name).exists():
            collisions.append(str(yolo_image_dir / image_path.name))
        if (yolo_label_dir / label_path.name).exists():
            collisions.append(str(yolo_label_dir / label_path.name))

    if collisions:
        sample = "\n".join(collisions[:20])
        raise ValueError(f"Refusing to overwrite existing files. Collision sample:\n{sample}")

    print(f"Augmented pairs found: {len(pairs)}")
    if args.dry_run:
        print("Dry run only. No files copied.")
        return

    yolo_image_dir.mkdir(parents=True, exist_ok=True)
    yolo_label_dir.mkdir(parents=True, exist_ok=True)

    for image_path, label_path in pairs:
        shutil.copy2(image_path, yolo_image_dir / image_path.name)
        shutil.copy2(label_path, yolo_label_dir / label_path.name)

    print(f"Copied augmented images to: {yolo_image_dir}")
    print(f"Copied augmented labels to: {yolo_label_dir}")


def collect_pairs(image_dir: Path, label_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = label_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            print(f"Skip missing label: {image_path.name}")
            continue
        pairs.append((image_path, label_path))
    return pairs


def first_existing_path(candidates: list[str]) -> Path:
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    return Path(candidates[0])


if __name__ == "__main__":
    main()
