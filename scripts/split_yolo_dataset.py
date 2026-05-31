from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split YOLO image/label pairs into train, val, and test folders.")
    parser.add_argument("--images", required=True, help="Input image directory.")
    parser.add_argument("--labels", required=True, help="Input label directory.")
    parser.add_argument("--out-root", default="data/yolo", help="Output YOLO dataset root.")
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.2)
    parser.add_argument("--test", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clear", action="store_true", help="Clear existing output split folders before copying.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if round(args.train + args.val + args.test, 6) != 1.0:
        raise ValueError("--train + --val + --test must equal 1.0")

    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    out_root = Path(args.out_root)

    pairs = collect_pairs(image_dir, label_dir)
    if not pairs:
        raise ValueError("No image/label pairs found.")

    random.seed(args.seed)
    random.shuffle(pairs)

    train_count = int(len(pairs) * args.train)
    val_count = int(len(pairs) * args.val)
    splits = {
        "train": pairs[:train_count],
        "val": pairs[train_count : train_count + val_count],
        "test": pairs[train_count + val_count :],
    }

    prepare_output_dirs(out_root, args.clear)

    for split_name, split_pairs in splits.items():
        for image_path, label_path in split_pairs:
            shutil.copy2(image_path, out_root / "images" / split_name / image_path.name)
            shutil.copy2(label_path, out_root / "labels" / split_name / label_path.name)
        print(f"{split_name}: {len(split_pairs)} pairs")

    print(f"Total: {len(pairs)} pairs")


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


def prepare_output_dirs(out_root: Path, clear: bool) -> None:
    for kind in ["images", "labels"]:
        for split in ["train", "val", "test"]:
            path = out_root / kind / split
            if clear and path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
