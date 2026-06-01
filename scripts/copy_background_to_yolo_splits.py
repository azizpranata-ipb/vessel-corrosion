from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Copy background YOLO tiles into train/val/test splits.")
    parser.add_argument("--bg-images", default="data/processed/background/images")
    parser.add_argument("--bg-labels", default="data/processed/background/labels")
    parser.add_argument("--out-root", default="data/yolo")
    parser.add_argument("--train", type=float, default=0.8)
    parser.add_argument("--val", type=float, default=0.15)
    parser.add_argument("--test", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if round(args.train + args.val + args.test, 6) != 1.0:
        raise ValueError("--train + --val + --test must equal 1.0")

    bg_image_dir = Path(args.bg_images)
    bg_label_dir = Path(args.bg_labels)
    out_root = Path(args.out_root)

    pairs = collect_pairs(bg_image_dir, bg_label_dir)
    if not pairs:
        raise ValueError("No background image/label pairs found.")

    rng = random.Random(args.seed)
    rng.shuffle(pairs)

    train_count = round(len(pairs) * args.train)
    val_count = round(len(pairs) * args.val)
    splits = {
        "train": pairs[:train_count],
        "val": pairs[train_count : train_count + val_count],
        "test": pairs[train_count + val_count :],
    }

    collisions = find_collisions(out_root, splits)
    if collisions:
        sample = "\n".join(collisions[:20])
        raise ValueError(f"Refusing to overwrite existing files. Collision sample:\n{sample}")

    print(f"Background pairs found: {len(pairs)}")
    for split, split_pairs in splits.items():
        print(f"{split}: {len(split_pairs)} background pairs")

    if args.dry_run:
        print("Dry run only. No files copied.")
        return

    for split, split_pairs in splits.items():
        out_image_dir = out_root / "images" / split
        out_label_dir = out_root / "labels" / split
        out_image_dir.mkdir(parents=True, exist_ok=True)
        out_label_dir.mkdir(parents=True, exist_ok=True)
        for image_path, label_path in split_pairs:
            shutil.copy2(image_path, out_image_dir / image_path.name)
            shutil.copy2(label_path, out_label_dir / label_path.name)

    print(f"Copied background splits to: {out_root}")


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


def find_collisions(out_root: Path, splits: dict[str, list[tuple[Path, Path]]]) -> list[str]:
    collisions = []
    for split, split_pairs in splits.items():
        for image_path, label_path in split_pairs:
            output_image = out_root / "images" / split / image_path.name
            output_label = out_root / "labels" / split / label_path.name
            if output_image.exists():
                collisions.append(str(output_image))
            if output_label.exists():
                collisions.append(str(output_label))
    return collisions


if __name__ == "__main__":
    main()
