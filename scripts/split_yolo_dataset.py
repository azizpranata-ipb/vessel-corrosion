from __future__ import annotations

import argparse
import random
import re
import shutil
import sys
from collections import defaultdict
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
    parser.add_argument(
        "--group-by-source",
        action="store_true",
        help="Keep all tiles from the same source image in one split.",
    )
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

    splits = split_pairs(
        pairs,
        train_ratio=args.train,
        val_ratio=args.val,
        test_ratio=args.test,
        seed=args.seed,
        group_by_source=args.group_by_source,
    )

    prepare_output_dirs(out_root, args.clear)

    for split_name, split_items in splits.items():
        for image_path, label_path in split_items:
            shutil.copy2(image_path, out_root / "images" / split_name / image_path.name)
            shutil.copy2(label_path, out_root / "labels" / split_name / label_path.name)
        source_count = len({source_stem(image_path.stem) for image_path, _ in split_items})
        print(f"{split_name}: {len(split_items)} pairs from {source_count} source images")

    print(f"Total: {len(pairs)} pairs")


def collect_pairs(image_dir: Path, label_dir: Path) -> list[tuple[Path, Path]]:
    pairs = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        label_path = find_label_path(label_dir, image_path.stem)
        if not label_path.exists():
            print(f"Skip missing label: {image_path.name}")
            continue
        pairs.append((image_path, label_path))
    return pairs


def split_pairs(
    pairs: list[tuple[Path, Path]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    group_by_source: bool,
) -> dict[str, list[tuple[Path, Path]]]:
    rng = random.Random(seed)
    if not group_by_source:
        shuffled = pairs.copy()
        rng.shuffle(shuffled)
        train_count = int(len(shuffled) * train_ratio)
        val_count = int(len(shuffled) * val_ratio)
        return {
            "train": shuffled[:train_count],
            "val": shuffled[train_count : train_count + val_count],
            "test": shuffled[train_count + val_count :],
        }

    groups: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for pair in pairs:
        groups[source_stem(pair[0].stem)].append(pair)

    shuffled_groups = list(groups.values())
    rng.shuffle(shuffled_groups)
    train_groups = round(len(shuffled_groups) * train_ratio)
    test_groups = round(len(shuffled_groups) * test_ratio)
    val_groups = len(shuffled_groups) - train_groups - test_groups
    group_splits = {
        "train": shuffled_groups[:train_groups],
        "val": shuffled_groups[train_groups : train_groups + val_groups],
        "test": shuffled_groups[train_groups + val_groups : train_groups + val_groups + test_groups],
    }
    return {
        split: [pair for group in split_groups for pair in group]
        for split, split_groups in group_splits.items()
    }


def source_stem(tile_stem: str) -> str:
    return re.sub(r"_tile_\d{4}_x\d+_y\d+$", "", tile_stem)


def find_label_path(label_dir: Path, image_stem: str) -> Path:
    exact = label_dir / f"{image_stem}.txt"
    if exact.exists():
        return exact

    suffix_matches = sorted(label_dir.glob(f"*-{image_stem}.txt"))
    if suffix_matches:
        return suffix_matches[0]

    contains_matches = sorted(label_dir.glob(f"*{image_stem}*.txt"))
    if contains_matches:
        return contains_matches[0]

    return exact


def prepare_output_dirs(out_root: Path, clear: bool) -> None:
    for kind in ["images", "labels"]:
        for split in ["train", "val", "test"]:
            path = out_root / kind / split
            if clear and path.exists():
                shutil.rmtree(path)
            path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
