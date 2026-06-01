from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from scripts.tile_yolo_dataset import (
    IMAGE_EXTENSIONS,
    find_label_path,
    generate_tiles,
    pad_to_size,
    read_yolo_labels,
    remap_labels_to_tile,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export background-only YOLO tiles with empty labels.")
    parser.add_argument("--images", default="data/raw/images")
    parser.add_argument("--labels", default="data/raw/labels")
    parser.add_argument("--out-images", default="data/processed/background/images")
    parser.add_argument("--out-labels", default="data/processed/background/labels")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--min-visibility", type=float, default=0.35)
    parser.add_argument("--count", type=int, default=762)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--prefix", default="bg")
    parser.add_argument("--clear", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    out_image_dir = Path(args.out_images)
    out_label_dir = Path(args.out_labels)

    prepare_output_dirs(out_image_dir, out_label_dir, args.clear)

    candidates = collect_background_candidates(
        image_dir=image_dir,
        label_dir=label_dir,
        tile_size=args.tile_size,
        overlap=args.overlap,
        min_visibility=args.min_visibility,
    )
    if len(candidates) < args.count:
        raise ValueError(f"Only {len(candidates)} background candidates available, requested {args.count}.")

    rng = random.Random(args.seed)
    selected = rng.sample(candidates, args.count)

    for item in selected:
        image_path = item["image_path"]
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skip unreadable image: {image_path}")
            continue

        x1, y1, x2, y2 = item["tile"]
        tile = image[y1:y2, x1:x2]
        output_tile = pad_to_size(tile, args.tile_size)
        output_stem = f"{args.prefix}_{image_path.stem}_tile_{item['tile_index']:04d}_x{x1}_y{y1}"

        output_image_path = out_image_dir / f"{output_stem}{image_path.suffix.lower()}"
        output_label_path = out_label_dir / f"{output_stem}.txt"

        cv2.imwrite(str(output_image_path), output_tile)
        output_label_path.write_text("", encoding="utf-8")

    print(f"Background candidates available: {len(candidates)}")
    print(f"Background tiles written: {len(selected)}")
    print(f"Images written to: {out_image_dir}")
    print(f"Labels written to: {out_label_dir}")


def prepare_output_dirs(out_image_dir: Path, out_label_dir: Path, clear: bool) -> None:
    for path in [out_image_dir, out_label_dir]:
        if clear and path.exists():
            for file_path in path.iterdir():
                if file_path.is_file():
                    file_path.unlink()
        path.mkdir(parents=True, exist_ok=True)


def collect_background_candidates(
    image_dir: Path,
    label_dir: Path,
    tile_size: int,
    overlap: int,
    min_visibility: float,
) -> list[dict]:
    candidates = []
    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skip unreadable image: {image_path}")
            continue

        height, width = image.shape[:2]
        labels = read_yolo_labels(find_label_path(label_dir, image_path.stem))

        for tile_index, (x1, y1, x2, y2) in enumerate(generate_tiles(width, height, tile_size, overlap)):
            tile_w = x2 - x1
            tile_h = y2 - y1
            remapped = remap_labels_to_tile(labels, width, height, x1, y1, tile_w, tile_h, tile_size, min_visibility)
            if remapped:
                continue
            candidates.append({"image_path": image_path, "tile_index": tile_index, "tile": (x1, y1, x2, y2)})

    return candidates


if __name__ == "__main__":
    main()
