from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class YoloBox:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Tile large images into fixed-size patches and remap YOLO detection labels."
    )
    parser.add_argument("--images", required=True, help="Input image directory.")
    parser.add_argument("--labels", default=None, help="Input YOLO label directory. Optional.")
    parser.add_argument("--out-images", required=True, help="Output tiled image directory.")
    parser.add_argument("--out-labels", default=None, help="Output tiled label directory. Optional.")
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument(
        "--min-visibility",
        type=float,
        default=0.35,
        help="Minimum visible bbox area ratio to keep a clipped label in a tile.",
    )
    parser.add_argument(
        "--keep-empty",
        action="store_true",
        help="Keep tiles without any label. Useful for negative/background samples.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.images)
    label_dir = Path(args.labels) if args.labels else None
    out_image_dir = Path(args.out_images)
    out_label_dir = Path(args.out_labels) if args.out_labels else None

    out_image_dir.mkdir(parents=True, exist_ok=True)
    if out_label_dir:
        out_label_dir.mkdir(parents=True, exist_ok=True)

    for image_path in sorted(image_dir.iterdir()):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skip unreadable image: {image_path}")
            continue

        height, width = image.shape[:2]
        labels = []
        if label_dir:
            label_path = find_label_path(label_dir, image_path.stem)
            labels = read_yolo_labels(label_path)

        tile_count = 0
        for tile_index, (x1, y1, x2, y2) in enumerate(generate_tiles(width, height, args.tile_size, args.overlap)):
            tile = image[y1:y2, x1:x2]
            tile_h, tile_w = tile.shape[:2]
            remapped = remap_labels_to_tile(labels, width, height, x1, y1, tile_w, tile_h, args.min_visibility)

            if not remapped and not args.keep_empty:
                continue

            output_stem = f"{image_path.stem}_tile_{tile_index:04d}_x{x1}_y{y1}"
            output_image_path = out_image_dir / f"{output_stem}{image_path.suffix.lower()}"
            cv2.imwrite(str(output_image_path), tile)

            if out_label_dir:
                output_label_path = out_label_dir / f"{output_stem}.txt"
                write_yolo_labels(output_label_path, remapped)

            tile_count += 1

        print(f"{image_path.name}: {tile_count} tiles saved")


def generate_tiles(width: int, height: int, tile_size: int, overlap: int) -> list[tuple[int, int, int, int]]:
    if overlap >= tile_size:
        raise ValueError("Overlap must be smaller than tile size.")

    step = tile_size - overlap
    xs = _positions(width, tile_size, step)
    ys = _positions(height, tile_size, step)

    tiles = []
    for y in ys:
        for x in xs:
            tiles.append((x, y, min(x + tile_size, width), min(y + tile_size, height)))
    return tiles


def _positions(length: int, tile_size: int, step: int) -> list[int]:
    if length <= tile_size:
        return [0]

    positions = list(range(0, length - tile_size + 1, step))
    last = length - tile_size
    if positions[-1] != last:
        positions.append(last)
    return positions


def read_yolo_labels(label_path: Path) -> list[YoloBox]:
    if not label_path.exists():
        return []

    labels = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        labels.append(
            YoloBox(
                class_id=int(parts[0]),
                x_center=float(parts[1]),
                y_center=float(parts[2]),
                width=float(parts[3]),
                height=float(parts[4]),
            )
        )
    return labels


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


def write_yolo_labels(label_path: Path, labels: list[YoloBox]) -> None:
    lines = [
        f"{label.class_id} {label.x_center:.6f} {label.y_center:.6f} {label.width:.6f} {label.height:.6f}"
        for label in labels
    ]
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def remap_labels_to_tile(
    labels: list[YoloBox],
    image_width: int,
    image_height: int,
    tile_x: int,
    tile_y: int,
    tile_width: int,
    tile_height: int,
    min_visibility: float,
) -> list[YoloBox]:
    remapped = []
    tile_x2 = tile_x + tile_width
    tile_y2 = tile_y + tile_height

    for label in labels:
        box_x1, box_y1, box_x2, box_y2 = yolo_to_xyxy(label, image_width, image_height)
        original_area = max(0.0, box_x2 - box_x1) * max(0.0, box_y2 - box_y1)
        if original_area <= 0:
            continue

        clipped_x1 = max(box_x1, tile_x)
        clipped_y1 = max(box_y1, tile_y)
        clipped_x2 = min(box_x2, tile_x2)
        clipped_y2 = min(box_y2, tile_y2)

        clipped_area = max(0.0, clipped_x2 - clipped_x1) * max(0.0, clipped_y2 - clipped_y1)
        if clipped_area / original_area < min_visibility:
            continue

        local_x1 = clipped_x1 - tile_x
        local_y1 = clipped_y1 - tile_y
        local_x2 = clipped_x2 - tile_x
        local_y2 = clipped_y2 - tile_y
        remapped.append(xyxy_to_yolo(label.class_id, local_x1, local_y1, local_x2, local_y2, tile_width, tile_height))

    return remapped


def yolo_to_xyxy(label: YoloBox, image_width: int, image_height: int) -> tuple[float, float, float, float]:
    box_width = label.width * image_width
    box_height = label.height * image_height
    center_x = label.x_center * image_width
    center_y = label.y_center * image_height
    x1 = center_x - box_width / 2.0
    y1 = center_y - box_height / 2.0
    x2 = center_x + box_width / 2.0
    y2 = center_y + box_height / 2.0
    return x1, y1, x2, y2


def xyxy_to_yolo(
    class_id: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    image_width: int,
    image_height: int,
) -> YoloBox:
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    x_center = x1 + width / 2.0
    y_center = y1 + height / 2.0
    return YoloBox(
        class_id=class_id,
        x_center=clamp(x_center / image_width),
        y_center=clamp(y_center / image_height),
        width=clamp(width / image_width),
        height=clamp(height / image_height),
    )


def clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


if __name__ == "__main__":
    main()
