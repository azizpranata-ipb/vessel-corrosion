from __future__ import annotations

import argparse
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


@dataclass(frozen=True)
class AugmentationPlan:
    brightness_contrast: int
    gaussian_blur: int
    salt_pepper: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create non-geometric augmentations for a YOLO train split.")
    parser.add_argument("--images", default="data/yolo/images/train", help="Input train image directory.")
    parser.add_argument("--labels", default="data/yolo/labels/train", help="Input train label directory.")
    parser.add_argument("--out-images", default="data/processed/augmented/images/train", help="Output image directory.")
    parser.add_argument("--out-labels", default="data/processed/augmented/labels/train", help="Output label directory.")
    parser.add_argument("--fraction", type=float, default=0.5, help="Fraction of train images to augment.")
    parser.add_argument("--brightness-ratio", type=float, default=0.60)
    parser.add_argument("--blur-ratio", type=float, default=0.25)
    parser.add_argument("--noise-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clear", action="store_true", help="Clear output folders before writing.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.images)
    label_dir = Path(args.labels)
    out_image_dir = Path(args.out_images)
    out_label_dir = Path(args.out_labels)

    rng = random.Random(args.seed)
    np_rng = np.random.default_rng(args.seed)

    pairs = collect_pairs(image_dir, label_dir)
    if not pairs:
        raise ValueError("No image/label pairs found.")

    target_count = round(len(pairs) * args.fraction)
    plan = build_plan(target_count, args.brightness_ratio, args.blur_ratio, args.noise_ratio)
    selected_pairs = rng.sample(pairs, target_count)

    prepare_output_dirs(out_image_dir, out_label_dir, args.clear)

    tasks = (
        [("brightness", pair) for pair in selected_pairs[: plan.brightness_contrast]]
        + [
            ("blur", pair)
            for pair in selected_pairs[
                plan.brightness_contrast : plan.brightness_contrast + plan.gaussian_blur
            ]
        ]
        + [("salt_pepper", pair) for pair in selected_pairs[-plan.salt_pepper :]]
    )
    rng.shuffle(tasks)

    counts = {"brightness": 0, "blur": 0, "salt_pepper": 0}
    for aug_name, (image_path, label_path) in tasks:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skip unreadable image: {image_path}")
            continue

        if aug_name == "brightness":
            augmented = augment_brightness_contrast(image, rng)
        elif aug_name == "blur":
            augmented = augment_gaussian_blur(image, rng)
        elif aug_name == "salt_pepper":
            augmented = augment_salt_pepper(image, np_rng)
        else:
            raise ValueError(f"Unknown augmentation: {aug_name}")

        output_stem = f"{image_path.stem}_aug_{aug_name}"
        output_image_path = out_image_dir / f"{output_stem}{image_path.suffix.lower()}"
        output_label_path = out_label_dir / f"{output_stem}.txt"

        cv2.imwrite(str(output_image_path), augmented)
        shutil.copy2(label_path, output_label_path)
        counts[aug_name] += 1

    print(f"Input train pairs: {len(pairs)}")
    print(f"Augmented target: {target_count}")
    print(f"brightness/contrast: {counts['brightness']}")
    print(f"gaussian blur: {counts['blur']}")
    print(f"salt and pepper: {counts['salt_pepper']}")
    print(f"Total augmented: {sum(counts.values())}")
    print(f"Images written to: {out_image_dir}")
    print(f"Labels written to: {out_label_dir}")


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


def build_plan(total: int, brightness_ratio: float, blur_ratio: float, noise_ratio: float) -> AugmentationPlan:
    ratio_sum = brightness_ratio + blur_ratio + noise_ratio
    if ratio_sum <= 0:
        raise ValueError("Augmentation ratios must sum to a positive value.")

    brightness = round(total * brightness_ratio / ratio_sum)
    blur = round(total * blur_ratio / ratio_sum)
    noise = total - brightness - blur
    return AugmentationPlan(brightness, blur, noise)


def prepare_output_dirs(out_image_dir: Path, out_label_dir: Path, clear: bool) -> None:
    for path in [out_image_dir, out_label_dir]:
        if clear and path.exists():
            shutil.rmtree(path)
        path.mkdir(parents=True, exist_ok=True)


def augment_brightness_contrast(image: np.ndarray, rng: random.Random) -> np.ndarray:
    alpha = rng.uniform(0.75, 1.30)
    beta = rng.randint(-35, 35)
    return cv2.convertScaleAbs(image, alpha=alpha, beta=beta)


def augment_gaussian_blur(image: np.ndarray, rng: random.Random) -> np.ndarray:
    kernel_size = rng.choice([3, 5])
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def augment_salt_pepper(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    output = image.copy()
    height, width = output.shape[:2]
    amount = float(rng.uniform(0.002, 0.008))
    pixel_count = max(1, int(height * width * amount))

    salt_y = rng.integers(0, height, pixel_count)
    salt_x = rng.integers(0, width, pixel_count)
    pepper_y = rng.integers(0, height, pixel_count)
    pepper_x = rng.integers(0, width, pixel_count)

    output[salt_y, salt_x] = 255
    output[pepper_y, pepper_x] = 0
    return output


if __name__ == "__main__":
    main()
