from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.corrosion.config import load_config
from src.corrosion.inference import CorrosionAnalyzer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLOv8 + OpenCV corrosion analysis on one image.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--weights", default=None, help="Override model path from config.")
    parser.add_argument("--config", default="configs/app.yaml")
    parser.add_argument("--output-dir", default="outputs/predictions")
    parser.add_argument("--mm-per-pixel", type=float, default=None)
    parser.add_argument("--gt-mask", default=None, help="Optional ground truth binary mask for Dice Coefficient.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    analyzer = CorrosionAnalyzer(config=config, model_path=args.weights)
    response = analyzer.analyze_image(
        image_path=args.image,
        output_dir=args.output_dir,
        mm_per_pixel=args.mm_per_pixel,
        gt_mask_path=args.gt_mask,
    )
    print(json.dumps(response.model_dump(), indent=2))


if __name__ == "__main__":
    main()
