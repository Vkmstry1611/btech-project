"""Display an annotated PP-StructureV3 result image."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from catalogbank_ocr.config import load_config
from catalogbank_ocr.visualization.visualize_results import display_image, find_image_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument("--image-path", type=Path, default=None, help="Annotated image to display")
    parser.add_argument("--save-figure", type=Path, default=None, help="Optional path to save the displayed figure")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config)

    image_path = args.image_path
    if image_path is None:
        image_files = find_image_files(config.outputs.visualizations_dir)
        if not image_files:
            raise FileNotFoundError(f"No annotated images found in {config.outputs.visualizations_dir}")
        image_path = image_files[0]

    display_image(image_path, figure_path=args.save_figure, show=True)


if __name__ == "__main__":
    main()
