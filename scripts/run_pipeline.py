"""Run the full CatalogBank OCR pipeline from the command line."""

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
from catalogbank_ocr.pipeline import run_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument("--samples-per-vendor", type=int, default=None, help="Override sample count per vendor")
    parser.add_argument("--dpi", type=int, default=None, help="Override PDF render DPI")
    parser.add_argument("--seed", type=int, default=None, help="Override random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config)

    if args.samples_per_vendor is not None:
        config.dataset.samples_per_vendor = args.samples_per_vendor
    if args.dpi is not None:
        config.preprocessing.dpi = args.dpi
    if args.seed is not None:
        config.dataset.seed = args.seed

    result = run_pipeline(config)
    logging.info("Selected PDFs: %s", [path.name for path in result.selected_pdfs])
    logging.info("Generated %d OCR run(s)", len(result.ocr_results))


if __name__ == "__main__":
    main()
