"""Run the stage-1 research pipeline: preprocessing, semantic blocks, hierarchy."""

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
from catalogbank_ocr.research_stage1_pipeline import run_stage1_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument("--samples-per-vendor", type=int, default=3, help="Number of PDFs to sample per vendor")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic sampling seed")
    parser.add_argument("--dpi", type=int, default=200, help="Page rendering DPI")
    parser.add_argument("--deskew", action="store_true", help="Enable optional deskew")
    parser.add_argument("--denoise", action="store_true", help="Enable optional denoising")
    parser.add_argument("--enhance-resolution", action="store_true", help="Enable optional resolution enhancement")
    parser.add_argument("--single-pdf", type=Path, default=None, help="Process only a single PDF")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config)
    result = run_stage1_pipeline(
        config=config,
        samples_per_vendor=args.samples_per_vendor,
        seed=args.seed,
        dpi=args.dpi,
        deskew=args.deskew,
        denoise=args.denoise,
        enhance_resolution=args.enhance_resolution,
        single_pdf=args.single_pdf,
    )
    logging.info("Selected PDFs: %s", [path.name for path in result.selected_pdfs])
    logging.info("Processed %d page(s)", len(result.page_results))


if __name__ == "__main__":
    main()
