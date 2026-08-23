"""Run the stage-1 research pipeline: preprocessing, semantic blocks, hierarchy.

Quick-start examples
--------------------
# Single PDF (any vendor):
python scripts/run_stage1_pipeline.py --input path/to/file.pdf

# All PDFs in a directory:
python scripts/run_stage1_pipeline.py --input path/to/pdf_folder/

# Deterministic dataset sample (1 Thorlabs + 1 McMaster, seed 42):
python scripts/run_stage1_pipeline.py --samples-per-vendor 1 --seed 42

# Control output location and resolution:
python scripts/run_stage1_pipeline.py --input file.pdf --output my_outputs/ --dpi 300
"""

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
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── Input ────────────────────────────────────────────────────────────────
    input_group = parser.add_mutually_exclusive_group()
    input_group.add_argument(
        "--input", "-i",
        type=Path,
        default=None,
        help="Path to a single PDF file or a directory of PDF files to process.",
    )
    input_group.add_argument(
        "--single-pdf",
        type=Path,
        default=None,
        dest="single_pdf_legacy",
        help="(legacy alias for --input) Path to a single PDF.",
    )
    # Dataset sampling — used when --input is not provided
    parser.add_argument(
        "--samples-per-vendor",
        type=int,
        default=1,
        help="Number of PDFs to sample per vendor from the CatalogBank dataset (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic sampling seed (default: 42).",
    )

    # ── Output ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--output", "-o",
        type=Path,
        default=None,
        help="Root output directory (default: outputs/ from config.yaml).",
    )

    # ── Processing ───────────────────────────────────────────────────────────
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Page rendering DPI (default: 200).",
    )
    parser.add_argument(
        "--deskew",
        action="store_true",
        help="Enable deskew preprocessing.",
    )
    parser.add_argument(
        "--denoise",
        action="store_true",
        help="Enable denoising preprocessing.",
    )
    parser.add_argument(
        "--enhance-resolution",
        action="store_true",
        help="Enable resolution enhancement preprocessing.",
    )

    # ── Config ───────────────────────────────────────────────────────────────
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="Path to config.yaml (default: configs/config.yaml).",
    )

    return parser.parse_args()


def _resolve_input(args: argparse.Namespace) -> list[Path] | Path | None:
    """Return the resolved --input value, handling the legacy --single-pdf alias."""
    if args.input is not None:
        return args.input
    if args.single_pdf_legacy is not None:
        return args.single_pdf_legacy
    return None


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = load_config(args.config)

    # Override output directory if --output was passed
    if args.output is not None:
        config.outputs.base_dir = args.output.resolve()

    resolved_input = _resolve_input(args)

    # Resolve --input: can be a single PDF or a directory of PDFs
    single_pdf: Path | None = None
    input_dir: Path | None = None

    if resolved_input is not None:
        resolved_input = resolved_input.resolve()
        if resolved_input.is_dir():
            input_dir = resolved_input
        elif resolved_input.suffix.lower() == ".pdf":
            single_pdf = resolved_input
        else:
            logging.error("--input must be a .pdf file or a directory: %s", resolved_input)
            sys.exit(1)

    # When a directory is given, collect all PDFs inside it
    if input_dir is not None:
        pdf_files = sorted(input_dir.glob("**/*.pdf"))
        if not pdf_files:
            logging.error("No PDF files found in directory: %s", input_dir)
            sys.exit(1)
        logging.info("Found %d PDF(s) in %s", len(pdf_files), input_dir)
        results = []
        for pdf in pdf_files:
            result = run_stage1_pipeline(
                config=config,
                dpi=args.dpi,
                deskew=args.deskew,
                denoise=args.denoise,
                enhance_resolution=args.enhance_resolution,
                single_pdf=pdf,
            )
            results.extend(result.page_results)
        _print_summary(results)
        return

    # Single PDF or dataset sampling
    result = run_stage1_pipeline(
        config=config,
        samples_per_vendor=args.samples_per_vendor,
        seed=args.seed,
        dpi=args.dpi,
        deskew=args.deskew,
        denoise=args.denoise,
        enhance_resolution=args.enhance_resolution,
        single_pdf=single_pdf,
    )
    logging.info("Selected PDFs: %s", [path.name for path in result.selected_pdfs])
    logging.info("Processed %d page(s)", len(result.page_results))
    _print_summary(result.page_results)


def _print_summary(page_results) -> None:
    for pr in page_results:
        logging.info("─" * 60)
        logging.info("PDF           : %s", pr.pdf_path.name)
        logging.info("Canonical JSON: %s", pr.canonical_json_path)
        logging.info("Hierarchy viz : %s", pr.hierarchy_visualization_path)
        logging.info("Semantic viz  : %s", pr.semantic_visualization_path)


if __name__ == "__main__":
    main()
