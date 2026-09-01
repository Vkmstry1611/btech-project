"""Inspect generated PP-StructureV3 JSON outputs."""

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
from catalogbank_ocr.postprocessing.json_inspection import find_json_files, inspect_json_document


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    parser.add_argument("--json-path", type=Path, default=None, help="Specific JSON file to inspect")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config)

    json_path = args.json_path
    if json_path is None:
        json_files = find_json_files(config.outputs.json_dir)
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {config.outputs.json_dir}")
        json_path = json_files[0]

    inspection = inspect_json_document(json_path)
    logging.info("JSON path: %s", inspection.path)
    logging.info("Root type: %s", inspection.root_type)
    logging.info("Top-level keys: %s", inspection.top_level_keys)
    logging.info("Preview:\n%s", inspection.preview)


if __name__ == "__main__":
    main()
