"""Clone the CatalogBank dataset and fetch the sample assets used by the notebook."""

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
from catalogbank_ocr.data.download import clone_catalogbank, pull_git_lfs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None, help="Path to config.yaml")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    config = load_config(args.config)
    clone_catalogbank(config.dataset.repo_url, config.dataset.repo_dir)
    pull_git_lfs(
        config.dataset.repo_dir,
        [
            "Catalogs/Sample/Thorlabs/OptoMechanics_v21/_pdfs/*",
            "Catalogs/Sample/Thorlabs/OptoMechanics_v21/_images/*",
            "Catalogs/Sample/McMasterCarr/_pdfs/*",
            "Catalogs/Sample/McMasterCarr/_images/*",
        ],
    )


if __name__ == "__main__":
    main()
