"""Print environment and GPU capability information for the project."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from catalogbank_ocr.environment import collect_environment_status, status_as_dict


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    status = collect_environment_status()
    info = status_as_dict(status)
    for key, value in info.items():
        logging.info("%s: %s", key, value)


if __name__ == "__main__":
    main()
