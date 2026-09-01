from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from catalogbank_ocr.data.dataset import select_sample_pdfs, validate_pdf_paths


class DatasetTests(unittest.TestCase):
    def test_select_sample_pdfs_is_deterministic(self) -> None:
        thorlabs = [Path(f"thorlabs_{index}.pdf") for index in range(6)]
        mcmaster = [Path(f"mcmaster_{index}.pdf") for index in range(6)]

        first = select_sample_pdfs(thorlabs, mcmaster, samples_per_vendor=3, seed=42)
        second = select_sample_pdfs(thorlabs, mcmaster, samples_per_vendor=3, seed=42)

        self.assertEqual(first, second)
        self.assertEqual(len(first), 6)
        self.assertTrue(all(path.suffix == ".pdf" for path in first))

    def test_validate_pdf_paths_raises_for_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            existing = Path(tmp_dir) / "exists.pdf"
            existing.write_bytes(b"dummy")
            missing = Path(tmp_dir) / "missing.pdf"

            with self.assertRaises(FileNotFoundError):
                validate_pdf_paths([existing, missing])


if __name__ == "__main__":
    unittest.main()
