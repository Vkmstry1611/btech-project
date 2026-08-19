from __future__ import annotations

import importlib.util
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from catalogbank_ocr.preprocessing.pdf_to_image import pdf_first_page_to_image


@unittest.skipUnless(importlib.util.find_spec("fitz") is not None, "PyMuPDF is not installed")
class PdfToImageTests(unittest.TestCase):
    def test_pdf_first_page_to_image_creates_png(self) -> None:
        import fitz

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            pdf_path = tmp_path / "sample.pdf"
            document = fitz.open()
            page = document.new_page()
            page.insert_text((72, 72), "CatalogBank test page")
            document.save(pdf_path)
            document.close()

            output_dir = tmp_path / "images"
            image_path = pdf_first_page_to_image(pdf_path, output_dir, dpi=72)

            self.assertTrue(image_path.exists())
            self.assertEqual(image_path.suffix, ".png")
            self.assertTrue(output_dir.exists())

    def test_pdf_first_page_to_image_raises_for_missing_pdf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            pdf_path = Path(tmp_dir) / "missing.pdf"
            with self.assertRaises(FileNotFoundError):
                pdf_first_page_to_image(pdf_path, Path(tmp_dir) / "images")


if __name__ == "__main__":
    unittest.main()
