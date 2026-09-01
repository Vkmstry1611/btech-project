from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from catalogbank_ocr.postprocessing.json_inspection import inspect_json_document, load_json


class JsonInspectionTests(unittest.TestCase):
    def test_inspect_json_document_reports_top_level_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            json_path = Path(tmp_dir) / "sample.json"
            json_path.write_text(json.dumps({"alpha": 1, "beta": [1, 2, 3]}), encoding="utf-8")

            inspection = inspect_json_document(json_path)
            self.assertEqual(inspection.root_type, "dict")
            self.assertEqual(inspection.top_level_keys, ["alpha", "beta"])

    def test_load_json_raises_for_missing_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            missing = Path(tmp_dir) / "missing.json"
            with self.assertRaises(FileNotFoundError):
                load_json(missing)


if __name__ == "__main__":
    unittest.main()
