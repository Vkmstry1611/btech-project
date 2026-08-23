"""Tests for the Stage-1 CLI argument parsing.

These tests validate argument parsing and input resolution only —
they do not invoke the OCR engine.
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


class CLIArgParseTests(unittest.TestCase):
    """run_stage1_pipeline.py argument parsing."""

    def _parse(self, argv):
        """Import and call parse_args with a custom argv."""
        import importlib.util, types

        spec = importlib.util.spec_from_file_location(
            "run_stage1", SCRIPTS / "run_stage1_pipeline.py"
        )
        mod = importlib.util.module_from_spec(spec)
        # Temporarily override sys.argv
        old_argv = sys.argv
        sys.argv = ["run_stage1_pipeline.py"] + argv
        try:
            spec.loader.exec_module(mod)
            return mod.parse_args(), mod
        finally:
            sys.argv = old_argv

    def test_input_pdf_parsed(self) -> None:
        args, _ = self._parse(["--input", "/tmp/test.pdf"])
        self.assertEqual(args.input, Path("/tmp/test.pdf"))

    def test_input_shortflag(self) -> None:
        args, _ = self._parse(["-i", "/tmp/test.pdf"])
        self.assertEqual(args.input, Path("/tmp/test.pdf"))

    def test_output_override(self) -> None:
        args, _ = self._parse(["--output", "/tmp/out"])
        self.assertEqual(args.output, Path("/tmp/out"))

    def test_dpi_default(self) -> None:
        args, _ = self._parse([])
        self.assertEqual(args.dpi, 200)

    def test_dpi_override(self) -> None:
        args, _ = self._parse(["--dpi", "300"])
        self.assertEqual(args.dpi, 300)

    def test_seed_default(self) -> None:
        args, _ = self._parse([])
        self.assertEqual(args.seed, 42)

    def test_samples_per_vendor_default(self) -> None:
        args, _ = self._parse([])
        self.assertEqual(args.samples_per_vendor, 1)

    def test_deskew_flag(self) -> None:
        args, _ = self._parse(["--deskew"])
        self.assertTrue(args.deskew)

    def test_legacy_single_pdf_alias(self) -> None:
        args, _ = self._parse(["--single-pdf", "/tmp/legacy.pdf"])
        self.assertEqual(args.single_pdf_legacy, Path("/tmp/legacy.pdf"))

    def test_resolve_input_from_input_flag(self) -> None:
        args, mod = self._parse(["--input", "/tmp/test.pdf"])
        resolved = mod._resolve_input(args)
        self.assertEqual(resolved, Path("/tmp/test.pdf"))

    def test_resolve_input_from_legacy_flag(self) -> None:
        args, mod = self._parse(["--single-pdf", "/tmp/legacy.pdf"])
        resolved = mod._resolve_input(args)
        self.assertEqual(resolved, Path("/tmp/legacy.pdf"))

    def test_resolve_input_none_when_no_flag(self) -> None:
        args, mod = self._parse([])
        resolved = mod._resolve_input(args)
        self.assertIsNone(resolved)


if __name__ == "__main__":
    unittest.main()
