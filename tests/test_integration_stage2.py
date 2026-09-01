"""Integration tests for the Stage-2 OCR→semantic→hierarchy pipeline.

These tests run entirely on the JSON fixtures that are already present in
``outputs/raw/`` from previous Stage-1 runs, so they do *not* require the OCR
engine to be available.  They validate:

1. The PP-StructureV3 JSON is correctly parsed via ``parsing_res_list``.
2. Semantic types are correctly inferred for real PP-StructureV3 labels.
3. The hierarchy is non-trivial (headings with children) for both fixtures.
4. The canonical document conforms to the expected schema.
"""

from __future__ import annotations

import json
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
import sys
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from catalogbank_ocr.research_stage1_pipeline import _build_canonical_document
from catalogbank_ocr.semantic.block_detector import extract_page_records, detect_page_candidates, load_json_document
from catalogbank_ocr.semantic.block_mapper import build_semantic_document
from catalogbank_ocr.hierarchy.hierarchy_builder import build_hierarchy_document
from catalogbank_ocr.semantic.block_types import HEADING_LABELS, SemanticType


THORLABS_RAW = ROOT / "outputs" / "raw" / "V21_1_Optomechanics_45" / "V21_1_Optomechanics_45_res.json"
MCMASTER_RAW = ROOT / "outputs" / "raw" / "mcmaster-125_3378_15" / "mcmaster-125_3378_15_res.json"


def _skip_if_missing(path: Path) -> None:
    if not path.exists():
        raise unittest.SkipTest(f"Fixture not found: {path}")


class TestParsingResListDetection(unittest.TestCase):
    """parsing_res_list should be the primary source — no OCR-line duplication."""

    def test_thorlabs_block_count_matches_parsing_res_list(self) -> None:
        _skip_if_missing(THORLABS_RAW)
        raw = load_json_document(THORLABS_RAW)
        expected = len(raw["parsing_res_list"])

        page_records = extract_page_records(raw)
        self.assertEqual(len(page_records), 1)
        self.assertTrue(page_records[0].get("_use_parsing_res_list"), "Should use parsing_res_list path")

        candidates = detect_page_candidates(page_records[0])
        self.assertEqual(len(candidates), expected,
                         f"Expected {expected} candidates from parsing_res_list, got {len(candidates)}")

    def test_mcmaster_block_count_matches_parsing_res_list(self) -> None:
        _skip_if_missing(MCMASTER_RAW)
        raw = load_json_document(MCMASTER_RAW)
        expected = len(raw["parsing_res_list"])
        page_records = extract_page_records(raw)
        candidates = detect_page_candidates(page_records[0])
        self.assertEqual(len(candidates), expected)


class TestSemanticTypeMapping(unittest.TestCase):
    """PP-StructureV3 labels must map to correct SemanticTypes."""

    def test_paragraph_title_maps_to_heading(self) -> None:
        self.assertIn("paragraph_title", HEADING_LABELS)
        from catalogbank_ocr.semantic.block_types import infer_semantic_type
        result = infer_semantic_type("paragraph_title", "Some Section Title")
        self.assertEqual(result, SemanticType.HEADING)

    def test_text_label_maps_to_paragraph(self) -> None:
        from catalogbank_ocr.semantic.block_types import infer_semantic_type
        result = infer_semantic_type("text", "Some body text here.")
        self.assertEqual(result, SemanticType.PARAGRAPH)

    def test_image_label_maps_to_figure(self) -> None:
        from catalogbank_ocr.semantic.block_types import infer_semantic_type
        result = infer_semantic_type("image", "")
        self.assertEqual(result, SemanticType.FIGURE)

    def test_figure_title_maps_to_figure(self) -> None:
        from catalogbank_ocr.semantic.block_types import infer_semantic_type
        result = infer_semantic_type("figure_title", "Some caption")
        self.assertEqual(result, SemanticType.FIGURE)

    def test_header_label_maps_to_paragraph_not_heading(self) -> None:
        """PP-StructureV3 'header' is a page running-head, not a structural heading."""
        from catalogbank_ocr.semantic.block_types import infer_semantic_type, HEADING_LABELS
        self.assertNotIn("header", HEADING_LABELS, "'header' must not be in HEADING_LABELS")
        result = infer_semantic_type("header", "For technical drawings and 3-D models, go to")
        self.assertEqual(result, SemanticType.PARAGRAPH,
                         "Page running-head 'header' should map to PARAGRAPH, not HEADING")

    def test_thorlabs_has_headings(self) -> None:
        _skip_if_missing(THORLABS_RAW)
        raw = load_json_document(THORLABS_RAW)
        semantic_doc = build_semantic_document(raw, source_json_path=THORLABS_RAW)
        types = {b.type for page in semantic_doc.pages for b in page.blocks}
        self.assertIn("heading", types, "Thorlabs document must contain heading blocks")

    def test_mcmaster_has_headings(self) -> None:
        _skip_if_missing(MCMASTER_RAW)
        raw = load_json_document(MCMASTER_RAW)
        semantic_doc = build_semantic_document(raw, source_json_path=MCMASTER_RAW)
        types = {b.type for page in semantic_doc.pages for b in page.blocks}
        self.assertIn("heading", types)


class TestHierarchyConstruction(unittest.TestCase):
    """Hierarchy must be non-trivial: headings must have child nodes."""

    def _get_heading_children(self, path: Path) -> int:
        raw = load_json_document(path)
        semantic_doc = build_semantic_document(raw, source_json_path=path)
        hierarchy_doc = build_hierarchy_document(semantic_doc)
        total_children_under_headings = sum(
            len(child.children)
            for child in hierarchy_doc.root.children
            if child.semantic_type == "heading"
        )
        return total_children_under_headings

    def test_thorlabs_headings_have_children(self) -> None:
        _skip_if_missing(THORLABS_RAW)
        count = self._get_heading_children(THORLABS_RAW)
        self.assertGreater(count, 0, "Thorlabs hierarchy: headings should have child nodes")

    def test_mcmaster_headings_have_children(self) -> None:
        _skip_if_missing(MCMASTER_RAW)
        count = self._get_heading_children(MCMASTER_RAW)
        self.assertGreater(count, 0, "McMaster hierarchy: headings should have child nodes")

    def test_hierarchy_not_flat(self) -> None:
        """The root should not have 100+ direct children (indicates de-duplication worked)."""
        _skip_if_missing(THORLABS_RAW)
        raw = load_json_document(THORLABS_RAW)
        semantic_doc = build_semantic_document(raw, source_json_path=THORLABS_RAW)
        hierarchy_doc = build_hierarchy_document(semantic_doc)
        root_children = len(hierarchy_doc.root.children)
        self.assertLess(root_children, 30,
                        f"Root has {root_children} direct children; hierarchy appears flat (duplication not fixed)")


class TestCanonicalDocument(unittest.TestCase):
    """Canonical JSON must conform to schema_version 1.0."""

    def _build(self, path: Path):
        raw = load_json_document(path)
        semantic_doc = build_semantic_document(raw, source_json_path=path)
        hierarchy_doc = build_hierarchy_document(semantic_doc)
        dummy_image = Path("/tmp/dummy.png")
        return _build_canonical_document(
            pdf_path=path,
            page_number=1,
            rendered_image_path=dummy_image,
            preprocessed_image_path=dummy_image,
            raw_json_path=path,
            semantic_document=semantic_doc,
            hierarchy_document=hierarchy_doc,
        )

    def test_schema_version_present(self) -> None:
        _skip_if_missing(THORLABS_RAW)
        doc = self._build(THORLABS_RAW)
        self.assertEqual(doc["schema_version"], "1.0")

    def test_source_fields_present(self) -> None:
        _skip_if_missing(THORLABS_RAW)
        doc = self._build(THORLABS_RAW)
        for key in ("pdf_path", "page_number", "rendered_image_path", "preprocessed_image_path", "raw_json_path"):
            self.assertIn(key, doc["source"])

    def test_pages_and_hierarchy_present(self) -> None:
        _skip_if_missing(THORLABS_RAW)
        doc = self._build(THORLABS_RAW)
        self.assertIn("pages", doc)
        self.assertIn("hierarchy", doc)
        self.assertTrue(len(doc["pages"]) > 0)
        self.assertIn("children", doc["hierarchy"])

    def test_serialisable_to_json(self) -> None:
        _skip_if_missing(MCMASTER_RAW)
        doc = self._build(MCMASTER_RAW)
        serialised = json.dumps(doc)
        self.assertGreater(len(serialised), 100)


class TestConfidenceScores(unittest.TestCase):
    """Confidence scores should be populated from layout_det_res.boxes."""

    def test_mcmaster_blocks_have_confidence(self) -> None:
        _skip_if_missing(MCMASTER_RAW)
        raw = load_json_document(MCMASTER_RAW)
        semantic_doc = build_semantic_document(raw, source_json_path=MCMASTER_RAW)
        blocks_with_conf = [
            b for page in semantic_doc.pages for b in page.blocks
            if b.confidence is not None
        ]
        self.assertGreater(
            len(blocks_with_conf), 0,
            "At least some blocks should have confidence scores joined from layout_det_res"
        )

    def test_thorlabs_blocks_have_confidence(self) -> None:
        _skip_if_missing(THORLABS_RAW)
        raw = load_json_document(THORLABS_RAW)
        semantic_doc = build_semantic_document(raw, source_json_path=THORLABS_RAW)
        blocks_with_conf = [
            b for page in semantic_doc.pages for b in page.blocks
            if b.confidence is not None
        ]
        self.assertGreater(len(blocks_with_conf), 0)


if __name__ == "__main__":
    unittest.main()
