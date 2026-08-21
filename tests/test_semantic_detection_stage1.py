from __future__ import annotations

import unittest

from catalogbank_ocr.semantic.block_detector import detect_semantic_candidates
from catalogbank_ocr.semantic.block_mapper import build_semantic_document


class SemanticDetectionStage1Tests(unittest.TestCase):
    def test_detects_and_maps_semantic_types(self) -> None:
        raw_document = {
            "layoutParsingResults": [
                {
                    "page_index": 0,
                    "prunedResult": {
                        "boxes": [
                            {
                                "block_label": "title",
                                "block_bbox": [10, 10, 300, 50],
                                "block_content": "Home",
                                "block_order": 1,
                                "confidence": 0.99,
                            },
                            {
                                "block_label": "product",
                                "block_bbox": [10, 60, 300, 120],
                                "block_content": "Armchair 200\n$199\nModel AX-3",
                                "block_order": 2,
                                "confidence": 0.92,
                            },
                            {
                                "block_label": "text",
                                "block_bbox": [10, 130, 300, 170],
                                "block_content": "Width: 20 in\nDepth: 18 in",
                                "block_order": 3,
                                "confidence": 0.88,
                            },
                            {
                                "block_label": "table",
                                "block_bbox": [10, 180, 300, 240],
                                "block_content": "Size | Color",
                                "block_order": 4,
                                "confidence": 0.97,
                            },
                        ]
                    },
                    "markdown": {"text": "# Home\n\nArmchair 200\n\nWidth: 20 in"},
                }
            ],
            "dataInfo": {"source": "synthetic"},
        }

        candidates = detect_semantic_candidates(raw_document)
        document = build_semantic_document(raw_document)

        semantic_types = {candidate.source_type for candidate in candidates}
        mapped_types = {block.type for page in document.pages for block in page.blocks}

        self.assertIn("title", semantic_types)
        self.assertIn("product", semantic_types)
        self.assertIn("table", semantic_types)
        self.assertIn("heading", mapped_types)
        self.assertIn("product_card", mapped_types)
        self.assertIn("specification", mapped_types)
        self.assertIn("table", mapped_types)


if __name__ == "__main__":
    unittest.main()
