from __future__ import annotations

import unittest

from catalogbank_ocr.hierarchy.hierarchy_builder import build_hierarchy_document
from catalogbank_ocr.schemas.semantic_block import PageSemanticBlocks, SemanticBlock, SemanticDocument


class HierarchyStage1Tests(unittest.TestCase):
    def test_builds_product_tree_from_semantic_blocks(self) -> None:
        page = PageSemanticBlocks(
            page=1,
            blocks=[
                SemanticBlock(id="p1_b001", page=1, type="heading", source_type="title", text="Home", bbox=[10, 10, 300, 50], order=1, heading_level=1),
                SemanticBlock(id="p1_b002", page=1, type="heading", source_type="heading", text="Seating", bbox=[10, 60, 300, 90], order=2, heading_level=2),
                SemanticBlock(id="p1_b003", page=1, type="product_card", source_type="product", text="Armchair", bbox=[20, 100, 280, 180], order=3),
                SemanticBlock(id="p1_b004", page=1, type="specification", source_type="text", text="Birch", bbox=[30, 190, 180, 220], order=4),
                SemanticBlock(id="p1_b005", page=1, type="specification", source_type="text", text="Black-Brown", bbox=[30, 225, 180, 255], order=5),
                SemanticBlock(id="p1_b006", page=1, type="heading", source_type="heading", text="Bookcase", bbox=[10, 300, 300, 340], order=6, heading_level=2),
            ],
        )
        semantic_document = SemanticDocument(pages=[page])
        hierarchy = build_hierarchy_document(semantic_document)

        root = hierarchy.root
        self.assertEqual(root.semantic_type, "document")
        self.assertEqual(len(root.children), 1)
        home = root.children[0]
        self.assertEqual(home.name, "Home")
        seating = home.children[0]
        self.assertEqual(seating.name, "Seating")
        self.assertEqual(seating.children[0].name, "Armchair")
        armchair = seating.children[0]
        child_names = [child.name for child in armchair.children]
        self.assertIn("Birch", child_names)
        self.assertIn("Black-Brown", child_names)


if __name__ == "__main__":
    unittest.main()
