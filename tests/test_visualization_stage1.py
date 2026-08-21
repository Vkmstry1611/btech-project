from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from catalogbank_ocr.schemas.hierarchy import HierarchyNode
from catalogbank_ocr.schemas.semantic_block import PageSemanticBlocks, SemanticBlock
from catalogbank_ocr.visualization.research_visualization import visualize_hierarchy_tree, visualize_semantic_blocks


class VisualizationStage1Tests(unittest.TestCase):
    def test_visualizations_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_path = tmp_path / "page.png"
            Image.new("RGB", (400, 400), color="white").save(image_path)

            page_blocks = PageSemanticBlocks(
                page=1,
                blocks=[
                    SemanticBlock(id="b1", page=1, type="heading", source_type="title", text="Home", bbox=[10, 10, 150, 50], order=1, heading_level=1),
                    SemanticBlock(id="b2", page=1, type="table", source_type="table", text="Size | Color", bbox=[10, 70, 200, 140], order=2),
                ],
            )
            semantic_out = tmp_path / "semantic.png"
            hierarchy_out = tmp_path / "hierarchy.png"

            visualize_semantic_blocks(image_path, page_blocks, semantic_out)
            visualize_hierarchy_tree(
                HierarchyNode(name="document", semantic_type="document", children=[HierarchyNode(name="Home", semantic_type="heading")]),
                hierarchy_out,
            )

            self.assertTrue(semantic_out.exists())
            self.assertTrue(hierarchy_out.exists())


if __name__ == "__main__":
    unittest.main()
