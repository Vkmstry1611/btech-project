from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from catalogbank_ocr.config import DatasetConfig, OCRConfig, OutputConfig, PreprocessingConfig, ProjectConfig
from catalogbank_ocr.data.dataset import VendorPdfs
from catalogbank_ocr.ocr.ppstructure import PPStructureRunResult
from catalogbank_ocr.pipeline import PipelineResult, run_pipeline
from catalogbank_ocr.postprocessing.json_inspection import JsonInspection


class PipelineTests(unittest.TestCase):
    def test_run_pipeline_copies_outputs_and_inspects_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo_dir = root / "CatalogBank"
            repo_dir.mkdir(parents=True, exist_ok=True)

            pdf_path = root / "sample.pdf"
            pdf_path.write_bytes(b"dummy")
            image_path = root / "outputs" / "images" / "sample.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            image_path.write_bytes(b"png")
            json_path = root / "sample.json"
            json_path.write_text('{"foo": "bar"}', encoding="utf-8")
            md_path = root / "sample.md"
            md_path.write_text("# sample", encoding="utf-8")

            config = ProjectConfig(
                dataset=DatasetConfig(
                    repo_url="https://example.com/catalogbank.git",
                    repo_dir=repo_dir,
                    thorlabs_pdf_dir=repo_dir / "thorlabs",
                    mcmaster_pdf_dir=repo_dir / "mcmaster",
                    samples_per_vendor=1,
                    seed=42,
                ),
                preprocessing=PreprocessingConfig(dpi=72),
                outputs=OutputConfig(
                    base_dir=root / "outputs",
                    json_dir=root / "outputs" / "json",
                    markdown_dir=root / "outputs" / "markdown",
                    images_dir=root / "outputs" / "images",
                    visualizations_dir=root / "outputs" / "visualizations",
                    ocr_dir=root / "outputs" / "ppstructurev3",
                ),
                ocr=OCRConfig(ppstructurev3={}),
            )

            vendor_pdfs = VendorPdfs(thorlabs=[pdf_path], mcmaster=[pdf_path])
            run_result = PPStructureRunResult(
                image_path=image_path,
                output_dir=root / "outputs" / "ppstructurev3" / "sample",
                prediction_count=1,
                json_files=[json_path],
                markdown_files=[md_path],
                image_files=[image_path],
            )

            with patch("catalogbank_ocr.pipeline.load_vendor_pdfs", return_value=vendor_pdfs), \
                patch("catalogbank_ocr.pipeline.select_sample_pdfs", return_value=[pdf_path]), \
                patch("catalogbank_ocr.pipeline.validate_pdf_paths", return_value=None), \
                patch("catalogbank_ocr.pipeline.initialize_ppstructurev3", return_value=object()), \
                patch("catalogbank_ocr.pipeline.pdf_first_page_to_image", return_value=image_path), \
                patch("catalogbank_ocr.pipeline.run_ppstructurev3", return_value=run_result), \
                patch("catalogbank_ocr.pipeline.inspect_json_document", return_value=JsonInspection(json_path, "dict", ["foo"], "{\n  \"foo\": \"bar\"\n}")), \
                patch("catalogbank_ocr.pipeline.display_image", return_value=None):
                result = run_pipeline(config)

            self.assertIsInstance(result, PipelineResult)
            self.assertEqual(result.selected_pdfs, [pdf_path])
            self.assertEqual(len(result.json_inspections), 1)
            self.assertTrue((config.outputs.json_dir / "sample" / "sample.json").exists())
            self.assertTrue((config.outputs.markdown_dir / "sample" / "sample.md").exists())
            self.assertTrue((config.outputs.images_dir / "sample.png").exists())
            self.assertTrue((config.outputs.visualizations_dir / "sample" / "sample.png").exists())


if __name__ == "__main__":
    unittest.main()
