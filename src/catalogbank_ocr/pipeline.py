"""End-to-end pipeline orchestration for the CatalogBank OCR workflow."""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from catalogbank_ocr.config import ProjectConfig, ensure_project_directories
from catalogbank_ocr.data.dataset import load_vendor_pdfs, select_sample_pdfs, validate_pdf_paths
from catalogbank_ocr.ocr.ppstructure import PPStructureRunResult, initialize_ppstructurev3, run_ppstructurev3
from catalogbank_ocr.postprocessing.json_inspection import JsonInspection, inspect_json_document
from catalogbank_ocr.preprocessing.pdf_to_image import pdf_first_page_to_image
from catalogbank_ocr.visualization.visualize_results import display_image

logger = logging.getLogger(__name__)


@dataclass()
class PipelineResult:
    """Summary of a completed pipeline run."""

    selected_pdfs: list[Path]
    image_paths: list[Path]
    ocr_results: list[PPStructureRunResult]
    json_inspections: list[JsonInspection]


def _copy_artifacts(run_result: PPStructureRunResult, config: ProjectConfig) -> None:
    stem = run_result.image_path.stem
    for file_path in run_result.json_files:
        target_dir = config.outputs.json_dir / stem
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, target_dir / file_path.name)
    for file_path in run_result.markdown_files:
        target_dir = config.outputs.markdown_dir / stem
        target_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, target_dir / file_path.name)
    for file_path in run_result.image_files:
        viz_target = config.outputs.visualizations_dir / stem
        viz_target.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, viz_target / file_path.name)


def run_pipeline(config: ProjectConfig) -> PipelineResult:
    """Run the full notebook workflow in a modular form."""

    ensure_project_directories(config)
    logger.info("Using CatalogBank repository at %s", config.dataset.repo_dir)

    vendor_pdfs = load_vendor_pdfs(
        config.dataset.repo_dir,
        str(config.dataset.thorlabs_pdf_dir.relative_to(config.dataset.repo_dir)),
        str(config.dataset.mcmaster_pdf_dir.relative_to(config.dataset.repo_dir)),
    )
    selected_pdfs = select_sample_pdfs(
        vendor_pdfs.thorlabs,
        vendor_pdfs.mcmaster,
        samples_per_vendor=config.dataset.samples_per_vendor,
        seed=config.dataset.seed,
    )
    validate_pdf_paths(selected_pdfs)

    engine = initialize_ppstructurev3(**config.ocr.ppstructurev3)

    image_paths: list[Path] = []
    ocr_results: list[PPStructureRunResult] = []
    json_inspections: list[JsonInspection] = []

    for pdf_path in selected_pdfs:
        image_path = pdf_first_page_to_image(pdf_path, config.outputs.images_dir, dpi=config.preprocessing.dpi)
        image_paths.append(image_path)

        output_dir = config.outputs.ocr_dir / pdf_path.stem
        run_result = run_ppstructurev3(image_path, output_dir, engine=engine)
        _copy_artifacts(run_result, config)
        ocr_results.append(run_result)

        if run_result.json_files:
            inspection = inspect_json_document(run_result.json_files[0])
            json_inspections.append(inspection)
            logger.info("JSON top-level keys for %s: %s", pdf_path.name, inspection.top_level_keys)

        if run_result.image_files:
            display_image(run_result.image_files[0], figure_path=config.outputs.visualizations_dir / pdf_path.stem / f"{pdf_path.stem}.png", show=False)

        logger.info("Completed processing for %s", pdf_path.name)

    return PipelineResult(
        selected_pdfs=selected_pdfs,
        image_paths=image_paths,
        ocr_results=ocr_results,
        json_inspections=json_inspections,
    )
