"""Stage-1 research pipeline: preprocessing, semantic blocks, and hierarchy reconstruction."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from catalogbank_ocr.config import ProjectConfig, ensure_project_directories
from catalogbank_ocr.data.dataset import load_vendor_pdfs, select_sample_pdfs, validate_pdf_paths
from catalogbank_ocr.hierarchy.hierarchy_builder import build_hierarchy_document
from catalogbank_ocr.ocr.ppstructure import initialize_ppstructurev3, run_ppstructurev3
from catalogbank_ocr.preprocessing.image_preprocessing import ImagePreprocessingConfig, render_and_preprocess_pdf_page
from catalogbank_ocr.semantic.block_detector import load_json_document
from catalogbank_ocr.semantic.block_mapper import build_semantic_document
from catalogbank_ocr.visualization.research_visualization import visualize_hierarchy_tree, visualize_semantic_blocks

logger = logging.getLogger(__name__)


@dataclass
class Stage1PageResult:
    """Result for a single processed page."""

    pdf_path: Path
    page_number: int
    rendered_image_path: Path
    preprocessed_image_path: Path
    raw_json_path: Path
    semantic_json_path: Path
    hierarchy_json_path: Path
    semantic_visualization_path: Path
    hierarchy_visualization_path: Path
    semantic_document: Dict[str, Any] = field(default_factory=dict)
    hierarchy_document: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Stage1RunResult:
    """Summary of a stage-1 pipeline run."""

    selected_pdfs: List[Path] = field(default_factory=list)
    page_results: List[Stage1PageResult] = field(default_factory=list)


def _save_json(path: Path, data: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _resolve_pdf_output_stem(pdf_path: Path) -> str:
    return pdf_path.stem


def run_stage1_for_pdf(
    pdf_path: Path,
    output_root: Path,
    preprocess_config: Optional[ImagePreprocessingConfig] = None,
    ocr_engine: Any = None,
) -> Stage1PageResult:
    """Run the stage-1 pipeline for one PDF using the first page only."""

    stem = _resolve_pdf_output_stem(pdf_path)
    rendered_dir = output_root / "images" / stem
    processed_dir = output_root / "processed" / stem
    raw_dir = output_root / "raw" / stem
    semantic_dir = output_root / "semantic" / stem
    hierarchy_dir = output_root / "hierarchy" / stem
    viz_dir = output_root / "visualizations" / stem

    preprocessing_result = render_and_preprocess_pdf_page(
        pdf_path=pdf_path,
        rendered_dir=rendered_dir,
        processed_dir=processed_dir,
        config=preprocess_config,
    )

    if ocr_engine is None:
        ocr_engine = initialize_ppstructurev3()

    ocr_output_dir = output_root / "ocr" / stem
    run_result = run_ppstructurev3(preprocessing_result.processed_image, ocr_output_dir, engine=ocr_engine)

    if not run_result.json_files:
        raise FileNotFoundError(f"PP-StructureV3 did not generate JSON output for {pdf_path}")

    raw_json_source = run_result.json_files[0]
    raw_json_target = raw_dir / raw_json_source.name
    raw_json_target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(raw_json_source, raw_json_target)

    raw_document = load_json_document(raw_json_target)
    semantic_document = build_semantic_document(raw_document, source_json_path=raw_json_target)
    hierarchy_document = build_hierarchy_document(semantic_document, source_pdf_path=pdf_path)

    semantic_json_path = _save_json(semantic_dir / f"{stem}_semantic_blocks.json", semantic_document.to_dict())
    hierarchy_json_path = _save_json(hierarchy_dir / f"{stem}_hierarchy.json", hierarchy_document.to_dict())

    semantic_visualization_path = visualize_semantic_blocks(
        source_image_path=preprocessing_result.rendered_image,
        page_blocks=semantic_document.pages[0],
        output_path=viz_dir / f"{stem}_semantic_blocks.png",
    )
    hierarchy_visualization_path = visualize_hierarchy_tree(
        root=hierarchy_document.root,
        output_path=viz_dir / f"{stem}_hierarchy.png",
    )

    return Stage1PageResult(
        pdf_path=pdf_path,
        page_number=1,
        rendered_image_path=preprocessing_result.rendered_image,
        preprocessed_image_path=preprocessing_result.processed_image,
        raw_json_path=raw_json_target,
        semantic_json_path=semantic_json_path,
        hierarchy_json_path=hierarchy_json_path,
        semantic_visualization_path=semantic_visualization_path,
        hierarchy_visualization_path=hierarchy_visualization_path,
        semantic_document=semantic_document.to_dict(),
        hierarchy_document=hierarchy_document.to_dict(),
    )


def run_stage1_pipeline(
    config: ProjectConfig,
    samples_per_vendor: Optional[int] = None,
    seed: Optional[int] = None,
    dpi: Optional[int] = None,
    deskew: bool = False,
    denoise: bool = False,
    enhance_resolution: bool = False,
    single_pdf: Optional[Path] = None,
) -> Stage1RunResult:
    """Run the stage-1 pipeline on the deterministic CatalogBank sample."""

    ensure_project_directories(config)

    preprocess_config = ImagePreprocessingConfig(
        dpi=dpi or config.preprocessing.dpi,
        first_page_only=True,
        deskew=deskew,
        denoise=denoise,
        enhance_resolution=enhance_resolution,
    )

    if single_pdf is not None:
        selected_pdfs = [single_pdf]
    else:
        vendor_pdfs = load_vendor_pdfs(
            config.dataset.repo_dir,
            str(config.dataset.thorlabs_pdf_dir.relative_to(config.dataset.repo_dir)),
            str(config.dataset.mcmaster_pdf_dir.relative_to(config.dataset.repo_dir)),
        )
        selected_pdfs = select_sample_pdfs(
            vendor_pdfs.thorlabs,
            vendor_pdfs.mcmaster,
            samples_per_vendor=samples_per_vendor or config.dataset.samples_per_vendor,
            seed=seed if seed is not None else config.dataset.seed,
        )

    validate_pdf_paths(selected_pdfs)

    output_root = config.outputs.base_dir
    page_results: List[Stage1PageResult] = []
    ocr_engine = initialize_ppstructurev3(**config.ocr.ppstructurev3)

    for pdf_path in selected_pdfs:
        logger.info("Running stage-1 pipeline for %s", pdf_path)
        page_results.append(
            run_stage1_for_pdf(
                pdf_path=pdf_path,
                output_root=output_root,
                preprocess_config=preprocess_config,
                ocr_engine=ocr_engine,
            )
        )

    return Stage1RunResult(selected_pdfs=selected_pdfs, page_results=page_results)
