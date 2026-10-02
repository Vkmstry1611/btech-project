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
from catalogbank_ocr.preprocessing.image_preprocessing import (
    ImagePreprocessingConfig,
    render_and_preprocess_pdf_page,
    render_and_preprocess_all_pages,
)
from catalogbank_ocr.schemas.semantic_block import SemanticDocument
from catalogbank_ocr.semantic.block_detector import load_json_document, extract_page_records, detect_page_candidates
from catalogbank_ocr.semantic.block_mapper import build_semantic_document, map_page_candidates
from catalogbank_ocr.visualization.research_visualization import visualize_hierarchy_tree, visualize_semantic_blocks

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

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
    canonical_json_path: Path
    semantic_visualization_path: Path
    hierarchy_visualization_path: Path
    semantic_document: Dict[str, Any] = field(default_factory=dict)
    hierarchy_document: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Stage1MultiPageResult:
    """Result for a full multi-page PDF processed by Stage 1."""

    pdf_path: Path
    total_pages: int
    page_results: List[Stage1PageResult] = field(default_factory=list)
    canonical_json_path: Optional[Path] = None
    semantic_json_path: Optional[Path] = None
    hierarchy_json_path: Optional[Path] = None
    semantic_visualization_paths: List[Path] = field(default_factory=list)
    hierarchy_visualization_path: Optional[Path] = None


@dataclass
class Stage1RunResult:
    """Summary of a stage-1 pipeline run."""

    selected_pdfs: List[Path] = field(default_factory=list)
    page_results: List[Stage1PageResult] = field(default_factory=list)
    multipage_results: List[Stage1MultiPageResult] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_json(path: Path, data: Dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _resolve_pdf_output_stem(pdf_path: Path) -> str:
    return pdf_path.stem


def _build_canonical_document(
    pdf_path: Path,
    page_number: int,
    rendered_image_path: Path,
    preprocessed_image_path: Path,
    raw_json_path: Path,
    semantic_document: Any,
    hierarchy_document: Any,
) -> Dict[str, Any]:
    """Build a single-page canonical JSON (schema_version 1.0)."""
    return {
        "schema_version": "1.0",
        "source": {
            "pdf_path": str(pdf_path),
            "page_number": page_number,
            "rendered_image_path": str(rendered_image_path),
            "preprocessed_image_path": str(preprocessed_image_path),
            "raw_json_path": str(raw_json_path),
        },
        "pages": semantic_document.to_dict().get("pages", []),
        "hierarchy": hierarchy_document.root.to_dict(),
    }


def _build_canonical_document_multipage(
    pdf_path: Path,
    total_pages: int,
    page_preprocessing_results: List[Any],
    semantic_document: Any,
    hierarchy_document: Any,
) -> Dict[str, Any]:
    """Build a multi-page canonical JSON (schema_version 1.1).

    Schema::

        {
          "schema_version": "1.1",
          "source": {
            "pdf_path": str,
            "total_pages": int,
            "pages": [
              { "page_number": int, "rendered_image_path": str, "preprocessed_image_path": str }
            ]
          },
          "pages": [ { "page": int, "blocks": [...] }, ... ],
          "hierarchy": { ... }
        }
    """
    source_pages = [
        {
            "page_number": idx + 1,
            "rendered_image_path": str(r.rendered_image),
            "preprocessed_image_path": str(r.processed_image),
        }
        for idx, r in enumerate(page_preprocessing_results)
    ]
    return {
        "schema_version": "1.1",
        "source": {
            "pdf_path": str(pdf_path),
            "total_pages": total_pages,
            "pages": source_pages,
        },
        "pages": semantic_document.to_dict().get("pages", []),
        "hierarchy": hierarchy_document.root.to_dict(),
    }


# ---------------------------------------------------------------------------
# Single-page pipeline (preserved for backward compat)
# ---------------------------------------------------------------------------

def run_stage1_for_pdf(
    pdf_path: Path,
    output_root: Path,
    preprocess_config: Optional[ImagePreprocessingConfig] = None,
    ocr_engine: Any = None,
) -> Stage1PageResult:
    """Run the stage-1 pipeline for one PDF using the first page only."""

    stem = _resolve_pdf_output_stem(pdf_path)
    rendered_dir  = output_root / "images" / stem
    processed_dir = output_root / "processed" / stem
    raw_dir       = output_root / "raw" / stem
    semantic_dir  = output_root / "semantic" / stem
    hierarchy_dir = output_root / "hierarchy" / stem
    viz_dir       = output_root / "visualizations" / stem

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

    raw_document      = load_json_document(raw_json_target)
    semantic_document = build_semantic_document(raw_document, source_json_path=raw_json_target)
    hierarchy_document = build_hierarchy_document(semantic_document, source_pdf_path=pdf_path)

    semantic_json_path  = _save_json(semantic_dir / f"{stem}_semantic_blocks.json", semantic_document.to_dict())
    hierarchy_json_path = _save_json(hierarchy_dir / f"{stem}_hierarchy.json", hierarchy_document.to_dict())

    canonical_data = _build_canonical_document(
        pdf_path=pdf_path,
        page_number=1,
        rendered_image_path=preprocessing_result.rendered_image,
        preprocessed_image_path=preprocessing_result.processed_image,
        raw_json_path=raw_json_target,
        semantic_document=semantic_document,
        hierarchy_document=hierarchy_document,
    )
    canonical_json_path = _save_json(
        output_root / "canonical" / stem / f"{stem}_document.json", canonical_data
    )

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
        canonical_json_path=canonical_json_path,
        semantic_visualization_path=semantic_visualization_path,
        hierarchy_visualization_path=hierarchy_visualization_path,
        semantic_document=semantic_document.to_dict(),
        hierarchy_document=hierarchy_document.to_dict(),
    )


# ---------------------------------------------------------------------------
# Multi-page pipeline
# ---------------------------------------------------------------------------

def run_stage1_for_pdf_multipage(
    pdf_path: Path,
    output_root: Path,
    preprocess_config: Optional[ImagePreprocessingConfig] = None,
    ocr_engine: Any = None,
    max_pages: Optional[int] = None,
) -> Stage1MultiPageResult:
    """Run Stage 1 for ALL pages of a PDF.

    Each page is rendered → preprocessed → OCR'd → semantic blocks extracted.
    All per-page blocks are assembled into one SemanticDocument and one
    HierarchyDocument, then written as a merged canonical JSON (schema 1.1).

    Args:
        pdf_path: Path to the input PDF.
        output_root: Root output directory.
        preprocess_config: Image preprocessing settings.
        ocr_engine: Initialized PPStructureV3 instance (created if None).
        max_pages: Process only the first N pages (None = all pages).

    Returns:
        Stage1MultiPageResult with per-page details and merged JSON paths.
    """
    stem = _resolve_pdf_output_stem(pdf_path)
    rendered_dir  = output_root / "images" / stem
    processed_dir = output_root / "processed" / stem
    raw_dir       = output_root / "raw" / stem
    semantic_dir  = output_root / "semantic" / stem
    hierarchy_dir = output_root / "hierarchy" / stem
    viz_dir       = output_root / "visualizations" / stem

    if preprocess_config is None:
        preprocess_config = ImagePreprocessingConfig()

    # ---- Render all pages --------------------------------------------------
    preprocessing_results = render_and_preprocess_all_pages(
        pdf_path=pdf_path,
        rendered_dir=rendered_dir,
        processed_dir=processed_dir,
        config=preprocess_config,
        max_pages=max_pages,
    )
    total_pages = len(preprocessing_results)
    logger.info("Rendered %d page(s) from %s", total_pages, pdf_path.name)

    if ocr_engine is None:
        ocr_engine = initialize_ppstructurev3()

    # ---- OCR + semantic blocks per page ------------------------------------
    all_page_semantic_blocks = []
    page_results: List[Stage1PageResult] = []
    semantic_viz_paths: List[Path] = []

    for page_idx, prep in enumerate(preprocessing_results):
        page_number = page_idx + 1
        ocr_out_dir = output_root / "ocr" / stem / f"page_{page_number:03d}"

        run_result = run_ppstructurev3(prep.processed_image, ocr_out_dir, engine=ocr_engine)
        if not run_result.json_files:
            logger.warning("No OCR output for page %d of %s — skipping", page_number, pdf_path.name)
            continue

        # Copy raw JSON and inject correct page_index
        raw_json_src    = run_result.json_files[0]
        raw_json_target = raw_dir / f"page_{page_number:03d}" / raw_json_src.name
        raw_json_target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(raw_json_src, raw_json_target)

        raw_doc = load_json_document(raw_json_target)
        if isinstance(raw_doc, dict):
            raw_doc["page_index"] = page_idx   # 0-based; extract_page_records adds 1

        # Build semantic blocks for this page
        page_records   = extract_page_records(raw_doc)
        page_candidates = []
        for pr in page_records:
            page_candidates.extend(detect_page_candidates(pr))

        page_sem = map_page_candidates(
            page_candidates,
            page_number=page_number,
            source_path=str(raw_json_target),
        )
        all_page_semantic_blocks.append(page_sem)

        # Per-page semantic visualization
        sem_viz = visualize_semantic_blocks(
            source_image_path=prep.rendered_image,
            page_blocks=page_sem,
            output_path=viz_dir / f"{stem}_p{page_number:03d}_semantic_blocks.png",
        )
        semantic_viz_paths.append(sem_viz)

        page_results.append(Stage1PageResult(
            pdf_path=pdf_path,
            page_number=page_number,
            rendered_image_path=prep.rendered_image,
            preprocessed_image_path=prep.processed_image,
            raw_json_path=raw_json_target,
            semantic_json_path=semantic_dir / f"{stem}_p{page_number:03d}_semantic_blocks.json",
            hierarchy_json_path=hierarchy_dir / f"{stem}_p{page_number:03d}_hierarchy.json",
            canonical_json_path=output_root / "canonical" / stem / f"{stem}_p{page_number:03d}_document.json",
            semantic_visualization_path=sem_viz,
            hierarchy_visualization_path=viz_dir / f"{stem}_p{page_number:03d}_hierarchy.png",
        ))

    # ---- Assemble full document --------------------------------------------
    semantic_doc   = SemanticDocument(pages=all_page_semantic_blocks)
    hierarchy_doc  = build_hierarchy_document(semantic_doc, source_pdf_path=pdf_path)

    # ---- Save merged outputs -----------------------------------------------
    semantic_json_path = _save_json(
        semantic_dir / f"{stem}_semantic_blocks.json",
        semantic_doc.to_dict(),
    )
    hierarchy_json_path = _save_json(
        hierarchy_dir / f"{stem}_hierarchy.json",
        hierarchy_doc.to_dict(),
    )

    canonical_data = _build_canonical_document_multipage(
        pdf_path=pdf_path,
        total_pages=total_pages,
        page_preprocessing_results=preprocessing_results,
        semantic_document=semantic_doc,
        hierarchy_document=hierarchy_doc,
    )
    canonical_json_path = _save_json(
        output_root / "canonical" / stem / f"{stem}_document.json",
        canonical_data,
    )

    hierarchy_viz_path = visualize_hierarchy_tree(
        root=hierarchy_doc.root,
        output_path=viz_dir / f"{stem}_hierarchy.png",
    )

    total_blocks = sum(len(p.blocks) for p in all_page_semantic_blocks)
    logger.info(
        "Stage 1 complete for %s: %d pages, %d total blocks",
        pdf_path.name, total_pages, total_blocks,
    )

    return Stage1MultiPageResult(
        pdf_path=pdf_path,
        total_pages=total_pages,
        page_results=page_results,
        canonical_json_path=canonical_json_path,
        semantic_json_path=semantic_json_path,
        hierarchy_json_path=hierarchy_json_path,
        semantic_visualization_paths=semantic_viz_paths,
        hierarchy_visualization_path=hierarchy_viz_path,
    )


# ---------------------------------------------------------------------------
# Batch pipeline entry point
# ---------------------------------------------------------------------------

def run_stage1_pipeline(
    config: ProjectConfig,
    samples_per_vendor: Optional[int] = None,
    seed: Optional[int] = None,
    dpi: Optional[int] = None,
    deskew: bool = False,
    denoise: bool = False,
    enhance_resolution: bool = False,
    single_pdf: Optional[Path] = None,
    multipage: bool = True,
    max_pages: Optional[int] = None,
) -> Stage1RunResult:
    """Run the stage-1 pipeline on the CatalogBank sample.

    Args:
        config: Project config.
        samples_per_vendor: Number of PDFs to sample per vendor.
        seed: Random seed for deterministic sampling.
        dpi: Rendering DPI (overrides config).
        deskew / denoise / enhance_resolution: Preprocessing toggles.
        single_pdf: If set, process only this one PDF (ignores dataset sampling).
        multipage: If True (default), use run_stage1_for_pdf_multipage for all pages.
                   If False, use the legacy first-page-only run_stage1_for_pdf.
        max_pages: Cap number of pages per PDF (only used when multipage=True).
    """
    ensure_project_directories(config)

    preprocess_config = ImagePreprocessingConfig(
        dpi=dpi or config.preprocessing.dpi,
        first_page_only=not multipage,
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
    ocr_engine  = initialize_ppstructurev3(**config.ocr.ppstructurev3)

    page_results: List[Stage1PageResult] = []
    multipage_results: List[Stage1MultiPageResult] = []

    for pdf_path in selected_pdfs:
        logger.info("Running stage-1 pipeline for %s (multipage=%s)", pdf_path, multipage)
        if multipage:
            result = run_stage1_for_pdf_multipage(
                pdf_path=pdf_path,
                output_root=output_root,
                preprocess_config=preprocess_config,
                ocr_engine=ocr_engine,
                max_pages=max_pages,
            )
            multipage_results.append(result)
            page_results.extend(result.page_results)
        else:
            result = run_stage1_for_pdf(
                pdf_path=pdf_path,
                output_root=output_root,
                preprocess_config=preprocess_config,
                ocr_engine=ocr_engine,
            )
            page_results.append(result)

    return Stage1RunResult(
        selected_pdfs=selected_pdfs,
        page_results=page_results,
        multipage_results=multipage_results,
    )
