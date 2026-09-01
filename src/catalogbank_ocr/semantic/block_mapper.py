"""Convert raw PP-StructureV3 candidates into canonical semantic blocks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from catalogbank_ocr.schemas.semantic_block import PageSemanticBlocks, SemanticBlock, SemanticDocument
from catalogbank_ocr.semantic.block_detector import BlockCandidate, detect_semantic_candidates, extract_page_records
from catalogbank_ocr.semantic.block_types import SemanticType, infer_heading_level, infer_semantic_type, normalize_label, normalize_text


def map_raw_candidate_to_semantic_block(candidate: BlockCandidate) -> SemanticBlock:
    """Map a raw candidate to the canonical semantic block schema."""

    block_type = infer_semantic_type(candidate.source_type, candidate.text, candidate.raw)
    heading_level = None
    if block_type == SemanticType.HEADING:
        heading_level = infer_heading_level(candidate.source_type, candidate.text, candidate.raw)

    block_id = f"p{candidate.page}_b{candidate.order:03d}"
    layout_hint = candidate.source_type
    return SemanticBlock(
        id=block_id,
        page=candidate.page,
        type=block_type.value,
        source_type=candidate.source_type,
        text=candidate.text,
        bbox=candidate.bbox,
        confidence=candidate.confidence,
        order=candidate.order,
        source_ref=candidate.source_ref,
        raw=candidate.raw,
        heading_level=heading_level,
        layout_hint=layout_hint,
    )


def map_page_candidates(page_candidates: List[BlockCandidate], page_number: int, source_path: Optional[str] = None) -> PageSemanticBlocks:
    """Map all page candidates to a page-level semantic block collection."""

    blocks = [map_raw_candidate_to_semantic_block(candidate) for candidate in page_candidates]
    blocks.sort(key=lambda block: (block.order, block.bbox[1] if block.bbox else 10**9, block.bbox[0] if block.bbox else 10**9))
    return PageSemanticBlocks(page=page_number, source_path=source_path, blocks=blocks)


def build_semantic_document(raw_document: Any, source_json_path: Optional[Path] = None) -> SemanticDocument:
    """Build a document-level semantic representation from raw PP-StructureV3 JSON."""

    pages: List[PageSemanticBlocks] = []
    for page_record in extract_page_records(raw_document):
        candidates = []
        page_number = int(page_record["page_number"])
        # We reuse the detector through the page record to preserve page context.
        from catalogbank_ocr.semantic.block_detector import detect_page_candidates

        candidates = detect_page_candidates(page_record)
        pages.append(map_page_candidates(candidates, page_number=page_number, source_path=str(source_json_path) if source_json_path else None))
    return SemanticDocument(source_json_path=source_json_path, pages=pages)
