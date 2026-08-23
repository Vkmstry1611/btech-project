"""Detect raw semantic candidates from PP-StructureV3 JSON output."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from catalogbank_ocr.semantic.block_types import normalize_label, normalize_text

@dataclass
class BlockCandidate:
    """A raw candidate extracted from PP-StructureV3 output."""

    page: int
    source_type: str
    text: str = ""
    bbox: Optional[List[float]] = None
    confidence: Optional[float] = None
    order: int = 0
    source_ref: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page": self.page,
            "source_type": self.source_type,
            "text": self.text,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "order": self.order,
            "source_ref": self.source_ref,
            "raw": self.raw,
        }


def load_json_document(json_path: Path) -> Dict[str, Any]:
    """Load a PP-StructureV3 JSON document."""

    with json_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def extract_page_records(raw_document: Any) -> List[Dict[str, Any]]:
    """Extract per-page records from a raw PP-StructureV3 document.

    PP-StructureV3 saves one JSON file per page; the top-level object is a
    ``dict`` with a ``parsing_res_list`` key that holds the clean, de-duplicated
    layout blocks.  This format is treated as the authoritative source.

    Older or alternative shapes (list of pages, ``layoutParsingResults``, …) are
    handled as fallbacks so that unit-test fixtures and future schema changes
    continue to work.
    """

    # ---- Primary: flat PP-StructureV3 per-page dict -------------------------
    if isinstance(raw_document, dict) and "parsing_res_list" in raw_document:
        page_number = 1
        raw_page_index = raw_document.get("page_index")
        if isinstance(raw_page_index, int) and raw_page_index >= 0:
            page_number = raw_page_index + 1
        return [
            {
                "page_number": page_number,
                "page_index": page_number,
                "page_item": raw_document,
                "top_meta": raw_document,
                "_use_parsing_res_list": True,
            }
        ]

    # ---- Fallback: list of pages or legacy layoutParsingResults -------------
    if isinstance(raw_document, list):
        pages = raw_document
        top_meta: Dict[str, Any] = {}
    elif isinstance(raw_document, dict):
        top_meta = raw_document
        if isinstance(raw_document.get("layoutParsingResults"), list):
            pages = raw_document["layoutParsingResults"]
        elif isinstance(raw_document.get("pages"), list):
            pages = raw_document["pages"]
        elif isinstance(raw_document.get("ocrResults"), list):
            pages = raw_document["ocrResults"]
        else:
            pages = [raw_document]
    else:
        raise TypeError(f"Unsupported PP-StructureV3 document type: {type(raw_document).__name__}")

    page_records: List[Dict[str, Any]] = []
    for index, page_item in enumerate(pages, start=1):
        if not isinstance(page_item, dict):
            continue
        page_number = _extract_page_number(page_item, index)
        page_records.append(
            {
                "page_number": page_number,
                "page_index": index,
                "page_item": page_item,
                "top_meta": top_meta,
                "_use_parsing_res_list": False,
            }
        )
    return page_records


def _extract_page_number(page_item: Dict[str, Any], default_index: int) -> int:
    for key in ("page_index", "pageIndex", "page", "page_number", "pageNumber"):
        value = page_item.get(key)
        if isinstance(value, int):
            return value + 1 if key in {"page_index", "pageIndex"} and value >= 0 else value
    return default_index


def _normalize_bbox(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    if isinstance(value, dict):
        left_top = value.get("left_top") or value.get("leftTop")
        right_bottom = value.get("right_bottom") or value.get("rightBottom")
        if isinstance(left_top, (list, tuple)) and isinstance(right_bottom, (list, tuple)) and len(left_top) >= 2 and len(right_bottom) >= 2:
            return [float(left_top[0]), float(left_top[1]), float(right_bottom[0]), float(right_bottom[1])]
        if "bbox" in value:
            return _normalize_bbox(value["bbox"])
        return None
    if isinstance(value, (list, tuple)):
        if len(value) == 4 and all(isinstance(item, (int, float)) for item in value):
            return [float(item) for item in value]
        if len(value) == 8 and all(isinstance(item, (int, float)) for item in value):
            xs = [float(value[i]) for i in range(0, 8, 2)]
            ys = [float(value[i + 1]) for i in range(0, 8, 2)]
            return [min(xs), min(ys), max(xs), max(ys)]
    return None


def _extract_text(node: Dict[str, Any]) -> str:
    for key in (
        "block_content",
        "block_text",
        "text",
        "content",
        "markdown_text",
        "markdownText",
        "rec_text",
        "label",
        "name",
    ):
        value = node.get(key)
        if isinstance(value, str) and value.strip():
            return normalize_text(value)
    if isinstance(node.get("markdown"), dict):
        md_text = node["markdown"].get("text")
        if isinstance(md_text, str) and md_text.strip():
            return normalize_text(md_text)
    if isinstance(node.get("texts"), list):
        parts = [normalize_text(item) for item in node["texts"] if isinstance(item, str) and item.strip()]
        if parts:
            return "\n".join(parts)
    return ""


def _extract_confidence(node: Dict[str, Any]) -> Optional[float]:
    for key in ("confidence", "score", "block_score", "rec_score", "dt_score"):
        value = node.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    if isinstance(node.get("scores"), list) and node["scores"]:
        scores = [float(item) for item in node["scores"] if isinstance(item, (int, float))]
        if scores:
            return sum(scores) / float(len(scores))
    return None


def _extract_order(node: Dict[str, Any], fallback: int) -> int:
    for key in ("block_order", "order", "seq", "idx", "index", "reading_order"):
        value = node.get(key)
        if isinstance(value, int):
            return value
    return fallback


def _dict_looks_like_block(node: Dict[str, Any]) -> bool:
    has_bbox = any(key in node for key in ("block_bbox", "bbox", "left_top", "right_bottom"))
    has_label = any(key in node for key in ("block_label", "label", "type", "name"))
    has_content = any(key in node for key in ("block_content", "block_text", "text", "content", "markdown_text", "rec_text"))
    return has_bbox and (has_label or has_content)


def _emit_candidate(node: Dict[str, Any], page: int, source_ref: str, fallback_order: int) -> Optional[BlockCandidate]:
    if not _dict_looks_like_block(node):
        return None
    source_type = normalize_label(node.get("block_label") or node.get("label") or node.get("type") or node.get("name"))
    text = _extract_text(node)
    bbox = _normalize_bbox(node.get("block_bbox") or node.get("bbox") or node)
    if bbox is None:
        bbox = _normalize_bbox(node.get("bbox"))
    return BlockCandidate(
        page=page,
        source_type=source_type or "unknown",
        text=text,
        bbox=bbox,
        confidence=_extract_confidence(node),
        order=_extract_order(node, fallback_order),
        source_ref=source_ref,
        raw=node,
    )


def _parse_markdown_blocks(markdown_text: str, page: int, source_ref_prefix: str) -> List[BlockCandidate]:
    lines = markdown_text.splitlines()
    candidates: List[BlockCandidate] = []
    paragraph_buffer: List[str] = []
    order = 0

    def flush_paragraph() -> None:
        nonlocal order
        if paragraph_buffer:
            text = "\n".join(paragraph_buffer).strip()
            if text:
                order += 1
                candidates.append(
                    BlockCandidate(
                        page=page,
                        source_type="markdown",
                        text=text,
                        bbox=None,
                        confidence=None,
                        order=order,
                        source_ref=f"{source_ref_prefix}.markdown.paragraph[{order}]",
                        raw={"text": text, "kind": "paragraph"},
                    )
                )
            paragraph_buffer.clear()

    for line in lines:
        stripped = line.strip()
        if not stripped:
            flush_paragraph()
            continue
        if stripped.startswith("#"):
            flush_paragraph()
            order += 1
            candidates.append(
                BlockCandidate(
                    page=page,
                    source_type="heading",
                    text=stripped,
                    bbox=None,
                    confidence=None,
                    order=order,
                    source_ref=f"{source_ref_prefix}.markdown.heading[{order}]",
                    raw={"text": stripped, "kind": "heading"},
                )
            )
            continue
        if "|" in stripped and stripped.count("|") >= 2:
            flush_paragraph()
            order += 1
            candidates.append(
                BlockCandidate(
                    page=page,
                    source_type="table",
                    text=stripped,
                    bbox=None,
                    confidence=None,
                    order=order,
                    source_ref=f"{source_ref_prefix}.markdown.table[{order}]",
                    raw={"text": stripped, "kind": "table"},
                )
            )
            continue
        if stripped.startswith(("- ", "* ", "+ ")):
            flush_paragraph()
            order += 1
            candidates.append(
                BlockCandidate(
                    page=page,
                    source_type="paragraph",
                    text=stripped,
                    bbox=None,
                    confidence=None,
                    order=order,
                    source_ref=f"{source_ref_prefix}.markdown.list[{order}]",
                    raw={"text": stripped, "kind": "list_item"},
                )
            )
            continue
        paragraph_buffer.append(stripped)

    flush_paragraph()
    return candidates


def _parse_ocr_lines(node: Dict[str, Any], page: int, source_ref_prefix: str) -> List[BlockCandidate]:
    texts = node.get("rec_texts") or []
    boxes = node.get("rec_boxes") or []
    scores = node.get("rec_scores") or []
    candidates: List[BlockCandidate] = []
    for idx, text in enumerate(texts):
        if not isinstance(text, str) or not text.strip():
            continue
        bbox = None
        if idx < len(boxes):
            bbox = _normalize_bbox(boxes[idx])
        score = None
        if idx < len(scores) and isinstance(scores[idx], (int, float)):
            score = float(scores[idx])
        candidates.append(
            BlockCandidate(
                page=page,
                source_type="ocr_line",
                text=normalize_text(text),
                bbox=bbox,
                confidence=score,
                order=idx + 1,
                source_ref=f"{source_ref_prefix}.ocr[{idx}]",
                raw={"text": text, "bbox": bbox, "score": score},
            )
        )
    return candidates


def _parse_output_images(node: Dict[str, Any], page: int, source_ref_prefix: str) -> List[BlockCandidate]:
    candidates: List[BlockCandidate] = []
    for key in ("outputImages", "markdownImages", "images"):
        value = node.get(key)
        if isinstance(value, dict):
            for idx, (name, resource) in enumerate(value.items(), start=1):
                candidates.append(
                    BlockCandidate(
                        page=page,
                        source_type="figure",
                        text=str(name),
                        bbox=None,
                        confidence=None,
                        order=1000 + idx,
                        source_ref=f"{source_ref_prefix}.{key}[{name}]",
                        raw={"resource": resource, "name": name, "kind": key},
                    )
                )
    return candidates


def _walk_candidates(node: Any, page: int, source_ref: str, results: List[BlockCandidate], visited: set) -> None:
    obj_id = id(node)
    if obj_id in visited:
        return
    visited.add(obj_id)

    if isinstance(node, dict):
        candidate = _emit_candidate(node, page, source_ref, len(results) + 1)
        if candidate is not None:
            results.append(candidate)

        if isinstance(node.get("rec_texts"), list) and isinstance(node.get("rec_boxes"), list):
            results.extend(_parse_ocr_lines(node, page, source_ref))

        results.extend(_parse_output_images(node, page, source_ref))

        markdown_text = None
        if isinstance(node.get("markdownText"), str):
            markdown_text = node["markdownText"]
        elif isinstance(node.get("markdown"), dict) and isinstance(node["markdown"].get("text"), str):
            markdown_text = node["markdown"]["text"]

        if markdown_text:
            results.extend(_parse_markdown_blocks(markdown_text, page, source_ref))

        for key, value in node.items():
            if key in {"markdown", "layoutParsingResults", "pages", "ocrResults"} and isinstance(value, list):
                for idx, item in enumerate(value):
                    _walk_candidates(item, page, f"{source_ref}.{key}[{idx}]", results, visited)
            elif key in {"prunedResult", "result", "raw"}:
                _walk_candidates(value, page, f"{source_ref}.{key}", results, visited)
            elif key in {"boxes", "results", "items", "blocks", "elements", "regions"} and isinstance(value, list):
                for idx, item in enumerate(value):
                    _walk_candidates(item, page, f"{source_ref}.{key}[{idx}]", results, visited)
            elif isinstance(value, dict):
                _walk_candidates(value, page, f"{source_ref}.{key}", results, visited)
            elif isinstance(value, list):
                for idx, item in enumerate(value):
                    if isinstance(item, (dict, list)):
                        _walk_candidates(item, page, f"{source_ref}.{key}[{idx}]", results, visited)
    elif isinstance(node, list):
        for idx, item in enumerate(node):
            _walk_candidates(item, page, f"{source_ref}[{idx}]", results, visited)


def detect_page_candidates(page_record: Dict[str, Any]) -> List[BlockCandidate]:
    """Detect raw block candidates for one page record.

    When the record was produced from a PP-StructureV3 per-page JSON
    (identified by ``_use_parsing_res_list=True``), the clean ``parsing_res_list``
    is used as the exclusive source.  Confidence scores are joined from
    ``layout_det_res.boxes`` when available.

    For legacy / alternative document shapes the generic recursive walker is
    used as a fallback.
    """

    page_number = int(page_record["page_number"])
    payload = page_record.get("page_item", {})

    if page_record.get("_use_parsing_res_list") and isinstance(payload.get("parsing_res_list"), list):
        layout_det_boxes = None
        ldr = payload.get("layout_det_res")
        if isinstance(ldr, dict):
            layout_det_boxes = ldr.get("boxes")
        return _detect_from_parsing_res_list(
            payload["parsing_res_list"], page_number, layout_det_boxes=layout_det_boxes
        )

    # Fallback: generic walk
    candidates: List[BlockCandidate] = []
    visited: set = set()
    _walk_candidates(payload, page_number, "page[" + str(page_number) + "]", candidates, visited)
    return candidates


def _detect_from_parsing_res_list(
    parsing_res_list: List[Dict[str, Any]], page_number: int,
    layout_det_boxes: Optional[List[Dict[str, Any]]] = None,
) -> List[BlockCandidate]:
    """Build candidates directly from PP-StructureV3 parsing_res_list blocks.

    Each entry has a fixed schema::

        {
            "block_label":   str,    # layout class, e.g. "paragraph_title"
            "block_content": str,    # extracted text / html
            "block_bbox":    [x1, y1, x2, y2],
            "block_id":      int,
            "block_order":   int,
        }

    Confidence scores are not included in ``parsing_res_list`` but are available
    in ``layout_det_res.boxes``.  When ``layout_det_boxes`` is supplied the
    scores are joined by matching bounding-box coordinates (rounded to integer
    pixels), providing per-block confidence in the output.

    Entries whose label belongs to the ``IGNORE_LABELS`` set (footer, header,
    number) are kept but tagged so downstream consumers can decide whether to
    drop them.
    """
    from catalogbank_ocr.semantic.block_types import IGNORE_LABELS, normalize_label, normalize_text  # local import to avoid circular

    # Build a bbox→score lookup from layout_det_res.boxes if provided.
    # layout_det_res.boxes are sorted by confidence (descending) and use float
    # coordinates.  parsing_res_list uses integer coordinates from the same
    # detections, so bboxes should match within ±2 px after rounding.
    # We build an approximate lookup: map (x1_round, y1_round) → score so a
    # single integer-coord key lookup is fast and tolerant of rounding.
    bbox_to_score: Dict[tuple, float] = {}
    if layout_det_boxes:
        for box in layout_det_boxes:
            coord = box.get("coordinate")
            score = box.get("score")
            if coord and isinstance(score, (int, float)) and len(coord) == 4:
                # Key on all four corners rounded — exact match preferred
                key_exact = tuple(round(float(v)) for v in coord)
                bbox_to_score[key_exact] = float(score)
                # Also store a key based on top-left + bottom-right rounded to nearest 5
                # for approximate matching of integer-coord parsing_res_list entries
                key_approx = tuple(round(float(v) / 5) * 5 for v in coord)
                if key_approx not in bbox_to_score:
                    bbox_to_score[key_approx] = float(score)

    def _lookup_score(bbox: Optional[List[float]]) -> Optional[float]:
        """Try exact then approximate bbox→score lookup."""
        if bbox is None or not bbox_to_score:
            return None
        # Exact integer match
        key = tuple(round(v) for v in bbox)
        if key in bbox_to_score:
            return bbox_to_score[key]
        # Approximate match (nearest-5 grid)
        key_approx = tuple(round(v / 5) * 5 for v in bbox)
        if key_approx in bbox_to_score:
            return bbox_to_score[key_approx]
        # Linear scan with ±3 px tolerance on each corner
        for stored_key, score in bbox_to_score.items():
            if len(stored_key) == 4 and all(abs(stored_key[i] - key[i]) <= 3 for i in range(4)):
                return score
        return None

    candidates: List[BlockCandidate] = []
    for entry in parsing_res_list:
        if not isinstance(entry, dict):
            continue
        label = normalize_label(entry.get("block_label") or entry.get("label") or "unknown")
        text = normalize_text(entry.get("block_content") or entry.get("block_text") or entry.get("text") or "")
        bbox = _normalize_bbox(entry.get("block_bbox") or entry.get("bbox"))
        order = entry.get("block_order") or entry.get("order") or len(candidates) + 1
        block_id = entry.get("block_id")
        source_ref = "parsing_res_list[" + str(block_id if block_id is not None else order) + "]"

        # Try to find confidence from layout_det_res.boxes via bbox match.
        confidence = _extract_confidence(entry)  # usually None for parsing_res_list entries
        if confidence is None and bbox is not None:
            confidence = _lookup_score(bbox)

        candidates.append(
            BlockCandidate(
                page=page_number,
                source_type=label,
                text=text,
                bbox=bbox,
                confidence=confidence,
                order=int(order),
                source_ref=source_ref,
                raw=entry,
            )
        )
    return candidates


def detect_semantic_candidates(raw_document: Any) -> List[BlockCandidate]:
    """Detect raw block candidates across an entire document."""

    all_candidates: List[BlockCandidate] = []
    for page_record in extract_page_records(raw_document):
        all_candidates.extend(detect_page_candidates(page_record))
    return all_candidates


def detect_semantic_document(raw_document: Any) -> List[BlockCandidate]:
    """Backward-compatible alias for document-level candidate detection."""

    return detect_semantic_candidates(raw_document)
