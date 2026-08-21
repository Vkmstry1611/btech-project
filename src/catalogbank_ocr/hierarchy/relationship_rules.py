"""Spatial and semantic rules for hierarchy reconstruction."""

from __future__ import annotations

from typing import Iterable, Optional, Sequence, Tuple

from catalogbank_ocr.schemas.semantic_block import SemanticBlock


def _normalize_bbox(bbox):
    if bbox is None:
        return None
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        try:
            return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]
        except (TypeError, ValueError):
            return None
    return None


def bbox_left(block: SemanticBlock) -> Optional[float]:
    bbox = _normalize_bbox(block.bbox)
    return bbox[0] if bbox else None


def bbox_top(block: SemanticBlock) -> Optional[float]:
    bbox = _normalize_bbox(block.bbox)
    return bbox[1] if bbox else None


def bbox_right(block: SemanticBlock) -> Optional[float]:
    bbox = _normalize_bbox(block.bbox)
    return bbox[2] if bbox else None


def bbox_bottom(block: SemanticBlock) -> Optional[float]:
    bbox = _normalize_bbox(block.bbox)
    return bbox[3] if bbox else None


def bbox_width(block: SemanticBlock) -> Optional[float]:
    bbox = _normalize_bbox(block.bbox)
    if bbox is None:
        return None
    return max(0.0, bbox[2] - bbox[0])


def bbox_height(block: SemanticBlock) -> Optional[float]:
    bbox = _normalize_bbox(block.bbox)
    if bbox is None:
        return None
    return max(0.0, bbox[3] - bbox[1])


def bbox_center_x(block: SemanticBlock) -> Optional[float]:
    bbox = _normalize_bbox(block.bbox)
    if bbox is None:
        return None
    return (bbox[0] + bbox[2]) / 2.0


def horizontal_overlap_ratio(a: SemanticBlock, b: SemanticBlock) -> float:
    box_a = _normalize_bbox(a.bbox)
    box_b = _normalize_bbox(b.bbox)
    if box_a is None or box_b is None:
        return 0.0
    left = max(box_a[0], box_b[0])
    right = min(box_a[2], box_b[2])
    overlap = max(0.0, right - left)
    width = min(max(1.0, box_a[2] - box_a[0]), max(1.0, box_b[2] - box_b[0]))
    return overlap / width


def vertical_gap(parent: SemanticBlock, child: SemanticBlock) -> Optional[float]:
    parent_bottom = bbox_bottom(parent)
    child_top = bbox_top(child)
    if parent_bottom is None or child_top is None:
        return None
    return child_top - parent_bottom


def same_column(a: SemanticBlock, b: SemanticBlock, min_overlap: float = 0.2) -> bool:
    return horizontal_overlap_ratio(a, b) >= min_overlap


def should_attach_to_container(container: SemanticBlock, child: SemanticBlock) -> bool:
    """Decide whether a child block belongs to a nearby product-card/container block."""

    if container.page != child.page:
        return False
    if container.bbox is None or child.bbox is None:
        return False

    gap = vertical_gap(container, child)
    if gap is None:
        return False
    if gap < -5:
        return False

    container_height = bbox_height(container) or 0.0
    child_height = bbox_height(child) or 0.0
    max_gap = max(60.0, container_height * 1.8, child_height * 1.2)
    if gap > max_gap:
        return False

    if same_column(container, child):
        return True

    # Allow slightly offset blocks when the child is narrow but text-heavy.
    if child.type in {"specification", "paragraph", "table"} and gap <= max_gap * 0.8:
        return True

    return False
