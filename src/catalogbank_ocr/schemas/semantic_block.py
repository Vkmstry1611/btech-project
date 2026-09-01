"""Canonical semantic block schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SemanticBlock:
    """A normalized document block preserving raw PP-StructureV3 information."""

    id: str
    page: int
    type: str
    source_type: str
    text: str = ""
    bbox: Optional[List[float]] = None
    confidence: Optional[float] = None
    order: int = 0
    source_ref: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)
    heading_level: Optional[int] = None
    layout_hint: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "page": self.page,
            "type": self.type,
            "source_type": self.source_type,
            "text": self.text,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "order": self.order,
            "source_ref": self.source_ref,
            "raw": self.raw,
            "parent_id": self.parent_id,
            "children_ids": list(self.children_ids),
            "heading_level": self.heading_level,
            "layout_hint": self.layout_hint,
        }


@dataclass
class PageSemanticBlocks:
    """Semantic blocks for a single page."""

    page: int
    source_path: Optional[str] = None
    blocks: List[SemanticBlock] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page": self.page,
            "source_path": self.source_path,
            "blocks": [block.to_dict() for block in self.blocks],
        }


@dataclass
class SemanticDocument:
    """Document-wide semantic block container."""

    source_json_path: Optional[Path] = None
    pages: List[PageSemanticBlocks] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_json_path": str(self.source_json_path) if self.source_json_path else None,
            "pages": [page.to_dict() for page in self.pages],
        }
