"""Hierarchy tree schemas."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class HierarchyNode:
    """A tree node in the reconstructed hierarchy."""

    name: str
    semantic_type: str
    block_id: Optional[str] = None
    page: Optional[int] = None
    source_type: Optional[str] = None
    bbox: Optional[List[float]] = None
    confidence: Optional[float] = None
    order: Optional[int] = None
    children: List["HierarchyNode"] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "semantic_type": self.semantic_type,
            "block_id": self.block_id,
            "page": self.page,
            "source_type": self.source_type,
            "bbox": self.bbox,
            "confidence": self.confidence,
            "order": self.order,
            "children": [child.to_dict() for child in self.children],
            "metadata": self.metadata,
        }


@dataclass
class HierarchyDocument:
    """Document-wide hierarchy result."""

    root: HierarchyNode
    source_json_path: Optional[Path] = None
    source_pdf_path: Optional[Path] = None
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": self.root.to_dict(),
            "source_json_path": str(self.source_json_path) if self.source_json_path else None,
            "source_pdf_path": str(self.source_pdf_path) if self.source_pdf_path else None,
            "page_number": self.page_number,
            "metadata": self.metadata,
        }
