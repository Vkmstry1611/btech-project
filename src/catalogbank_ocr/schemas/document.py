"""Document-level schemas used by the stage-1 research pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


def _to_path_string(value: Optional[Path]) -> Optional[str]:
    if value is None:
        return None
    return str(value)


@dataclass
class DocumentSource:
    """Source file metadata for a processed document."""

    pdf_path: Path
    page_number: int
    rendered_image_path: Path
    preprocessed_image_path: Path
    raw_json_path: Path

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pdf_path": str(self.pdf_path),
            "page_number": self.page_number,
            "rendered_image_path": str(self.rendered_image_path),
            "preprocessed_image_path": str(self.preprocessed_image_path),
            "raw_json_path": str(self.raw_json_path),
        }


@dataclass
class DocumentPage:
    """A page with extracted semantic blocks and hierarchy."""

    page_number: int
    source: DocumentSource
    raw_page_result: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "page_number": self.page_number,
            "source": self.source.to_dict(),
            "raw_page_result": self.raw_page_result,
            "metadata": self.metadata,
        }


@dataclass
class Stage1Document:
    """Top-level output of the stage-1 research pipeline."""

    source: DocumentSource
    pages: List[DocumentPage] = field(default_factory=list)
    semantic_json_path: Optional[Path] = None
    hierarchy_json_path: Optional[Path] = None
    semantic_visualization_path: Optional[Path] = None
    hierarchy_visualization_path: Optional[Path] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "pages": [page.to_dict() for page in self.pages],
            "semantic_json_path": _to_path_string(self.semantic_json_path),
            "hierarchy_json_path": _to_path_string(self.hierarchy_json_path),
            "semantic_visualization_path": _to_path_string(self.semantic_visualization_path),
            "hierarchy_visualization_path": _to_path_string(self.hierarchy_visualization_path),
        }
