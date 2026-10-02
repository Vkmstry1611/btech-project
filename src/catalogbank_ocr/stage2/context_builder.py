"""Stage 2 — Step 1: Context Builder.

Reads a canonical JSON document (schema_version 1.0 or 1.1) produced by Stage 1
and groups the content into *ContextChunks* — one chunk per heading section.

Works with both single-page (schema 1.0) and multi-page (schema 1.1) canonical
documents. Each chunk's page number is read directly from the heading node in
the hierarchy tree, which carries the correct page from Stage 1.

Each chunk captures everything the LLM needs to extract entities and relations
from one section of the catalog page:
  - The section heading text and level
  - All specification blocks (key-value attributes)
  - All table blocks (raw text preserved; parsed rows added by table_parser)
  - All product-card blocks
  - Supporting paragraph text (up to a char limit)
  - Figure references (name only — images are not sent to the LLM)

Design decisions (POC-lean):
  - Pure Python + stdlib only (no heavy deps)
  - Works directly from the dict produced by json.load() — no re-import of
    Stage 1 code needed
  - Flat iteration over hierarchy tree (recursive DFS) rather than re-reading
    pages[].blocks[], so the heading→children relationship is preserved
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class RawBlock:
    """Lightweight view of a canonical JSON block, enough for Stage 2."""

    block_id: str
    page: int
    block_type: str          # heading | table | figure | product_card | specification | paragraph
    source_type: str         # raw PP-StructureV3 label
    text: str
    heading_level: Optional[int]
    confidence: Optional[float]
    order: int
    bbox: Optional[List[float]]
    raw: Dict[str, Any]

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RawBlock":
        return cls(
            block_id=d.get("id", ""),
            page=d.get("page", 1),
            block_type=d.get("type", "paragraph"),
            source_type=d.get("source_type", ""),
            text=(d.get("text") or "").strip(),
            heading_level=d.get("heading_level"),
            confidence=d.get("confidence"),
            order=d.get("order", 0),
            bbox=d.get("bbox"),
            raw=d.get("raw") or {},
        )


@dataclass
class ContextChunk:
    """One section of a catalog page, ready to be sent to the LLM.

    Attributes
    ----------
    chunk_id        Stable identifier: ``{doc_stem}__p{page}__c{index}``
    doc_stem        Source document stem (filename without extension)
    page            Page number (always 1 for current Stage 1 output)
    section_path    Breadcrumb of heading names from root to this section,
                    e.g. ``["Pipe Fittings", "Stainless Steel Variants"]``
    heading_level   Depth in the heading stack (1 = top-level)
    headings        All heading blocks directly in this section
    tables          Table blocks (raw text intact; rows populated by table_parser)
    specifications  Specification blocks (key-value rich text)
    product_cards   Product-card blocks
    paragraphs      Paragraph / supporting text blocks
    figures         Figure reference blocks (name only)
    source_doc      Path to the source canonical JSON (for traceability)
    """

    chunk_id: str
    doc_stem: str
    page: int
    section_path: List[str]
    heading_level: int
    headings: List[RawBlock] = field(default_factory=list)
    tables: List[RawBlock] = field(default_factory=list)
    specifications: List[RawBlock] = field(default_factory=list)
    product_cards: List[RawBlock] = field(default_factory=list)
    paragraphs: List[RawBlock] = field(default_factory=list)
    figures: List[RawBlock] = field(default_factory=list)
    source_doc: Optional[str] = None

    # ---- convenience -------------------------------------------------------

    @property
    def section_name(self) -> str:
        return self.section_path[-1] if self.section_path else "(root)"

    @property
    def is_empty(self) -> bool:
        """True if the chunk has no extractable content beyond its heading."""
        return not (self.tables or self.specifications or self.product_cards or self.paragraphs)

    def block_count(self) -> int:
        return sum(
            len(lst)
            for lst in (
                self.headings, self.tables, self.specifications,
                self.product_cards, self.paragraphs, self.figures,
            )
        )

    def to_dict(self) -> Dict[str, Any]:
        def _blk(b: RawBlock) -> Dict[str, Any]:
            return {
                "id": b.block_id,
                "type": b.block_type,
                "text": b.text,
                "confidence": b.confidence,
                "heading_level": b.heading_level,
            }

        return {
            "chunk_id": self.chunk_id,
            "doc_stem": self.doc_stem,
            "page": self.page,
            "section_path": self.section_path,
            "heading_level": self.heading_level,
            "section_name": self.section_name,
            "block_count": self.block_count(),
            "headings": [_blk(b) for b in self.headings],
            "tables": [_blk(b) for b in self.tables],
            "specifications": [_blk(b) for b in self.specifications],
            "product_cards": [_blk(b) for b in self.product_cards],
            "paragraphs": [_blk(b) for b in self.paragraphs],
            "figures": [_blk(b) for b in self.figures],
            "source_doc": self.source_doc,
        }


# ---------------------------------------------------------------------------
# Hierarchy walker
# ---------------------------------------------------------------------------

def _node_to_raw_block(node: Dict[str, Any]) -> RawBlock:
    """Convert a hierarchy node dict into a RawBlock."""
    return RawBlock(
        block_id=node.get("block_id") or node.get("id") or "",
        page=node.get("page") or 1,
        block_type=node.get("semantic_type") or node.get("type") or "paragraph",
        source_type=node.get("source_type") or "",
        text=(node.get("name") or node.get("text") or "").strip(),
        heading_level=node.get("metadata", {}).get("heading_level") if node.get("metadata") else None,
        confidence=node.get("confidence"),
        order=node.get("order") or 0,
        bbox=node.get("bbox"),
        raw=node.get("metadata", {}).get("raw") or {},
    )


def _walk_hierarchy(
    node: Dict[str, Any],
    parent_path: List[str],
    doc_stem: str,
    source_doc: Optional[str],
    chunks: List[ContextChunk],
    chunk_counter: List[int],   # mutable counter via list
) -> None:
    """Recursively walk a hierarchy node and emit one ContextChunk per heading."""

    node_type = node.get("semantic_type") or node.get("type") or "paragraph"
    node_name = (node.get("name") or "").strip()
    children: List[Dict[str, Any]] = node.get("children") or []

    # Skip the synthetic root node — just recurse into its children
    if node_type == "document":
        for child in children:
            _walk_hierarchy(child, parent_path, doc_stem, source_doc, chunks, chunk_counter)
        return

    # Every heading node becomes a ContextChunk
    if node_type == "heading":
        current_path = parent_path + [node_name]
        heading_level = len(current_path)

        chunk_counter[0] += 1
        page = node.get("page") or 1
        chunk = ContextChunk(
            chunk_id=f"{doc_stem}__p{page}__c{chunk_counter[0]:03d}",
            doc_stem=doc_stem,
            page=page,
            section_path=current_path,
            heading_level=heading_level,
            source_doc=source_doc,
        )
        chunk.headings.append(_node_to_raw_block(node))

        # Distribute direct children into the appropriate bucket
        for child in children:
            child_type = child.get("semantic_type") or child.get("type") or "paragraph"

            if child_type == "heading":
                # Nested heading — recurse; do NOT add to this chunk's headings
                _walk_hierarchy(child, current_path, doc_stem, source_doc, chunks, chunk_counter)
            elif child_type == "table":
                chunk.tables.append(_node_to_raw_block(child))
            elif child_type == "specification":
                chunk.specifications.append(_node_to_raw_block(child))
            elif child_type == "product_card":
                chunk.product_cards.append(_node_to_raw_block(child))
                # Product-card children (specs / paragraphs nested inside) also belong here
                for grandchild in child.get("children") or []:
                    gc_type = grandchild.get("semantic_type") or grandchild.get("type") or "paragraph"
                    if gc_type == "specification":
                        chunk.specifications.append(_node_to_raw_block(grandchild))
                    elif gc_type == "table":
                        chunk.tables.append(_node_to_raw_block(grandchild))
                    elif gc_type == "paragraph":
                        chunk.paragraphs.append(_node_to_raw_block(grandchild))
            elif child_type == "figure":
                chunk.figures.append(_node_to_raw_block(child))
            else:
                # paragraph / unknown — keep as supporting text
                chunk.paragraphs.append(_node_to_raw_block(child))

        chunks.append(chunk)
        return

    # Non-heading, non-document nodes at the top level (edge case — e.g. a
    # page that starts with a table before any heading).  Attach to a synthetic
    # "Unlabelled" chunk or just skip figures/paragraphs without a heading.
    if node_type not in {"figure", "paragraph"}:
        # Worth capturing (table / spec / product_card at root level)
        chunk_counter[0] += 1
        page = node.get("page") or 1
        chunk = ContextChunk(
            chunk_id=f"{doc_stem}__p{page}__c{chunk_counter[0]:03d}",
            doc_stem=doc_stem,
            page=page,
            section_path=parent_path + ["(unlabelled)"],
            heading_level=len(parent_path) + 1,
            source_doc=source_doc,
        )
        if node_type == "table":
            chunk.tables.append(_node_to_raw_block(node))
        elif node_type == "specification":
            chunk.specifications.append(_node_to_raw_block(node))
        elif node_type == "product_card":
            chunk.product_cards.append(_node_to_raw_block(node))
        chunks.append(chunk)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_context_chunks(
    canonical_doc: Dict[str, Any],
    source_doc_path: Optional[str] = None,
    skip_empty: bool = True,
) -> List[ContextChunk]:
    """Build ContextChunks from a Stage 1 canonical document dict.

    Parameters
    ----------
    canonical_doc
        The dict produced by ``json.load()`` on a Stage 1 canonical JSON.
    source_doc_path
        Path string recorded for traceability (optional).
    skip_empty
        If True (default), chunks that contain only a heading and no
        extractable content (tables, specs, product cards, paragraphs) are
        dropped.  Useful to avoid sending empty LLM prompts.

    Returns
    -------
    List[ContextChunk]
        One chunk per heading section found in the hierarchy.
    """
    # Derive a document stem from the source path or a fallback
    if source_doc_path:
        doc_stem = Path(source_doc_path).stem
    else:
        doc_stem = canonical_doc.get("source", {}).get("pdf_path", "unknown")
        doc_stem = Path(doc_stem).stem if doc_stem else "unknown"

    hierarchy: Dict[str, Any] = canonical_doc.get("hierarchy") or {}

    chunks: List[ContextChunk] = []
    counter: List[int] = [0]
    _walk_hierarchy(hierarchy, [], doc_stem, source_doc_path, chunks, counter)

    if skip_empty:
        chunks = [c for c in chunks if not c.is_empty]

    return chunks


def build_context_chunks_from_file(
    canonical_json_path: Path,
    skip_empty: bool = True,
) -> List[ContextChunk]:
    """Convenience wrapper — loads the JSON file then calls build_context_chunks."""
    with canonical_json_path.open("r", encoding="utf-8") as fh:
        doc = json.load(fh)
    return build_context_chunks(doc, source_doc_path=str(canonical_json_path), skip_empty=skip_empty)


def summarise_chunks(chunks: List[ContextChunk]) -> str:
    """Return a human-readable summary string — useful for quick inspection."""
    lines = [f"Total chunks: {len(chunks)}"]
    for c in chunks:
        indent = "  " * (c.heading_level - 1)
        tag_parts = []
        if c.tables:
            tag_parts.append(f"{len(c.tables)} table(s)")
        if c.specifications:
            tag_parts.append(f"{len(c.specifications)} spec(s)")
        if c.product_cards:
            tag_parts.append(f"{len(c.product_cards)} product(s)")
        if c.paragraphs:
            tag_parts.append(f"{len(c.paragraphs)} para(s)")
        tags = f"  [{', '.join(tag_parts)}]" if tag_parts else "  [empty]"
        lines.append(f"{indent}• [{c.chunk_id}] {c.section_name}{tags}")
    return "\n".join(lines)
