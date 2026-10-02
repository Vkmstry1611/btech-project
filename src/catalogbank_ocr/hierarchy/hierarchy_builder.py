"""Deterministic hierarchy reconstruction for catalog pages."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from catalogbank_ocr.hierarchy.relationship_rules import bbox_top, should_attach_to_container, vertical_gap
from catalogbank_ocr.schemas.hierarchy import HierarchyDocument, HierarchyNode
from catalogbank_ocr.schemas.semantic_block import PageSemanticBlocks, SemanticBlock, SemanticDocument


@dataclass
class HierarchyBuildContext:
    """Internal state used while building a hierarchy."""

    heading_stack: List[Tuple[int, HierarchyNode]] = field(default_factory=list)
    active_container: Optional[HierarchyNode] = None


def _clean_name(text: str, fallback: str) -> str:
    stripped = (text or "").strip()
    if not stripped:
        return fallback
    stripped = stripped.lstrip("#").strip()
    if not stripped:
        return fallback
    return stripped.replace("\n", " ")


def _node_from_block(block: SemanticBlock) -> HierarchyNode:
    text = block.text.strip()
    if block.type == "heading":
        name = _clean_name(text, f"heading-{block.order}")
    elif block.type == "product_card":
        first_line = text.splitlines()[0].strip() if text else ""
        name = _clean_name(first_line or text, f"product-card-{block.order}")
    elif block.type == "table":
        name = _clean_name(text.splitlines()[0] if text else "", f"table-{block.order}")
    elif block.type == "figure":
        name = _clean_name(text or "figure", f"figure-{block.order}")
    elif block.type == "specification":
        name = _clean_name(text.splitlines()[0] if text else "", f"specification-{block.order}")
    else:
        name = _clean_name(text.splitlines()[0] if text else "", f"paragraph-{block.order}")

    return HierarchyNode(
        name=name,
        semantic_type=block.type,
        block_id=block.id,
        page=block.page,
        source_type=block.source_type,
        bbox=block.bbox,
        confidence=block.confidence,
        order=block.order,
        children=[],
        metadata={
            "source_ref": block.source_ref,
            "raw": block.raw,
            "layout_hint": block.layout_hint,
            "heading_level": block.heading_level,
        },
    )


def _push_heading(context: HierarchyBuildContext, root: HierarchyNode, heading_node: HierarchyNode, heading_level: int) -> HierarchyNode:
    while context.heading_stack and context.heading_stack[-1][0] >= heading_level:
        context.heading_stack.pop()
    parent = context.heading_stack[-1][1] if context.heading_stack else root
    parent.children.append(heading_node)
    context.heading_stack.append((heading_level, heading_node))
    context.active_container = None
    return heading_node


def _attach_to_current_heading(context: HierarchyBuildContext, root: HierarchyNode, node: HierarchyNode) -> HierarchyNode:
    parent = context.heading_stack[-1][1] if context.heading_stack else root
    parent.children.append(node)
    return parent


def build_hierarchy_from_page(page: PageSemanticBlocks, root_name: str = "document") -> HierarchyNode:
    """Build a hierarchy tree for a single page from semantic blocks."""

    root = HierarchyNode(name=root_name, semantic_type="document", page=page.page, metadata={"source_path": page.source_path})
    context = HierarchyBuildContext()

    sorted_blocks = sorted(
        page.blocks,
        key=lambda block: (
            block.order,
            bbox_top(block) if bbox_top(block) is not None else 10**9,
            block.bbox[0] if block.bbox else 10**9,
        ),
    )

    for block in sorted_blocks:
        node = _node_from_block(block)

        if block.type == "heading":
            level = block.heading_level or 2
            _push_heading(context, root, node, level)
            continue

        if block.type == "product_card":
            parent = _attach_to_current_heading(context, root, node)
            context.active_container = node
            continue

        attached = False
        if context.active_container is not None:
            # attach if spatially close enough to the active product-card/container.
            if should_attach_to_container(_semantic_block_from_node(context.active_container), block):
                context.active_container.children.append(node)
                attached = True

        if not attached:
            _attach_to_current_heading(context, root, node)

        # Avoid letting long text paragraphs capture too much after a large gap.
        if context.active_container is not None and not attached:
            gap = vertical_gap(_semantic_block_from_node(context.active_container), block)
            if gap is not None and gap > 250:
                context.active_container = None

    return root


def _semantic_block_from_node(node: HierarchyNode) -> SemanticBlock:
    return SemanticBlock(
        id=node.block_id or node.name,
        page=node.page or 0,
        type=node.semantic_type,
        source_type=node.source_type or node.semantic_type,
        text=node.name,
        bbox=node.bbox,
        confidence=node.confidence,
        order=node.order or 0,
        source_ref=node.metadata.get("source_ref", ""),
        raw=node.metadata.get("raw", {}),
        heading_level=node.metadata.get("heading_level"),
        layout_hint=node.metadata.get("layout_hint"),
    )


def build_hierarchy_document(semantic_document: SemanticDocument, source_pdf_path: Optional[Path] = None) -> HierarchyDocument:
    """Build a document-level hierarchy from a semantic document.

    The heading stack is shared across all pages so that sections spanning
    page boundaries are correctly nested under their parent heading.
    """

    root = HierarchyNode(name="document", semantic_type="document")
    # Shared context — carries heading_stack across page boundaries
    shared_context = HierarchyBuildContext()

    for page in semantic_document.pages:
        sorted_blocks = sorted(
            page.blocks,
            key=lambda block: (
                block.order,
                block.bbox[1] if block.bbox else 10**9,
                block.bbox[0] if block.bbox else 10**9,
            ),
        )
        for block in sorted_blocks:
            node = _node_from_block(block)

            if block.type == "heading":
                level = block.heading_level or 2
                _push_heading(shared_context, root, node, level)
                continue

            if block.type == "product_card":
                _attach_to_current_heading(shared_context, root, node)
                shared_context.active_container = node
                continue

            attached = False
            if shared_context.active_container is not None:
                if should_attach_to_container(
                    _semantic_block_from_node(shared_context.active_container), block
                ):
                    shared_context.active_container.children.append(node)
                    attached = True

            if not attached:
                _attach_to_current_heading(shared_context, root, node)

            if shared_context.active_container is not None and not attached:
                gap = vertical_gap(_semantic_block_from_node(shared_context.active_container), block)
                if gap is not None and gap > 250:
                    shared_context.active_container = None

    return HierarchyDocument(
        root=root,
        source_json_path=semantic_document.source_json_path,
        source_pdf_path=source_pdf_path,
        metadata={"pages": len(semantic_document.pages)},
    )
