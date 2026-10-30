"""Visualize semantic blocks, reconstructed hierarchies, and knowledge graphs."""

from __future__ import annotations

import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw, ImageFont

from catalogbank_ocr.schemas.hierarchy import HierarchyNode
from catalogbank_ocr.schemas.semantic_block import PageSemanticBlocks, SemanticBlock

logger = logging.getLogger(__name__)

SEMANTIC_COLORS = {
    "heading": (255, 87, 34),
    "table": (33, 150, 243),
    "figure": (156, 39, 176),
    "product_card": (76, 175, 80),
    "specification": (255, 193, 7),
    "paragraph": (96, 125, 139),
    "document": (0, 0, 0),
}


def _load_font(size: int = 14):
    try:
        return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Unicode.ttf", size)
    except Exception:
        try:
            return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", size)
        except Exception:
            return ImageFont.load_default()


def _bbox_is_valid(bbox) -> bool:
    return isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(v, (int, float)) for v in bbox)


def visualize_semantic_blocks(
    source_image_path: Path,
    page_blocks: PageSemanticBlocks,
    output_path: Path,
    show_labels: bool = True,
) -> Path:
    """Overlay semantic blocks on the source image and save the result."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source_image_path) as image:
        canvas = image.convert("RGB")
        draw = ImageDraw.Draw(canvas)
        font = _load_font(14)

        for block in page_blocks.blocks:
            if not _bbox_is_valid(block.bbox):
                continue
            color = SEMANTIC_COLORS.get(block.type, (0, 0, 0))
            bbox = [float(value) for value in block.bbox]  # type: ignore[arg-type]
            draw.rectangle(bbox, outline=color, width=3)
            label = f"{block.type} #{block.order}"
            if show_labels and block.text:
                short_text = block.text.replace("\n", " ")[:60]
                label = f"{label}: {short_text}"
            text_bbox = draw.textbbox((0, 0), label, font=font)
            text_w = text_bbox[2] - text_bbox[0]
            text_h = text_bbox[3] - text_bbox[1]
            x1 = bbox[0]
            y1 = max(0, bbox[1] - text_h - 4)
            draw.rectangle([x1, y1, x1 + text_w + 8, y1 + text_h + 4], fill=color)
            draw.text((x1 + 4, y1 + 2), label, fill=(255, 255, 255), font=font)

        canvas.save(output_path)
    logger.info("Saved semantic block visualization to %s", output_path)
    return output_path


def _tree_metrics(root: HierarchyNode) -> Tuple[Dict[int, int], int]:
    levels = defaultdict(int)

    def walk(node: HierarchyNode, depth: int) -> None:
        levels[depth] += 1
        for child in node.children:
            walk(child, depth + 1)

    walk(root, 0)
    max_depth = max(levels.keys()) if levels else 0
    return levels, max_depth


def _assign_positions(node: HierarchyNode, depth: int, x_cursor: List[int], positions: Dict[int, Tuple[float, float]], node_order: Dict[int, HierarchyNode]) -> float:
    node_id = id(node)
    node_order[node_id] = node
    if not node.children:
        x = float(x_cursor[0])
        x_cursor[0] += 1
        positions[node_id] = (x, float(-depth))
        return x

    child_xs = []
    for child in node.children:
        child_xs.append(_assign_positions(child, depth + 1, x_cursor, positions, node_order))
    x = sum(child_xs) / float(len(child_xs))
    positions[node_id] = (x, float(-depth))
    return x


def visualize_hierarchy_tree(root: HierarchyNode, output_path: Path) -> Path:
    """Render a hierarchy tree visualization as a simple matplotlib diagram."""

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        raise RuntimeError("matplotlib is required for hierarchy visualization") from exc

    positions: Dict[int, Tuple[float, float]] = {}
    node_order: Dict[int, HierarchyNode] = {}
    _assign_positions(root, 0, [0], positions, node_order)

    # Cap figure width to avoid matplotlib's 65535-pixel limit at 200 DPI.
    # A sensible upper bound is ~40 inches (8000 px at 200 DPI).
    node_count = len(positions)
    fig_width = min(40.0, max(8.0, node_count * 0.8))
    fig, ax = plt.subplots(figsize=(fig_width, 8))
    ax.axis("off")

    def draw_node(node: HierarchyNode) -> None:
        node_id = id(node)
        x, y = positions[node_id]
        # Escape matplotlib math special characters in node labels
        label = node.name.replace("$", r"\$").replace("%", r"\%").replace("_", r"\_")
        # Truncate long labels (HTML table content etc.) to avoid rendering issues
        if len(label) > 80:
            label = label[:77] + "..."
        if node.semantic_type and node.semantic_type != "document":
            label = f"{label}\n[{node.semantic_type}]"
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            bbox=dict(boxstyle="round,pad=0.4", fc="white", ec="black", lw=1.2),
            fontsize=9,
        )
        for child in node.children:
            cx, cy = positions[id(child)]
            ax.plot([x, cx], [y - 0.1, cy + 0.1], color="black", linewidth=1)
            draw_node(child)

    draw_node(root)
    ax.relim()
    ax.autoscale_view()
    try:
        fig.tight_layout()
    except Exception:
        pass  # tight_layout can fail with unusual glyphs — safe to skip
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved hierarchy visualization to %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Knowledge graph visualization
# ---------------------------------------------------------------------------

# Node type → color mapping for the KG diagram
_KG_NODE_COLORS = {
    "Document":  "#4A90D9",   # blue
    "Section":   "#7B68EE",   # medium slate blue
    "Product":   "#2ECC71",   # green
    "Attribute": "#F39C12",   # orange
    "Category":  "#E74C3C",   # red
}

_KG_EDGE_COLORS = {
    "HAS_SECTION":   "#7B68EE",
    "CONTAINS":      "#2ECC71",
    "HAS_ATTRIBUTE": "#F39C12",
    "BELONGS_TO":    "#E74C3C",
    "IS_VARIANT_OF": "#95A5A6",
    "HAS_MATERIAL":  "#1ABC9C",
}


def visualize_knowledge_graph(
    normalized_result: Any,
    output_path: Path,
    doc_stem: str = "",
    max_nodes: int = 80,
) -> Path:
    """Render a knowledge graph from a NormalizedResult as a matplotlib PNG.

    Nodes: Document, Section (from chunk IDs), Product, Category, Attribute.
    Edges: HAS_SECTION, CONTAINS, HAS_ATTRIBUTE, BELONGS_TO, IS_VARIANT_OF.

    Args:
        normalized_result: A NormalizedResult from normalization.py.
        output_path: Where to save the PNG.
        doc_stem: Document name shown as the Document node label.
        max_nodes: Cap total nodes to keep the diagram readable.

    Returns:
        Path to the saved PNG.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        import networkx as nx
    except ImportError as exc:
        raise RuntimeError("matplotlib and networkx are required for KG visualization") from exc

    G = nx.DiGraph()

    # ---- Document node -----------------------------------------------
    doc_label = doc_stem or "Document"
    doc_id = f"doc::{doc_label}"
    G.add_node(doc_id, label=doc_label[:30], node_type="Document")

    # ---- Section nodes (from chunk IDs) ------------------------------
    seen_sections: set = set()
    for chunk_id in (normalized_result.source_chunk_ids or []):
        parts = chunk_id.split("__")
        page_str = parts[1] if len(parts) >= 2 else "p?"
        sec_label = f"p{page_str.lstrip('p')}"
        sec_id = f"sec::{chunk_id}"
        if sec_id not in seen_sections:
            seen_sections.add(sec_id)
            G.add_node(sec_id, label=sec_label, node_type="Section")
            G.add_edge(doc_id, sec_id, rel="HAS_SECTION")

    # ---- Product nodes -----------------------------------------------
    prod_id_map: dict = {}
    for p in (normalized_result.products or []):
        name = (p.name or "Product")[:25]
        pid = f"prod::{p.name}::{p.sku or ''}"
        prod_id_map[p.name.lower()] = pid
        G.add_node(pid, label=name, node_type="Product")

        # Link product → section via first source chunk
        if p.source_chunks:
            sec_id = f"sec::{p.source_chunks[0]}"
            if sec_id in G:
                G.add_edge(sec_id, pid, rel="CONTAINS")
            else:
                G.add_edge(doc_id, pid, rel="CONTAINS")
        else:
            G.add_edge(doc_id, pid, rel="CONTAINS")

        # Category node
        if p.category:
            cat_id = f"cat::{p.category}"
            if cat_id not in G:
                G.add_node(cat_id, label=p.category[:20], node_type="Category")
            G.add_edge(pid, cat_id, rel="BELONGS_TO")

    # ---- Attribute nodes (limit to keep diagram readable) ------------
    attr_count = 0
    for a in (normalized_result.attributes or []):
        if attr_count >= max_nodes // 3:
            break
        key_label = f"{a.key[:15]}: {a.value[:10]}"
        attr_id = f"attr::{a.entity}::{a.key}"
        G.add_node(attr_id, label=key_label, node_type="Attribute")

        entity_prod_id = prod_id_map.get(a.entity.lower())
        if entity_prod_id:
            G.add_edge(entity_prod_id, attr_id, rel="HAS_ATTRIBUTE")
        attr_count += 1

    # ---- Variant relations -------------------------------------------
    for r in (normalized_result.relations or []):
        if r.predicate.upper() == "IS_VARIANT_OF":
            subj_id = prod_id_map.get(r.subject.lower())
            obj_id  = prod_id_map.get(r.object.lower())
            if subj_id and obj_id and subj_id != obj_id:
                G.add_edge(subj_id, obj_id, rel="IS_VARIANT_OF")

    # ---- Trim if too large -------------------------------------------
    if len(G.nodes) > max_nodes:
        # Keep document + all products + top-k attributes
        keep = {doc_id} | set(prod_id_map.values())
        for n in list(G.nodes):
            nt = G.nodes[n].get("node_type", "")
            if nt in ("Section", "Category"):
                keep.add(n)
        for n in list(G.nodes):
            if n not in keep:
                G.remove_node(n)
            if len(G.nodes) <= max_nodes:
                break

    if len(G.nodes) == 0:
        # Nothing to draw — write a placeholder
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No products or attributes extracted",
                ha="center", va="center", fontsize=14, color="gray")
        ax.axis("off")
        fig.savefig(output_path, dpi=100, bbox_inches="tight")
        plt.close(fig)
        return output_path

    # ---- Layout & draw -----------------------------------------------
    # Use spring layout with a fixed seed for reproducibility
    try:
        pos = nx.spring_layout(G, seed=42, k=2.5 / max(1, len(G.nodes) ** 0.5))
    except Exception:
        pos = nx.random_layout(G, seed=42)

    fig_size = max(10, min(24, len(G.nodes) * 0.5))
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.7))
    ax.axis("off")
    ax.set_title(f"Knowledge Graph — {doc_label}", fontsize=13, pad=12)

    # Draw edges grouped by relation type
    edge_groups: dict = {}
    for u, v, data in G.edges(data=True):
        rel = data.get("rel", "")
        edge_groups.setdefault(rel, []).append((u, v))

    for rel, edges in edge_groups.items():
        color = _KG_EDGE_COLORS.get(rel, "#AAAAAA")
        nx.draw_networkx_edges(
            G, pos, edgelist=edges, ax=ax,
            edge_color=color, arrows=True,
            arrowsize=15, arrowstyle="-|>",
            width=1.5, connectionstyle="arc3,rad=0.08",
            min_source_margin=18, min_target_margin=18,
        )

    # Draw nodes grouped by type
    node_groups: dict = {}
    for n, data in G.nodes(data=True):
        nt = data.get("node_type", "Document")
        node_groups.setdefault(nt, []).append(n)

    for node_type, nodes in node_groups.items():
        color = _KG_NODE_COLORS.get(node_type, "#CCCCCC")
        size  = {"Document": 1800, "Section": 900, "Product": 1200,
                 "Category": 1000, "Attribute": 600}.get(node_type, 800)
        nx.draw_networkx_nodes(G, pos, nodelist=nodes, ax=ax,
                               node_color=color, node_size=size, alpha=0.92)

    # Draw labels
    labels = {n: d.get("label", n.split("::")[-1][:20]) for n, d in G.nodes(data=True)}
    nx.draw_networkx_labels(G, pos, labels=labels, ax=ax,
                            font_size=7, font_color="white", font_weight="bold")

    # Legend
    legend_handles = [
        mpatches.Patch(color=color, label=nt)
        for nt, color in _KG_NODE_COLORS.items()
        if any(G.nodes[n].get("node_type") == nt for n in G.nodes)
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper left",
                  fontsize=8, framealpha=0.8, title="Node Types")

    try:
        fig.tight_layout()
    except Exception:
        pass
    fig.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved knowledge graph visualization to %s", output_path)
    return output_path
